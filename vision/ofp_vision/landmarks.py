"""MediaPipe wrappers.

The only module in the package that imports mediapipe. Everything downstream
works on the plain `Frame` dataclass from `signals.py`, which is what makes
the math testable without a camera.

Note on the GPU delegate: MediaPipe's Windows wheels ship without the GPU
build flags, so `Delegate.GPU` fails at graph init with
"ImageCloneCalculator: GPU processing is disabled in build flags" for all
three models. We run on CPU (XNNPACK), which measures ~27ms/frame for
face+hand+pose combined - comfortably inside a 30fps budget.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import MODEL_FILES, REQUIRED_MODELS
from .signals import Frame, HandObservation


log = logging.getLogger(__name__)


class ModelMissingError(RuntimeError):
    """Raised when a .task model file is missing."""


# The skin segmenter takes ~90ms, far longer than a frame: it runs in the
# background (MediaPipe's live-stream mode, its own thread) on every Nth
# frame at reduced size, and each frame carries the latest result.
SEGMENT_EVERY = 3
SEGMENT_SCALE = 0.5


class SkinSegmenter:
    """MediaPipe's multi-class selfie segmenter: a label for every pixel
    (background, hair, body skin, face skin, clothes, other)."""

    def __init__(self, model_path: Path) -> None:
        from mediapipe.tasks.python import vision

        self.latest = None
        self._frames = 0
        self._model = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=_build_options(model_path, "cpu"),
                running_mode=vision.RunningMode.LIVE_STREAM,
                output_category_mask=True,
                output_confidence_masks=False,
                result_callback=self._on_result,
            )
        )

    def _on_result(self, result, image, timestamp_ms) -> None:
        import numpy as np

        labels = result.category_mask.numpy_view()
        # Copied: the result's memory is only valid inside this callback.
        self.latest = np.array(labels).reshape(labels.shape[0], labels.shape[1])

    def submit(self, bgr, timestamp_ms: int) -> None:
        """Queue every SEGMENT_EVERY-th frame; returns at once."""
        import cv2
        import mediapipe as mp
        import numpy as np

        self._frames += 1
        if self._frames % SEGMENT_EVERY:
            return
        small = cv2.resize(bgr, None, fx=SEGMENT_SCALE, fy=SEGMENT_SCALE, interpolation=cv2.INTER_AREA)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small[:, :, ::-1]))
        self._model.segment_async(image, timestamp_ms)

    def close(self) -> None:
        self._model.close()


def _build_options(model_path: Path, delegate: str):
    from mediapipe.tasks import python as mp_python

    d = (
        mp_python.BaseOptions.Delegate.GPU
        if delegate == "gpu"
        else mp_python.BaseOptions.Delegate.CPU
    )
    return mp_python.BaseOptions(model_asset_path=str(model_path), delegate=d)


class Tracker:
    """Runs the face, hand and pose models over video frames."""

    def __init__(
        self,
        models_dir: Path,
        delegate: str = "cpu",
        swap_handedness: bool = True,
        num_faces: int = 1,
        num_hands: int = 2,
    ) -> None:
        from mediapipe.tasks.python import vision

        missing = [name for name in REQUIRED_MODELS if not (models_dir / MODEL_FILES[name]).exists()]
        if missing:
            raise ModelMissingError(
                f"Missing model files for {missing} in {models_dir}. "
                "Run: python tools/fetch_models.py"
            )

        self._vision = vision
        self.swap_handedness = swap_handedness
        segment_model = models_dir / MODEL_FILES["segment"]
        self.segmenter = SkinSegmenter(segment_model) if segment_model.exists() else None
        if self.segmenter is None:
            log.info("No skin segmentation model (%s): run fetch_models for sharper skin edges", segment_model.name)
        self._last_ts = -1

        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=_build_options(models_dir / MODEL_FILES["face"], delegate),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=num_faces,
                output_face_blendshapes=True,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=_build_options(models_dir / MODEL_FILES["hand"], delegate),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=num_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=_build_options(models_dir / MODEL_FILES["pose"], delegate),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

    # --- inference ----------------------------------------------------------

    def _timestamp(self, t_ms: float) -> int:
        """Clamp a millisecond timestamp into what MediaPipe can accept.

        MediaPipe stores the frame timestamp in an int32, which caps it at
        ~24 days of milliseconds. Feeding it epoch milliseconds (~1.8e12)
        overflows, wraps negative, and silently breaks video-mode tracking:
        the models still run but report almost no detections. Timestamps also
        have to increase strictly, or the tracker throws.

        Clamping here means a caller cannot accidentally poison the tracker
        with a bad clock.
        """
        limit = 2**31 - 1
        ts = int(t_ms)
        if ts < 0:
            ts = 0
        if ts > limit:
            ts = limit
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def process(self, bgr, t_ms: float) -> Frame:
        """Run all three models on one BGR frame (and queue a segmentation)."""
        import mediapipe as mp
        import numpy as np

        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame = Frame(t_ms=t_ms)

        ts = self._timestamp(t_ms)
        face_result = self.face.detect_for_video(image, ts)
        if face_result.face_landmarks:
            frame.face = [_pt(p) for p in face_result.face_landmarks[0]]
            if face_result.face_blendshapes:
                frame.blendshapes = {
                    c.category_name: float(c.score) for c in face_result.face_blendshapes[0]
                }

        hand_result = self.hand.detect_for_video(image, ts)
        for i, landmarks in enumerate(hand_result.hand_landmarks):
            label = "Right"
            if hand_result.handedness and i < len(hand_result.handedness):
                label = hand_result.handedness[i][0].category_name
            # MediaPipe assumes the input image is already mirrored (selfie).
            # A raw webcam capture is not, so its labels are inverted.
            if self.swap_handedness:
                label = "Left" if label == "Right" else "Right"
            frame.hands.append(
                HandObservation(handedness=label, landmarks=[_pt(p) for p in landmarks])
            )

        pose_result = self.pose.detect_for_video(image, ts)
        if pose_result.pose_landmarks:
            frame.pose = [_pt(p) for p in pose_result.pose_landmarks[0]]

        if self.segmenter is not None:
            self.segmenter.submit(bgr, ts)
            frame.segmentation = self.segmenter.latest
        return frame

    def close(self) -> None:
        for model in (self.face, self.hand, self.pose, self.segmenter):
            if model is None:
                continue
            try:
                model.close()
            except Exception as exc:  # closing is best effort at shutdown
                log.debug("Closing a model failed: %s", exc)


def _pt(p) -> tuple[float, float, float]:
    return (float(p.x), float(p.y), float(p.z))
