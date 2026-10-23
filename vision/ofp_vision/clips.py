"""Clips: short recordings of the real camera, for tuning the stretch on them.

Press v in the lab to start one and v again to stop it. A clip is a folder:

    video.avi          the raw camera pictures (MJPG, high quality)
    frames.jsonl       each frame's tracking (face, hands, pose, time) and
                       which segmentation it had: an index, or null
    segmentation.npz   the skin segmentations, exactly as the lab had them

`read_clip` plays one back - pictures and tracking together - with no camera
and no MediaPipe, so the stretch can be re-run on the same real footage as
often as it takes (`tools/render_clip.py`).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .signals import Frame, HandObservation

log = logging.getLogger(__name__)

# A forgotten recording stops itself here (~5MB a second).
CLIP_MAX_SECONDS = 30.0
JPEG_QUALITY = 95
VIDEO_FILE = "video.avi"
FRAMES_FILE = "frames.jsonl"
SEGMENTATION_FILE = "segmentation.npz"


def frame_payload(frame: Frame) -> dict:
    """A frame's tracking, as JSON-ready data."""
    return {
        "t": round(frame.t_ms, 2),
        "face": [[round(v, 5) for v in p] for p in frame.face] if frame.face else None,
        "bs": {k: round(v, 4) for k, v in frame.blendshapes.items()},
        "hands": [
            {"h": h.handedness, "lm": [[round(v, 5) for v in p] for p in h.landmarks]}
            for h in frame.hands
        ],
        "pose": [[round(v, 5) for v in p] for p in frame.pose] if frame.pose else None,
    }


def frame_from_payload(data: dict) -> Frame:
    """The inverse of `frame_payload`."""
    hands = [
        HandObservation(handedness=h["h"], landmarks=[tuple(p) for p in h["lm"]])
        for h in data.get("hands", [])
    ]
    return Frame(
        t_ms=float(data["t"]),
        face=[tuple(p) for p in data["face"]] if data.get("face") else None,
        blendshapes=data.get("bs", {}) or {},
        hands=hands,
        pose=[tuple(p) for p in data["pose"]] if data.get("pose") else None,
    )


def new_clip_folder(root: Path) -> Path:
    """A fresh folder name for a clip, named by the time it starts."""
    return root / time.strftime("clip-%Y%m%d-%H%M%S")


class ClipRecorder:
    """Writes one clip (see the module docstring)."""

    def __init__(
        self, folder: Path, size: tuple[int, int], fps: float, max_seconds: float = CLIP_MAX_SECONDS
    ) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.folder = folder
        self._size = size
        self._video = cv2.VideoWriter(
            str(folder / VIDEO_FILE), cv2.CAP_OPENCV_MJPEG, cv2.VideoWriter_fourcc(*"MJPG"), fps, size
        )
        if not self._video.isOpened():
            raise OSError(f"cannot write a video into {folder}")
        self._video.set(cv2.VIDEOWRITER_PROP_QUALITY, JPEG_QUALITY)
        try:
            self._frames = (folder / FRAMES_FILE).open("w", encoding="utf-8")
        except OSError:
            self._video.release()
            raise
        self._limit = max(1, int(round(max_seconds * fps)))
        self._segmentations: list[np.ndarray] = []
        self._last_segmentation: np.ndarray | None = None
        self.count = 0

    @property
    def full(self) -> bool:
        return self.count >= self._limit

    def write(self, bgr: np.ndarray, frame: Frame) -> None:
        """Add one frame - the raw camera picture and its tracking. Frames
        past the length limit are dropped."""
        if self.full:
            return
        if (bgr.shape[1], bgr.shape[0]) != self._size:
            raise ValueError(f"frame is {bgr.shape[1]}x{bgr.shape[0]}, the clip is {self._size[0]}x{self._size[1]}")
        self._video.write(bgr)
        payload = {**frame_payload(frame), "seg": self._segmentation_index(frame.segmentation)}
        self._frames.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.count += 1

    def _segmentation_index(self, labels: np.ndarray | None) -> int | None:
        """Each new segmentation is stored once; frames refer to it."""
        if labels is None:
            return None
        if labels is not self._last_segmentation:
            self._segmentations.append(np.array(labels, dtype=np.uint8))
            self._last_segmentation = labels
        return len(self._segmentations) - 1

    def close(self) -> None:
        self._video.release()
        self._frames.close()
        if self._segmentations:
            np.savez_compressed(self.folder / SEGMENTATION_FILE, labels=np.stack(self._segmentations))


class ClipSwitch:
    """The lab's v key: start a clip, stop it - or it stops itself when full."""

    def __init__(self, root: Path, fps: float, max_seconds: float = CLIP_MAX_SECONDS) -> None:
        self._root = root
        self._fps = fps
        self._max_seconds = max_seconds
        self._recorder: ClipRecorder | None = None

    def toggle(self, size: tuple[int, int]) -> None:
        if self._recorder is not None:
            self.stop()
            return
        folder = new_clip_folder(self._root)
        try:
            self._recorder = ClipRecorder(folder, size, self._fps, self._max_seconds)
        except OSError:
            log.exception("Could not start a clip in %s", folder)
            return
        log.info("Recording a clip to %s (v to stop; it stops itself at %.0fs)", folder, self._max_seconds)

    def write(self, bgr: np.ndarray, frame: Frame) -> None:
        """Add the raw camera picture and its tracking, if recording."""
        if self._recorder is None:
            return
        try:
            self._recorder.write(bgr, frame)
        except (OSError, ValueError):
            log.exception("The clip could not take this frame; stopping it")
            self.stop()
            return
        if self._recorder.full:
            log.info("The clip reached %.0fs", self._max_seconds)
            self.stop()

    def stop(self) -> None:
        if self._recorder is None:
            return
        recorder, self._recorder = self._recorder, None
        try:
            recorder.close()
        except OSError:
            log.exception("The clip in %s could not be saved", recorder.folder)
            return
        log.info(
            "Saved a %d-frame clip. Render it with: python -m ofp_vision.tools.render_clip %s",
            recorder.count,
            recorder.folder,
        )

    def status(self) -> str | None:
        """A line for the lab while recording; None otherwise."""
        if self._recorder is None:
            return None
        return f"REC {self._recorder.count / self._fps:4.1f}s of {self._max_seconds:.0f}s - v to stop"


def read_clip(folder: Path) -> Iterator[tuple[np.ndarray, Frame]]:
    """Play a clip back: each frame's camera picture and tracking, in order."""
    segmentation_path = folder / SEGMENTATION_FILE
    labels = np.load(segmentation_path)["labels"] if segmentation_path.exists() else None
    video = cv2.VideoCapture(str(folder / VIDEO_FILE))
    if not video.isOpened():
        raise OSError(f"cannot read the video in {folder}")
    # Frames that shared a segmentation live share one array again: the
    # room memory learns once per segmentation, as it does live.
    current, current_index = None, None
    try:
        with (folder / FRAMES_FILE).open("r", encoding="utf-8") as lines:
            for line in lines:
                if not line.strip():
                    continue
                ok, bgr = video.read()
                if not ok:
                    return  # the video was cut short: play what there is
                data = json.loads(line)
                frame = frame_from_payload(data)
                index = data.get("seg")
                if index is not None and labels is not None:
                    if index != current_index:
                        current, current_index = labels[index], index
                    frame.segmentation = current
                yield bgr, frame
    finally:
        video.release()
