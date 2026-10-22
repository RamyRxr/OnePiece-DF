"""Quick liveness probe: how many faces / hands / poses are in the camera right now.

Run:  python -m ofp_vision.tools.probe [frames]
"""

from __future__ import annotations

import sys
import time

from ..capture import CameraSource
from ..config import MODELS_DIR, CaptureConfig
from ..landmarks import Tracker


def main() -> int:
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cam = CameraSource(CaptureConfig())
    if not cam.open():
        print("CAMERA FAILED TO OPEN")
        return 1
    print("opened:", cam.opened)

    tracker = Tracker(MODELS_DIR, delegate="cpu", swap_handedness=True)
    counts = {"face": 0, "hand": 0, "pose": 0}
    t0 = time.perf_counter()
    # MediaPipe takes the timestamp in *milliseconds* and stores it in an int32.
    # Epoch milliseconds (~1.8e12) overflow that, wrap negative, and break
    # video-mode tracking - the models then report almost no detections. Use a
    # clock relative to process start, exactly like the server does.
    clock = time.perf_counter()
    try:
        for i in range(frames):
            ok, bgr = cam.read()
            if not ok or bgr is None:
                print("read failed at frame", i)
                break
            t_ms = (time.perf_counter() - clock) * 1000.0
            frame = tracker.process(bgr, t_ms)
            if frame.face:
                counts["face"] += 1
            counts["hand"] += len(frame.hands)
            if frame.pose:
                counts["pose"] += 1
    finally:
        tracker.close()
        cam.release()

    dt = time.perf_counter() - t0
    print(f"frames={frames} elapsed={dt:.1f}s fps={frames / max(dt, 1e-6):.1f}")
    print(f"face_frames={counts['face']} hand_detections={counts['hand']} pose_frames={counts['pose']}")
    if counts["face"] == 0 and counts["pose"] == 0:
        print("RESULT: nothing detected - is anyone in frame?")
        return 2
    print("RESULT: something detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
