"""Camera probe: is the source real, and is the picture good enough?

Checks the four things that actually break a camera session, in the order
they break it:

  1. the source opens at all;
  2. it delivers *new* frames - a virtual/stale camera returns one frozen
     image forever and every channel then reads zero forever;
  3. the frame is exposed well enough to see a face in;
  4. the frame is sharp enough to be worth tracking.

Works on a local device or on a phone's network stream:

  python -m ofp_vision.tools.probe_camera
  python -m ofp_vision.tools.probe_camera --url http://192.168.1.42:8080/video
  python -m ofp_vision.tools.probe_camera --device "Integrated Camera"

It writes one still so the picture can be judged by eye as well.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ..capture import CameraSource
from ..config import RECORDS_DIR, CaptureConfig

# Measurements are taken on a 640-wide copy so the numbers mean the same
# thing whether the source is 480p or 1080p.
MEASURE_WIDTH = 640


def _small(gray: np.ndarray) -> np.ndarray:
    if gray.shape[1] <= MEASURE_WIDTH:
        return gray
    height = max(1, round(gray.shape[0] * MEASURE_WIDTH / gray.shape[1]))
    return cv2.resize(gray, (MEASURE_WIDTH, height), interpolation=cv2.INTER_AREA)


def _sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: higher means more edge detail.

    Sensor noise inflates this, so a noisy dark frame can score like a
    crisp one - read it together with the brightness line.
    """
    return float(cv2.Laplacian(_small(gray), cv2.CV_64F).var())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probe-camera", description="camera quality and liveness probe")
    parser.add_argument("--url", type=str, default=None, help="network camera URL (phone MJPEG/RTSP stream)")
    parser.add_argument("--device", type=str, default=None, help="local camera name substring")
    parser.add_argument("--frames", type=int, default=60, help="how many frames to sample (default 60)")
    parser.add_argument(
        "--still",
        type=str,
        default=None,
        help=f"where to save one frame (default: {RECORDS_DIR / 'probe-still.jpg'})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = CaptureConfig()
    if args.device:
        config.device_name = args.device
    if args.url:
        config.source_url = args.url

    camera = CameraSource(config)
    if not camera.open():
        print("RESULT: could not open the camera (see the log line above)")
        return 1
    if camera.opened is not None:
        print(f"opened: {camera.opened.name} via {camera.opened.backend}")

    sharpness: list[float] = []
    brightness: list[float] = []
    motion: list[float] = []
    clipped_low = 0
    clipped_high = 0
    pixels = 0
    duplicates = 0
    previous_gray: np.ndarray | None = None
    previous_raw: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    width = height = 0

    started = time.perf_counter()
    read = 0
    try:
        for _ in range(max(args.frames, 1)):
            ok, bgr = camera.read()
            if not ok or bgr is None:
                print(f"RESULT: the stream stopped after {read} frames")
                return 2
            read += 1
            last_frame = bgr
            height, width = bgr.shape[:2]

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            sharpness.append(_sharpness(gray))
            brightness.append(float(gray.mean()))
            clipped_low += int(np.count_nonzero(gray < 8))
            clipped_high += int(np.count_nonzero(gray > 247))
            pixels += gray.size

            if previous_gray is not None:
                # Mean absolute difference: zero means the source is frozen.
                motion.append(float(cv2.absdiff(previous_gray, _small(gray)).mean()))
            if previous_raw is not None and np.array_equal(previous_raw, bgr):
                duplicates += 1
            previous_gray = _small(gray)
            previous_raw = bgr
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        camera.release()

    if last_frame is None:
        print("RESULT: no frames")
        return 2

    elapsed = max(time.perf_counter() - started, 1e-6)
    measured_fps = read / elapsed
    mean_sharp = float(np.mean(sharpness))
    mean_bright = float(np.mean(brightness))
    mean_motion = float(np.mean(motion)) if motion else 0.0
    dark = 100.0 * clipped_low / max(pixels, 1)
    blown = 100.0 * clipped_high / max(pixels, 1)

    still = Path(args.still) if args.still else RECORDS_DIR / "probe-still.jpg"
    still.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(still), last_frame)

    print()
    print(f"  frames            {read}")
    print(f"  measured fps      {measured_fps:.1f}")
    print(f"  frame size        {width}x{height}")
    print(f"  brightness        {mean_bright:.0f} / 255   (good: 80-180)")
    print(f"  crushed / blown   {dark:.1f}% / {blown:.1f}%")
    print(f"  sharpness         {mean_sharp:.0f}         (higher = crisper; > 40 is healthy)")
    print(f"  motion            {mean_motion:.2f}       (0 = frozen source)")
    print(f"  identical frames  {duplicates}")
    print(f"  still             {still}")
    print()

    if duplicates > read * 0.9 or mean_motion < 0.05:
        print("RESULT: the source is FROZEN - it is returning the same frame.")
        print("        Start the phone app / virtual camera first, then re-run.")
        return 2
    if measured_fps < 10:
        print("RESULT: too few frames arrive - the tracking will stutter.")
        return 2
    if mean_bright < 50:
        print("RESULT: too DARK - add light on your face, or lower gain/exposure.")
        return 3
    if mean_bright > 210 or blown > 15:
        print("RESULT: over-exposed - light is behind you, or exposure is too high.")
        return 3
    if mean_sharp < 25:
        print("RESULT: SOFT - clean the lens, check focus, and prefer the back camera.")
        return 4

    print("RESULT: OK - the source is live and usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
