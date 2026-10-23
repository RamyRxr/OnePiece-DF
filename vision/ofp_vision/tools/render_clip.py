"""Render a clip through the stretch, to judge the result frame by frame.

    python -m ofp_vision.tools.render_clip records/clips/clip-20261023-101500
    python -m ofp_vision.tools.render_clip records/clips/clip-20261023-101500 --every 3

Writes, inside the clip's folder:

    render/rendered.avi     camera | stretch, side by side, mirrored like the lab
    render/still-0042.jpg   every Nth frame while a grab is held, labelled with
                            what each hand holds

The clip plays with no camera and no MediaPipe: the tracking and the skin
segmentation are the ones recorded live. So a change to the stretch can be
judged on exactly the same real footage, before and after.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..clips import read_clip
from ..config import Config
from ..grab import GrabState, TwoHandGrabber
from ..server import make_effect

STILL_QUALITY = 92


@dataclass(frozen=True)
class RenderReport:
    frames: int
    # Frames on which at least one hand held a grab.
    held: int
    video: Path
    still_paths: tuple[Path, ...]

    @property
    def stills(self) -> int:
        return len(self.still_paths)


def render(folder: Path, every: int = 6, config: Config | None = None) -> RenderReport:
    """Run a clip's frames through the grab and the stretch, as the lab does."""
    config = config or Config()
    grabber = TwoHandGrabber(config.grab, mirror=config.capture.mirror)
    effect = make_effect()
    out = folder / "render"
    out.mkdir(exist_ok=True)
    for stale in out.glob("still-*.jpg"):
        stale.unlink()
    writer = None
    frames = held = 0
    stills: list[Path] = []
    try:
        for bgr, frame in read_clip(folder):
            size = (bgr.shape[1], bgr.shape[0])
            grabber.set_aspect(size[0] / size[1])
            frame = grabber.relabel(frame)
            grabs = grabber.update(frame, frame.t_ms)
            effect.update(grabs, frame, size)
            pair = _side_by_side(bgr, effect.warp(bgr), config.capture.mirror)
            if writer is None:
                writer = _open_writer(out / "rendered.avi", (pair.shape[1], pair.shape[0]))
            writer.write(pair)
            holding = {hand: grab for hand, grab in grabs.items() if grab.active}
            if holding:
                held += 1
                if (held - 1) % every == 0:
                    stills.append(_still(out, frames, pair, holding))
            frames += 1
    finally:
        if writer is not None:
            writer.release()
    return RenderReport(frames, held, out / "rendered.avi", tuple(stills))


def _side_by_side(camera: np.ndarray, stretched: np.ndarray, mirror: bool) -> np.ndarray:
    if mirror:
        camera, stretched = cv2.flip(camera, 1), cv2.flip(stretched, 1)
    return np.hstack([camera, stretched])


def _open_writer(path: Path, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.CAP_OPENCV_MJPEG, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, size)
    if not writer.isOpened():
        raise OSError(f"cannot write {path}")
    return writer


def _still(out: Path, index: int, pair: np.ndarray, holding: dict[str, GrabState]) -> Path:
    """One frame as a picture, labelled with what each hand holds."""
    labelled = pair.copy()
    text = f"frame {index}   " + "   ".join(f"{hand}: {g.target} {g.stretch:.2f}" for hand, g in holding.items())
    for colour, thickness in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(labelled, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, thickness, cv2.LINE_AA)
    path = out / f"still-{index:04d}.jpg"
    cv2.imwrite(str(path), labelled, [cv2.IMWRITE_JPEG_QUALITY, STILL_QUALITY])
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a clip through the stretch")
    parser.add_argument("clip", type=Path, help="a clip folder (records/clips/clip-...)")
    parser.add_argument("--every", type=int, default=6, help="a still every N held frames")
    args = parser.parse_args()
    if not (args.clip / "frames.jsonl").exists():
        print(f"not a clip folder: {args.clip}", file=sys.stderr)
        return 1
    report = render(args.clip, every=max(1, args.every))
    print(f"{report.frames} frames, a grab held on {report.held}")
    print(f"video:  {report.video}")
    print(f"stills: {report.stills} in {report.video.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
