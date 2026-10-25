"""The picture behind the hands, remembered - so a part pulled away can leave it.

When a finger (or an ear) is pulled off, the spot where it was has to show
what is behind it, or the real one stays in the picture next to its stretched
self. The webcam does not move, and everything but the hands changes slowly,
so everything but the hands is remembered, frame by frame: the room, the
shirt, the face. Where a finger is now, the picture from before it got there
is still remembered. Hands are never learned: they move fast, and they are
what gets pulled away.

Where nothing was ever seen behind it (a hand that has not moved since the
start), a hole is filled from the colours around it instead (`filled`) - and
given the picture's own grain (`grain`), or it reads as a flat cut-out.
"""

from __future__ import annotations

import cv2
import numpy as np

from .skin import Window

# Each new look moves a remembered colour this far: a change is learned in a
# few frames.
ROOM_LEARNING = 0.3
# Seen this much (0..1, two looks) and the remembered colour is trusted.
ROOM_KNOWN = 0.5
# Remembered at this share of the camera's size: smooth anyway, and cheap.
ROOM_SCALE = 0.5


class RoomMemory:
    """The picture behind the hands, and how sure it is of each pixel."""

    def __init__(self) -> None:
        self._room: np.ndarray | None = None
        self._seen: np.ndarray | None = None
        self._frame_size: tuple[int, int] | None = None

    def update(self, bgr: np.ndarray, exclude: np.ndarray | None) -> None:
        """Learn a camera frame - all of it but `exclude` (uint8, nonzero over
        the hands; any size, it is scaled to the memory's)."""
        h, w = bgr.shape[:2]
        size = (max(1, round(w * ROOM_SCALE)), max(1, round(h * ROOM_SCALE)))
        if self._room is None or self._frame_size != (w, h):
            self._room = np.zeros((size[1], size[0], 3), dtype=np.float32)
            self._seen = np.zeros((size[1], size[0]), dtype=np.float32)
            self._frame_size = (w, h)
        small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        learn = np.full((size[1], size[0]), 255, dtype=np.uint8)
        if exclude is not None:
            learn[cv2.resize(exclude, size, interpolation=cv2.INTER_NEAREST) > 0] = 0
        # Running averages from zero, divided by how much was seen (`fill`):
        # unbiased from the first look.
        cv2.accumulateWeighted(small, self._room, ROOM_LEARNING, mask=learn)
        cv2.accumulateWeighted(np.ones_like(self._seen), self._seen, ROOM_LEARNING, mask=learn)

    def fill(self, window: Window) -> tuple[np.ndarray, np.ndarray]:
        """(remembered colours, how far each is trusted 0..1) over `window`,
        a region of the camera frame."""
        h, w = window.shape
        if self._room is None or self._seen is None:
            return np.zeros((h, w, 3), dtype=np.float32), np.zeros((h, w), dtype=np.float32)
        # Memory pixel i's centre is at camera pixel (i + 0.5) / scale - 0.5.
        to_window = np.float32([
            [1.0 / ROOM_SCALE, 0.0, 0.5 / ROOM_SCALE - 0.5 - window.x0],
            [0.0, 1.0 / ROOM_SCALE, 0.5 / ROOM_SCALE - 0.5 - window.y0],
        ])
        size = (w, h)
        colours = cv2.warpAffine(self._room, to_window, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # Both read past the edge the same way, or the frame's outer pixels
        # come back too bright (colour kept, how much it was seen halved).
        seen = cv2.warpAffine(self._seen, to_window, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        colours = colours / np.maximum(seen, 1e-6)[..., None]
        return colours, np.clip(seen / ROOM_KNOWN, 0.0, 1.0)


def grain(image: np.ndarray, where: np.ndarray) -> float:
    """How grainy `image` (float32) is where `where` is set: the spread of
    its fine detail, in levels. Robust (from the median): a few sharp edges
    in the picture are not grain."""
    detail = image - cv2.GaussianBlur(image, (5, 5), 0)
    picked = detail[where]
    return 1.4826 * float(np.median(np.abs(picked))) if picked.size else 0.0


def filled(image: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """`image` (float32, HxWx3) with `hole` (0..1, HxW) filled in from the
    colours around it: they are averaged over wider and wider areas until
    every hole pixel has some (a pull-push fill). Outside the hole, unchanged."""
    keep = (1.0 - np.clip(hole, 0.0, 1.0)).astype(np.float32)
    levels = [(image * keep[..., None], keep)]
    while min(levels[-1][1].shape) > 4:
        colour, weight = levels[-1]
        levels.append((cv2.pyrDown(colour), cv2.pyrDown(weight)))
    colour, weight = levels[-1]
    # Where even the coarsest level saw nothing (a tiny picture), the average
    # of everything that is not hole.
    seen = float(keep.sum())
    mean = (image * keep[..., None]).reshape(-1, 3).sum(axis=0) / seen if seen > 0 else np.zeros(3)
    guess = np.where((weight > 1e-6)[..., None], colour / np.maximum(weight, 1e-6)[..., None], mean)
    for colour, weight in reversed(levels[:-1]):
        up = cv2.pyrUp(guess, dstsize=(weight.shape[1], weight.shape[0]))
        own = colour / np.maximum(weight, 1e-6)[..., None]
        trust = np.clip(2.0 * weight, 0.0, 1.0)[..., None]
        guess = own * trust + up * (1.0 - trust)
    return image * keep[..., None] + guess * (1.0 - keep[..., None])
