"""The room behind the player, remembered - so a stretched part can leave it.

When a finger or an ear is pulled away, the spot where it was has to show
what is behind it, or the real one stays in the picture next to its stretched
copy. The webcam does not move, so the room behind the player can be
remembered: every pixel the segmentation calls background is blended into a
picture of the room - slowly, so a passing shadow does not stick.

Where the room was never seen (behind the body, which is always there), a
hole is filled from the colours around it instead (`filled`).
"""

from __future__ import annotations

import cv2
import numpy as np

from .skin import SEGMENT_BACKGROUND, Window

# Each new look at a background pixel moves its remembered colour this far.
ROOM_LEARNING = 0.1
# Seen this much (0..1, ~7 looks) and the remembered colour is trusted.
ROOM_KNOWN = 0.5
# Background this close to the body is not learned: the labels are coarse,
# and a few frames late.
SAFETY_PX = 12


class RoomMemory:
    """A slowly learned picture of the room, and how sure it is of each pixel."""

    def __init__(self) -> None:
        self._room: np.ndarray | None = None
        self._seen: np.ndarray | None = None
        self._last_labels: np.ndarray | None = None

    def is_new(self, labels: np.ndarray | None) -> bool:
        """Whether `labels` is a segmentation not yet learned from."""
        return labels is not None and labels is not self._last_labels

    def update(self, bgr: np.ndarray, labels: np.ndarray | None, exclude: np.ndarray | None = None) -> None:
        """Learn from a camera frame and its segmentation - once per new
        segmentation (the frames in between carry the old one). `exclude`
        (uint8, frame-sized) marks pixels not to learn, such as the hands."""
        if not self.is_new(labels):
            return
        self._last_labels = labels
        h, w = bgr.shape[:2]
        if self._room is None or self._room.shape[:2] != (h, w):
            self._room = np.zeros((h, w, 3), dtype=np.float32)
            self._seen = np.zeros((h, w), dtype=np.float32)
        background = cv2.resize(
            (labels == SEGMENT_BACKGROUND).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
        reach = 2 * SAFETY_PX + 1
        background = cv2.erode(background, np.ones((reach, reach), dtype=np.uint8))
        if exclude is not None:
            background[exclude > 0] = 0
        # Running averages from zero, divided by how much was seen (`fill`):
        # unbiased from the first look.
        cv2.accumulateWeighted(bgr.astype(np.float32), self._room, ROOM_LEARNING, mask=background)
        cv2.accumulateWeighted(np.ones((h, w), dtype=np.float32), self._seen, ROOM_LEARNING, mask=background)

    def fill(self, window: Window) -> tuple[np.ndarray, np.ndarray]:
        """(room colours, how far each is trusted 0..1) over `window`."""
        room, seen = self._room, self._seen
        if room is None or seen is None or window.y1 > room.shape[0] or window.x1 > room.shape[1]:
            return np.zeros((*window.shape, 3), dtype=np.float32), np.zeros(window.shape, dtype=np.float32)
        seen = window.crop(seen)
        colours = window.crop(room) / np.maximum(seen, 1e-6)[..., None]
        return colours, np.clip(seen / ROOM_KNOWN, 0.0, 1.0)


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
