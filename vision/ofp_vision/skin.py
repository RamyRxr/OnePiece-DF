"""What counts as skin, for the lab's stretch effect.

Only skin stretches. The room, hair, clothes and teeth never do. This module
draws the masks that say which pixels are which, and only inside a small
window around the pinch - building them over the whole frame every frame
was most of the effect's cost.

- **Face**: the hull of the face outline. Its interior counts as is; its
  outer band only where it is skin-coloured, because a convex hull takes in
  background beside the jaw and hair at the temples.
- **Ears**: an ellipse where the ear should be (MediaPipe has no ear
  landmarks), counted only where skin-coloured - stricter than the face,
  since this is where the estimate is weakest.
- **Mouth opening**: the inside of the lips. Never skin; an opened mouth
  shows its dark inside instead (see `cavity_colour`).
- **Fingers**: one finger, knuckle to tip, when a finger is grabbed; whole
  hands for the hands that stay on top.

Skin colour is the player's own, measured from their cheeks and forehead in
(Y, Cr, Cb), with brightness counted loosely - hair can share skin's hue,
but not its brightness.

When the segmentation model is available (`landmarks.SkinSegmenter`), the
masks come from its per-pixel labels instead - real edges rather than
guessed shapes (`segmented_face_masks`; a finger's in `fingers.py`). The
colour-based ones above remain the fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .grab import ear_points
from .indices import (
    FACE_CHEEK_L,
    FACE_CHEEK_R,
    FACE_EYE_OUTER_L,
    FACE_EYE_OUTER_R,
    FACE_FOREHEAD_CENTRE,
    FACE_INNER_LIPS,
    FACE_NOSE_TIP,
    FACE_OVAL,
    HAND_CONNECTIONS,
    HAND_DIGITS,
    HAND_MIDDLE_MCP,
    HAND_PALM,
    HAND_WRIST,
)
from .signals import HandObservation, Point

# Ear half-size, in interocular distances (half-width, half-height of the
# ellipse). An ear is about 0.6 interocular tall and is seen nearly edge-on
# from the front, so it is narrow.
EAR_SIZE = (0.13, 0.30)
# Skin colour match, as a squared Mahalanobis distance in (Y, Cr, Cb): full
# match inside NEAR, none beyond FAR (EAR_FAR for the ears).
COLOUR_NEAR = 12.0
COLOUR_FAR = 30.0
EAR_COLOUR_FAR = 20.0
# Shading varies across a face, so brightness is allowed this many times its
# measured variance (plus a floor).
BRIGHTNESS_LOOSENESS = 4.0
# The outer band of the face outline that must be skin-coloured, in
# interocular distances.
EDGE_BAND = 0.15
# Half a finger's thickness, in palm lengths (a finger is ~0.2 palm wide;
# a little more covers the landmarks' jitter). Beyond it is the air.
FINGER_HALF_WIDTH = 0.12
FINGER_WIDTH = 2 * FINGER_HALF_WIDTH
# Softening of mask edges, in pixels (odd).
FEATHER_PX = 7
# Colour of an opened mouth if the mouth shows nothing dark.
DEFAULT_CAVITY = (40.0, 30.0, 70.0)
# An opened mouth is never brighter than this (BT.601 luma).
CAVITY_MAX_LUMA = 70.0
LUMA = np.array([0.114, 0.587, 0.299])


@dataclass(frozen=True)
class Window:
    """A region of the frame, in pixels: x0 <= x < x1, y0 <= y < y1."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0

    @property
    def empty(self) -> bool:
        return self.x0 >= self.x1 or self.y0 >= self.y1

    def grown(self, by: int, size: tuple[int, int]) -> "Window":
        """This window grown on every side, clipped to a frame of `size`."""
        return Window(
            max(0, self.x0 - by), max(0, self.y0 - by), min(size[0], self.x1 + by), min(size[1], self.y1 + by)
        )

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y0 : self.y1, self.x0 : self.x1]


@dataclass(frozen=True)
class Masks:
    """Masks (0..1) over `window`: what may move (`skin`, feathered for soft
    edges, and `skin_solid`, unsoftened - exactly the skin pixels), the mouth
    opening, and what stays on top. `cavity` is the colour an opened mouth
    shows - the darkest part of the player's own mouth."""

    window: Window
    skin: np.ndarray
    skin_solid: np.ndarray
    mouth: np.ndarray | None = None
    front: np.ndarray | None = None
    cavity: tuple[float, float, float] = DEFAULT_CAVITY

    @classmethod
    def of(cls, window: Window, skin: np.ndarray, **others) -> "Masks":
        """From a uint8 skin mask (0..255); feathers it."""
        return cls(window, feathered(skin), skin.astype(np.float32) / 255.0, **others)


def _to_px(point: Point | tuple[float, float], size: tuple[int, int]) -> tuple[int, int]:
    return int(point[0] * size[0]), int(point[1] * size[1])


def _local(points, size: tuple[int, int], window: Window) -> np.ndarray:
    """Landmarks as int32 pixel coordinates inside `window`."""
    return np.array(
        [(int(p[0] * size[0]) - window.x0, int(p[1] * size[1]) - window.y0) for p in points], dtype=np.int32
    )


def _pt(point) -> tuple[int, int]:
    """A point OpenCV's drawing functions accept: a tuple of Python ints."""
    return int(point[0]), int(point[1])


def patch_around(image: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    """The square of `image` within `radius` of (x, y), clipped to the frame.

    Empty when the point is off-frame. (Plain slicing with a negative stop
    wraps around to the far side of the image - a forehead above the frame
    would be "sampled" from the room at the bottom.)
    """
    height, width = image.shape[:2]
    x0, x1 = max(0, x - radius), min(width, x + radius)
    y0, y1 = max(0, y - radius), min(height, y + radius)
    if x0 >= x1 or y0 >= y1:
        return image[0:0, 0:0]
    return image[y0:y1, x0:x1]


def _length_px(a: Point, b: Point, size: tuple[int, int]) -> float:
    return math.hypot((a[0] - b[0]) * size[0], (a[1] - b[1]) * size[1])


def interocular(face: list[Point], size: tuple[int, int]) -> float:
    return _length_px(face[FACE_EYE_OUTER_R], face[FACE_EYE_OUTER_L], size)


def feathered(mask: np.ndarray) -> np.ndarray:
    """A uint8 mask, edges softened, as float32 0..1."""
    return cv2.GaussianBlur(mask, (FEATHER_PX, FEATHER_PX), 0).astype(np.float32) / 255.0


# --- skin colour ------------------------------------------------------------


def skin_colour(bgr: np.ndarray, face: list[Point]) -> tuple[np.ndarray, np.ndarray] | None:
    """Mean and inverse covariance of the face's skin colour, in (Y, Cr, Cb).

    Sampled from both cheeks and the forehead, so it is the player's own
    skin tone under the room's own light. None if all three are off-frame.
    """
    size = (bgr.shape[1], bgr.shape[0])
    radius = max(2, int(0.1 * interocular(face, size)))
    samples = []
    for index in (FACE_CHEEK_R, FACE_CHEEK_L, FACE_FOREHEAD_CENTRE):
        patch = patch_around(bgr, *_to_px(face[index], size), radius)
        if patch.size:
            samples.append(cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb).reshape(-1, 3))
    if not samples or sum(len(s) for s in samples) < 3:
        return None
    data = np.concatenate(samples).astype(np.float64)
    covariance = np.cov(data.T)
    covariance[0, 0] = covariance[0, 0] * BRIGHTNESS_LOOSENESS + 100.0
    covariance += np.diag([0.0, 4.0, 4.0])
    return data.mean(axis=0), np.linalg.inv(covariance)


def colour_distance(region: np.ndarray, model: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Squared Mahalanobis distance of each pixel from the skin colour.

    `region` is any BGR image, including an (N, 1, 3) column of pixels.
    """
    mean, inverse = model
    ycc = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    d0, d1, d2 = (ycc[..., i] - np.float32(mean[i]) for i in range(3))
    a = inverse.astype(np.float32)
    return (
        a[0, 0] * d0 * d0 + a[1, 1] * d1 * d1 + a[2, 2] * d2 * d2
        + 2.0 * (a[0, 1] * d0 * d1 + a[0, 2] * d0 * d2 + a[1, 2] * d1 * d2)
    )


def colour_match(distance: np.ndarray, far: float = COLOUR_FAR) -> np.ndarray:
    """How skin-coloured each pixel is, 0..1, from its colour distance."""
    return np.clip((far - distance) / (far - COLOUR_NEAR), 0.0, 1.0)


# --- masks ------------------------------------------------------------------


def face_masks(bgr: np.ndarray, face: list[Point], window: Window) -> tuple[np.ndarray, np.ndarray]:
    """(skin, mouth opening) for a face, as uint8 masks over `window`."""
    size = (bgr.shape[1], bgr.shape[0])
    eyes_apart = interocular(face, size)
    depth = max(1, int(EDGE_BAND * eyes_apart))
    # Work over a slightly bigger window, so the band is right at its edges.
    work = window.grown(depth + FEATHER_PX, size)

    hull = np.zeros(work.shape, dtype=np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(_local([face[i] for i in FACE_OVAL], size, work)), 255)
    interior = cv2.distanceTransform(hull, cv2.DIST_L2, 3) > depth
    ears = np.zeros(work.shape, dtype=np.uint8)
    axes = (max(1, int(EAR_SIZE[0] * eyes_apart)), max(1, int(EAR_SIZE[1] * eyes_apart)))
    for centre in _local(list(ear_points(face).values()), size, work):
        cv2.ellipse(ears, _pt(centre), axes, 0, 0, 360, 255, -1)

    # Colour is only checked where it matters: the hull's outer band and the
    # ears. The hull's interior is skin as it is.
    band = (hull > 0) & ~interior
    rows, cols = np.nonzero(band | (ears > 0))
    skin = np.where(interior, hull, 0).astype(np.uint8)
    model = skin_colour(bgr, face)
    # Without a colour model (the sample spots are off-frame) the band and
    # the ears stay out: better a smaller stretch than dragging the room.
    if len(rows) and model is not None:
        distance = colour_distance(work.crop(bgr)[rows, cols][:, np.newaxis, :], model)[:, 0]
        skin[rows, cols] = np.maximum(
            hull[rows, cols] * colour_match(distance), ears[rows, cols] * colour_match(distance, EAR_COLOUR_FAR)
        ).astype(np.uint8)

    mouth = np.zeros(work.shape, dtype=np.uint8)
    cv2.fillPoly(mouth, [_local([face[i] for i in FACE_INNER_LIPS], size, work)], 255)
    skin[mouth > 0] = 0

    inner = Window(window.x0 - work.x0, window.y0 - work.y0, window.x1 - work.x0, window.y1 - work.y0)
    return inner.crop(skin), inner.crop(mouth)


def cavity_colour(bgr: np.ndarray, mouth: np.ndarray, window: Window) -> tuple[float, float, float]:
    """The colour of the darkest part of the mouth opening (its shadow)."""
    pixels = window.crop(bgr)[mouth > 0].astype(np.float64)
    if len(pixels) < 20:
        return DEFAULT_CAVITY
    luma = pixels @ LUMA
    colour = pixels[luma <= np.percentile(luma, 25)].mean(axis=0)
    scale = min(1.0, CAVITY_MAX_LUMA / max(1.0, float(colour @ LUMA)))
    return (float(colour[0] * scale), float(colour[1] * scale), float(colour[2] * scale))


def hand_mask(hands: list[HandObservation], size: tuple[int, int], window: Window) -> np.ndarray:
    """The skin of these hands over `window`: the palm plus a stroke per bone."""
    mask = np.zeros(window.shape, dtype=np.uint8)
    for hand in hands:
        lm = hand.landmarks
        width = max(3, int(FINGER_WIDTH * _length_px(lm[HAND_WRIST], lm[HAND_MIDDLE_MCP], size)))
        points = _local(lm, size, window)
        cv2.fillConvexPoly(mask, cv2.convexHull(points[list(HAND_PALM)]), 255)
        for a, b in HAND_CONNECTIONS:
            cv2.line(mask, _pt(points[a]), _pt(points[b]), 255, width)
        for digit in HAND_DIGITS.values():
            cv2.circle(mask, _pt(points[digit[3]]), width // 2, 255, -1)
    return mask


# --- masks from the segmentation model ----------------------------------------

# The segmentation model's labels.
SEGMENT_BACKGROUND, SEGMENT_HAIR, SEGMENT_BODY_SKIN, SEGMENT_FACE_SKIN, SEGMENT_CLOTHES, SEGMENT_OTHER = range(6)

def _labels_as(segmentation: np.ndarray, wanted, size: tuple[int, int], window: Window) -> np.ndarray:
    """uint8 0..255 over `window`: where the labels are any of `wanted`.

    Taken at the segmentation's own resolution, then scaled up smoothly to
    the window, so the edge is soft rather than blocky.
    """
    seg_h, seg_w = segmentation.shape[:2]
    fx, fy = seg_w / size[0], seg_h / size[1]
    sx0, sy0 = int(window.x0 * fx), int(window.y0 * fy)
    sx1, sy1 = min(seg_w, int(np.ceil(window.x1 * fx)) + 1), min(seg_h, int(np.ceil(window.y1 * fy)) + 1)
    if sx0 >= sx1 or sy0 >= sy1:
        return np.zeros(window.shape, dtype=np.uint8)
    chosen = np.isin(segmentation[sy0:sy1, sx0:sx1], wanted).astype(np.float32)
    # Place the low-resolution cells on the window's pixel grid.
    matrix = np.float32([[1.0 / fx, 0.0, sx0 / fx - window.x0], [0.0, 1.0 / fy, sy0 / fy - window.y0]])
    matrix[:, 2] += 0.5 / np.float32([fx, fy]) - 0.5
    scaled = cv2.warpAffine(chosen, matrix, (window.shape[1], window.shape[0]), flags=cv2.INTER_LINEAR)
    return (np.clip(scaled, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def segmented_face_masks(
    segmentation: np.ndarray, face: list[Point], size: tuple[int, int], window: Window
) -> tuple[np.ndarray, np.ndarray]:
    """(skin, mouth opening) for a face, from the segmentation's labels.

    Skin is face or body skin, within the head (the face outline's hull and
    the ears) - plus the beard: hair below the nose, inside the face, which
    is attached to the skin under it. Scalp hair, the room and clothes never
    are; the mouth opening is cut out.
    """
    skin = _labels_as(segmentation, (SEGMENT_FACE_SKIN, SEGMENT_BODY_SKIN), size, window)
    hair = _labels_as(segmentation, (SEGMENT_HAIR,), size, window)

    eyes_apart = interocular(face, size)
    head = np.zeros(window.shape, dtype=np.uint8)
    cv2.fillConvexPoly(head, cv2.convexHull(_local([face[i] for i in FACE_OVAL], size, window)), 255)
    face_only = head.copy()
    axes = (max(1, int(EAR_SIZE[0] * eyes_apart)), max(1, int(EAR_SIZE[1] * eyes_apart)))
    for centre in _local(list(ear_points(face).values()), size, window):
        cv2.ellipse(head, _pt(centre), axes, 0, 0, 360, 255, -1)

    nose_row = int(face[FACE_NOSE_TIP][1] * size[1]) - window.y0
    beard = hair.copy()
    beard[: max(0, min(window.shape[0], nose_row))] = 0
    skin = np.maximum(skin, np.minimum(beard, face_only))
    skin = np.minimum(skin, head)

    mouth = np.zeros(window.shape, dtype=np.uint8)
    cv2.fillPoly(mouth, [_local([face[i] for i in FACE_INNER_LIPS], size, window)], 255)
    skin[mouth > 0] = 0
    return skin, mouth


def digit_mask(hand: HandObservation, digit: str, size: tuple[int, int], window: Window) -> np.ndarray:
    """One finger's skin over `window`, knuckle to tip. Not its neighbours."""
    mask = np.zeros(window.shape, dtype=np.uint8)
    lm = hand.landmarks
    width = max(3, int(FINGER_WIDTH * _length_px(lm[HAND_WRIST], lm[HAND_MIDDLE_MCP], size)))
    joints = _local([lm[i] for i in HAND_DIGITS[digit]], size, window)
    for a, b in zip(joints, joints[1:]):
        cv2.line(mask, _pt(a), _pt(b), 255, width)
    cv2.circle(mask, _pt(joints[-1]), width // 2, 255, -1)
    return mask
