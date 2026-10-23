"""Tests for skin masks built from the segmentation model's labels.

The model labels every pixel: background, hair, body skin, face skin,
clothes, other. These masks turn that into "what may stretch":

- a face grab: face skin - and the beard (hair below the nose, inside the
  face), which is attached to skin - but never scalp hair, the room, or the
  teeth. (A finger's own skin is tested with the finger: test_fingers.py.)

The segmentation here is synthetic, at half the frame's resolution, like
the real one.
"""

from __future__ import annotations

import cv2
import numpy as np
from face_fixture import H, RADIUS, W, landmarks, norm, px

from ofp_vision.skin import (
    SEGMENT_FACE_SKIN,
    SEGMENT_HAIR,
    Window,
    segmented_face_masks,
)

FACE = landmarks()
FULL = Window(0, 0, W, H)
SCALE = 0.5


def half(point) -> tuple[int, int]:
    x, y = px(point)
    return int(x * SCALE), int(y * SCALE)


def face_segmentation() -> np.ndarray:
    labels = np.zeros((int(H * SCALE), int(W * SCALE)), dtype=np.uint8)
    cv2.circle(labels, half(norm(0.0, 0.0)), int(RADIUS * H * SCALE), SEGMENT_FACE_SKIN, -1)
    # Hair over the forehead, and a beard over the chin and jaw.
    cv2.rectangle(labels, half(norm(-0.12, -0.2)), half(norm(0.12, -0.12)), SEGMENT_HAIR, -1)
    cv2.rectangle(labels, half(norm(-0.12, 0.13)), half(norm(0.12, 0.19)), SEGMENT_HAIR, -1)
    return labels


def at(mask: np.ndarray, point) -> int:
    x, y = px(point)
    return int(mask[y, x])


def test_face_skin_is_skin_and_the_room_beside_it_is_not():
    skin, _ = segmented_face_masks(face_segmentation(), FACE, (W, H), FULL)
    assert at(skin, norm(0.1, 0.04)) > 200
    assert at(skin, norm(RADIUS + 0.02, 0.0)) == 0


def test_scalp_hair_is_not_skin_but_a_beard_is():
    skin, _ = segmented_face_masks(face_segmentation(), FACE, (W, H), FULL)
    assert at(skin, norm(0.0, -0.16)) == 0, "hair above the eyes stays hair"
    assert at(skin, norm(0.0, 0.16)) > 200, "the beard stretches with the skin under it"


def test_the_mouth_opening_is_never_skin():
    skin, mouth = segmented_face_masks(face_segmentation(), FACE, (W, H), FULL)
    assert at(skin, norm(0.0, 0.09)) == 0
    assert at(mouth, norm(0.0, 0.09)) == 255


def test_the_skin_edge_follows_the_segmentation_not_a_guess():
    """Shrink the face in the labels: the mask shrinks with it, although the
    landmarks (and so the old outline guess) have not changed."""
    labels = face_segmentation()
    cv2.circle(labels, half(norm(RADIUS * 0.8, 0.0)), int(0.05 * H * SCALE), 0, -1)
    skin, _ = segmented_face_masks(labels, FACE, (W, H), FULL)
    assert at(skin, norm(RADIUS * 0.8, 0.0)) == 0
