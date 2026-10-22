"""Tests for skin masks built from the segmentation model's labels.

The model labels every pixel: background, hair, body skin, face skin,
clothes, other. These masks turn that into "what may stretch":

- a face grab: face skin - and the beard (hair below the nose, inside the
  face), which is attached to skin - but never scalp hair, the room, or the
  teeth;
- a finger grab: that finger's own skin pixels - not its neighbours, not
  the wall between them.

The segmentation here is synthetic, at half the frame's resolution, like
the real one.
"""

from __future__ import annotations

import cv2
import numpy as np
from face_fixture import H, RADIUS, W, landmarks, norm, px

from ofp_vision.indices import HAND_DIGITS, HAND_MIDDLE_MCP, HAND_WRIST
from ofp_vision.signals import HandObservation
from ofp_vision.skin import (
    SEGMENT_BODY_SKIN,
    SEGMENT_FACE_SKIN,
    SEGMENT_HAIR,
    Window,
    segmented_digit_mask,
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


# --- fingers -----------------------------------------------------------------


def spread_hand() -> HandObservation:
    """A left hand, fingers up and spread 0.04 apart (normalised x)."""
    x, y = 0.75, 0.7
    points = [(x, y, 0.0)] * 21
    points = list(points)
    points[HAND_WRIST] = (x, y + 0.2, 0.0)
    points[HAND_MIDDLE_MCP] = (x, y, 0.0)
    for offset, digit in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        fx = x + (offset - 2) * 0.04
        for j, joint in enumerate(HAND_DIGITS[digit]):
            points[joint] = (fx, y - 0.06 * j, 0.0)
    points[HAND_MIDDLE_MCP] = (x, y, 0.0)
    return HandObservation(handedness="Left", landmarks=points)


def hand_segmentation(hand: HandObservation) -> np.ndarray:
    labels = np.zeros((int(H * SCALE), int(W * SCALE)), dtype=np.uint8)
    for digit in HAND_DIGITS.values():
        a, b = half(hand.landmarks[digit[0]]), half(hand.landmarks[digit[3]])
        cv2.line(labels, a, b, SEGMENT_BODY_SKIN, 8)
    return labels


def test_a_finger_mask_is_that_fingers_own_skin():
    hand = spread_hand()
    mask = segmented_digit_mask(hand_segmentation(hand), hand, "index", (W, H), FULL)
    index = HAND_DIGITS["index"]
    middle = HAND_DIGITS["middle"]
    between = ((hand.landmarks[index[2]][0] + hand.landmarks[middle[2]][0]) / 2, hand.landmarks[index[2]][1])
    assert at(mask, hand.landmarks[index[2]]) > 200, "the grabbed finger"
    assert at(mask, hand.landmarks[middle[2]]) == 0, "not its neighbour"
    assert at(mask, between) == 0, "not the wall between them"


def test_skin_the_model_smears_around_a_finger_is_not_that_finger():
    """At its low resolution the model labels a blur of skin around a hand,
    over the air beside and beyond the fingers. A finger is only its own
    width: the air next to it never stretches with it."""
    hand = spread_hand()
    labels = np.zeros((int(H * SCALE), int(W * SCALE)), dtype=np.uint8)
    cv2.rectangle(labels, half((0.6, 0.4)), half((0.95, 0.95)), SEGMENT_BODY_SKIN, -1)
    mask = segmented_digit_mask(labels, hand, "pinky", (W, H), FULL)
    pinky = HAND_DIGITS["pinky"]
    middle_joint, tip = hand.landmarks[pinky[2]], hand.landmarks[pinky[3]]
    assert at(mask, middle_joint) > 200, "the finger itself"
    assert at(mask, (middle_joint[0] + 0.03, middle_joint[1])) == 0, "not the air beside it"
    assert at(mask, (tip[0], tip[1] - 0.03)) == 0, "not the air beyond its tip"
