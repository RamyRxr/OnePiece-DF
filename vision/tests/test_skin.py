"""Tests for what counts as skin in the stretch effect.

Only skin may stretch. These lock in how the masks decide it: the player's
own skin colour, measured from their face; the face outline's interior as
is, its outer band and the ears only where skin-coloured; the mouth opening
never; one finger, not its neighbours.
"""

from __future__ import annotations

import numpy as np
from face_fixture import EYE_L, H, RADIUS, SKIN, W, image, landmarks, norm, px  # noqa: F401

from ofp_vision.grab import ear_points
from ofp_vision.indices import FACE_INNER_LIPS, HAND_DIGITS, HAND_MIDDLE_MCP, HAND_WRIST
from ofp_vision.signals import HandObservation
from ofp_vision.skin import (
    CAVITY_MAX_LUMA,
    Window,
    cavity_colour,
    colour_distance,
    colour_match,
    digit_mask,
    face_masks,
    skin_colour,
)

FACE = landmarks()
FULL = Window(0, 0, W, H)
LUMA = np.array([0.114, 0.587, 0.299])


def at(mask: np.ndarray, point) -> int:
    x, y = px(point)
    return int(mask[y, x])


def match(colour, img=None) -> float:
    model = skin_colour(image() if img is None else img, FACE)
    return float(colour_match(colour_distance(np.uint8([[colour]]), model))[0, 0])


# --- colour ----------------------------------------------------------------------


def test_the_players_own_skin_colour_matches():
    assert match(SKIN) == 1.0


def test_the_room_does_not_match():
    assert match((200, 120, 40)) == 0.0


def test_hair_of_a_similar_hue_but_much_darker_does_not_match():
    """Brightness counts: brown hair can share skin's hue."""
    hair = tuple(int(c * 0.4) for c in SKIN)
    assert match(hair) < 0.5


# --- the face ------------------------------------------------------------------


def test_the_face_interior_is_skin_whatever_its_colour():
    """Eyes, brows, nostrils are inside the face: they belong to it."""
    skin, _ = face_masks(image(), FACE, FULL)
    assert at(skin, norm(*EYE_L)) == 255


def test_background_in_the_outlines_outer_band_is_not_skin():
    img = image()
    x, y = px(norm(RADIUS - 0.01, 0.0))
    img[y - 4 : y + 5, x - 4 : x + 5] = (200, 120, 40)
    skin, _ = face_masks(img, FACE, FULL)
    assert skin[y, x] == 0


def test_the_same_colour_deep_inside_the_face_is_still_skin():
    img = image()
    x, y = px(norm(-0.05, 0.03))
    img[y - 4 : y + 5, x - 4 : x + 5] = (200, 120, 40)
    skin, _ = face_masks(img, FACE, FULL)
    assert skin[y, x] == 255


def test_the_mouth_opening_is_not_skin():
    skin, mouth = face_masks(image(), FACE, FULL)
    teeth = norm(0.0, 0.09)
    assert at(skin, teeth) == 0
    assert at(mouth, teeth) == 255


def test_an_ear_counts_only_where_it_is_skin_coloured():
    ear = ear_points(FACE)["earL"]
    skin, _ = face_masks(image(), FACE, FULL)
    assert at(skin, ear) == 0, "the fixture's ear spot is background"
    img = image()
    x, y = px(ear)
    img[y - 6 : y + 7, x - 4 : x + 5] = SKIN
    skin, _ = face_masks(img, FACE, FULL)
    assert skin[y, x] > 200


def test_masks_built_for_a_window_match_the_whole_frame():
    img = image()
    window = Window(500, 320, 760, 460)
    full_skin, full_mouth = face_masks(img, FACE, FULL)
    skin, mouth = face_masks(img, FACE, window)
    assert skin.shape == window.shape
    assert np.array_equal(skin, window.crop(full_skin))
    assert np.array_equal(mouth, window.crop(full_mouth))


def test_an_opened_mouth_is_shadow_dark():
    _, mouth = face_masks(image(), FACE, FULL)
    colour = np.array(cavity_colour(image(), mouth, FULL))
    assert colour @ LUMA <= CAVITY_MAX_LUMA + 1e-6


def test_the_mouth_outline_is_the_inner_lips():
    assert len(FACE_INNER_LIPS) == 20


# --- fingers -------------------------------------------------------------------


def test_a_finger_mask_covers_that_finger_and_not_its_neighbours():
    x, y = 0.8, 0.6
    points = [(x, y, 0.0)] * 21
    points = list(points)
    points[HAND_WRIST] = (x, y + 0.2, 0.0)
    points[HAND_MIDDLE_MCP] = (x, y + 0.1, 0.0)
    for offset, digit in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        fx = x + (offset - 2) * 0.03
        for j, joint in enumerate(HAND_DIGITS[digit]):
            points[joint] = (fx, y + 0.1 - 0.03 * j, 0.0)
    hand = HandObservation(handedness="Left", landmarks=points)
    mask = digit_mask(hand, "index", (W, H), FULL)
    index_tip = points[HAND_DIGITS["index"][3]]
    pinky_tip = points[HAND_DIGITS["pinky"][3]]
    assert at(mask, index_tip) == 255
    assert at(mask, pinky_tip) == 0



# --- regressions from review ---------------------------------------------------


def test_a_window_wholly_inside_the_face_still_gets_its_masks():
    """No edge band or ear in the window: nothing to colour-check."""
    window = Window(620, 330, 660, 350)
    skin, mouth = face_masks(image(), FACE, window)
    assert skin.shape == window.shape
    assert skin.min() == 255


def test_off_frame_sample_spots_never_wrap_around_to_the_room():
    """A forehead above the frame used to be sampled from the bottom rows."""
    face, img = landmarks(centre=(0.5, 0.08)), image(centre=(0.5, 0.08))
    model = skin_colour(img, face)
    room = tuple(int(c) for c in img[700, 100])
    assert float(colour_match(colour_distance(np.uint8([[room]]), model))[0, 0]) == 0.0


def test_a_face_with_every_sample_spot_off_frame_keeps_only_its_interior():
    face, img = landmarks(centre=(0.5, -0.12)), image(centre=(0.5, -0.12))
    assert skin_colour(img, face) is None
    skin, _ = face_masks(img, face, FULL)
    assert skin.shape == (H, W)
