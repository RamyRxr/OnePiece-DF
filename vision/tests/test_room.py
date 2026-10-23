"""Tests for the remembered room: what is behind a part that is pulled away.

- the room is learned where the segmentation sees background, and only there
- background right next to the body is not trusted: the labels are coarse and
  a few frames late
- one odd frame (a passing shadow) does not stick
- where the room was never seen, a hole is filled from the colours around it
"""

from __future__ import annotations

import numpy as np

from ofp_vision.room import SAFETY_PX, RoomMemory, filled
from ofp_vision.skin import SEGMENT_BACKGROUND, SEGMENT_BODY_SKIN, Window

W, H = 320, 240
FULL = Window(0, 0, W, H)
WALL = (200, 60, 30)
PERSON = (0, 0, H // 2, W // 2)  # rows, cols: the person fills the top left


def scene(colour=WALL) -> tuple[np.ndarray, np.ndarray]:
    bgr = np.zeros((H, W, 3), dtype=np.uint8)
    bgr[:] = colour
    bgr[: H // 2, : W // 2] = (90, 140, 210)
    labels = np.full((H // 2, W // 2), SEGMENT_BACKGROUND, dtype=np.uint8)
    labels[: H // 4, : W // 4] = SEGMENT_BODY_SKIN
    return bgr, labels


def learned(looks: int = 30) -> RoomMemory:
    memory = RoomMemory()
    for _ in range(looks):
        bgr, labels = scene()
        memory.update(bgr, labels)
    return memory


def test_the_room_is_learned_where_the_segmentation_sees_background():
    room, known = learned().fill(FULL)
    assert known[H - 10, W - 10] > 0.99
    assert np.abs(room[H - 10, W - 10] - WALL).max() < 2
    assert known[10, 10] == 0.0, "the person was never seen through"


def test_background_right_next_to_the_body_is_not_trusted():
    _, known = learned().fill(FULL)
    assert known[10, W // 2 + SAFETY_PX // 2] == 0.0
    assert known[10, W // 2 + 3 * SAFETY_PX] > 0.99


def test_one_odd_frame_does_not_stick():
    memory = learned()
    bgr, labels = scene(colour=(0, 0, 0))
    memory.update(bgr, labels)
    room, _ = memory.fill(FULL)
    assert np.abs(room[H - 10, W - 10] - WALL).max() < 0.25 * 200


def test_the_same_segmentation_is_learned_from_once():
    """Labels arrive every third frame; frames in between carry the old ones,
    which no longer match the picture."""
    memory = RoomMemory()
    bgr, labels = scene()
    memory.update(bgr, labels)
    once = memory.fill(FULL)[1].copy()
    memory.update(bgr, labels)
    assert np.array_equal(memory.fill(FULL)[1], once)


def test_a_hole_is_filled_from_the_colours_around_it():
    image = np.zeros((120, 160, 3), dtype=np.float32)
    image[:, :80] = (0, 0, 255)
    image[:, 80:] = (255, 0, 0)
    hole = np.zeros((120, 160), dtype=np.float32)
    hole[40:80, 20:60] = 1.0
    image[40:80, 20:60] = (0, 255, 0)  # what was in the hole must not survive
    out = filled(image, hole)
    assert np.abs(out[60, 40] - (0, 0, 255)).max() < 30, "red from around it"
    assert np.abs(out[60, 120] - image[60, 120]).max() < 1e-3, "outside the hole, untouched"


def test_a_hole_in_a_tiny_picture_is_filled_not_painted_black():
    """At the frame's corner the spot to fill can be a few pixels across."""
    image = np.full((4, 4, 3), 200.0, dtype=np.float32)
    hole = np.zeros((4, 4), dtype=np.float32)
    hole[1:3, 1:3] = 1.0
    assert np.abs(filled(image, hole)[1:3, 1:3] - 200.0).max() < 1.0
