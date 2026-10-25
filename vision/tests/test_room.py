"""Tests for the remembered picture behind the hands: what shows where a
pulled part used to be.

- everything but the hands is learned, within a few frames
- the hands never are: where a hand is now, the picture from before it got
  there is still remembered
- the memory is read back where it was learned, at any window
- a hole is filled from the colours around it - never black - and given the
  picture's own grain, so it does not read as a flat cut-out
"""

from __future__ import annotations

import numpy as np

from ofp_vision.room import RoomMemory, filled, grain
from ofp_vision.skin import Window

W, H = 320, 240
FULL = Window(0, 0, W, H)
WALL = (200, 60, 30)
SKIN = (120, 160, 215)


def frame(colour=WALL) -> np.ndarray:
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[:] = colour
    return image


def test_everything_but_the_hands_is_learned_in_a_few_frames():
    memory = RoomMemory()
    for _ in range(8):
        memory.update(frame(), None)
    room, known = memory.fill(FULL)
    assert known[H // 2, W // 2] > 0.99
    assert np.abs(room[H // 2, W // 2] - WALL).max() < 2


def test_a_hand_is_never_learned_what_was_behind_it_is_kept():
    memory = RoomMemory()
    for _ in range(8):
        memory.update(frame(), None)
    hand = np.zeros((H, W), dtype=np.uint8)
    hand[50:150, 100:200] = 255
    scene = frame()
    scene[50:150, 100:200] = SKIN
    for _ in range(20):
        memory.update(scene, hand)
    room, known = memory.fill(FULL)
    assert known[100, 150] > 0.99
    assert np.abs(room[100, 150] - WALL).max() < 2, "the wall behind the hand, as it was"


def test_nothing_is_known_before_it_is_seen():
    memory = RoomMemory()
    hand = np.full((H, W), 255, dtype=np.uint8)
    memory.update(frame(), hand)
    assert memory.fill(FULL)[1].max() == 0.0


def test_a_window_reads_the_memory_where_it_was_learned():
    image = frame()
    image[:, 160:] = SKIN
    memory = RoomMemory()
    for _ in range(8):
        memory.update(image, None)
    room, _ = memory.fill(Window(100, 50, 220, 150))
    assert np.abs(room[50, 40] - WALL).max() < 2, "x = 140"
    assert np.abs(room[50, 80] - SKIN).max() < 2, "x = 180"


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


def test_the_grain_of_a_picture_is_measured_where_asked():
    rng = np.random.default_rng(1)
    grainy = (120.0 + rng.normal(0.0, 8.0, (60, 60, 3))).astype(np.float32)
    grainy[:, 30:] = 120.0  # the right half is smooth
    left = np.zeros((60, 60), dtype=bool)
    left[:, :25] = True
    assert 4.0 < grain(grainy, left) < 12.0
    assert grain(grainy, ~left & (np.arange(60) > 35)[None, :]) < 0.5


def test_the_frame_edge_reads_back_as_bright_as_it_was():
    memory = RoomMemory()
    for _ in range(8):
        memory.update(frame((100, 100, 100)), None)
    room, _ = memory.fill(FULL)
    for y, x in ((0, 0), (0, W // 2), (H // 2, W - 1), (H - 1, W - 1)):
        assert np.abs(room[y, x] - 100).max() < 1.5
