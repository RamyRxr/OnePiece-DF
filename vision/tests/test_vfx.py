"""The lab's release burst: sparks, and only when a pinch lets go.

The game has its own effects; this is so that a pull in the lab reads as an
event when the fingers open, instead of a finger quietly returning.
"""

from __future__ import annotations

import numpy as np

from ofp_vision.vfx import Sparks


def frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_nothing_is_drawn_before_a_release():
    still = frame()
    assert Sparks().draw(still) is still


def test_a_release_leaves_a_light_where_it_happened():
    sparks = Sparks()
    still = frame()
    sparks.release(160.0, 120.0, 2.0)
    out = sparks.draw(still)
    assert out is not still, "a copy, so the camera frame is not painted over"
    assert still.sum() == 0.0
    assert out[112:128, 152:168].sum() > 0.0, "a burst at the pinch"
    assert out[0:40, 0:40].sum() == 0.0, "and nowhere else"


def test_a_stronger_release_is_a_bigger_burst():
    weak, strong = Sparks(), Sparks()
    weak.release(160.0, 120.0, 0.2)
    strong.release(160.0, 120.0, 2.0)
    dim = weak.draw(frame())
    bright = strong.draw(frame())
    assert bright.sum() > dim.sum()
