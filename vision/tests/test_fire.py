"""The fire effect's state: ignition, fingertip flames, and being put out.

The hand is synthetic (`hand_fixture`): a fist that opens is an ignition, a
side-to-side wrist motion is a shake, and each raised finger earns a flame.
No camera, no MediaPipe.
"""

from __future__ import annotations

import numpy as np
from hand_fixture import DIGITS, make_hand, moved, snap_closed

from ofp_vision.fire import Fire
from ofp_vision.signals import Frame

SIZE = (640, 480)
FRAME_MS = 33.0


def only(*digits: str) -> dict[str, bool]:
    return {digit: digit in digits for digit in DIGITS}


def _step(fire: Fire, hand, t_ms: float) -> None:
    fire.update(Frame(t_ms=t_ms, hands=[hand]), SIZE)


def _frames(fire: Fire, hand, t_ms: float, count: int = 10) -> float:
    """Step `count` frames, returning the next timestamp."""
    for i in range(count):
        _step(fire, hand, t_ms + i * FRAME_MS)
    return t_ms + count * FRAME_MS


def _light(fire: Fire) -> float:
    """Fist, then open: light the hand and let the first flame settle."""
    t = _frames(fire, make_hand(only()), 0.0, 1)
    t = _frames(fire, make_hand(), t, 10)
    assert fire.status()["Right"]["lit"] == 1.0
    return t


def test_reset_puts_every_fire_out() -> None:
    fire = Fire()
    _light(fire)

    fire.reset()

    assert fire.status() == {}
    # Nothing left to draw: the image comes back untouched.
    image = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    assert fire.draw(image) is image


def test_a_fist_then_open_lights_the_hand() -> None:
    fire = Fire()
    _step(fire, make_hand(only()), 0.0)
    _step(fire, make_hand(), FRAME_MS)
    assert fire.status()["Right"]["lit"] == 1.0


def test_a_snap_does_not_light_the_hand() -> None:
    fire = Fire()
    open_hand = make_hand()
    _step(fire, open_hand, 0.0)
    _step(fire, snap_closed(open_hand), FRAME_MS)
    _step(fire, open_hand, 2 * FRAME_MS)
    assert fire.status()["Right"]["lit"] == 0.0


def test_opening_the_whole_hand_lights_only_one_finger() -> None:
    fire = Fire()
    _light(fire)

    status = fire.status()["Right"]
    assert status["fingers"] == 1.0
    assert status["flames"] == 1.0


def test_fingers_are_earned_one_at_a_time() -> None:
    fire = Fire()
    t = _light(fire)
    assert fire.status()["Right"]["fingers"] == 1.0

    for expected, digits in enumerate(
        (
            ("index", "middle"),
            ("index", "middle", "ring"),
            ("index", "middle", "ring", "pinky"),
        ),
        start=2,
    ):
        t = _frames(fire, make_hand(only(*digits)), t)
        assert fire.status()["Right"]["fingers"] == float(expected)

    # The thumb is the capstone: it lights only after all four fingers.
    t = _frames(fire, make_hand(), t)
    assert fire.status()["Right"]["fingers"] == 4.0
    assert fire.status()["Right"]["flames"] == 5.0


def test_a_jump_does_not_light_the_missing_fingers() -> None:
    fire = Fire()
    t = _light(fire)
    assert fire.status()["Right"]["fingers"] == 1.0

    # The hand is already wide open, but the player never showed two, three
    # and four. It must stay at one.
    t = _frames(fire, make_hand(), t)
    assert fire.status()["Right"]["fingers"] == 1.0


def test_a_shake_puts_the_fire_out() -> None:
    fire = Fire()
    t = _light(fire)
    open_hand = make_hand()

    for dx in (0.0, 0.08, -0.08, 0.08, -0.08, 0.08):
        _step(fire, moved(open_hand, dx), t)
        t += FRAME_MS

    assert fire.status()["Right"]["lit"] == 0.0


def test_the_fire_draws_something() -> None:
    fire = Fire()
    _light(fire)
    image = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)

    out = fire.draw(image)

    assert out is not image
    assert out.any()
