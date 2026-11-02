"""Tests for the hand signs behind the fire: snap, fist, pointing finger.

The hand is synthetic (`hand_fixture.make_hand`): a finger that is out points
straight up, one that is in folds into the palm. No camera, no MediaPipe.
"""

from __future__ import annotations

from hand_fixture import ASPECT, DIGITS, fingertip, make_hand, snap_closed

from ofp_vision.config import FireConfig
from ofp_vision.gestures import SignTracker, palm_length, signs

FRAME_MS = 33.0


def only(*digits: str) -> dict[str, bool]:
    return {digit: digit in digits for digit in DIGITS}


def test_palm_length_is_the_wrist_to_middle_knuckle() -> None:
    hand = make_hand()
    assert palm_length(hand.landmarks, ASPECT) > 0.1


def test_an_open_hand_has_no_fist_and_no_point() -> None:
    found = signs(make_hand(), aspect=ASPECT)
    assert all(found.out.values())
    assert not found.fist
    assert not found.index_only


def test_a_fist_is_all_four_fingers_curled() -> None:
    found = signs(make_hand(only()), aspect=ASPECT)
    assert found.fist
    assert not found.index_only
    assert not any(found.out[digit] for digit in ("index", "middle", "ring", "pinky"))


def test_a_curled_thumb_does_not_stop_a_point() -> None:
    hand = make_hand(only("index"), side="Left")
    found = signs(hand, aspect=ASPECT)
    assert found.index_only
    assert not found.fist


def test_a_point_with_the_middle_finger_too_is_not_index_only() -> None:
    hand = make_hand(only("index", "middle"))
    assert not signs(hand, aspect=ASPECT).index_only


class Session:
    """One tracker, stepped with a running clock."""

    def __init__(self) -> None:
        self.tracker = SignTracker(FireConfig())
        self.t = 0.0

    def step(self, hand):
        found = self.tracker.update(hand, self.t, ASPECT)
        self.t += FRAME_MS
        return found


def test_a_snap_is_one_close_and_fast_open() -> None:
    session = Session()
    open_hand = make_hand()
    found = [
        session.step(open_hand),
        session.step(snap_closed(open_hand)),
        session.step(open_hand),
        session.step(open_hand),
    ]
    assert [sign.snap for sign in found] == [False, False, True, False]


def test_a_pinch_held_does_not_snap_when_it_lets_go() -> None:
    session = Session()
    open_hand = make_hand()
    session.step(open_hand)
    for _ in range(12):  # held well past the snap window
        assert not session.step(snap_closed(open_hand)).snap
    assert not session.step(open_hand).snap


def test_two_snaps_are_two_events() -> None:
    session = Session()
    hand = make_hand()
    session.step(hand)
    session.step(snap_closed(hand))
    assert session.step(hand).snap
    for _ in range(12):  # the cooldown
        session.step(hand)
    session.step(snap_closed(hand))
    assert session.step(hand).snap


def test_closing_a_fist_does_not_snap() -> None:
    """The thumb brushes the curled fingers on the way into a fist; that is
    not a snap, because neither the index nor the middle finger is out when
    the gap opens again."""
    session = Session()
    fist = make_hand(only())
    session.step(fist)
    session.step(snap_closed(fist))  # thumb onto the curled middle tip
    assert not session.step(fist).snap


def test_out_fingers_stay_out_through_the_hysteresis_band() -> None:
    """A finger that is out stays out while it is only mildly changed, so a
    sign near the threshold cannot flicker (and take the fireball with it)."""
    session = Session()
    assert session.step(make_hand(only("index"))).out["index"]
    # Straight but shorter than the out threshold: it stays out.
    half = make_hand(only("index"))
    mcp = half.landmarks[5]
    for joint, length in ((6, 0.03), (7, 0.06), (8, 0.09)):
        half.landmarks[joint] = (mcp[0], mcp[1] - length, 0.0)
    assert session.step(half).out["index"]
    # Now clearly folded: it goes in.
    folded = make_hand(only())
    assert not session.step(folded).out["index"]
