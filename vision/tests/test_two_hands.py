"""Tests for grabbing with both hands, and for grabbing fingers precisely.

- each hand pinches and pulls on its own: one hand, or both at once
- a finger is grabbed where the pinch lands on it (anywhere along it), and
  only when the pinch is actually on the finger - not in the air beside it
"""

from __future__ import annotations

import pytest
from test_grab import ASPECT, CHEEK_L, EAR_R, FRAME_MS, OPEN, PINCHED, make_face, make_hand, other_hand

from ofp_vision.config import GRAB_CHANNELS, GrabConfig
from ofp_vision.grab import GrabState, TwoHandGrabber, followed_grab
from ofp_vision.signals import Frame


class Session:
    def __init__(self) -> None:
        self.grabber = TwoHandGrabber(GrabConfig(), mirror=True, aspect=ASPECT)
        self.t = 0.0

    def step(self, hands, face=None):
        frame = Frame(t_ms=self.t, face=make_face() if face is None else face, hands=list(hands))
        states = self.grabber.update(frame, self.t)
        self.t += FRAME_MS
        return states


def test_each_hand_grabs_on_its_own_at_the_same_time():
    session = Session()
    right_open, left_open = make_hand(CHEEK_L, OPEN, "Right"), make_hand(EAR_R, OPEN, "Left")
    session.step([right_open, left_open])
    states = session.step([make_hand(CHEEK_L, PINCHED, "Right"), make_hand(EAR_R, PINCHED, "Left")])
    assert states["Right"].active and states["Right"].target == "cheekL"
    assert states["Left"].active and states["Left"].target == "earR"


def test_one_hand_letting_go_leaves_the_other_holding():
    session = Session()
    session.step([make_hand(CHEEK_L, OPEN, "Right"), make_hand(EAR_R, OPEN, "Left")])
    session.step([make_hand(CHEEK_L, PINCHED, "Right"), make_hand(EAR_R, PINCHED, "Left")])
    states = session.step([make_hand(CHEEK_L, OPEN, "Right"), make_hand(EAR_R, PINCHED, "Left")])
    assert states["Right"].just_released and not states["Right"].active
    assert states["Left"].active


def test_both_hands_report_on_the_wire():
    session = Session()
    session.step([make_hand(CHEEK_L, OPEN, "Right")])
    session.step([make_hand(CHEEK_L, PINCHED, "Right")])
    features = session.grabber.features()
    assert set(features) == set(GRAB_CHANNELS)
    assert features["grab.Right.active"] == 1.0
    assert features["grab.Left.active"] == 0.0


def test_a_one_grab_effect_keeps_following_the_hand_it_started_with():
    """The CPU effect draws one grab. A second hand pinching mid-stretch must
    not snatch it away - only once the first lets go."""
    left = GrabState(active=True, hand="Left", target="cheekR")
    right = GrabState(active=True, hand="Right", target="cheekL")
    released = GrabState(hand="Right", target="cheekL", just_released=True)
    assert followed_grab({"Left": GrabState(), "Right": right}, None) == ("Right", right)
    assert followed_grab({"Left": left, "Right": right}, "Right") == ("Right", right)
    assert followed_grab({"Left": left, "Right": released}, "Right") == ("Right", released)
    assert followed_grab({"Left": left, "Right": GrabState()}, "Right") == ("Left", left)
    assert followed_grab({}, "Right") == ("Right", GrabState())


# --- fingers -----------------------------------------------------------------

TIP = (0.85, 0.30)


def finger_session(pinch_at):
    session = Session()
    left = other_hand(TIP)
    session.step([make_hand(pinch_at, OPEN, "Right"), left])
    return session, session.step([make_hand(pinch_at, PINCHED, "Right"), left]), left


def test_a_finger_is_grabbed_where_the_pinch_lands_along_it():
    """Pinch halfway down the index finger: it grabs there, not the tip."""
    halfway = (TIP[0], TIP[1] + 0.075)
    _, states, _ = finger_session(halfway)
    assert states["Right"].target == "finger.Left.index"
    assert states["Right"].anchor == pytest.approx(halfway, abs=1e-6)


def test_pinching_the_air_beside_a_finger_grabs_nothing():
    """0.6 of the gap to the next finger (0.04 away): in the air, not on it."""
    beside = (TIP[0] + 0.024, TIP[1] + 0.05)
    _, states, _ = finger_session(beside)
    assert not states["Right"].active


def test_a_grabbed_finger_spot_stays_put_along_the_finger():
    """Move the grabbed hand: the anchor follows the same place on the finger."""
    halfway = (TIP[0], TIP[1] + 0.075)
    session, _, _ = finger_session(halfway)
    moved_tip = (TIP[0] + 0.05, TIP[1] - 0.02)
    moved = other_hand(moved_tip)
    pinch = (halfway[0] + 0.05, halfway[1] - 0.02)
    states = session.step([make_hand(pinch, PINCHED, "Right"), moved])
    assert states["Right"].active
    assert states["Right"].stretch == pytest.approx(0.0, abs=0.02)


# --- unstable labels -------------------------------------------------------------


def test_a_hand_relabelled_mid_pull_keeps_its_grab():
    """MediaPipe sometimes swaps a hand's label. The pinch is the same pinch:
    the grab carries on, no release fires, and the other hand grabs nothing."""
    session = Session()
    at = (CHEEK_L[0] + 0.05, CHEEK_L[1])
    session.step([make_hand(CHEEK_L, OPEN, "Right")])
    session.step([make_hand(CHEEK_L, PINCHED, "Right")])
    for _ in range(5):
        session.step([make_hand(at, PINCHED, "Right")])
    for _ in range(15):  # ~500ms under the other label
        states = session.step([make_hand(at, PINCHED, "Left")])
        assert not states["Right"].just_released
        assert not states["Left"].active
    assert states["Right"].active and states["Right"].stretch > 0.2


def test_two_hands_given_the_same_label_both_keep_their_grabs():
    """MediaPipe can call both hands "Right". Each pinch still belongs to its
    own hand - and one held loosely (inside the hysteresis band) holds."""
    session = Session()
    session.step([make_hand(CHEEK_L, OPEN, "Right"), make_hand(EAR_R, OPEN, "Left")])
    session.step([make_hand(CHEEK_L, PINCHED, "Right"), make_hand(EAR_R, PINCHED, "Left")])
    loose = 0.2
    for _ in range(10):  # past the 200ms a lost hand's grab is kept for
        states = session.step([make_hand(EAR_R, PINCHED, "Right"), make_hand(CHEEK_L, loose, "Right")])
    assert states["Right"].active and states["Right"].target == "cheekL"
    assert states["Left"].active and states["Left"].target == "earR"


def test_hands_given_the_same_label_before_a_grab_can_both_grab():
    session = Session()
    session.step([make_hand(CHEEK_L, OPEN, "Right"), make_hand(EAR_R, OPEN, "Left")])
    states = session.step([make_hand(EAR_R, PINCHED, "Right"), make_hand(CHEEK_L, PINCHED, "Right")])
    assert states["Right"].target == "cheekL"
    assert states["Left"].target == "earR"


def test_the_settled_labels_are_what_everything_downstream_sees():
    """The effect and the lab look hands up by label: they get the fixed ones."""
    from ofp_vision.config import Config
    from ofp_vision.pipeline import Pipeline

    pipeline = Pipeline(Config(), {"face.cheekL": 1.0})
    hands = [make_hand(CHEEK_L, OPEN, "Right"), make_hand(EAR_R, OPEN, "Right")]
    result = pipeline.step(Frame(t_ms=0.0, face=make_face(), hands=hands), 30.0)
    assert sorted(h.handedness for h in result.frame.hands) == ["Left", "Right"]
