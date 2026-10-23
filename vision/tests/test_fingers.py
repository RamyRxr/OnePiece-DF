"""Tests for the rubber finger: a tube along the finger, with physics.

- at rest it is the finger, exactly: drawn where its own skin is
- pulled from a pinch, the part past the pinch travels whole, the knuckle
  end stays on the hand, and the part in between gets longer - a little
  thinner, never a noodle
- let go, it snaps back and settles
- moving the hand carries it along, with no lag
"""

from __future__ import annotations

import math

import numpy as np

from ofp_vision.fingers import BASE_NODES, MIN_THINNING, FingerSession, apply, similarity
from ofp_vision.indices import HAND_DIGITS, HAND_MIDDLE_MCP, HAND_WRIST
from ofp_vision.signals import HandObservation
from ofp_vision.skin import SEGMENT_BODY_SKIN, Window

W, H = 1280, 720
SIZE = (W, H)
PART = "finger.Left.index"


def hand(dx: float = 0.0, dy: float = 0.0, spread: float = 0.04) -> HandObservation:
    """A left hand, fingers up, `spread` apart; wrist to knuckles 0.2 frame heights."""
    x, y = 0.6 + dx, 0.75 + dy
    points = [(x, y, 0.0)] * 21
    points[HAND_WRIST] = (x, y + 0.2, 0.0)
    for offset, digit in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        fx = x + (offset - 2) * spread
        for j, joint in enumerate(HAND_DIGITS[digit]):
            points[joint] = (fx, y - 0.07 * j, 0.0)
    points[HAND_MIDDLE_MCP] = (x, y, 0.0)
    return HandObservation(handedness="Left", landmarks=points)


def tip_px(observed: HandObservation) -> np.ndarray:
    tip = observed.landmarks[HAND_DIGITS["index"][3]]
    return np.array([tip[0] * W, tip[1] * H])


def run(session: FingerSession, observed: HandObservation, pulls: dict, frames: int) -> np.ndarray:
    rest = None
    for _ in range(frames):
        rest = session.follow(observed, SIZE)
        session.step(rest, pulls)
    return rest


def pulled(by=(150.0, -40.0), frames: int = 30):
    observed = hand()
    session = FingerSession.start(PART, observed, SIZE)
    hold = session.hold(PART, tip_px(observed))
    session.holds = {"Right": hold}
    rest = None
    for f in range(1, frames + 1):
        k = min(1.0, f / 8)
        rest = run(session, observed, {"Right": np.array([by[0] * k, by[1] * k, 0.0])}, 1)
    return session, hold, rest, observed


def test_the_palm_fit_recovers_a_turn_a_scale_and_a_shift():
    src = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 20.0], [7.0, 3.0]])
    turn = np.array([[math.cos(0.3), -math.sin(0.3)], [math.sin(0.3), math.cos(0.3)]])
    dst = 1.5 * src @ turn.T + [4.0, -2.0]
    assert np.allclose(apply(similarity(src, dst), src), dst)


def test_at_rest_the_tube_is_the_finger_exactly():
    observed = hand()
    session = FingerSession.start(PART, observed, SIZE)
    rest = run(session, observed, {}, 3)
    drawn, source = session.tube(rest)
    assert np.abs(drawn[:, :2] - source).max() < 1e-6, "drawn exactly where its skin is"
    nearest = np.linalg.norm(rest[:, :2] - tip_px(observed), axis=1).min()
    assert nearest < 4.0, "along the index finger, up to its tip"


def test_pulled_from_its_tip_the_finger_gets_longer_and_stays_a_finger():
    session, hold, rest, _ = pulled()
    p = session.body.positions
    assert np.abs(p[hold.core, :2] - (rest[hold.core, :2] + [150.0, -40.0])).max() < 0.5, "the tip goes whole"
    assert np.abs(p[:BASE_NODES] - rest[:BASE_NODES]).max() < 1e-6, "the knuckle end stays on the hand"
    length = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()
    resting = np.linalg.norm(np.diff(rest[:, :2], axis=0), axis=1).sum()
    assert length > 1.4 * resting, "it got longer"
    drawn, source = session.tube(rest)
    across = drawn.reshape(len(p), -1, 3)[:, [0, -1], :2]
    width = np.linalg.norm(across[:, 1] - across[:, 0], axis=1)
    resting_width = np.linalg.norm(source.reshape(len(p), -1, 2)[:, -1] - source.reshape(len(p), -1, 2)[:, 0], axis=1)
    ratio = width / resting_width
    assert ratio.min() >= MIN_THINNING - 1e-6, "never a noodle"
    assert ratio.min() < 0.95, "but a stretched finger is a little thinner"


def test_let_go_the_finger_snaps_back_and_settles():
    session, _, _, observed = pulled()
    session.holds = {}
    rest = None
    for frame in range(15):
        rest = run(session, observed, {}, 1)
        if session.body.settled(rest):
            break
    assert session.body.settled(rest), "back and still within half a second"


def test_moving_the_hand_carries_the_finger_without_lag():
    session = FingerSession.start(PART, hand(), SIZE)
    rest = None
    for f in range(1, 8):
        rest = run(session, hand(dx=0.01 * f, dy=-0.004 * f), {}, 1)
    assert np.abs(session.body.positions - rest).max() < 0.01


def test_a_hand_shrunk_to_a_point_is_ignored_not_an_error():
    """The tracker can return a hand collapsed to one point for a frame."""
    session = FingerSession.start(PART, hand(), SIZE)
    before = session.rest()
    rest = session.follow(HandObservation("Left", [(0.5, 0.5, 0.0)] * 21), SIZE)
    assert np.allclose(rest, before), "the finger stays where it was"


def skin_of(observed: HandObservation) -> np.ndarray:
    """The index finger's skin over the whole frame, the segmentation calling
    everything around the hand body skin (it blurs over the gaps)."""
    labels = np.zeros((H // 2, W // 2), dtype=np.uint8)
    labels[100:340, 200:500] = SEGMENT_BODY_SKIN
    session = FingerSession.start(PART, observed, SIZE)
    rest = run(session, observed, {}, 1)
    return session.skin(rest, labels, SIZE, Window(0, 0, W, H), hand=observed)


def at(mask: np.ndarray, x: float, y: float) -> int:
    return int(mask[int(y * H), int(x * W)])


def test_the_finger_skin_is_its_own_not_its_neighbours():
    """Fingers together: the skin between them belongs to the nearer one, so
    the tube never takes a sliver of the next finger along."""
    observed = hand(spread=0.02)
    index = observed.landmarks[HAND_DIGITS["index"][2]]
    mask = skin_of(observed)
    assert at(mask, index[0], index[1]) > 200, "the finger itself"
    assert at(mask, index[0] + 0.006, index[1]) > 200, "its own side of the gap"
    assert at(mask, index[0] + 0.014, index[1]) == 0, "the middle finger's side"


def test_the_finger_skin_ends_with_the_fingertip():
    observed = hand()
    tip = observed.landmarks[HAND_DIGITS["index"][3]]
    mask = skin_of(observed)
    assert at(mask, tip[0], tip[1]) > 200
    assert at(mask, tip[0], tip[1] - 0.03) == 0, "not the air beyond the tip"
