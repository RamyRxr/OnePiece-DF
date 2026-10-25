"""Tests for the rubber finger: a chain with physics, warping the picture.

The physics:
- pulled from a pinch, the part past the pinch travels whole (turning to
  follow the pull), the knuckle end stays on the hand, and the part in
  between gets longer - a little thinner, gradually
- let go, it snaps back and settles; moving the hand carries it, no lag

The picture:
- the finger is the camera's own picture, warped - nothing is erased and
  nothing new is drawn: at rest the warp changes nothing
- the finger's pixels move whole with it; a thin skirt of what is around it
  goes along less and less, so the stretch blends into the picture; beyond
  the skirt, nothing moves - and never the next finger
"""

from __future__ import annotations

import math

import numpy as np

from ofp_vision.fingers import (
    BASE_NODES,
    FINGER_HALF,
    SKIRT,
    FingerSession,
    _thinning,
    apply,
    similarity,
)
from ofp_vision.indices import HAND_DIGITS, HAND_MIDDLE_MCP, HAND_WRIST
from ofp_vision.signals import HandObservation

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


def pulled(by=(150.0, -40.0), frames: int = 30, part: str = PART, observed: HandObservation | None = None):
    observed = observed or hand()
    session = FingerSession.start(part, observed, SIZE)
    joints = HAND_DIGITS[part.split(".")[2]]
    tip = observed.landmarks[joints[3]]
    hold = session.hold(part, np.array([tip[0] * W, tip[1] * H]))
    session.holds = {"Right": hold}
    rest = None
    for f in range(1, frames + 1):
        k = min(1.0, f / 8)
        rest = run(session, observed, {"Right": np.array([by[0] * k, by[1] * k, 0.0])}, 1)
    return session, hold, rest, observed


# --- the physics -------------------------------------------------------------


def test_the_palm_fit_recovers_a_turn_a_scale_and_a_shift():
    src = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 20.0], [7.0, 3.0]])
    turn = np.array([[math.cos(0.3), -math.sin(0.3)], [math.sin(0.3), math.cos(0.3)]])
    dst = 1.5 * src @ turn.T + [4.0, -2.0]
    assert np.allclose(apply(similarity(src, dst), src), dst)


def test_pulled_from_its_tip_the_finger_gets_longer_and_stays_a_finger():
    session, hold, rest, _ = pulled()
    p = session.body.positions
    held, resting = p[hold.core, :2], rest[hold.core, :2]
    under = hold.pivot
    assert np.abs(p[under, :2] - (rest[under, :2] + [150.0, -40.0])).max() < 0.5, "the pinched point goes with the pinch"
    spacing = np.linalg.norm(np.diff(held, axis=0), axis=1)
    assert np.allclose(spacing, np.linalg.norm(np.diff(resting, axis=0), axis=1), atol=0.5), "the tip with it, whole"
    assert np.abs(p[:BASE_NODES] - rest[:BASE_NODES]).max() < 1e-6, "the knuckle end stays on the hand"
    length = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()
    resting_length = np.linalg.norm(np.diff(rest[:, :2], axis=0), axis=1).sum()
    assert length > 1.4 * resting_length, "it got longer"


def test_a_stretched_finger_thins_gradually_not_in_a_step():
    """Where the stretched part meets the knuckle and the held end, its width
    changes smoothly - a sudden notch reads as a seam."""
    session, _, rest, _ = pulled()
    thin = _thinning(session.body.positions[:, :2], rest[:, :2])
    assert thin.min() < 0.95, "a stretched finger is a little thinner"
    assert np.abs(np.diff(thin)).max() < 0.06


def test_let_go_the_finger_snaps_back_and_settles():
    session, _, _, observed = pulled()
    session.holds = {}
    rest = None
    for _ in range(15):
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


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return abs(math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], a @ b)))


def test_pulled_sideways_from_halfway_the_held_end_turns_toward_the_pull():
    """A rubber finger's end swings round to follow the stretch - not an L
    with the tip still pointing where it did."""
    observed = hand()
    session = FingerSession.start(PART, observed, SIZE)
    hold = session.hold(PART, tip_px(observed) + [0.0, 0.1 * H])
    session.holds = {"Right": hold}
    rest = None
    for f in range(1, 31):
        rest = run(session, observed, {"Right": np.array([200.0, 0.0, 0.0]) * min(1.0, f / 8)}, 1)
    p = session.body.positions[:, :2]
    held = p[-1] - p[hold.core[0]]
    stretch = p[hold.core[0]] - p[0]
    resting = rest[-1, :2] - rest[hold.core[0], :2]
    assert angle_between(held, stretch) < 0.5 * angle_between(resting, stretch)


def first_frame_of_a_pull(pull: np.ndarray) -> np.ndarray:
    observed = hand()
    session = FingerSession.start(PART, observed, SIZE)
    session.holds = {"Right": session.hold(PART, tip_px(observed) + [0.0, 0.1 * H])}
    run(session, observed, {"Right": pull}, 1)
    return session.body.positions[session.holds["Right"].core, :2].copy()


def test_dragged_back_past_the_knuckle_the_end_does_not_flip():
    """Straight back over the knuckle the turn has no side; a hair's width
    either way must not swing the held end from one side to the other. (The
    free part between may buckle either way - that is just physics.)"""
    down = np.array([0.0, 2.5 * 0.21 * H, 0.0])
    left = first_frame_of_a_pull(down + [-0.01, 0.0, 0.0])
    right = first_frame_of_a_pull(down + [0.01, 0.0, 0.0])
    assert np.abs(left - right).max() < 1.0


def test_the_pinched_spot_stays_with_the_pinch_as_the_end_turns():
    observed = hand()
    session = FingerSession.start(PART, observed, SIZE)
    anchor = tip_px(observed) + [0.0, 0.1 * H]
    session.holds = {"Right": session.hold(PART, anchor)}
    rest = None
    for f in range(1, 31):
        rest = run(session, observed, {"Right": np.array([200.0, 0.0, 0.0]) * min(1.0, f / 8)}, 1)
    place = int(np.argmin(np.linalg.norm(rest[:, :2] - anchor, axis=1)))
    assert np.abs(session.body.positions[place, :2] - (rest[place, :2] + [200.0, 0.0])).max() < 0.5


def test_a_thumb_stretches_from_its_knuckle_not_from_inside_the_palm():
    """The tracker's first thumb point is by the wrist, inside the palm. A
    stretched thumb leaves the hand at its knuckle, the next one."""
    observed = hand()
    session = FingerSession.start("finger.Left.thumb", observed, SIZE)
    knuckle = observed.landmarks[HAND_DIGITS["thumb"][1]]
    assert np.abs(session.rest()[0, :2] - [knuckle[0] * W, knuckle[1] * H]).max() < 1e-6


# --- the picture -------------------------------------------------------------


def moved(warp) -> np.ndarray:
    """How far each point of the warp has moved from where its pixel is."""
    return np.linalg.norm(warp.positions[:, :2] - warp.source, axis=1)


def test_at_rest_the_warp_changes_nothing():
    """The finger is the camera's own picture: until it is pulled, exactly so."""
    observed = hand()
    session = FingerSession.start(PART, observed, SIZE)
    rest = run(session, observed, {}, 3)
    warp = session.warp(rest, observed, SIZE)
    assert moved(warp).max() < 1e-6


def test_the_finger_moves_whole_and_the_picture_around_it_barely():
    session, hold, rest, observed = pulled()
    warp = session.warp(rest, observed, SIZE)
    grid = moved(warp).reshape(warp.rows, warp.cols)
    middle = warp.cols // 2
    assert grid[-1 - warp.cols // 4 :, middle].max() > 100.0, "down the finger's middle, it went with the pull"
    assert grid[:, 0].max() < 1e-6 and grid[:, -1].max() < 1e-6, "the skirt's edge never moves"
    assert grid[0].max() < 1e-6, "nor the knuckle"


def test_the_stretch_blends_into_the_picture_across_its_skirt():
    """From the finger's edge out, the picture goes along less and less - no
    step where the moving part meets the still one."""
    session, _, rest, observed = pulled()
    warp = session.warp(rest, observed, SIZE)
    row = moved(warp).reshape(warp.rows, warp.cols)[warp.rows // 2]
    outward = row[warp.cols // 2 :]
    assert np.all(np.diff(outward) <= 1e-6), "less and less"
    assert np.abs(np.diff(outward)).max() < 0.5 * outward[0], "never all at once"


def test_a_thumb_moves_wider_than_a_little_finger():
    """A thumb is thicker: more of the picture around its middle goes with it."""
    assert FINGER_HALF["thumb"] > FINGER_HALF["index"] > FINGER_HALF["pinky"]
    assert SKIRT < FINGER_HALF["index"] * 1.5, "the skirt is thin: the picture around barely moves"


def test_the_next_finger_is_never_dragged_along():
    """Fingers together: pulling the index toward the middle finger, the
    middle finger's pixels stay where they are."""
    together = hand(spread=0.02)
    session, _, rest, _ = pulled(by=(80.0, 0.0), observed=together)
    warp = session.warp(rest, together, SIZE)
    middle = [together.landmarks[i] for i in HAND_DIGITS["middle"]]
    line = np.array([[p[0] * W, p[1] * H] for p in middle])
    near_middle = np.min([np.linalg.norm(warp.source - point, axis=1) for point in line], axis=0) < 6.0
    assert near_middle.any()
    assert moved(warp)[near_middle].max() < 1e-6
