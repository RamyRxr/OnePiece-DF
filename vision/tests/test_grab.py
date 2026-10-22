"""Tests for pinch-grab-pull, the main interaction.

The face here is laid out in raw (unmirrored) camera coordinates, which is
what MediaPipe reports: the subject's LEFT is on the image's RIGHT (larger x).

No camera, no MediaPipe: frames are built by hand.
"""

from __future__ import annotations

import pytest

from ofp_vision.config import GrabConfig
from ofp_vision.grab import GrabDetector
from ofp_vision.indices import (
    FACE_CHEEK_BONE_L,
    FACE_CHEEK_BONE_R,
    FACE_CHEEK_L,
    FACE_CHEEK_R,
    FACE_CHIN,
    FACE_EAR_LEVEL_L,
    FACE_EAR_LEVEL_R,
    FACE_EYE_OUTER_L,
    FACE_EYE_OUTER_R,
    FACE_JAW_L,
    FACE_JAW_R,
    FACE_LIP_BOTTOM_OUTER,
    FACE_LIP_TOP_OUTER,
    FACE_MOUTH_L,
    FACE_MOUTH_R,
    FACE_NOSE_TIP,
    HAND_DIGITS,
    HAND_MIDDLE_MCP,
    HAND_WRIST,
)
from ofp_vision.signals import Frame, HandObservation

ASPECT = 4 / 3
FRAME_MS = 33.0

CHEEK_L = (0.58, 0.55)
NOSE = (0.50, 0.50)
EAR_R_SILHOUETTE = (0.30, 0.50)
# The ear sits outside the face mesh, out from the silhouette point.
EAR_R = (0.264, 0.50)


def make_face(dx: float = 0.0, dy: float = 0.0) -> list[tuple[float, float, float]]:
    points = [(0.5 + dx, 0.5 + dy, 0.0)] * 478
    layout = {
        FACE_EYE_OUTER_R: (0.40, 0.40),
        FACE_EYE_OUTER_L: (0.60, 0.40),
        FACE_CHEEK_R: (0.42, 0.55),
        FACE_CHEEK_L: CHEEK_L,
        FACE_CHEEK_BONE_R: EAR_R_SILHOUETTE,
        FACE_CHEEK_BONE_L: (0.70, 0.50),
        FACE_EAR_LEVEL_R: EAR_R_SILHOUETTE,
        FACE_EAR_LEVEL_L: (0.70, 0.50),
        FACE_NOSE_TIP: NOSE,
        FACE_CHIN: (0.50, 0.72),
        FACE_JAW_R: (0.36, 0.64),
        FACE_JAW_L: (0.64, 0.64),
        FACE_MOUTH_R: (0.46, 0.61),
        FACE_MOUTH_L: (0.54, 0.61),
        FACE_LIP_TOP_OUTER: (0.50, 0.59),
        FACE_LIP_BOTTOM_OUTER: (0.50, 0.63),
    }
    points = list(points)
    for index, (x, y) in layout.items():
        points[index] = (x + dx, y + dy, 0.0)
    return points


def make_hand(at: tuple[float, float], gap: float, side: str = "Right") -> HandObservation:
    """A hand whose thumb and index tips straddle `at`, `gap` palm lengths apart."""
    x, y = at
    palm = 0.1
    points = [(x, y + 0.1, 0.0)] * 21
    points = list(points)
    points[HAND_WRIST] = (x, y + 0.15, 0.0)
    points[HAND_MIDDLE_MCP] = (x, y + 0.05, 0.0)
    points[HAND_DIGITS["thumb"][3]] = (x, y - gap * palm / 2, 0.0)
    points[HAND_DIGITS["index"][3]] = (x, y + gap * palm / 2, 0.0)
    return HandObservation(handedness=side, landmarks=points)


OPEN = 0.6
PINCHED = 0.05


class Session:
    """Drives a detector frame by frame with a running clock."""

    def __init__(self, config: GrabConfig | None = None, mirror: bool = True) -> None:
        self.detector = GrabDetector(config or GrabConfig(), mirror=mirror, aspect=ASPECT)
        self.t = 0.0

    def step(self, hands, face=None):
        face = make_face() if face is None else face
        frame = Frame(t_ms=self.t, face=face or None, hands=list(hands))
        state = self.detector.update(frame, self.t)
        self.t += FRAME_MS
        return state

    def idle(self, frames: int):
        state = None
        for _ in range(frames):
            state = self.detector.update(Frame(t_ms=self.t, face=make_face()), self.t)
            self.t += FRAME_MS
        return state

    def grab(self, at=CHEEK_L):
        self.step([make_hand(at, OPEN)])
        return self.step([make_hand(at, PINCHED)])

    def hold(self, at, frames: int = 20, face=None):
        state = None
        for _ in range(frames):
            state = self.step([make_hand(at, PINCHED)], face=face)
        return state


def interocular() -> float:
    return 0.20 * ASPECT


# --- grabbing ----------------------------------------------------------------


def test_pinching_on_a_cheek_grabs_it():
    state = Session().grab()
    assert state.active is True
    assert state.just_grabbed is True
    assert state.target == "cheekL"


def test_the_nearest_target_wins():
    state = Session().grab(at=EAR_R)
    assert state.target == "earR"


def test_pinching_away_from_the_face_does_not_grab():
    state = Session().grab(at=(0.9, 0.9))
    assert state.active is False


def test_an_already_pinched_hand_does_not_grab_what_it_passes_over():
    session = Session()
    session.hold((0.9, 0.9), frames=15)
    state = session.hold(CHEEK_L, frames=5)
    assert state.active is False


def test_a_grab_starts_with_zero_stretch():
    session = Session()
    session.grab()
    state = session.hold(CHEEK_L, frames=5)
    assert state.stretch == pytest.approx(0.0, abs=1e-6)


# --- pulling -----------------------------------------------------------------


def test_pulling_away_grows_the_stretch_in_face_units():
    session = Session()
    session.grab()
    state = session.hold((CHEEK_L[0] + 0.1, CHEEK_L[1]))
    assert state.stretch == pytest.approx(0.1 * ASPECT / interocular(), abs=0.02)
    assert state.peak >= state.stretch - 1e-9


def test_moving_the_head_with_the_hand_is_not_a_stretch():
    """The anchor follows the face. It used to be frozen at grab time, so
    any head movement read as a pull."""
    session = Session()
    session.grab()
    moved = make_face(dx=0.1, dy=0.05)
    state = session.hold((CHEEK_L[0] + 0.1, CHEEK_L[1] + 0.05), face=moved)
    assert state.active is True
    assert state.stretch == pytest.approx(0.0, abs=0.01)


def test_stretch_x_is_reported_from_the_players_point_of_view():
    """+x on the raw image is towards the subject's left. The wire uses +x =
    the player's right, so a pull towards the image's +x reads negative."""
    session = Session(mirror=True)
    session.grab()
    state = session.hold((CHEEK_L[0] + 0.1, CHEEK_L[1]))
    assert state.stretch_x < 0
    assert state.dir_x == pytest.approx(-1.0, abs=0.01)


def test_an_unmirrored_setup_keeps_image_x():
    session = Session(mirror=False)
    session.grab()
    state = session.hold((CHEEK_L[0] + 0.1, CHEEK_L[1]))
    assert state.stretch_x > 0


# --- releasing ---------------------------------------------------------------


def test_unpinching_releases_with_the_peak_of_the_pull():
    session = Session()
    session.grab()
    held = session.hold((CHEEK_L[0] + 0.1, CHEEK_L[1]))
    state = session.step([make_hand((CHEEK_L[0] + 0.1, CHEEK_L[1]), OPEN)])
    assert state.active is False
    assert state.just_released is True
    assert state.release_peak == pytest.approx(held.peak)


def test_the_release_frame_still_names_what_was_let_go():
    session = Session()
    session.grab()
    session.hold(CHEEK_L, frames=3)
    released = session.step([make_hand(CHEEK_L, OPEN)])
    after = session.step([make_hand(CHEEK_L, OPEN)])
    assert released.features()["grab.target"] == 1.0
    assert after.features()["grab.target"] == 0.0
    assert after.just_released is False


def test_a_wobbly_pinch_does_not_drop_the_grab():
    """Between pinch_on and pinch_off the pinch holds (hysteresis)."""
    config = GrabConfig()
    wobble = (config.pinch_on + config.pinch_off) / 2
    session = Session(config)
    session.grab()
    state = session.step([make_hand(CHEEK_L, wobble)])
    assert state.active is True


def test_overpulling_snaps_and_counts_as_a_release():
    config = GrabConfig()
    session = Session(config)
    session.grab()
    far = CHEEK_L[0] - (config.snap_distance + 0.5) * interocular() / ASPECT
    state = session.hold((far, CHEEK_L[1]), frames=30)
    assert state.active is False


def test_overpulling_fires_exactly_one_release():
    config = GrabConfig()
    session = Session(config)
    session.grab()
    far = CHEEK_L[0] - (config.snap_distance + 0.5) * interocular() / ASPECT
    releases = 0
    for _ in range(30):
        state = session.step([make_hand((far, CHEEK_L[1]), PINCHED)])
        releases += state.just_released
    assert releases == 1


# --- tracking dropouts -------------------------------------------------------


def test_a_brief_dropout_keeps_the_grab():
    session = Session()
    session.grab()
    session.idle(3)  # ~100ms with no hands
    state = session.hold(CHEEK_L, frames=1)
    assert state.active is True


def test_a_long_dropout_cancels_without_firing():
    session = Session()
    session.grab()
    state = session.idle(15)  # ~500ms
    assert state.active is False
    assert state.release_peak == 0.0


# --- wire features -----------------------------------------------------------


def test_features_cover_every_grab_channel():
    from ofp_vision.config import GRAB_CHANNELS

    state = Session().grab()
    features = state.features("grab.Right")
    assert set(features) == {name for name in GRAB_CHANNELS if name.startswith("grab.Right.")}
    assert features["grab.Right.active"] == 1.0
    assert features["grab.Right.target"] == 1.0


def test_idle_features_are_all_zero():
    state = Session().idle(1)
    assert all(value == 0.0 for value in state.features().values())


# --- review follow-ups -------------------------------------------------------


def test_a_fast_yank_keeps_most_of_its_peak():
    """A 1.5-interocular yank in ~165ms, released on the next frame.

    The filter used to be tuned in image units, which left it a fixed 2Hz
    low-pass: the release reported ~60% of the real pull.
    """
    session = Session()
    session.grab()
    target = 1.5
    steps = 5
    for i in range(1, steps + 1):
        dx = -target * interocular() / ASPECT * i / steps
        held = session.step([make_hand((CHEEK_L[0] + dx, CHEEK_L[1]), PINCHED)])
    released = session.step([make_hand((CHEEK_L[0] + dx, CHEEK_L[1]), OPEN)])
    assert released.just_released is True
    assert released.release_peak > 0.85 * target
    assert held.peak == released.release_peak


def test_a_handedness_flip_mid_pull_does_not_drop_the_grab():
    """MediaPipe sometimes relabels a hand. The pinch is the same pinch."""
    session = Session()
    session.grab()
    at = (CHEEK_L[0] + 0.05, CHEEK_L[1])
    session.hold(at, frames=10)
    state = None
    for _ in range(15):  # ~500ms under the other label
        state = session.step([make_hand(at, PINCHED, side="Left")])
    assert state.active is True
    assert state.stretch > 0.2


# --- target set --------------------------------------------------------------


@pytest.mark.parametrize(
    "at, expected",
    [
        (NOSE, "nose"),
        ((0.50, 0.72), "chin"),
        ((0.64, 0.64), "jawL"),
        ((0.54, 0.61), "mouthL"),
        ((0.50, 0.585), "lipTop"),
        ((0.736, 0.50), "earL"),
    ],
)
def test_every_listed_part_can_be_grabbed(at, expected):
    state = Session().grab(at=at)
    assert state.target == expected


def test_temples_are_no_longer_targets():
    state = Session().grab(at=(0.32, 0.38))
    assert state.active is False


def test_target_codes_are_unique():
    from ofp_vision.grab import TARGET_CODES

    assert len(set(TARGET_CODES.values())) == len(TARGET_CODES)
    assert 0 not in TARGET_CODES.values()


# --- fingers -----------------------------------------------------------------


def other_hand(tip_at: tuple[float, float]) -> HandObservation:
    """A left hand pointing up with its index fingertip at `tip_at`.

    Palm length 0.1 (vertical), fingers fanned out so each tip is distinct.
    """
    x, y = tip_at
    points = [(x, y + 0.2, 0.0)] * 21
    points = list(points)
    points[HAND_WRIST] = (x, y + 0.25, 0.0)
    points[HAND_MIDDLE_MCP] = (x, y + 0.15, 0.0)
    for offset, digit in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        mcp, pip, dip, tip = HAND_DIGITS[digit]
        fx = x + (offset - 1) * 0.04
        points[mcp] = (fx, y + 0.15, 0.0)
        points[pip] = (fx, y + 0.10, 0.0)
        points[dip] = (fx, y + 0.05, 0.0)
        points[tip] = (fx, y, 0.0)
    points[HAND_MIDDLE_MCP] = (x, y + 0.15, 0.0)  # keep the palm exactly 0.1
    return HandObservation(handedness="Left", landmarks=points)


FINGER_TIP = (0.85, 0.30)


def test_pinching_the_other_hands_fingertip_grabs_that_finger():
    session = Session()
    left = other_hand(FINGER_TIP)
    session.step([make_hand(FINGER_TIP, OPEN), left])
    state = session.step([make_hand(FINGER_TIP, PINCHED), left])
    assert state.active is True
    assert state.target == "finger.Left.index"


def test_a_finger_can_be_grabbed_with_no_face_in_view():
    session = Session()
    left = other_hand(FINGER_TIP)
    session.step([make_hand(FINGER_TIP, OPEN), left], face=[])
    state = session.step([make_hand(FINGER_TIP, PINCHED), left], face=[])
    assert state.target == "finger.Left.index"


def test_pulling_a_finger_stretches_it_in_palm_lengths():
    session = Session()
    left = other_hand(FINGER_TIP)
    session.step([make_hand(FINGER_TIP, OPEN), left])
    session.step([make_hand(FINGER_TIP, PINCHED), left])
    pulled = (FINGER_TIP[0], FINGER_TIP[1] - 0.1)  # one palm length up
    state = None
    for _ in range(20):
        state = session.step([make_hand(pulled, PINCHED), left])
    assert state.stretch == pytest.approx(1.0, abs=0.05)


def test_a_hand_cannot_grab_its_own_fingers():
    """The pinching hand's own fingertips are where the pinch is."""
    session = Session()
    state = session.grab(at=FINGER_TIP)  # only one hand, far from the face
    assert state.active is False
