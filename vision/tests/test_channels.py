"""Tests for the per-channel state machine.

The behaviour that matters here is the *sign* of a pull. A tug that shrinks
a measurement is as much of a pull as one that grows it, and the ear gesture
(a downward tug) only produces the shrinking kind. A sign-sensitive
threshold silently made the whole ear gesture undetectable, which is exactly
the class of bug the Signal Lab exists to catch.

No camera, no MediaPipe: features are fed in directly.
"""

from __future__ import annotations

import pytest

from ofp_vision.channels import ChannelManager
from ofp_vision.config import Config


def make_manager() -> ChannelManager:
    manager = ChannelManager(Config())
    manager.set_baseline({"face.cheekL": 0.0, "face.earL": 0.0})
    return manager


def drive(manager: ChannelManager, values: list[float], channel: str = "face.cheekL"):
    """Feed a sequence of raw values for one channel, return (events, states)."""
    events = []
    t = 0.0
    for value in values:
        events += manager.update({channel: value}, t)
        t += 33.0
    return events, manager.channels[channel]


def rest(frames: int = 10) -> list[float]:
    return [0.0] * frames


# --- activation --------------------------------------------------------------


def test_a_positive_excursion_activates():
    manager = make_manager()
    _, state = drive(manager, rest() + [0.05] * 10)
    assert state.active is True
    assert state.magnitude == pytest.approx(0.05, abs=0.01)


def test_a_negative_excursion_activates_too():
    """A downward tug shrinks the measurement. It must still fire."""
    manager = make_manager()
    _, state = drive(manager, rest() + [-0.05] * 10)
    assert state.active is True
    assert state.magnitude == pytest.approx(0.05, abs=0.01)


def test_a_small_excursion_stays_below_the_threshold():
    manager = make_manager()
    _, state = drive(manager, rest() + [0.01] * 20)
    assert state.active is False


def test_magnitude_is_the_absolute_value():
    manager = make_manager()
    _, state = drive(manager, rest() + [-0.06] * 10)
    assert state.magnitude == pytest.approx(abs(state.value), abs=1e-9)
    assert state.direction == -1


# --- release -----------------------------------------------------------------


def test_a_positive_pull_fires_a_release():
    manager = make_manager()
    events, _ = drive(manager, rest() + [0.06] * 20 + rest(20))
    assert [e.channel for e in events] == ["face.cheekL"]
    assert events[0].peak == pytest.approx(0.06, abs=0.02)


def test_a_negative_pull_fires_a_release():
    """The bug this test exists for: the ear gesture's downward tug."""
    manager = make_manager()
    events, _ = drive(manager, rest() + [-0.06] * 20 + rest(20))
    assert [e.channel for e in events] == ["face.cheekL"]


def test_release_peak_is_a_magnitude_even_for_a_negative_pull():
    manager = make_manager()
    events, _ = drive(manager, rest() + [-0.08] * 20 + rest(20))
    assert events[0].peak > 0.0
    assert events[0].signed_peak < 0.0


def test_a_single_pull_does_not_fire_twice():
    manager = make_manager()
    events, _ = drive(manager, rest() + [0.06] * 20 + rest(20) + [0.06] * 20 + rest(20))
    assert len(events) == 2, "one release per pull, two pulls expected"


def test_a_second_pull_after_a_rest_fires_again():
    manager = make_manager()
    events, _ = drive(manager, rest() + [0.06] * 20 + rest(20) + [0.06] * 20 + rest(20))
    assert len(events) == 2


def test_no_release_fires_while_the_pull_is_held():
    manager = make_manager()
    events, _ = drive(manager, rest() + [0.06] * 40)
    assert events == []


def test_a_fast_snap_back_fires_before_reaching_rest():
    """Releasing hard should fire on the contraction, not on the slow decay."""
    manager = make_manager()
    events, _ = drive(manager, rest() + [0.08] * 20 + [0.0] * 2)
    assert len(events) == 1


def test_a_slow_decay_still_fires_the_release():
    """A gentle let-go must not be swallowed by the velocity test.

    The decay here is slow enough that the magnitude velocity never reaches
    the contraction limit, so the release has to come from the "relaxed back
    to rest" branch instead.
    """
    manager = make_manager()
    decay = [max(0.0, 0.08 - 0.003 * i) for i in range(30)]
    events, _ = drive(manager, rest() + [0.08] * 20 + decay)
    assert len(events) == 1
    assert events[0].peak == pytest.approx(0.08, abs=0.02)


# --- lost sources ------------------------------------------------------------


def test_a_lost_source_clears_the_pull():
    manager = make_manager()
    events, state = drive(manager, rest() + [0.06] * 20)
    assert state.latched is True
    # Face leaves the frame: the channel is marked lost and the pull reset.
    manager.update({}, 1000.0)
    state = manager.channels["face.cheekL"]
    assert state.lost is True
    assert state.latched is False
    assert state.active is False


# --- protocol ----------------------------------------------------------------


def test_the_protocol_carries_the_magnitude():
    from ofp_vision.protocol import PROTOCOL_VERSION, build_message

    manager = make_manager()
    manager.update({"face.cheekL": -0.05}, 0.0)
    message = build_message(0.0, manager.channels, [], {"face": 1})
    channel = message["ch"]["face.cheekL"]
    assert channel["m"] == pytest.approx(abs(channel["v"]), abs=1e-4)
    assert channel["v"] < 0.0
    assert message["v"] == PROTOCOL_VERSION


# --- composite channels ------------------------------------------------------


def composite_manager() -> ChannelManager:
    """A manager with the face pull composites wired up."""
    manager = ChannelManager(Config())
    manager.set_baseline(
        {
            "face.tragChinL": 1.0,
            "face.tragForeheadL": 1.0,
            "face.tragChinR": 1.0,
            "face.tragForeheadR": 1.0,
            "face.jawOpen": 0.0,
        }
    )
    return manager


def test_the_pull_composite_is_the_norm_of_its_sources():
    manager = composite_manager()
    manager.update(
        {
            "face.tragChinL": 1.03,
            "face.tragForeheadL": 1.04,
            "face.tragChinR": 1.0,
            "face.tragForeheadR": 1.0,
            "face.jawOpen": 0.0,
        },
        0.0,
    )
    state = manager.channels["face.pullL"]
    assert state.value == pytest.approx(0.05, abs=0.01)
    assert manager.channels["face.pullR"].value == pytest.approx(0.0, abs=1e-6)


def test_the_pull_composite_fires_in_either_direction():
    """The whole point: the two sources can move in any combination."""
    manager = composite_manager()
    for chin, brow in ((1.05, 1.0), (1.0, 1.05), (0.95, 1.0), (1.0, 0.95)):
        manager = composite_manager()
        manager.update(
            {
                "face.tragChinL": chin,
                "face.tragForeheadL": brow,
                "face.tragChinR": 1.0,
                "face.tragForeheadR": 1.0,
                "face.jawOpen": 0.0,
            },
            0.0,
        )
        state = manager.channels["face.pullL"]
        assert state.value == pytest.approx(0.05, abs=0.01), (chin, brow)
        assert state.active is True, (chin, brow)


def test_the_pull_asymmetry_separates_the_two_sides():
    manager = composite_manager()
    manager.update(
        {
            "face.tragChinL": 1.05,
            "face.tragForeheadL": 1.05,
            "face.tragChinR": 1.0,
            "face.tragForeheadR": 1.0,
            "face.jawOpen": 0.0,
        },
        0.0,
    )
    assert manager.channels["face.pullAsym"].value == pytest.approx(0.0707, abs=0.01)
    assert manager.channels["face.pullAsym"].active is True


def test_a_composite_is_lost_when_its_source_is_lost():
    manager = composite_manager()
    manager.update(
        {
            "face.tragChinL": 1.05,
            "face.tragForeheadL": 1.05,
            "face.tragChinR": 1.0,
            "face.tragForeheadR": 1.0,
            "face.jawOpen": 0.0,
        },
        0.0,
    )
    manager.update({}, 33.0)
    assert manager.channels["face.pullL"].lost is True
    assert manager.channels["face.pullAsym"].lost is True


# --- jaw gate ----------------------------------------------------------------


def test_an_open_jaw_suppresses_the_face_pull_channels():
    """Opening the jaw swings the chin, which would look like a pull."""
    manager = composite_manager()
    manager.update(
        {
            "face.tragChinL": 1.05,
            "face.tragForeheadL": 1.05,
            "face.tragChinR": 1.0,
            "face.tragForeheadR": 1.0,
            "face.jawOpen": 0.4,
        },
        0.0,
    )
    state = manager.channels["face.pullL"]
    assert state.value == pytest.approx(0.0)
    assert state.active is False


def test_a_closed_jaw_does_not_suppress_anything():
    manager = composite_manager()
    manager.update(
        {
            "face.tragChinL": 1.05,
            "face.tragForeheadL": 1.05,
            "face.tragChinR": 1.0,
            "face.tragForeheadR": 1.0,
            "face.jawOpen": 0.0,
        },
        0.0,
    )
    assert manager.channels["face.pullL"].active is True


# --- time units --------------------------------------------------------------


def test_the_filter_smooths_jitter_at_real_frame_timing():
    """The server passes millisecond timestamps. The filters work in seconds.

    When the milliseconds went straight in, every frame looked 33 *seconds*
    apart, the OneEuroFilter's cutoff became irrelevant and its output was
    exactly as jittery as its input - the smoothing silently did nothing.
    """
    import random
    import statistics

    rng = random.Random(3)
    manager = make_manager()
    raw = [0.05 + rng.uniform(-0.01, 0.01) for _ in range(200)]
    out = []
    for i, value in enumerate(raw):
        manager.update({"face.cheekL": value}, i * 33.0)
        out.append(manager.channels["face.cheekL"].value)
    assert statistics.pstdev(out[50:]) < statistics.pstdev(raw[50:]) * 0.5


def test_velocity_is_reported_per_second():
    manager = make_manager()
    manager.update({"face.cheekL": 0.0}, 0.0)
    manager.update({"face.cheekL": 0.0}, 33.0)
    manager.update({"face.cheekL": 0.033}, 66.0)
    # A 0.033 step over 33ms is ~1/s before smoothing; per-millisecond units
    # would report ~0.001.
    assert manager.channels["face.cheekL"].velocity > 0.1


# --- missing baseline --------------------------------------------------------


def test_a_channel_without_a_baseline_learns_one_instead_of_staying_dead():
    """Hands are rarely in view while calibrating a neutral face.

    A channel with no baseline used to subtract itself and read 0 forever,
    so every finger channel was dead for the whole session.
    """
    manager = make_manager()  # no hand baseline
    frames = Config().detection.lazy_baseline_frames
    t = 0.0
    for _ in range(frames + 5):
        manager.update({"hand.Right.index.reach": 1.0}, t)
        t += 33.0
    for _ in range(20):
        manager.update({"hand.Right.index.reach": 1.6}, t)
        t += 33.0
    state = manager.channels["hand.Right.index.reach"]
    assert manager.baseline["hand.Right.index.reach"] == pytest.approx(1.0)
    assert state.value == pytest.approx(0.6, abs=0.05)
    assert state.active is True


def test_a_channel_reads_zero_while_its_baseline_is_being_learned():
    manager = make_manager()
    manager.update({"hand.Right.index.reach": 1.4}, 0.0)
    state = manager.channels["hand.Right.index.reach"]
    assert state.value == 0.0
    assert state.active is False
    assert state.lost is False


def test_losing_the_source_restarts_baseline_learning():
    manager = make_manager()
    frames = Config().detection.lazy_baseline_frames
    for i in range(frames - 1):
        manager.update({"hand.Right.index.reach": 1.0}, i * 33.0)
    manager.update({}, frames * 33.0)  # hand leaves before the baseline is set
    assert "hand.Right.index.reach" not in manager.baseline


# --- grab channels -----------------------------------------------------------


def grab_features(active: float, stretch: float, target: float, hand: str = "Right") -> dict[str, float]:
    return {
        f"grab.{hand}.active": active,
        f"grab.{hand}.stretch": stretch,
        f"grab.{hand}.stretchX": stretch,
        f"grab.{hand}.stretchY": 0.0,
        f"grab.{hand}.dirX": 1.0 if stretch else 0.0,
        f"grab.{hand}.dirY": 0.0,
        f"grab.{hand}.target": target,
    }


def test_grab_channels_pass_through_unfiltered():
    """grab.target is a category code. Smoothing blended 0 into 4 via 3.99,
    which Unity's int cast read as 3 - the wrong ear."""
    manager = make_manager()
    manager.update(grab_features(0.0, 0.0, 0.0), 0.0)
    manager.update(grab_features(1.0, 0.7, 4.0), 33.0)
    assert manager.channels["grab.Right.target"].value == 4.0
    assert manager.channels["grab.Right.stretch"].value == 0.7
    assert manager.channels["grab.Right.stretch"].active is True


def test_each_hands_grab_channels_are_active_only_while_that_hand_holds():
    manager = make_manager()
    manager.update({**grab_features(1.0, 0.7, 4.0, "Left"), **grab_features(0.0, 0.0, 0.0, "Right")}, 0.0)
    assert manager.channels["grab.Left.stretch"].active is True
    assert manager.channels["grab.Right.stretch"].active is False


def test_grab_channels_never_fire_threshold_releases():
    """The grab has its own release (an unpinch). Direction channels swinging
    about must not produce release events of their own."""
    import math

    manager = make_manager()
    events = []
    for i in range(90):
        features = grab_features(1.0, 0.5, 1.0)
        features["grab.Right.dirX"] = math.cos(i / 5)
        events += manager.update(features, i * 33.0)
    assert events == []


def test_a_channel_resting_slightly_off_zero_still_fires_every_pull():
    """Re-arming used to need the value back under the release threshold
    (0.015). A channel whose rest drifted to 0.025 fired once, then never."""
    manager = make_manager()
    resting = [0.025] * 20
    pull = [0.10] * 15
    events, _ = drive(manager, rest() + pull + resting + pull + resting)
    assert len(events) == 2
