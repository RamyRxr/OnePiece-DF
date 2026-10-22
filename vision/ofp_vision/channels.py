"""Per-channel state machine: filter, activate, detect release.

Each channel runs through the same pipeline every frame:

    raw feature -> minus baseline -> OneEuroFilter -> value
                                              -> Slope   -> velocity
                                              -> activation / release detection

Release detection is the interesting part. A pull is a motion, not a pose, so
the signal to watch is the *velocity*: while the player pulls, the value
climbs; the instant they let go, it drops. We fire the event on that
contraction, edge-triggered, and carry the peak value reached since the pull
started so the game knows how hard it was yanked.

Sign convention
---------------
Activation is thresholded on the **magnitude** of the deviation, not its
sign. A displacement is unsigned: tugging a cheek down shrinks the
mouth-to-cheekbone distance while tugging it out grows it, but the player
stretched their face either way. An earlier version only activated on
positive excursions, which silently broke the ear gesture - a downward tug
made every face channel negative and nothing ever fired.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field

from .config import ALL_CHANNELS, COMPOSITE_CHANNELS, GRAB_CHANNELS, Config, group_of
from .filters import OneEuroFilter, Slope

log = logging.getLogger(__name__)


@dataclass
class ChannelState:
    """Everything the game and the lab overlay need to know about one channel."""

    name: str
    value: float = 0.0
    velocity: float = 0.0
    peak: float = 0.0
    active: bool = False
    released: bool = False
    lost: bool = True
    # Set while a pull is in progress; cleared when the release fires so a
    # single pull cannot fire twice.
    latched: bool = False
    # Size of the largest excursion seen during the current pull, regardless
    # of direction. `peak` keeps the signed value it happened at.
    peak_magnitude: float = 0.0
    # Set when a release fires on the snap-back while the value is still
    # high. The channel cannot start a new pull until it has dropped out of
    # the active zone, or the tail of the same pull would read as a second one.
    disarmed: bool = False

    @property
    def group(self) -> str:
        return group_of(self.name)

    @property
    def magnitude(self) -> float:
        """How far the channel is from neutral, ignoring direction.

        This is the number the avatar cares about: a pull stretches the
        character whether the player tugged up or down.
        """
        return abs(self.value)

    @property
    def direction(self) -> int:
        """+1 if the excursion was positive, -1 if negative, 0 at rest."""
        if self.value > 0:
            return 1
        if self.value < 0:
            return -1
        return 0


@dataclass
class ReleaseEvent:
    """Fired when a pull ends.

    `peak` is the *magnitude* of the pull (always >= 0) - that is "how hard
    was it yanked". `signed_peak` keeps the direction it happened in.
    """

    channel: str
    peak: float
    value: float
    t_ms: float
    signed_peak: float = 0.0


@dataclass
class ChannelManager:
    """Owns the filter + state for every channel."""

    config: Config
    baseline: dict[str, float] = field(default_factory=dict)
    channels: dict[str, ChannelState] = field(default_factory=dict)
    events: list[ReleaseEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._filters: dict[str, OneEuroFilter] = {}
        self._slopes: dict[str, Slope] = {}
        # Samples collected for channels that have no baseline yet.
        self._pending: dict[str, list[float]] = {}
        for name in ALL_CHANNELS:
            self.channels[name] = ChannelState(name=name)
            self._filters[name] = OneEuroFilter()
            self._slopes[name] = Slope()

    # --- configuration ------------------------------------------------------

    def _filter_params(self, group: str) -> tuple[float, float]:
        f = self.config.filters
        if group == "hand":
            return f.hand_min_cutoff, f.hand_beta
        if group == "body":
            return f.body_min_cutoff, f.body_beta
        return f.face_min_cutoff, f.face_beta

    def set_baseline(self, baseline: dict[str, float]) -> None:
        self.baseline = dict(baseline)
        self.reset_state()

    def reset_state(self) -> None:
        """Clear filters and channel state, keeping the baseline."""
        self._pending = {}
        for name in ALL_CHANNELS:
            min_cutoff, beta = self._filter_params(group_of(name))
            self._filters[name] = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
            self._slopes[name] = Slope()
            self.channels[name] = ChannelState(name=name)

    # --- per-frame update ---------------------------------------------------

    def _mark_lost(self, state: ChannelState, name: str) -> None:
        """The source vanished (hand out of frame, face turned away).

        Decay toward rest rather than freezing the last value, and forget any
        pull in progress: the gesture is over.
        """
        state.lost = True
        state.value *= 0.75
        state.velocity = 0.0
        state.active = False
        state.released = False
        state.peak = 0.0
        state.peak_magnitude = 0.0
        state.latched = False
        state.disarmed = False
        self._filters[name].reset()
        self._slopes[name].reset()
        # A half-learned baseline is discarded: the next appearance may be
        # in a different pose.
        self._pending.pop(name, None)

    def _learn_baseline(self, name: str, raw: float) -> bool:
        """Collect a sample for a channel with no baseline.

        Returns True once the baseline is set. Until then the channel reads
        zero rather than subtracting itself into a permanent zero.
        """
        samples = self._pending.setdefault(name, [])
        samples.append(raw)
        if len(samples) < self.config.detection.lazy_baseline_frames:
            return False
        self.baseline = {**self.baseline, name: statistics.median(samples)}
        del self._pending[name]
        log.info("Learned a baseline for %s (it was not visible during calibration)", name)
        return True

    def _pass_through(self, name: str, features: dict[str, float], t: float) -> None:
        """Grab channels are already deviations from rest: no baseline, no
        filter, and no threshold release (the grab has its own)."""
        state = self.channels[name]
        if name not in features:
            state.lost = True
            state.value = 0.0
            state.active = False
            return
        state.lost = False
        state.value = features[name]
        state.velocity = self._slopes[name](state.value, t)
        # Active while that hand holds its grab (grab.<hand>.active).
        state.active = features.get(name.rsplit(".", 1)[0] + ".active", 0.0) > 0.5

    def _advance(
        self,
        state: ChannelState,
        filtered: float,
        velocity_limit: float,
        detection,
        t: float,
    ) -> None:
        """Run the activation / release state machine for one channel."""
        # --- activation ---
        #
        # A pull is a *displacement*, and displacement has no sign. Tugging
        # a cheek down shrinks the mouth-to-cheekbone distance while tugging
        # it out grows it - but the player stretched their face either way,
        # so the channel must fire either way. Thresholding the magnitude
        # is what makes the ear gesture (a downward tug) work at all.
        magnitude = abs(filtered)
        # Re-arm once the value has left the active zone. Waiting for a full
        # return to rest would strand a channel whose rest has drifted.
        if state.disarmed and magnitude < detection.activate_threshold:
            state.disarmed = False
        if magnitude >= detection.activate_threshold and not state.disarmed:
            state.active = True
            state.latched = True
            if magnitude > state.peak_magnitude:
                state.peak_magnitude = magnitude
                state.peak = filtered
        else:
            state.active = False

        # --- release ---
        #
        # The magnitude has to be *contracting* for a release. The velocity
        # of the signed value is not that: a negative excursion climbing
        # back toward zero has a positive velocity while the magnitude
        # shrinks. Multiplying the velocity by the sign of the value
        # converts it into "how fast is the magnitude growing".
        released = False
        if state.latched and state.peak_magnitude >= detection.release_threshold:
            if magnitude <= detection.release_threshold:
                # Relaxed all the way back to rest.
                released = True
            else:
                sign = 1.0 if filtered >= 0.0 else -1.0
                magnitude_velocity = state.velocity * sign
                if (
                    magnitude_velocity <= velocity_limit
                    and magnitude < state.peak_magnitude
                ):
                    # Snapped back from the peak.
                    released = True

        if released:
            state.released = True
            state.latched = False
            state.active = False
            state.disarmed = magnitude > detection.release_threshold
            self.events.append(
                ReleaseEvent(
                    channel=state.name,
                    peak=state.peak_magnitude,
                    value=filtered,
                    t_ms=t,
                    signed_peak=state.peak,
                )
            )
            state.peak = 0.0
            state.peak_magnitude = 0.0
        else:
            state.released = False

    def _composite_value(self, rule: str, sources: list[float]) -> float:
        """Combine the source channel values into one."""
        if rule == "norm":
            return math.sqrt(sum(value * value for value in sources))
        if rule == "diff":
            return sources[0] - sources[1]
        raise ValueError(f"unknown composite rule: {rule!r}")

    def update(self, features: dict[str, float], t_ms: float) -> list[ReleaseEvent]:
        """Advance every channel by one frame. Returns any release events.

        `t_ms` is in milliseconds, like every timestamp on the wire. The
        filters and slopes work in seconds - their cutoffs are in Hz and the
        release velocity is "per second" - so it is converted here, once.
        """
        self.events = []
        t = t_ms / 1000.0
        detection = self.config.detection
        # Scale-aware contraction threshold: expressed as a multiple of the
        # activation threshold per second, so it means the same thing for a
        # cheek feature (range ~0.05) and a finger reach feature (range ~3).
        velocity_limit = -detection.release_velocity_ratio * detection.activate_threshold

        values: dict[str, float] = {}
        lost: set[str] = set()

        # --- 1. measured channels ---
        for name in ALL_CHANNELS:
            if name in COMPOSITE_CHANNELS:
                continue
            if name in GRAB_CHANNELS:
                self._pass_through(name, features, t)
                continue
            state = self.channels[name]

            if name not in features:
                self._mark_lost(state, name)
                values[name] = state.value
                lost.add(name)
                continue

            state.lost = False

            if name not in self.baseline and not self._learn_baseline(name, features[name]):
                state.value = 0.0
                state.active = False
                values[name] = 0.0
                continue

            raw = features[name] - self.baseline[name]

            filtered = self._filters[name](raw, t)
            state.value = filtered
            state.velocity = self._slopes[name](filtered, t)
            values[name] = filtered

            self._advance(state, filtered, velocity_limit, detection, t_ms)

        # --- 2. jaw gate ---
        #
        # The face pull composites measure the tragion against the chin, and
        # the chin swings when the jaw opens. A player who opens their mouth
        # mid-pull would otherwise read as a hard tug, so the face pull
        # channels are suppressed while the jaw is away from its calibrated
        # aperture.
        jaw = values.get(detection.jaw_gate_feature, 0.0)
        jaw_blocked = jaw > detection.jaw_gate_threshold

        # --- 3. composite channels ---
        for name in ALL_CHANNELS:
            if name not in COMPOSITE_CHANNELS:
                continue
            state = self.channels[name]
            rule, sources = COMPOSITE_CHANNELS[name]

            if any(source in lost for source in sources):
                self._mark_lost(state, name)
                values[name] = state.value
                lost.add(name)
                continue

            if jaw_blocked and name.startswith("face."):
                # Suppressed: report rest, and do not fire a release event.
                state.value = 0.0
                state.velocity = 0.0
                state.active = False
                state.released = False
                state.peak = 0.0
                state.peak_magnitude = 0.0
                state.latched = False
                self._slopes[name].reset()
                values[name] = 0.0
                continue

            # A composite is already a deviation from rest, so it is not
            # baseline-subtracted again. Its sources are filtered, so it does
            # not need filtering either.
            filtered = self._composite_value(rule, [values[source] for source in sources])
            state.lost = False
            state.value = filtered
            state.velocity = self._slopes[name](filtered, t)
            values[name] = filtered

            self._advance(state, filtered, velocity_limit, detection, t_ms)

        return self.events

    # --- introspection ------------------------------------------------------

    def snapshot(self) -> dict[str, ChannelState]:
        return self.channels

    def gameplay_values(self) -> dict[str, float]:
        """Just the values the game binds to, for quick inspection."""
        from .config import GAMEPLAY_CHANNELS

        return {name: self.channels[name].value for name in GAMEPLAY_CHANNELS}
