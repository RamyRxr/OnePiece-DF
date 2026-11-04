"""Hand signs for the fire: ignition, shake, fist, and the pointing finger.

The stretch interaction (`grab.py`) watches a pinch down to millimetres. These
signs are coarser, whole-hand shapes, which is what a webcam can read from a
hand at arm's length:

    ignite      a fist that opens again quickly - light the hand
    shake       a rapid side-to-side wrist shake - put the fire out
    fist        all four fingers curled (swells the flame)
    index_only  the index out, the other three curled, thumb anywhere
    snap        thumb tip flicks off the index or middle tip (kept as a sign;
                it is no longer the fire's trigger)

Pure geometry over landmarks, scale-invariant against the hand's own palm
length, so it holds at any distance from the lens. Kept apart from the fire
effect so the signs can be tested - and later reused on the wire - without
drawing anything. Thresholds live in `config.FireConfig`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import FireConfig
from .indices import HAND_DIGIT_ORDER, HAND_DIGITS, HAND_MIDDLE_MCP, HAND_WRIST
from .signals import HandObservation

FINGERS = ("index", "middle", "ring", "pinky")
# The order fingertips light up in. The thumb is the capstone: it is often
# already "out" on an open hand, so letting it drive the count would let a
# player skip straight to five flames. It only lights once all four fingers
# have been earned.
FLAME_FINGERS = FINGERS + ("thumb",)

THUMB_TIP = HAND_DIGITS["thumb"][3]
INDEX_TIP = HAND_DIGITS["index"][3]
MIDDLE_TIP = HAND_DIGITS["middle"][3]


def _dist(a, b, aspect: float) -> float:
    """Distance with x scaled by the frame's aspect, so a circle stays round."""
    return math.hypot((a[0] - b[0]) * aspect, a[1] - b[1])


def palm_length(landmarks, aspect: float) -> float:
    """Palm length (wrist -> middle knuckle) in normalised, aspect-corrected
    units. Everything below is a ratio against it. Zero for a bad hand."""
    if len(landmarks) <= HAND_MIDDLE_MCP:
        return 0.0
    return _dist(landmarks[HAND_WRIST], landmarks[HAND_MIDDLE_MCP], aspect)


def _cos(a, b) -> float:
    na = math.hypot(a[0], a[1])
    nb = math.hypot(b[0], b[1])
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return (a[0] * b[0] + a[1] * b[1]) / (na * nb)


def _straightness(lm, mcp: int, pip: int, dip: int, tip: int) -> float:
    """Mean cosine of the finger's consecutive segments. 1 = straight."""
    segments = []
    for a, b in ((mcp, pip), (pip, dip), (dip, tip)):
        segments.append((lm[b][0] - lm[a][0], lm[b][1] - lm[a][1]))
    return (_cos(segments[0], segments[1]) + _cos(segments[1], segments[2])) / 2.0


def digit_geometry(lm, digit: str, palm: float, aspect: float) -> tuple[float, float]:
    """(reach, straightness) for one finger: tip-to-knuckle in palm lengths,
    and how straight its joints are (1 = straight, negative = folded back)."""
    mcp, pip, dip, tip = HAND_DIGITS[digit]
    reach = _dist(lm[tip], lm[mcp], aspect) / palm
    return reach, _straightness(lm, mcp, pip, dip, tip)


def _out(reach: float, straight: float, config: FireConfig, was_out: bool) -> bool:
    """Whether a finger is out, with hysteresis: once out it stays out while
    it is either still straight or still long, so a sign hovering on a
    threshold (or a finger foreshortened by the camera angle) cannot flicker."""
    if was_out:
        return straight > config.in_straight or reach > config.in_reach
    return straight > config.out_straight and reach > config.out_reach


def _signs(out: dict[str, bool]) -> tuple[bool, bool]:
    fist = not any(out[digit] for digit in FINGERS)
    index_only = out["index"] and not any(out[digit] for digit in ("middle", "ring", "pinky"))
    return fist, index_only


@dataclass(frozen=True)
class HandSigns:
    """One hand's signs this frame."""

    fist: bool = False
    index_only: bool = False
    # True on the one frame a snap was heard (a close and fast open).
    snap: bool = False
    # True on the one frame a fist opened quickly enough to light the fire.
    ignite: bool = False
    # True on the one frame a rapid side-to-side shake was detected.
    shake: bool = False
    out: dict[str, bool] = field(default_factory=dict)


def signs(hand: HandObservation, config: FireConfig | None = None, aspect: float = 4 / 3) -> HandSigns:
    """The hand's signs now, without any memory (for tests and diagnostics)."""
    config = config or FireConfig()
    lm = hand.landmarks
    palm = palm_length(lm, aspect)
    if palm < 1e-6:
        return HandSigns()
    out = {digit: _out(*digit_geometry(lm, digit, palm, aspect), config, False) for digit in HAND_DIGIT_ORDER}
    fist, index_only = _signs(out)
    return HandSigns(fist=fist, index_only=index_only, out=out)


class SignTracker:
    """One hand's signs across frames: hysteresis, ignition, shake, snap.

    Ignition is a fist that opens again inside `ignite_window_ms`: remember
    when the fist started, and fire once when it opens. A fist held longer is
    just a fist, not a delayed ignition.

    Extinguish is a shake: the wrist's horizontal position (in palm lengths)
    must swing back and forth enough times inside `shake_window_ms`. Normal
    drifting and a single sweep have no direction reversals and cannot fire.

    A snap is a pinch (thumb tip to index or middle tip) that goes from closed
    to open in less than `snap_window_ms`, with the snapped finger coming back
    out. It is still reported as a sign, but the fire no longer uses it.
    """

    def __init__(self, config: FireConfig | None = None) -> None:
        self.config = config or FireConfig()
        self._out: dict[str, bool] = {}
        self._armed_at: float | None = None
        # Which finger the snap was against: the pair that touched is the one
        # that has to open, or a nearby index finger would mask a middle-finger
        # snap's release.
        self._armed_digit = "middle"
        # True while the hand is still pinched. A pinch held past the snap
        # window disarms (it is too slow to be a snap), but it must not re-arm
        # on the next still-closed frame - otherwise letting go of a long
        # pinch reads as a snap.
        self._pinched = False
        self._last_snap_ms = -1e9
        # Ignition: when the current fist began, and the last time we lit.
        self._was_fist = False
        self._fist_since: float | None = None
        self._last_ignite_ms = -1e9
        # Shake: recent (time, wrist x in palm lengths) samples.
        self._motion: list[tuple[float, float]] = []
        self._last_shake_ms = -1e9

    def _is_shake(self, t_ms: float) -> bool:
        """True when the recent wrist motion is a rapid back-and-forth."""
        if len(self._motion) < 4:
            return False
        if t_ms - self._last_shake_ms < self.config.shake_cooldown_ms:
            return False
        values = [value for _, value in self._motion]
        if max(values) - min(values) < self.config.shake_span:
            return False
        directions: list[int] = []
        for (_, a), (_, b) in zip(self._motion, self._motion[1:]):
            delta = b - a
            if abs(delta) >= self.config.shake_step:
                directions.append(1 if delta > 0 else -1)
        reversals = sum(1 for a, b in zip(directions, directions[1:]) if a != b)
        if reversals < self.config.shake_reversals:
            return False
        # One shake, one event: start the window over so the same swing cannot
        # fire again on the next frame.
        self._last_shake_ms = t_ms
        self._motion.clear()
        return True

    def update(self, hand: HandObservation, t_ms: float, aspect: float = 4 / 3) -> HandSigns:
        lm = hand.landmarks
        palm = palm_length(lm, aspect)
        if palm < 1e-6:
            self._out = {}
            self._fist_since = None
            self._was_fist = False
            self._motion.clear()
            return HandSigns()

        out: dict[str, bool] = {}
        raw: dict[str, bool] = {}
        for digit in HAND_DIGIT_ORDER:
            reach, straight = digit_geometry(lm, digit, palm, aspect)
            raw[digit] = _out(reach, straight, self.config, False)
            out[digit] = _out(reach, straight, self.config, self._out.get(digit, False))
        self._out = out
        fist, index_only = _signs(out)

        # Ignition is a fist that opens again quickly. Remember when the fist
        # started; opening inside the window is the event. A fist held longer
        # is just a fist.
        ignite = False
        if fist:
            if not self._was_fist:
                self._fist_since = t_ms
        else:
            if (
                self._was_fist
                and self._fist_since is not None
                and t_ms - self._fist_since <= self.config.ignite_window_ms
                and t_ms - self._last_ignite_ms >= self.config.ignite_cooldown_ms
            ):
                ignite = True
                self._last_ignite_ms = t_ms
            self._fist_since = None
        self._was_fist = fist

        # Extinguish is a fast shake. The wrist's x in palm lengths makes the
        # thresholds hold at any distance from the camera.
        self._motion.append((t_ms, lm[HAND_WRIST][0] * aspect / palm))
        cutoff = t_ms - self.config.shake_window_ms
        self._motion = [sample for sample in self._motion if sample[0] >= cutoff]
        shake = self._is_shake(t_ms)

        gaps = {
            "index": _dist(lm[THUMB_TIP], lm[INDEX_TIP], aspect),
            "middle": _dist(lm[THUMB_TIP], lm[MIDDLE_TIP], aspect),
        }
        gaps = {digit: gap / palm for digit, gap in gaps.items()}
        closest = min(gaps, key=gaps.get)

        snap = False
        if gaps[closest] <= self.config.snap_gap:
            if self._armed_at is None and not self._pinched:
                self._armed_at = t_ms
                self._armed_digit = closest
            self._pinched = True
        else:
            if self._armed_at is not None and gaps[self._armed_digit] >= self.config.snap_open:
                elapsed = t_ms - self._armed_at
                if (
                    elapsed <= self.config.snap_window_ms
                    and t_ms - self._last_snap_ms >= self.config.snap_cooldown_ms
                    and raw[self._armed_digit]
                ):
                    snap = True
                    self._last_snap_ms = t_ms
                self._armed_at = None
            if self._pinched and gaps[self._armed_digit] >= self.config.snap_open:
                self._pinched = False
        if self._armed_at is not None and t_ms - self._armed_at > self.config.snap_window_ms:
            self._armed_at = None

        return HandSigns(
            fist=fist,
            index_only=index_only,
            snap=snap,
            ignite=ignite,
            shake=shake,
            out=out,
        )
