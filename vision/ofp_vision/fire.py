"""Ace's fire: the hand alight, a flame on every earned fingertip.

The lab's rendering of the devil fruit, driven by `gestures.py`:

    fist, then open                   light the hand
    shake the hand                    put the fire out
    fist                              the flame swells
    raise fingers one by one          earn a flame on each fingertip
    only the index up                 a fireball floats above its tip

The hand is lit rather than painted over: its real silhouette (the
segmentation's body skin, or the landmark glove when that is missing) gets a
hot rim and a faint body tint, and flame streaks rise off its edge. Each
fingertip carries its own teardrop flame. Flames are earned sequentially:
opening the whole hand at once lights only the first finger, and the count can
only grow by one at a time. The thumb is the capstone and lights only after
all four fingers.

Ignition is a flash and a burst of embers; putting it out is a puff of smoke
and embers falling away. Everything is advanced by the frame clock
(`Frame.t_ms`), never by wall time, so a recording replays the same and tests
can step it. Particles live in image space but ride the hand's translation,
so the fire stays attached while the flames still rise upward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import FireConfig
from .gestures import FLAME_FINGERS, SignTracker, palm_length
from .skin import Window, hand_silhouette
from .signals import Frame, HandObservation
from .indices import HAND_DIGITS

# Flame colour ramp (BGR). Low heat is the dark red edge of a flame, high heat
# its white core. The colour is picked from this by heat, so the shape of the
# hot region is all the fire's shape needs.
RAMP_AT = np.array([0.00, 0.12, 0.30, 0.52, 0.72, 0.88, 1.00], dtype=np.float32)
RAMP_BGR = np.array(
    [
        (0.0, 0.0, 0.0),        # nothing
        (0.0, 0.0, 70.0),       # ember red
        (0.0, 45.0, 170.0),     # red-orange
        (0.0, 115.0, 235.0),    # orange
        (0.0, 195.0, 255.0),    # amber
        (95.0, 240.0, 255.0),   # yellow
        (235.0, 255.0, 255.0),  # white-hot
    ],
    dtype=np.float32,
)
# Heat at which the fire becomes fully opaque; below it it fades out.
ALPHA_HEAT = 0.78
# How much the hottest part is added on top (a glow, not a flat colour).
CORE_GLOW = 0.75
SMOKE_COLOUR = np.array([62.0, 64.0, 66.0], dtype=np.float32)
SMOKE_ALPHA = 0.38
# Precomputed smooth noise, scrolled upward to make the fire flicker.
NOISE_SIZE = 96
# Particles are capped per hand: a stuck frame cannot grow the list forever.
MAX_PARTICLES = 520
# Ignition / extinguish bursts.
IGNITE_EMBERS = 46
EXTINGUISH_PUFFS = 26
# How long a hand may vanish before its fire gives up (tracking dropouts).
LOST_GRACE_MS = 350.0
# Forget a hand entirely this long after it was last seen and went cold.
FORGET_MS = 4000.0


@dataclass
class _Particle:
    x: float
    y: float
    vx: float
    vy: float
    age: float
    life: float
    size: float
    grow: float
    heat: float
    buoyancy: float
    phase: float
    drag: float = 1.4


@dataclass
class _HandFire:
    """One hand's fire: its signs, envelopes and particles."""

    tracker: SignTracker
    hand: HandObservation | None = None
    size: tuple[int, int] = (0, 0)
    palm: float = 0.0
    since_ms: float = 0.0
    missing_ms: float = 0.0
    lit: bool = False
    level: float = 0.0
    fist: float = 0.0
    orb: float = 0.0
    flash: float = 0.0
    orb_flash: float = 0.0
    # How many of the four counting fingers have been earned (0..4). The
    # thumb is separate: it lights only once all four are earned.
    lit_fingers: int = 0
    last_finger_count: int = 0
    finger_levels: dict[str, float] = field(default_factory=dict)
    signs: object = None
    window: Window | None = None
    silhouette: np.ndarray | None = None
    orb_at: tuple[float, float] | None = None
    anchor: tuple[float, float] | None = None
    spawn_debt: float = 0.0
    finger_debt: float = 0.0
    orb_debt: float = 0.0
    particles: list[_Particle] = field(default_factory=list)
    orb_particles: list[_Particle] = field(default_factory=list)
    smoke: list[_Particle] = field(default_factory=list)


def _approach(value: float, target: float, step: float) -> float:
    """Move `value` toward `target` by at most `step`."""
    if step <= 0.0 or value == target:
        return value
    if target > value:
        return min(target, value + step)
    return max(target, value - step)


class Fire:
    """The flame, drawn over the camera image."""

    def __init__(self, config: FireConfig | None = None, seed: int = 7) -> None:
        self.config = config or FireConfig()
        self._rng = np.random.default_rng(seed)
        self._states: dict[str, _HandFire] = {}
        self._last_t: float | None = None
        self._seconds = 0.0
        self._noise = self._make_noise()
        self._lut = self._make_lut()

    # --- setup --------------------------------------------------------------

    @staticmethod
    def _make_noise() -> np.ndarray:
        rng = np.random.default_rng(11)
        noise = rng.random((NOISE_SIZE, NOISE_SIZE)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), NOISE_SIZE / 10.0)
        low, high = float(noise.min()), float(noise.max())
        return (noise - low) / max(1e-6, high - low)

    @staticmethod
    def _make_lut() -> np.ndarray:
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        return np.stack([np.interp(x, RAMP_AT, RAMP_BGR[:, c]) for c in range(3)], axis=1).astype(np.float32)

    # --- public -------------------------------------------------------------

    def status(self) -> dict[str, dict[str, float]]:
        """What each hand is doing, for the lab's status line and for tests."""
        return {
            hand: {
                "lit": 1.0 if state.lit else 0.0,
                "level": state.level,
                "fist": state.fist,
                "orb": state.orb,
                "fingers": float(state.lit_fingers),
                "flames": float(self._flame_count(state)),
            }
            for hand, state in self._states.items()
        }

    def status_line(self) -> str:
        active = []
        for hand, state in self._states.items():
            if not state.lit and state.level <= 0.01 and state.orb <= 0.01:
                continue
            bits = [f"{hand[0]} {'LIT' if state.lit else 'out'}"]
            if state.level > 0.01:
                bits.append(f"{state.level:.0%}")
            flames = self._flame_count(state)
            if flames:
                bits.append(f"{flames} flame{'s' if flames != 1 else ''}")
            if state.fist > 0.05:
                bits.append(f"fist {state.fist:.0%}")
            if state.orb > 0.01:
                bits.append(f"orb {state.orb:.0%}")
            active.append(" ".join(bits))
        return "fire: " + ("   ".join(active) if active else "off (fist, then open)")

    @staticmethod
    def _flame_count(state: _HandFire) -> int:
        return sum(1 for digit in FLAME_FINGERS if state.finger_levels.get(digit, 0.0) > 0.5)

    def reset(self) -> None:
        """Put every fire out and forget it: the fruit was switched away.

        Without this, switching to the Gum-Gum fruit and back would resume
        whatever was lit (and whatever embers were mid-air) as if the player
        had never left.
        """
        self._states.clear()
        self._last_t = None

    def update(self, frame: Frame, size: tuple[int, int]) -> None:
        """Advance the gestures, envelopes and particles by one frame."""
        t = float(frame.t_ms)
        dt = 0.0 if self._last_t is None else min(0.12, max(0.0, (t - self._last_t) / 1000.0))
        self._last_t = t
        self._seconds = t / 1000.0
        aspect = size[0] / max(1, size[1])

        present: set[str] = set()
        for hand in frame.hands:
            if len(hand.landmarks) < 21:
                continue
            state = self._states.get(hand.handedness)
            if state is None:
                state = _HandFire(tracker=SignTracker(self.config))
                self._states[hand.handedness] = state
            state.hand = hand
            state.size = size
            state.since_ms = t
            state.missing_ms = 0.0
            state.palm = max(6.0, palm_length(hand.landmarks, aspect) * size[1])
            present.add(hand.handedness)

            state.signs = state.tracker.update(hand, t, aspect)
            if state.signs is not None:
                if state.signs.ignite and not state.lit:
                    self._light(state, size)
                elif state.signs.shake and state.lit:
                    self._extinguish(state)

            # The fire rides the hand's translation; the flames themselves
            # still rise in image space.
            anchor = (
                float(np.mean([p[0] for p in hand.landmarks]) * size[0]),
                float(np.mean([p[1] for p in hand.landmarks]) * size[1]),
            )
            if state.anchor is not None and dt > 0.0:
                dx, dy = anchor[0] - state.anchor[0], anchor[1] - state.anchor[1]
                if abs(dx) < size[0] * 0.5 and abs(dy) < size[1] * 0.5:
                    for particle in state.particles + state.orb_particles:
                        particle.x += dx
                        particle.y += dy
            state.anchor = anchor

        for hand, state in self._states.items():
            if hand in present:
                continue
            state.missing_ms += dt * 1000.0
            if state.missing_ms > LOST_GRACE_MS:
                if state.lit:
                    state.lit = False
                    state.lit_fingers = 0
                    state.last_finger_count = 0
                    self._puff(state)
                state.signs = None

        for state in self._states.values():
            self._advance(state, frame, dt)

        for hand in [
            hand
            for hand, state in self._states.items()
            if state.missing_ms > FORGET_MS
            and state.level <= 0.0
            and state.orb <= 0.0
            and self._flame_count(state) == 0
            and not state.smoke
        ]:
            del self._states[hand]

    def draw(self, bgr: np.ndarray) -> np.ndarray:
        """The image with the fire drawn in. Returns `bgr` itself when cold."""
        out = bgr
        for state in self._states.values():
            if (
                state.level <= 0.0
                and state.orb <= 0.0
                and self._flame_count(state) == 0
                and not state.smoke
            ):
                continue
            if out is bgr:
                out = bgr.copy()
            if state.level > 0.0 or state.flash > 0.0:
                self._draw_hand(out, state)
            if any(level > 0.01 for level in state.finger_levels.values()):
                self._draw_fingers(out, state)
            if state.smoke:
                self._draw_smoke(out, state)
            if state.orb > 0.0:
                self._draw_orb(out, state)
        return out

    # --- state --------------------------------------------------------------

    def _light(self, state: _HandFire, size: tuple[int, int]) -> None:
        """The fist opened: light the hand, starting from one fingertip."""
        state.lit = True
        state.lit_fingers = 0
        state.last_finger_count = 0
        # Re-lighting while old fingertip flames are still fading must still
        # start from one, not inherit the previous hand's flames.
        state.finger_levels = {digit: 0.0 for digit in FLAME_FINGERS}
        state.flash = 1.0
        lm = state.hand.landmarks if state.hand is not None else []
        if not lm:
            return
        for _ in range(IGNITE_EMBERS):
            point = lm[int(self._rng.integers(len(lm)))]
            x = point[0] * size[0] + self._rng.normal(0.0, state.palm * 0.12)
            y = point[1] * size[1] + self._rng.normal(0.0, state.palm * 0.12)
            angle = self._rng.uniform(0.0, 2.0 * math.pi)
            speed = state.palm * self._rng.uniform(1.2, 4.0)
            self._spawn_particle(
                state.particles,
                x,
                y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed - state.palm * 0.8,
                life=self._rng.uniform(0.35, 0.8),
                size=state.palm * self._rng.uniform(0.012, 0.035),
                grow=state.palm * 0.02,
                heat=1.0,
                buoyancy=state.palm * 0.6,
            )

    def _extinguish(self, state: _HandFire) -> None:
        """The hand shook: put every flame out and leave smoke behind."""
        if not state.lit:
            return
        state.lit = False
        state.lit_fingers = 0
        state.last_finger_count = 0
        self._puff(state)

    def _puff(self, state: _HandFire) -> None:
        """Smoke where the hand was: what is left when the fire goes out."""
        lm = state.hand.landmarks if state.hand is not None else []
        if not lm:
            return
        size = state.size
        for _ in range(EXTINGUISH_PUFFS):
            point = lm[int(self._rng.integers(len(lm)))]
            x = point[0] * size[0] + self._rng.normal(0.0, state.palm * 0.15)
            y = point[1] * size[1] + self._rng.normal(0.0, state.palm * 0.15)
            self._spawn_particle(
                state.smoke,
                x,
                y,
                vx=self._rng.normal(0.0, state.palm * 0.25),
                vy=-state.palm * self._rng.uniform(0.2, 0.7),
                life=self._rng.uniform(0.7, 1.4),
                size=state.palm * self._rng.uniform(0.04, 0.08),
                grow=state.palm * self._rng.uniform(0.10, 0.20),
                heat=1.0,
                buoyancy=state.palm * 0.25,
            )

    def _advance(self, state: _HandFire, frame: Frame, dt: float) -> None:
        state.level = _approach(
            state.level,
            1.0 if state.lit else 0.0,
            dt / max(0.01, self.config.ignite_s if state.lit else self.config.extinguish_s),
        )
        state.flash = max(0.0, state.flash - dt / max(0.01, self.config.flash_s))
        state.fist = _approach(
            state.fist,
            1.0 if state.lit and state.signs is not None and state.signs.fist else 0.0,
            dt / max(0.01, self.config.fist_s),
        )

        # Fingertip flames are earned one at a time. Opening the whole hand
        # from a fist lights only the first finger; the count can only grow by
        # exactly one, so a jump from one to five does nothing. The count is
        # kept while fingers are lowered (the flames themselves fade out), so
        # the player does not have to re-earn a flame every time they curl.
        signs = state.signs
        extended = [
            digit
            for digit in FLAME_FINGERS[:-1]
            if signs is not None and signs.out.get(digit, False)
        ]
        count = len(extended)
        if state.lit:
            if state.lit_fingers == 0 and count > 0:
                state.lit_fingers = 1
            elif (
                count == state.lit_fingers + 1
                and count != state.last_finger_count
            ):
                state.lit_fingers = min(len(FLAME_FINGERS) - 1, state.lit_fingers + 1)
        else:
            state.lit_fingers = 0
        state.last_finger_count = count
        lit_digits = set(extended[: state.lit_fingers]) if state.lit else set()
        if (
            state.lit
            and state.lit_fingers >= len(FLAME_FINGERS) - 1
            and count > 0
            and signs is not None
            and signs.out.get("thumb", False)
        ):
            lit_digits.add("thumb")
        for digit in FLAME_FINGERS:
            state.finger_levels[digit] = _approach(
                state.finger_levels.get(digit, 0.0),
                1.0 if digit in lit_digits else 0.0,
                dt / max(0.01, self.config.finger_light_s),
            )

        index_only = state.lit and state.signs is not None and state.signs.index_only
        was_orb = state.orb
        state.orb = _approach(state.orb, 1.0 if index_only else 0.0, dt / max(0.01, self.config.orb_s))
        if state.orb > 0.0 and was_orb <= 0.0:
            state.orb_flash = 1.0
        state.orb_flash = max(0.0, state.orb_flash - dt / 0.25)

        if state.level > 0.0 or state.flash > 0.0:
            self._refresh_silhouette(state, frame)
        if state.orb > 0.0 and state.hand is not None:
            self._float_orb(state, dt)

        if state.level > 0.0:
            self._spawn(state, dt)
        if any(level > 0.01 for level in state.finger_levels.values()):
            self._spawn_fingers(state, dt)
        if state.orb > 0.0:
            self._spawn_orb(state, dt)
        self._move(state, dt)

    # --- geometry -----------------------------------------------------------

    def _refresh_silhouette(self, state: _HandFire, frame: Frame) -> None:
        if state.hand is None or state.size == (0, 0):
            return
        window = self._hand_window(state)
        if window is None:
            return
        state.window = window
        state.silhouette = hand_silhouette(state.hand, frame.segmentation, state.size, window)

    def _hand_window(self, state: _HandFire) -> Window | None:
        """The hand's box, with room above for the plume."""
        size = state.size
        palm = state.palm
        xs = [point[0] * size[0] for point in state.hand.landmarks]
        ys = [point[1] * size[1] for point in state.hand.landmarks]
        margin = palm * 0.45
        top = palm * (self.config.plume + 0.35)
        x0 = max(0, int(min(xs) - margin))
        y0 = max(0, int(min(ys) - top))
        x1 = min(size[0], int(max(xs) + margin) + 1)
        y1 = min(size[1], int(max(ys) + margin) + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        return Window(x0, y0, x1, y1)

    def _orb_point(self, state: _HandFire) -> tuple[float, float]:
        lm = state.hand.landmarks
        tip = lm[HAND_DIGITS["index"][3]]
        return (
            tip[0] * state.size[0],
            tip[1] * state.size[1] - self.config.orb_lift * state.palm,
        )

    def _float_orb(self, state: _HandFire, dt: float) -> None:
        target = self._orb_point(state)
        if state.orb_at is None:
            state.orb_at = target
        else:
            follow = min(1.0, max(0.15, dt * 9.0))
            state.orb_at = (
                state.orb_at[0] + (target[0] - state.orb_at[0]) * follow,
                state.orb_at[1] + (target[1] - state.orb_at[1]) * follow,
            )

    # --- particles ----------------------------------------------------------

    def _spawn_particle(
        self,
        into: list[_Particle],
        x: float,
        y: float,
        *,
        vx: float,
        vy: float,
        life: float,
        size: float,
        grow: float,
        heat: float,
        buoyancy: float,
    ) -> None:
        if len(into) >= MAX_PARTICLES:
            del into[: len(into) - MAX_PARTICLES + 1]
        into.append(
            _Particle(
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                age=0.0,
                life=life,
                size=size,
                grow=grow,
                heat=heat,
                buoyancy=buoyancy,
                phase=self._rng.uniform(0.0, 2.0 * math.pi),
            )
        )

    def _spawn(self, state: _HandFire, dt: float) -> None:
        if state.silhouette is None or state.window is None:
            return
        palm = state.palm
        rate = self.config.spawn_rate * state.level * (1.0 + 1.8 * state.fist)
        state.spawn_debt += rate * dt
        count = min(int(state.spawn_debt), 40)
        state.spawn_debt -= count
        if count <= 0:
            return
        edge, top = self._edges(state.silhouette)
        if len(edge[0]) == 0:
            return
        window = state.window
        for _ in range(count):
            if top is not None and self._rng.random() < 0.55:
                ys, xs = top
            else:
                ys, xs = edge
            choice = int(self._rng.integers(len(xs)))
            x = window.x0 + float(xs[choice]) + self._rng.uniform(-1.0, 1.0)
            y = window.y0 + float(ys[choice]) - 1.0
            ember = self._rng.random() < 0.11
            if ember:
                self._spawn_particle(
                    state.particles,
                    x,
                    y,
                    vx=self._rng.normal(0.0, palm * 0.35),
                    vy=-palm * self._rng.uniform(1.2, 2.6),
                    life=self._rng.uniform(0.5, 1.0),
                    size=palm * self._rng.uniform(0.008, 0.02),
                    grow=palm * 0.01,
                    heat=1.0,
                    buoyancy=palm * 1.2,
                )
            else:
                self._spawn_particle(
                    state.particles,
                    x,
                    y,
                    vx=self._rng.normal(0.0, palm * 0.18),
                    vy=-palm * self._rng.uniform(0.7, 1.6),
                    life=self._rng.uniform(0.3, 0.65),
                    size=palm * self._rng.uniform(0.025, 0.06),
                    grow=palm * self._rng.uniform(0.05, 0.12),
                    heat=self._rng.uniform(0.7, 1.0),
                    buoyancy=palm * 1.5,
                )

    def _spawn_fingers(self, state: _HandFire, dt: float) -> None:
        """A few embers off the earned fingertips, so the candles feel hot."""
        if state.hand is None:
            return
        palm = state.palm
        lit = [
            digit
            for digit in FLAME_FINGERS
            if state.finger_levels.get(digit, 0.0) > 0.15
        ]
        if not lit:
            return
        state.finger_debt += self.config.finger_spawn_rate * len(lit) * dt
        count = min(int(state.finger_debt), 12)
        state.finger_debt -= count
        lm = state.hand.landmarks
        for _ in range(count):
            digit = lit[int(self._rng.integers(len(lit)))]
            tip = lm[HAND_DIGITS[digit][3]]
            x = tip[0] * state.size[0] + self._rng.normal(0.0, palm * 0.06)
            y = tip[1] * state.size[1] - self._rng.uniform(0.0, palm * 0.10)
            self._spawn_particle(
                state.particles,
                x,
                y,
                vx=self._rng.normal(0.0, palm * 0.20),
                vy=-palm * self._rng.uniform(0.8, 1.8),
                life=self._rng.uniform(0.25, 0.55),
                size=palm * self._rng.uniform(0.008, 0.022),
                grow=palm * 0.012,
                heat=self._rng.uniform(0.8, 1.0),
                buoyancy=palm * 1.4,
            )

    @staticmethod
    def _edges(silhouette: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray] | None]:
        """Boundary pixels of the hand's fire, and its top edge (where fire
        licks off the fingers)."""
        mask = silhouette > 0.5
        if not mask.any():
            empty = np.array([], dtype=np.intp)
            return (empty, empty), None
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        edge = mask & ~eroded
        above = np.zeros_like(mask)
        above[1:] = mask[:-1]
        top = mask & ~above
        edge_y, edge_x = np.nonzero(edge)
        top_y, top_x = np.nonzero(top)
        if len(top_x) == 0:
            return (edge_y, edge_x), None
        return (edge_y, edge_x), (top_y, top_x)

    def _spawn_orb(self, state: _HandFire, dt: float) -> None:
        if state.orb_at is None:
            return
        palm = state.palm
        state.orb_debt += self.config.spawn_rate * 0.7 * state.orb * dt
        count = min(int(state.orb_debt), 16)
        state.orb_debt -= count
        cx, cy = state.orb_at
        radius = max(4.0, palm * self.config.orb_radius)
        for _ in range(count):
            angle = self._rng.uniform(0.0, 2.0 * math.pi)
            distance = radius * self._rng.uniform(0.05, 0.6)
            self._spawn_particle(
                state.orb_particles,
                cx + math.cos(angle) * distance,
                cy + math.sin(angle) * distance * 0.8,
                vx=self._rng.normal(0.0, palm * 0.25),
                vy=-palm * self._rng.uniform(0.6, 1.6),
                life=self._rng.uniform(0.22, 0.5),
                size=palm * self._rng.uniform(0.02, 0.05),
                grow=palm * 0.04,
                heat=self._rng.uniform(0.75, 1.0),
                buoyancy=palm * 1.2,
            )

    @staticmethod
    def _move_particles(particles: list[_Particle], dt: float) -> list[_Particle]:
        if dt <= 0.0:
            return particles
        for particle in particles:
            particle.age += dt
            particle.vy -= particle.buoyancy * dt
            particle.vx += math.sin(particle.age * 7.0 + particle.phase) * particle.buoyancy * 0.3 * dt
            damp = max(0.0, 1.0 - particle.drag * dt)
            particle.vx *= damp
            particle.vy *= damp
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.size += particle.grow * dt
        return [particle for particle in particles if particle.age < particle.life]

    def _move(self, state: _HandFire, dt: float) -> None:
        state.particles = self._move_particles(state.particles, dt)
        state.orb_particles = self._move_particles(state.orb_particles, dt)
        state.smoke = self._move_particles(state.smoke, dt)

    # --- drawing ------------------------------------------------------------

    def _noise_field(self, shape: tuple[int, int], scroll: float) -> np.ndarray:
        """Smooth 0..1 noise over `shape`, scrolled to animate the flame."""
        height, width = shape
        if height <= 0 or width <= 0:
            return np.zeros((max(0, height), max(0, width)), dtype=np.float32)
        grid = 28
        shift = int(scroll * 85.0) % NOISE_SIZE
        first = cv2.resize(
            np.roll(self._noise, -shift, axis=0), (grid, grid), interpolation=cv2.INTER_AREA
        )
        second = cv2.resize(
            np.roll(self._noise, -int(scroll * 150.0) % NOISE_SIZE, axis=0),
            (grid, grid),
            interpolation=cv2.INTER_AREA,
        )
        second = np.roll(second, int(scroll * 37.0) % grid, axis=1)
        coarse = 0.62 * first + 0.38 * second
        return cv2.resize(coarse.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)

    def _draw_hand(self, out: np.ndarray, state: _HandFire) -> None:
        if state.window is None or state.silhouette is None:
            return
        window = state.window
        height, width = window.shape
        palm = state.palm
        mask = state.silhouette
        noise = self._noise_field((height, width), self._seconds)

        # The hand is lit, not replaced: a hot rim, a faint body tint that is
        # strongest toward the fingers, and flame streaks rising off the edge.
        # The old version painted the whole silhouette nearly opaque, which
        # read as an orange glove over the camera image.
        binary = (mask > 0.5).astype(np.uint8)
        eroded = cv2.erode(binary, np.ones((3, 3), np.uint8)).astype(np.float32)
        rim = np.clip(mask - eroded, 0.0, 1.0)
        rim = cv2.GaussianBlur(rim, (0, 0), max(1.0, palm * 0.025))
        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, np.newaxis]
        top_weight = np.clip(1.0 - 0.85 * yy, 0.15, 1.0)
        body = mask * (0.08 + 0.20 * noise) * top_weight
        if state.flash > 0.0:
            body = body + state.flash * 0.5 * mask

        plume = np.zeros((height, width), dtype=np.float32)
        for particle in state.particles:
            x, y = particle.x - window.x0, particle.y - window.y0
            radius = max(1, int(particle.size))
            if -radius <= x < width + radius and -radius <= y < height + radius:
                fade = 1.0 - particle.age / particle.life
                heat = float(particle.heat * fade)
                speed = math.hypot(particle.vx, particle.vy)
                if speed > 1e-3:
                    tail = min(palm * 0.16, speed * 0.035)
                    ex = x - particle.vx / speed * tail
                    ey = y - particle.vy / speed * tail
                else:
                    ex, ey = x, y
                cv2.line(
                    plume,
                    (int(ex), int(ey)),
                    (int(x), int(y)),
                    heat,
                    radius,
                )
        plume = cv2.GaussianBlur(plume, (0, 0), max(1.2, palm * 0.035))
        glow = cv2.GaussianBlur(mask, (0, 0), max(1.0, palm * 0.10)) * 0.22
        heat = np.clip(body + rim * 0.95 + glow + plume * 0.90, 0.0, 1.0)
        self._paint(
            out,
            window,
            heat,
            state.level,
            alpha_scale=self.config.hand_alpha,
        )

    def _draw_fingers(self, out: np.ndarray, state: _HandFire) -> None:
        if state.hand is None or state.size == (0, 0):
            return
        for digit in FLAME_FINGERS:
            level = state.finger_levels.get(digit, 0.0)
            if level > 0.01:
                self._draw_finger_flame(out, state, digit, level)

    def _draw_finger_flame(
        self,
        out: np.ndarray,
        state: _HandFire,
        digit: str,
        level: float,
    ) -> None:
        """A teardrop flame standing on one fingertip."""
        if state.hand is None:
            return
        tip = state.hand.landmarks[HAND_DIGITS[digit][3]]
        cx = tip[0] * state.size[0]
        cy = tip[1] * state.size[1]
        palm = max(6.0, state.palm)
        width = max(3.0, palm * self.config.finger_flame_width)
        height = max(5.0, palm * self.config.finger_flame_height)
        reach_x = int(width * 2.4 + 2)
        reach_y = int(height * 1.35 + 2)
        x0 = max(0, int(cx) - reach_x)
        y0 = max(0, int(cy) - reach_y)
        x1 = min(state.size[0], int(cx) + reach_x + 1)
        y1 = min(state.size[1], int(cy) + int(height * 0.25) + 2)
        if x0 >= x1 or y0 >= y1:
            return
        window = Window(x0, y0, x1, y1)
        yy, xx = np.mgrid[0 : window.shape[0], 0 : window.shape[1]].astype(np.float32)
        dx = (xx + window.x0 - cx) / width
        up = (cy - (yy + window.y0)) / height
        phase = float(HAND_DIGITS[digit][3]) * 0.9
        dx = dx - math.sin(self._seconds * 6.0 + phase) * 0.16 * np.clip(up, 0.0, 1.0)
        taper = np.clip(1.0 - 0.72 * np.clip(up, 0.0, 1.0), 0.18, 1.0)
        body = np.clip(1.0 - (dx / taper) ** 2, 0.0, 1.0) * np.clip(1.0 - up, 0.0, 1.0) ** 0.7
        base = np.exp(
            -((dx / 0.55) ** 2 + (np.clip(up, -0.35, 0.55) / 0.20) ** 2)
        )
        noise = self._noise_field(window.shape, self._seconds * 2.2 + phase)
        heat = np.clip(body * (0.70 + 0.30 * noise) + base * 0.60, 0.0, 1.0)
        self._paint(
            out,
            window,
            heat,
            level,
            additive=1.05,
            alpha_scale=self.config.finger_alpha,
        )

    def _draw_orb(self, out: np.ndarray, state: _HandFire) -> None:
        if state.orb_at is None:
            return
        palm = state.palm
        radius = max(6.0, palm * self.config.orb_radius * (0.65 + 0.35 * state.orb))
        cx, cy = state.orb_at
        cy += math.sin(self._seconds * 3.3) * palm * 0.04
        size = state.size
        reach = int(radius * 2.1 + palm * 0.35)
        x0, y0 = max(0, int(cx) - reach), max(0, int(cy) - reach)
        x1, y1 = min(size[0], int(cx) + reach), min(size[1], int(cy) + reach)
        if x0 >= x1 or y0 >= y1:
            return
        window = Window(x0, y0, x1, y1)

        yy, xx = np.mgrid[0 : window.shape[0], 0 : window.shape[1]].astype(np.float32)
        dx = (xx + window.x0 - cx) / radius
        dy = (yy + window.y0 - cy) / (radius * 1.15)
        ball = np.clip(1.0 - np.sqrt(dx * dx + dy * dy), 0.0, 1.0) ** 1.4
        # A tongue of flame licking up from the ball.
        tail_x = (xx + window.x0 - cx) / (radius * 0.55)
        tail_y = (yy + window.y0 - (cy - radius * 0.75)) / (radius * 1.5)
        tail = np.clip(1.0 - np.sqrt(tail_x * tail_x + tail_y * tail_y), 0.0, 1.0) ** 1.6
        noise = self._noise_field(window.shape, self._seconds * 1.3)
        heat = ball * (0.80 + 0.20 * noise) + tail * 0.30
        heat = heat + state.orb_flash * 0.5 * ball

        sparks = np.zeros(window.shape, dtype=np.float32)
        for particle in state.orb_particles:
            x, y = particle.x - window.x0, particle.y - window.y0
            r = max(1, int(particle.size))
            if -r <= x < window.shape[1] + r and -r <= y < window.shape[0] + r:
                fade = 1.0 - particle.age / particle.life
                cv2.circle(sparks, (int(x), int(y)), r, float(particle.heat * fade), -1)
        sparks = cv2.GaussianBlur(sparks, (0, 0), max(1.2, palm * 0.03))
        heat = np.clip(heat + sparks * 0.8, 0.0, 1.0)
        self._paint(out, window, heat, state.orb, additive=CORE_GLOW)

    def _draw_smoke(self, out: np.ndarray, state: _HandFire) -> None:
        window = state.window
        if window is None:
            return
        layer = np.zeros(window.shape, dtype=np.float32)
        for particle in state.smoke:
            x, y = particle.x - window.x0, particle.y - window.y0
            radius = max(1, int(particle.size))
            if -radius <= x < window.shape[1] + radius and -radius <= y < window.shape[0] + radius:
                fade = 1.0 - particle.age / particle.life
                cv2.circle(layer, (int(x), int(y)), radius, float(fade), -1)
        if not layer.any():
            return
        layer = cv2.GaussianBlur(layer, (0, 0), max(2.0, state.palm * 0.08))
        alpha = np.clip(layer, 0.0, 1.0) * SMOKE_ALPHA
        patch = out[window.y0 : window.y1, window.x0 : window.x1].astype(np.float32)
        a = alpha[..., None]
        patch = patch * (1.0 - a) + SMOKE_COLOUR * a
        out[window.y0 : window.y1, window.x0 : window.x1] = np.clip(patch, 0.0, 255.0).astype(np.uint8)

    def _paint(
        self,
        out: np.ndarray,
        window: Window,
        heat: np.ndarray,
        level: float,
        additive: float = CORE_GLOW,
        alpha_scale: float = 1.0,
    ) -> None:
        """Composite a heat field over the window: colour by heat, alpha by
        heat, and the hottest part added as a glow.

        `alpha_scale` below 1 keeps the camera image readable under the flame;
        the additive core still glows through it.
        """
        indices = np.clip(heat * 255.0, 0.0, 255.0).astype(np.uint8)
        colour = self._lut[indices]
        alpha = np.clip(heat / ALPHA_HEAT, 0.0, 1.0) * level * 0.97 * alpha_scale
        patch = out[window.y0 : window.y1, window.x0 : window.x1].astype(np.float32)
        a = alpha[..., None]
        patch = patch * (1.0 - a) + colour * a
        core = (np.clip((heat - 0.52) / 0.48, 0.0, 1.0) ** 1.5)[..., None]
        patch = patch + colour * core * (level * additive)
        out[window.y0 : window.y1, window.x0 : window.x1] = np.clip(patch, 0.0, 255.0).astype(np.uint8)
