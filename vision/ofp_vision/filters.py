"""Signal conditioning: OneEuroFilter and a simple derivative.

Why OneEuro and not a plain exponential moving average:
an EMA has a fixed tradeoff between jitter and lag. When the player pulls
hard, an EMA visibly trails the motion. OneEuro adapts its cutoff to the
signal's speed - heavy smoothing when the finger is still (no jitter),
light smoothing when it is moving fast (no lag). That matters because the
whole feel of the mechanic depends on the stretch tracking the hand tightly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class LowPassFilter:
    """Exponential smoothing with an alpha derived from a cutoff frequency."""

    def __init__(self) -> None:
        self._value: float | None = None

    def reset(self) -> None:
        self._value = None

    @property
    def value(self) -> float | None:
        return self._value

    def filter(self, value: float, alpha: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value


def alpha_for_cutoff(cutoff: float, dt: float) -> float:
    """Smoothing factor for a given cutoff frequency and timestep."""
    if dt <= 0.0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


@dataclass
class OneEuroFilter:
    """Adaptive low-pass filter.

    Parameters
    ----------
    min_cutoff:
        Cutoff frequency at rest. Lower = smoother when still, more lag.
    beta:
        How much the cutoff rises with signal speed. Higher = more responsive
        to fast motion, more jitter.
    d_cutoff:
        Cutoff for the derivative estimate itself.
    """

    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0

    def __post_init__(self) -> None:
        self._x = LowPassFilter()
        self._dx = LowPassFilter()
        self._last_raw: float | None = None
        self._last_t: float | None = None

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._last_raw = None
        self._last_t = None

    @property
    def value(self) -> float | None:
        return self._x.value

    def __call__(self, value: float, t: float) -> float:
        dt = 1.0
        if self._last_t is not None:
            dt = max(t - self._last_t, 1e-3)
        self._last_t = t

        dx = 0.0
        if self._last_raw is not None:
            dx = (value - self._last_raw) / dt
        self._last_raw = value

        dx_hat = self._dx.filter(dx, alpha_for_cutoff(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        return self._x.filter(value, alpha_for_cutoff(cutoff, dt))


class Slope:
    """Velocity of a signal, in units per second.

    Used by the release detector: a pull that stops growing and starts
    contracting is the trigger to fire.
    """

    def __init__(self) -> None:
        self._prev: float | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._prev = None
        self._t = None

    def __call__(self, value: float, t: float) -> float:
        if self._prev is None or self._t is None:
            self._prev = value
            self._t = t
            return 0.0
        dt = max(t - self._t, 1e-3)
        slope = (value - self._prev) / dt
        self._prev = value
        self._t = t
        return slope
