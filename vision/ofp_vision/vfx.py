"""The snap: sparks where a stretch lets go.

This is the lab's own effect, not the game's. The game gets the release on
the wire (`grab.<hand>.stretch` fires with the peak), and will draw its own
- fire, a flash, a sound. This is here so that a pull in the lab reads as
something happening when you open your fingers, instead of a finger quietly
returning.

A burst is a ring thrown outward, a handful of spokes, and a bright core,
added to the picture and fading as it goes. Strength (the peak stretch that
was released) makes it bigger and longer-lived.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

# How long a burst lasts (seconds), and how fast its ring travels (pixels per
# second at full strength).
LIFE = 0.24
GROWTH = 150.0
# Spokes around the ring, and where they start and end (share of the radius).
SPOKES = 9
SPOKE_FROM, SPOKE_TO = 0.55, 1.35
# The spark's colour (BGR): warm white-gold, like a struck match.
COLOUR = np.array([150.0, 225.0, 255.0], dtype=np.float32)
# How strong a release has to be for a full burst.
FULL_STRENGTH = 1.5
# Never keep more than this many (a stuck release event cannot pile up).
MAX_BURSTS = 8


class Sparks:
    """Bursts of sparks, one per release, drawn over the camera image."""

    def __init__(self) -> None:
        self._bursts: list[tuple[float, float, float, float]] = []

    def release(self, x: float, y: float, strength: float) -> None:
        """Start a burst at (x, y) in image pixels; `strength` scales it."""
        share = float(np.clip(strength / FULL_STRENGTH, 0.2, 1.0))
        self._bursts.append((float(x), float(y), share, time.perf_counter()))
        self._bursts = self._bursts[-MAX_BURSTS:]

    def draw(self, bgr: np.ndarray) -> np.ndarray:
        """Add this frame's bursts to a copy of `bgr` (or return it as is)."""
        if not self._bursts:
            return bgr
        now = time.perf_counter()
        live = [burst for burst in self._bursts if now - burst[3] < LIFE]
        self._bursts = live
        if not live:
            return bgr
        out = bgr.copy()
        for x, y, share, born in live:
            self._one(out, x, y, share, (now - born) / LIFE)
        return out

    def _one(self, out: np.ndarray, x: float, y: float, share: float, age: float) -> None:
        """One burst, `age` through its life (0..1), added into `out`."""
        fade = (1.0 - age) ** 2
        radius = 4.0 + share * GROWTH * age * LIFE
        reach = int(radius * SPOKE_TO + 12)
        x0, y0 = max(0, int(x) - reach), max(0, int(y) - reach)
        x1, y1 = min(out.shape[1], int(x) + reach), min(out.shape[0], int(y) + reach)
        if x0 >= x1 or y0 >= y1:
            return
        cx, cy = int(x) - x0, int(y) - y0
        glow = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        cv2.circle(glow, (cx, cy), max(1, int(radius)), 1.0, 2)
        for spoke in range(SPOKES):
            angle = 2.0 * math.pi * spoke / SPOKES + 0.4
            a = (int(cx + math.cos(angle) * radius * SPOKE_FROM), int(cy + math.sin(angle) * radius * SPOKE_FROM))
            b = (int(cx + math.cos(angle) * radius * SPOKE_TO), int(cy + math.sin(angle) * radius * SPOKE_TO))
            cv2.line(glow, a, b, 0.6, 1)
        cv2.circle(glow, (cx, cy), max(1, int(3.0 * share * (1.0 - age))), 1.0, -1)
        glow = cv2.GaussianBlur(glow, (5, 5), 0) * fade
        patch = out[y0:y1, x0:x1].astype(np.float32)
        patch += glow[..., None] * COLOUR
        out[y0:y1, x0:x1] = np.clip(patch, 0.0, 255.0).astype(np.uint8)
