"""Skin physics: points with mass, joined by springs, tethered to the face.

Each point of the skin mesh (`skinmesh.py`) has mass. Springs join it to its
neighbours; a soft tether holds it to where it rests on the face. The
fingers hold the pinched points; everything else moves under the springs:

- pull, and the skin around follows - less and less further out, because
  each tether takes up some of the pull. How far it spreads is set by how
  stiff the springs are against the tethers;
- because the points have mass, a yank reaches the surrounding skin a
  little late, and it catches up - that lag and catch-up is the "weight";
- let go, and the tethers pull it home - quickly and cleanly, like a
  rubber band snapping back: a flick past rest at most, then still. How
  fast it snaps is set by tether stiffness and mass; that it stops instead
  of wobbling, by the damping (near critical).

Positions are (x, y, lift): image pixels, and how far the skin has come off
the face toward the camera. The solver is XPBD (extended position-based
dynamics, the method games use for cloth and soft bodies), run in small
substeps for stability, compiled with numba so a frame of it takes a few
milliseconds.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

FRAME_DT = 1.0 / 30.0
SUBSTEPS = 12
# Stiffness of the tether holding each point to its resting place, per unit
# mass. Tuned by measurement against "too slimy": released from a 60px
# pull, the skin is back in 2 frames (60 -> 12 -> 0px), flicks ~2% past
# rest and is still by the 3rd (0.1s); while pulled, the skin around keeps
# up (two thirds of the way each frame) - a little weight, no goo.
TETHER_STIFFNESS = (2.0 * math.pi * 6.0) ** 2
# Springs are this many times stiffer than tethers. The pull fades by about
# e every sqrt(this) points, so this sets how far a pull spreads: at 40, a
# 60px pull moves skin 60px away by ~7px and 100px away by ~2px.
SPRING_RATIO = 40.0
# Shear and bending springs are softer than stretch ones: skin shears and
# folds more easily than it stretches.
SHEAR_SOFTNESS = 2.0
BEND_SOFTNESS = 20.0
# A grip weaker than 1 holds its point with a spring this soft at grip 0.5
# (relative to the tethers): the rim of a pinch follows, but not rigidly.
GRIP_SOFTNESS = 0.02
# Velocity damping, per second: near critical for the tether, so the skin
# stops where it rests instead of wobbling around it.
DAMPING = 55.0
# Below these the skin counts as at rest.
SETTLED_PX = 0.5
SETTLED_SPEED = 8.0


class SoftBody:
    """The skin's points, their springs, and the state of the simulation.

    `rest` is (N, 3): where each point rests this frame (lift 0). `edges` is
    (E, 2) point pairs joined by springs. `pinned` points never move.
    """

    def __init__(self, rest: np.ndarray, edges: np.ndarray, pinned: np.ndarray) -> None:
        self.positions = rest.astype(np.float64).copy()
        self.velocities = np.zeros_like(self.positions)
        self.edges = np.ascontiguousarray(edges, dtype=np.int64)
        self._pinned = pinned.astype(bool)
        lengths = np.linalg.norm(rest[self.edges[:, 1]] - rest[self.edges[:, 0]], axis=1)
        self._compliance = _spring_compliance(lengths)
        self._targets: np.ndarray | None = None
        self._grabbed = np.zeros(0, dtype=np.int64)

    def step(self, rest: np.ndarray, grabbed: np.ndarray, targets: np.ndarray, grip=None) -> None:
        """Advance one frame. `grabbed` points are held at `targets` (K, 3).

        `grip` (K,) in (0, 1] is how firmly each is held: 1 exactly (the
        middle of a pinch), less with a spring (its rim). Default: all 1.
        """
        rest = rest.astype(np.float64)
        inverse_mass = np.where(self._pinned, 0.0, 1.0)
        grabbed = np.asarray(grabbed, dtype=np.int64)
        grip = np.ones(len(grabbed)) if grip is None else np.asarray(grip, dtype=np.float64)
        firm = grip >= 0.999
        inverse_mass[grabbed[firm]] = 0.0
        grip_compliance = GRIP_SOFTNESS / TETHER_STIFFNESS * (1.0 - grip) / np.maximum(grip, 1e-3)
        # The same points held as last frame move on from last frame's
        # targets; a new grip starts from where its skin is.
        same_grip = self._targets is not None and np.array_equal(self._grabbed, grabbed)
        start = self._targets if same_grip else self.positions[grabbed]
        end = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
        rest_lengths = np.linalg.norm(rest[self.edges[:, 1]] - rest[self.edges[:, 0]], axis=1)
        _simulate(
            self.positions, self.velocities, rest, inverse_mass, self.edges, rest_lengths, self._compliance,
            grabbed, np.ascontiguousarray(start), np.ascontiguousarray(end), firm, grip_compliance,
            FRAME_DT, SUBSTEPS, 1.0 / TETHER_STIFFNESS, DAMPING,
        )
        self._targets = end.copy() if len(grabbed) else None
        self._grabbed = grabbed.copy()

    @property
    def pinned(self) -> np.ndarray:
        return self._pinned

    def settled(self, rest: np.ndarray) -> bool:
        """True once the skin is back at rest and still."""
        return (
            float(np.abs(self.positions - rest).max()) < SETTLED_PX
            and float(np.abs(self.velocities).max()) < SETTLED_SPEED
        )


def _spring_compliance(lengths: np.ndarray) -> np.ndarray:
    """XPBD compliance (1 / stiffness) per spring, by kind.

    The grid's springs come in three lengths: neighbours (stretch), sqrt 2
    of that (shear), twice it (bending).
    """
    shortest = float(lengths.min()) if len(lengths) else 1.0
    ratio = lengths / shortest
    softness = np.where(ratio < 1.2, 1.0, np.where(ratio < 1.7, SHEAR_SOFTNESS, BEND_SOFTNESS))
    return softness / (SPRING_RATIO * TETHER_STIFFNESS)


@njit(cache=True)
def _simulate(pos, vel, rest, inv_mass, edges, rest_len, compliance, grabbed, start, end, firm, grip_compliance,
              dt, substeps, tether_compliance, damping):  # pragma: no cover - compiled
    count = pos.shape[0]
    prev = np.empty_like(pos)
    h = dt / substeps
    decay = math.exp(-damping * h)
    for s in range(substeps):
        blend = (s + 1.0) / substeps
        for i in range(count):
            for k in range(3):
                prev[i, k] = pos[i, k]
            if inv_mass[i] > 0.0:
                for k in range(3):
                    vel[i, k] *= decay
                    pos[i, k] += vel[i, k] * h
        for g in range(grabbed.shape[0]):
            if firm[g]:
                for k in range(3):
                    pos[grabbed[g], k] = start[g, k] + (end[g, k] - start[g, k]) * blend
        _solve_springs(pos, inv_mass, edges, rest_len, compliance, h, s % 2 == 1)
        for g in range(grabbed.shape[0]):
            if not firm[g]:
                i = grabbed[g]
                pull = inv_mass[i] / (inv_mass[i] + grip_compliance[g] / (h * h))
                for k in range(3):
                    goal = start[g, k] + (end[g, k] - start[g, k]) * blend
                    pos[i, k] += (goal - pos[i, k]) * pull
        _solve_tethers(pos, rest, inv_mass, tether_compliance / (h * h))
        for i in range(count):
            if inv_mass[i] == 0.0:
                if not _is_grabbed(i, grabbed, firm):
                    for k in range(3):
                        pos[i, k] = rest[i, k]
            for k in range(3):
                vel[i, k] = (pos[i, k] - prev[i, k]) / h


@njit(cache=True)
def _is_grabbed(i, grabbed, firm):  # pragma: no cover - compiled
    for g in range(grabbed.shape[0]):
        if grabbed[g] == i and firm[g]:
            return True
    return False


@njit(cache=True)
def _solve_springs(pos, inv_mass, edges, rest_len, compliance, h, backwards):  # pragma: no cover - compiled
    total = edges.shape[0]
    for n in range(total):
        e = total - 1 - n if backwards else n
        i, j = edges[e, 0], edges[e, 1]
        w = inv_mass[i] + inv_mass[j]
        if w == 0.0:
            continue
        dx, dy, dz = pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1], pos[j, 2] - pos[i, 2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-9:
            continue
        stretch = length - rest_len[e]
        step = -stretch / (w + compliance[e] / (h * h)) / length
        for k, d in ((0, dx), (1, dy), (2, dz)):
            pos[i, k] -= inv_mass[i] * step * d
            pos[j, k] += inv_mass[j] * step * d


@njit(cache=True)
def _solve_tethers(pos, rest, inv_mass, scaled_compliance):  # pragma: no cover - compiled
    for i in range(pos.shape[0]):
        if inv_mass[i] == 0.0:
            continue
        pull = inv_mass[i] / (inv_mass[i] + scaled_compliance)
        for k in range(3):
            pos[i, k] += (rest[i, k] - pos[i, k]) * pull
