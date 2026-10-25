"""A finger, stretched like rubber: a chain with physics, warping the picture.

A finger is not a sheet. Pulled, it gets longer - a little thinner - and the
part past the pinch (the tip, the nail) goes whole with the fingers that
hold it. So a finger is a chain of points along its centre line, from the
knuckle to just past the tip:

    knuckle end   pinned: the finger stays on the hand
    the pinch     the points from the pinch to the tip are held by the
                  pinching fingers and travel with them, unstretched
    in between    springs along the chain (it stretches, and snaps back),
                  bending springs (it stays straight-ish), and weak tethers
                  to where it rests - so, held, it runs nearly straight from
                  knuckle to pinch, like a rubber band

The finger on screen is the camera's own picture of it, warped (`warp`):
nothing is erased and nothing new is drawn. The picture along the finger
moves with the chain - the finger's own pixels whole, a thin skirt of what
is around it less and less, so the stretch blends into the picture with no
edge - and beyond the skirt nothing moves, never the next finger.

Where the chain rests follows the palm - which the tracker holds well even
when the other hand covers the finger - with the finger's own shape eased in
slowly: its landmarks wobble while it is held.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .indices import HAND_FINGER_JOINTS, HAND_MIDDLE_MCP, HAND_WRIST
from .signals import HandObservation
from .skinmesh import Hold
from .softbody import SoftBody

# Points along the chain.
NODES = 24
# The fingertip's flesh past its landmark, in palm lengths.
TIP_EXTENSION = 0.1
# The first points of the chain, at the knuckle, never move.
BASE_NODES = 2
# Tethers on a held chain, as a share of the face's: a held finger runs nearly
# straight from knuckle to pinch instead of bowing back to where it was.
# Let go, it is tethered fully again, and snaps home.
CHAIN_TETHER = 0.15
# A stretched finger thins (it keeps its volume), but never below this.
MIN_THINNING = 0.85
# The width's change along the chain is eased this many times (each over
# three points).
THIN_EASE_PASSES = 4
# How far each frame's measured finger shape moves the one used: slowly
# while it is held (its landmarks wobble under the other hand), quickly not.
EASE_HELD, EASE_FREE = 0.1, 0.5
# The finger is held from this far below the pinch (palm lengths) to its tip.
HOLD_BELOW = 0.08
# How far the held end turns to follow the stretch (1: all the way).
TURN_TO_PULL = 0.8
# The palm: the wrist and the four knuckles (not the thumb, which moves).
PALM_FRAME = (HAND_WRIST, 5, 9, 13, 17)
# How far from its centre line (palm lengths) a finger's picture moves whole
# with it - a thumb is the thickest - and how much further a thin skirt of
# what is around it goes along, less and less: the stretch blends into the
# picture there instead of having an edge.
FINGER_HALF = {"thumb": 0.14, "index": 0.105, "middle": 0.105, "ring": 0.1, "pinky": 0.085}
SKIRT = 0.1
# Past the fingertip, the same blend, along the finger (palm lengths).
TIP_SKIRT = 0.12
# The warp's mesh: a row every this many pixels along the finger, this many
# points across it (odd: one on the centre line).
ROW_PX = 4.0
WARP_COLS = 17


def similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The turn, uniform scale and shift (2x3) that best map `src` points
    onto `dst` (least squares; Umeyama's method)."""
    mean_src, mean_dst = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mean_src, dst - mean_dst
    u, s, vt = np.linalg.svd(b.T @ a / len(src))
    flip = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    turn = u @ flip @ vt
    scale = float(np.trace(np.diag(s) @ flip) / max((a**2).sum() / len(src), 1e-12))
    return np.hstack([scale * turn, (mean_dst - scale * turn @ mean_src)[:, None]])


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def _inverse(matrix: np.ndarray) -> np.ndarray:
    linear = np.linalg.inv(matrix[:, :2])
    return np.hstack([linear, (-linear @ matrix[:, 2])[:, None]])


def _px(hand: HandObservation, indices, size: tuple[int, int]) -> np.ndarray:
    return np.array([[hand.landmarks[i][0] * size[0], hand.landmarks[i][1] * size[1]] for i in indices])


def _palm_length(hand: HandObservation, size: tuple[int, int]) -> float:
    wrist, knuckle = _px(hand, (HAND_WRIST, HAND_MIDDLE_MCP), size)
    return float(np.linalg.norm(knuckle - wrist))


def _finger_line(hand: HandObservation, digit: str, size: tuple[int, int]) -> np.ndarray:
    """Knuckle, the joints, tip - and the flesh just past the tip."""
    joints = _px(hand, HAND_FINGER_JOINTS[digit], size)
    last = joints[-1] - joints[-2]
    beyond = joints[-1] + last / max(float(np.linalg.norm(last)), 1e-6) * TIP_EXTENSION * _palm_length(hand, size)
    return np.vstack([joints, beyond])


def resample(line: np.ndarray, count: int) -> np.ndarray:
    """`count` points evenly spaced along a polyline."""
    lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(lengths)])
    wanted = np.linspace(0.0, along[-1], count)
    return np.stack([np.interp(wanted, along, line[:, k]) for k in range(2)], axis=1)


def arc(points: np.ndarray) -> np.ndarray:
    """How far along a chain (by its x, y) each point is."""
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))])


def normals(points: np.ndarray) -> np.ndarray:
    """Unit normals (left of the direction of travel) along a chain; zero
    where the chain has no direction."""
    tangents = np.gradient(points, axis=0)
    tangents = tangents / np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)


def _thinning(points: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """How thin each part of the chain is drawn: stretched by r, a finger
    keeps its volume at width 1/sqrt(r) - eased along the chain, so where the
    stretch starts and ends its width changes smoothly, not in a notch."""
    span = np.linalg.norm(np.gradient(points, axis=0), axis=1)
    resting = np.maximum(np.linalg.norm(np.gradient(rest, axis=0), axis=1), 1e-9)
    thin = np.clip(1.0 / np.sqrt(np.maximum(span / resting, 1e-9)), MIN_THINNING, 1.0)
    for _ in range(THIN_EASE_PASSES):
        thin = np.convolve(np.pad(thin, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    return thin


def grid_triangles(rows: int, cols: int) -> np.ndarray:
    """Two triangles per cell of a rows x cols grid of points (row-major)."""
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    a = (r * cols + c).ravel()
    b, d, e = a + cols, a + cols + 1, a + 1
    return np.concatenate([np.stack([a, b, d], axis=1), np.stack([a, d, e], axis=1)]).astype(np.int32)


def _chain_edges(count: int) -> np.ndarray:
    stretch = [(i, i + 1) for i in range(count - 1)]
    bend = [(i, i + 2) for i in range(count - 2)]
    return np.array(stretch + bend, dtype=np.int64)


def _held_targets(rest: np.ndarray, hold: Hold, pull: np.ndarray) -> np.ndarray:
    """Where a hand's held points go: with the pinch, and turned about the
    point under it toward the stretch (TURN_TO_PULL of the way) - a rubber
    finger's end swings round to follow the pull, rather than make an L."""
    core = hold.core
    pivot = rest[hold.pivot if hold.pivot is not None else core[0]]
    before = pivot[:2] - rest[0, :2]
    after = before + pull[:2]
    angle = float(np.arctan2(before[0] * after[1] - before[1] * after[0], before @ after))
    # Dragged straight back past the knuckle the turn has no side, and with
    # the pinch at the knuckle no direction: fade it out there, so it never
    # flips from one side to the other.
    room = float(np.linalg.norm(after)) / max(float(np.linalg.norm(before)), 1e-6)
    fade = min(1.0, 1.0 + np.cos(angle)) * min(1.0, room)
    turn = TURN_TO_PULL * angle * fade
    c, s = np.cos(turn), np.sin(turn)
    offsets = rest[core, :2] - pivot[:2]
    targets = rest[core] + pull
    targets[:, :2] = pivot[:2] + pull[:2] + offsets @ np.array([[c, s], [-s, c]])
    return targets


def _extended(chain: np.ndarray, chain_arc: np.ndarray, s: np.ndarray) -> np.ndarray:
    """A chain's points (x, y, ...) at distances `s` along it - straight on
    past its end, the way its end points."""
    inside = np.stack([np.interp(s, chain_arc, chain[:, k]) for k in range(chain.shape[1])], axis=1)
    end = chain[-1, :2] - chain[-2, :2]
    direction = end / max(float(np.linalg.norm(end)), 1e-9)
    past = np.maximum(s - chain_arc[-1], 0.0)[:, None]
    inside[:, :2] += past * direction
    return inside


def _smooth(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _distance_to(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Each point's distance to a polyline."""
    best = np.full(len(points), np.inf)
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(points - (a + t[:, None] * ab), axis=1))
    return best


@dataclass(frozen=True, eq=False)
class FingerWarp:
    """The picture around a finger, as a mesh of `rows` x `cols` points:
    where each point is drawn now (x, y, lift), and where its pixel is in
    the camera's picture."""

    positions: np.ndarray
    source: np.ndarray
    triangles: np.ndarray
    rows: int
    cols: int


def finger_usable(hand: HandObservation, part: str, size: tuple[int, int]) -> bool:
    """Whether a hand's landmarks make a finger at all (not collapsed to a
    point, as the tracker sometimes reports for a frame)."""
    line = _finger_line(hand, part.split(".")[2], size)
    return _palm_length(hand, size) >= 1.0 and float(np.linalg.norm(line[-1] - line[0])) >= 1.0


@dataclass(eq=False)
class FingerSession:
    """One finger's chain and physics, from the first pinch on it until it
    has settled. The hand pinching it has a `Hold` (the chain's held points)."""

    part: str
    owner: str
    digit: str
    body: SoftBody
    # The finger's shape, kept in the coordinates of the palm as it was at
    # the start; and the map from there to the image now.
    palm_start: np.ndarray
    shape: np.ndarray
    to_image: np.ndarray
    palm_px: float
    holds: dict[str, Hold] = field(default_factory=dict)

    @classmethod
    def start(cls, part: str, hand: HandObservation, size: tuple[int, int]) -> FingerSession:
        _, owner, digit = part.split(".")
        shape = _finger_line(hand, digit, size)
        rest = np.zeros((NODES, 3))
        rest[:, :2] = resample(shape, NODES)
        pinned = np.arange(NODES) < BASE_NODES
        body = SoftBody(rest, _chain_edges(NODES), pinned)
        identity = np.hstack([np.eye(2), np.zeros((2, 1))])
        return cls(part, owner, digit, body, _px(hand, PALM_FRAME, size), shape, identity, _palm_length(hand, size))

    def rest(self) -> np.ndarray:
        """Where the chain rests now: (NODES, 3), lift 0."""
        rest = np.zeros((NODES, 3))
        rest[:, :2] = resample(apply(self.to_image, self.shape), NODES)
        return rest

    def follow(self, hand: HandObservation | None, size: tuple[int, int]) -> np.ndarray:
        """Follow the hand this frame (keep still if it is not in view) and
        carry the chain along with it. Returns where the chain rests now."""
        before = self.rest()
        matrix = similarity(self.palm_start, _px(hand, PALM_FRAME, size)) if hand is not None else None
        if hand is not None and matrix is not None and abs(float(np.linalg.det(matrix[:, :2]))) > 1e-6:
            self.to_image = matrix
            measured = apply(_inverse(self.to_image), _finger_line(hand, self.digit, size))
            ease = EASE_HELD if self.holds else EASE_FREE
            self.shape = self.shape + ease * (measured - self.shape)
            self.palm_px = _palm_length(hand, size)
        rest = self.rest()
        self.body.carry(rest - before)
        return rest

    def hold(self, target: str, anchor_px: np.ndarray) -> Hold | None:
        """The chain's points a pinch at `anchor_px` holds: from just below
        it to the tip. None if another hand holds them all already."""
        rest = self.rest()[:, :2]
        place = int(np.argmin(np.linalg.norm(rest - anchor_px, axis=1)))
        spacing = max(float(np.linalg.norm(rest[1] - rest[0])), 1e-6)
        first = max(BASE_NODES, place - int(round(HOLD_BELOW * self.palm_px / spacing)))
        taken = {int(i) for other in self.holds.values() for i in other.core}
        core = np.array([i for i in range(first, NODES) if i not in taken], dtype=np.int64)
        if len(core) == 0:
            return None
        return Hold(target, core, np.ones(len(core)), pivot=max(place, int(core[0])))

    def step(self, rest: np.ndarray, pulls: dict[str, np.ndarray]) -> None:
        """One frame of physics; each holding hand's points go with its pull
        (pixels, and lift), turned to follow the stretch (`_held_targets`)."""
        holds = [(hand, hold) for hand, hold in self.holds.items() if hand in pulls]
        self.body.tether = CHAIN_TETHER if holds else 1.0
        if not holds:
            self.body.step(rest, np.zeros(0, dtype=np.int64), np.zeros((0, 3)))
            return
        grabbed = np.concatenate([hold.core for _, hold in holds])
        targets = np.concatenate([_held_targets(rest, hold, pulls[hand]) for hand, hold in holds])
        self.body.step(rest, grabbed, targets, np.concatenate([hold.grip for _, hold in holds]))

    def warp(self, rest: np.ndarray, hand: HandObservation | None, size: tuple[int, int]) -> FingerWarp | None:
        """The picture along the finger as a mesh: each point where its pixel
        is at rest, moved with the chain now - fully across the finger, less
        and less across a thin skirt beside it and past its tip, not at all
        beyond (nor near the next finger, or at the knuckle)."""
        points = self.body.positions
        rest_arc = arc(rest)
        if rest_arc[-1] < 1.0:
            return None
        half = FINGER_HALF[self.digit] * self.palm_px
        reach = half + SKIRT * self.palm_px
        tip = TIP_SKIRT * self.palm_px
        s = np.arange(0.0, rest_arc[-1] + tip + ROW_PX, ROW_PX)
        across = np.linspace(-reach, reach, WARP_COLS)
        still, now = _extended(rest, rest_arc, s), _extended(points, rest_arc, s)
        thin = np.interp(s, rest_arc, _thinning(points[:, :2], rest[:, :2]))
        source = still[:, None, :2] + across[None, :, None] * normals(still[:, :2])[:, None, :]
        moved = now[:, None, :2] + (thin[:, None] * across[None, :])[..., None] * normals(now[:, :2])[:, None, :]
        knuckle = np.clip(s / max(rest_arc[BASE_NODES], 1e-6), 0.0, 1.0)
        along = _smooth((rest_arc[-1] + tip - s) / tip) * knuckle
        weight = along[:, None] * _smooth((reach - np.abs(across)) / (reach - half))[None, :]
        if hand is not None:
            weight = weight * ~self._nearer_another_finger(source.reshape(-1, 2), hand, size).reshape(weight.shape)
        xy = source + weight[..., None] * (moved - source)
        lift = (weight * now[:, None, 2]).reshape(-1, 1)
        rows, cols = len(s), WARP_COLS
        return FingerWarp(np.hstack([xy.reshape(-1, 2), lift]), source.reshape(-1, 2), grid_triangles(rows, cols), rows, cols)

    def _nearer_another_finger(self, points: np.ndarray, hand: HandObservation, size: tuple[int, int]) -> np.ndarray:
        """Which points are nearer another finger of the hand than this one:
        where fingers touch, those pixels are the next finger's, and stay."""
        own = _distance_to(points, apply(self.to_image, self.shape[:-1]))
        others = [_px(hand, HAND_FINGER_JOINTS[d], size) for d in HAND_FINGER_JOINTS if d != self.digit]
        return np.min([_distance_to(points, line) for line in others], axis=0) < own

    def mark_resting(self, mask: np.ndarray, scale: float) -> None:
        """Mark where the finger rests on `mask` (a frame `scale` times the
        camera's): what is being stretched is never learned as background."""
        line = np.round(self.rest()[:, :2] * scale).astype(np.int32)
        reach = (FINGER_HALF[self.digit] + SKIRT) * self.palm_px
        cv2.polylines(mask, [line], False, 255, max(1, int(round(2 * reach * scale))))
