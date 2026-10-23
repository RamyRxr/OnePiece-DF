"""A finger, stretched like rubber: a tube along the finger, with physics.

A finger is not a sheet. Pulled, it gets longer - a little thinner, still
round - and the part past the pinch (the tip, the nail) goes whole with the
fingers that hold it. So a finger is a chain of points along its centre
line, from the knuckle to just past the tip:

    knuckle end   pinned: the finger stays on the hand
    the pinch     the points from the pinch to the tip are held by the
                  pinching fingers and travel with them, unstretched
    in between    springs along the chain (it stretches, and snaps back),
                  bending springs (it stays straight-ish), and weak tethers
                  to where it rests - so, held, it runs nearly straight from
                  knuckle to pinch, like a rubber band

The tube drawn around the chain is painted with the finger's own pixels,
taken straight across the finger where it rests: a stretched finger is that
finger, drawn longer. Where it rests follows the palm - which the tracker
holds well even when the other hand covers the finger - with the finger's
own shape eased in slowly: its landmarks wobble while it is held.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .indices import HAND_DIGITS, HAND_MIDDLE_MCP, HAND_WRIST
from .signals import HandObservation
from .skin import FINGER_HALF_WIDTH, Window, body_skin, distance_to_chain
from .skinmesh import Hold
from .softbody import SoftBody

# Points along the chain, and vertices across the tube (odd: one on the
# centre line).
NODES = 24
ACROSS = 7
ACROSS_OFFSETS = np.linspace(-1.0, 1.0, ACROSS)
# The fingertip's flesh past its landmark, in palm lengths.
TIP_EXTENSION = 0.1
# The tube's half width, in palm lengths: wider than the finger - the
# finger's own skin mask trims it to the finger's real outline.
TUBE_HALF_WIDTH = 0.18
# The first points of the chain, at the knuckle, never move.
BASE_NODES = 2
# Tethers on a held chain, as a share of the face's: a held finger runs nearly
# straight from knuckle to pinch instead of bowing back to where it was.
# Let go, it is tethered fully again, and snaps home.
CHAIN_TETHER = 0.15
# A stretched finger thins (it keeps its volume), but never below this.
MIN_THINNING = 0.7
# How far each frame's measured finger shape moves the one used: slowly
# while it is held (its landmarks wobble under the other hand), quickly not.
EASE_HELD, EASE_FREE = 0.1, 0.5
# The finger is held from this far below the pinch (palm lengths) to its tip.
HOLD_BELOW = 0.08
# The palm: the wrist and the four knuckles (not the thumb, which moves).
PALM_FRAME = (HAND_WRIST, 5, 9, 13, 17)


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
    """Knuckle, two joints, tip - and the flesh just past the tip."""
    joints = _px(hand, HAND_DIGITS[digit], size)
    last = joints[-1] - joints[-2]
    beyond = joints[-1] + last / max(float(np.linalg.norm(last)), 1e-6) * TIP_EXTENSION * _palm_length(hand, size)
    return np.vstack([joints, beyond])


def resample(line: np.ndarray, count: int) -> np.ndarray:
    """`count` points evenly spaced along a polyline."""
    lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(lengths)])
    wanted = np.linspace(0.0, along[-1], count)
    return np.stack([np.interp(wanted, along, line[:, k]) for k in range(2)], axis=1)


def _normals(points: np.ndarray) -> np.ndarray:
    """Unit normals (left of the direction of travel) along a chain."""
    tangents = np.gradient(points, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)


def _thinning(points: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """How thin each part of the chain is drawn: stretched by r, a finger
    keeps its volume at width 1/sqrt(r)."""
    span = np.linalg.norm(np.gradient(points, axis=0), axis=1)
    resting = np.maximum(np.linalg.norm(np.gradient(rest, axis=0), axis=1), 1e-9)
    return np.clip(1.0 / np.sqrt(np.maximum(span / resting, 1e-9)), MIN_THINNING, 1.0)


def tube_triangles(count: int, across: int) -> np.ndarray:
    rows, cols = np.meshgrid(np.arange(count - 1), np.arange(across - 1), indexing="ij")
    a = (rows * across + cols).ravel()
    b, c, d = a + across, a + across + 1, a + 1
    return np.concatenate([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)]).astype(np.int32)


def _chain_edges(count: int) -> np.ndarray:
    stretch = [(i, i + 1) for i in range(count - 1)]
    bend = [(i, i + 2) for i in range(count - 2)]
    return np.array(stretch + bend, dtype=np.int64)


TRIANGLES = tube_triangles(NODES, ACROSS)


def finger_usable(hand: HandObservation, part: str, size: tuple[int, int]) -> bool:
    """Whether a hand's landmarks make a finger at all (not collapsed to a
    point, as the tracker sometimes reports for a frame)."""
    line = _finger_line(hand, part.split(".")[2], size)
    return _palm_length(hand, size) >= 1.0 and float(np.linalg.norm(line[-1] - line[0])) >= 1.0


@dataclass
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
        if matrix is not None and abs(float(np.linalg.det(matrix[:, :2]))) > 1e-6:
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
        return Hold(target, core, np.ones(len(core)))

    def step(self, rest: np.ndarray, pulls: dict[str, np.ndarray]) -> None:
        """One frame of physics; each holding hand's points go to where they
        rest plus that hand's pull (pixels, and lift)."""
        holds = [(hand, hold) for hand, hold in self.holds.items() if hand in pulls]
        self.body.tether = CHAIN_TETHER if holds else 1.0
        if not holds:
            self.body.step(rest, np.zeros(0, dtype=np.int64), np.zeros((0, 3)))
            return
        grabbed = np.concatenate([hold.core for _, hold in holds])
        targets = np.concatenate([rest[hold.core] + pulls[hand] for hand, hold in holds])
        self.body.step(rest, grabbed, targets, np.concatenate([hold.grip for _, hold in holds]))

    def tube(self, rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(drawn vertices (V, 3), where each one's skin rests (V, 2)): the
        tube around the chain now, and the same tube around it at rest."""
        width = TUBE_HALF_WIDTH * self.palm_px
        offsets = width * ACROSS_OFFSETS[None, :, None]
        source = rest[:, None, :2] + offsets * _normals(rest[:, :2])[:, None, :]
        points = self.body.positions
        thin = _thinning(points[:, :2], rest[:, :2])[:, None, None]
        drawn = points[:, None, :2] + thin * offsets * _normals(points[:, :2])[:, None, :]
        lift = np.repeat(points[:, 2], ACROSS)[:, None]
        return np.hstack([drawn.reshape(-1, 2), lift]), source.reshape(-1, 2)

    def skin(
        self,
        rest: np.ndarray,
        labels: np.ndarray | None,
        size: tuple[int, int],
        window: Window,
        hand: HandObservation | None = None,
        moving_only: bool = False,
    ) -> np.ndarray:
        """The finger's own skin over `window` (uint8): its outline, knuckle
        to fingertip, at rest; cut to the body skin the segmentation sees,
        and - given its `hand` - to what is nearer this finger than any
        other. `moving_only` leaves out the knuckle end, which never moves."""
        joints = apply(self.to_image, self.shape[:-1])
        line = np.vstack([rest[BASE_NODES, :2], joints[1:]]) if moving_only else joints
        outline = _outline(line, self.palm_px, window)
        if hand is not None:
            others = [_px(hand, HAND_DIGITS[d], size) for d in HAND_DIGITS if d != self.digit]
            outline = _own_side(outline, joints, others, window)
        if labels is not None:
            outline = np.minimum(outline, body_skin(labels, size, window))
        return outline


def _outline(line: np.ndarray, palm_px: float, window: Window) -> np.ndarray:
    """A finger's outline around its centre line, over `window` (uint8)."""
    outline = np.zeros(window.shape, dtype=np.uint8)
    points = np.round(line - [window.x0, window.y0]).astype(np.int32)
    thickness = max(3, int(round(2 * FINGER_HALF_WIDTH * palm_px)))
    cv2.polylines(outline, [points], False, 255, thickness, lineType=cv2.LINE_8)
    for end in (points[0], points[-1]):
        cv2.circle(outline, (int(end[0]), int(end[1])), thickness // 2, 255, -1)
    return outline


def _own_side(outline: np.ndarray, own: np.ndarray, others: list[np.ndarray], window: Window) -> np.ndarray:
    """`outline`, cut to the pixels nearer to this finger than to the others."""
    rows, cols = np.nonzero(outline)
    if len(rows) == 0 or not others:
        return outline
    points = np.stack([cols + window.x0 + 0.5, rows + window.y0 + 0.5], axis=1).astype(np.float64)
    mine = distance_to_chain(points, own)
    theirs = np.min([distance_to_chain(points, other) for other in others], axis=0)
    cut = outline.copy()
    cut[rows[theirs < mine], cols[theirs < mine]] = 0
    return cut
