"""The face, stretched: a fine skin mesh glued to the face's own landmarks.

When a pinch on the face starts, a grid of a few thousand points is laid over
the skin around it (`skinmesh.py`). Each point is glued to the tracked face
landmarks nearest it - a fixed blend of them, nearer ones counting for more
(affine moving least squares) - so every frame it rests exactly where its bit
of skin is now: when the head turns, the mouth opens, a cheek rises in a
smile. Physics (`softbody.py`) adds only the stretch.

The mouth is part of the skin like the rest: pulled at a corner it widens,
lips and opening together, the way Luffy's does - not a strip of skin pulled
out beside a mouth that stays put.

What of it shows is worked out every frame from the face as it is now - the
face inside its outline, mouth and all; beyond it (an ear, the neck) what the
segmentation calls skin - not from a mask taken at the pinch, which stops
fitting the face as soon as it moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .indices import FACE_EYE_POINTS, FACE_OVAL
from .signals import Point
from .skin import Window, face_masks, feathered, segmented_face_masks
from .skinmesh import Hold, SkinPatch, build_patch, face_frame
from .softbody import SoftBody

FACE_PART = "face"
# The mesh: this far around the pinch, points this far apart (both in
# eye-distances): ~3,700 points, ~5px apart at webcam distance.
FACE_PATCH_HALF = 0.9
FACE_SPACING = 0.03
# Points within this of the eye outline are pinned: the eyes never move.
EYE_PIN = 0.1
# The mesh's outer rows are pinned too: the skin beyond holds it.
BORDER_ROWS = 2
# How much skin a pinch holds, as a radius in eye-distances: a mouth's corner
# is held broadly - the lips go with it - and the nose takes its nostrils.
CORE_PER_FACE = 0.12
CORE_BY_TARGET = {"nose": 0.17, "mouthL": 0.15, "mouthR": 0.15, "lipTop": 0.1, "lipBottom": 0.1}
# The fingers hold the inner part of the pinch firmly (this fraction of its
# radius); beyond, the grip fades to a gentle pull at its rim.
GRIP_FIRM = 0.5
# A second pinch joins the mesh if it lands this far inside its edge (a
# fraction of its half size); nearer the pinned edge it gets its own.
JOIN_INSIDE = 0.7
# The face's own landmarks (the last ten are the irises: they move with the
# gaze, not the skin).
FACE_LANDMARKS = 468
# Each point is glued to this many landmarks around it.
GLUE_NEIGHBOURS = 32
# The visible edge starts this far inside the mask's soft edge (which is 0.5
# on the skin's outline), where the camera's pixels are pure skin.
VISIBLE_FROM, VISIBLE_TO = 0.75, 0.98


def glue_weights(points: np.ndarray, landmarks: np.ndarray, neighbours: int = GLUE_NEIGHBOURS):
    """(indices, weights), each (P, K): every point as a fixed blend of the
    K landmarks nearest it, such that `glued` moves it with the landmarks'
    local motion - exactly, for any move, turn or scale of the face (affine
    moving least squares). Weights fall to zero at the K-th neighbour, so
    neighbouring points' blends never jump."""
    k = min(neighbours, len(landmarks))
    distance = np.linalg.norm(points[:, None, :] - landmarks[None, :, :], axis=2)
    indices = np.argsort(distance, axis=1)[:, :k]
    near = np.maximum(np.take_along_axis(distance, indices, axis=1), 0.5)
    weight = np.maximum(1.0 / near**2 - 1.0 / near[:, -1:] ** 2, 0.0) + 1e-9
    marks = landmarks[indices]
    centre = (weight[..., None] * marks).sum(axis=1) / weight.sum(axis=1)[:, None]
    spread = marks - centre[:, None, :]
    moment = np.einsum("pk,pka,pkb->pab", weight, spread, spread) + 1e-6 * np.eye(2)
    lean = np.einsum("pa,pab->pb", points - centre, np.linalg.inv(moment))
    blend = weight * np.einsum("pb,pkb->pk", lean, spread) + weight / weight.sum(axis=1)[:, None]
    return indices, blend


def glued(indices: np.ndarray, weights: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Where the glued points are, with the landmarks where they are now."""
    return (weights[..., None] * landmarks[indices]).sum(axis=1)


def _landmarks_px(face: list[Point], size: tuple[int, int]) -> np.ndarray:
    return np.array([[p[0] * size[0], p[1] * size[1]] for p in face[:FACE_LANDMARKS]])


def _distinct(points: np.ndarray) -> np.ndarray:
    """Indices of the points at distinct places (the first of each)."""
    _, first = np.unique(np.round(points, 2), axis=0, return_index=True)
    return np.sort(first)


@dataclass
class FaceSession:
    """The face's mesh and physics, from the first pinch on it until it has
    settled. Each hand on it has a `Hold`. (A finger is a `FingerSession`.)"""

    part: str
    patch: SkinPatch
    body: SoftBody
    # The glue (indices into `marks`, weights), the landmarks it uses, and
    # where the points rested last frame.
    glue: tuple[np.ndarray, np.ndarray]
    marks: np.ndarray
    rest_xy: np.ndarray
    # The distance between the eyes when it started (pixels).
    scale: float
    holds: dict[str, Hold] = field(default_factory=dict)

    @classmethod
    def start(cls, face: list[Point], size: tuple[int, int], anchor_px: np.ndarray) -> FaceSession | None:
        frame = face_frame(face, size)
        if frame is None:
            return None
        patch = build_patch(frame.to_local(anchor_px[None, :])[0], FACE_PATCH_HALF, FACE_SPACING)
        rest = frame.to_image(patch.local)
        landmarks = _landmarks_px(face, size)
        marks = _distinct(landmarks)
        # Every point moves - the ones that are not skin just stay invisible.
        eyes = landmarks[list(FACE_EYE_POINTS)]
        near_eyes = np.min(np.linalg.norm(rest[:, None, :] - eyes[None, :, :], axis=2), axis=1) < EYE_PIN * frame.scale
        rest3 = np.zeros((len(rest), 3))
        rest3[:, :2] = rest
        body = SoftBody(rest3, patch.edges, _border(patch) | near_eyes)
        return cls(FACE_PART, patch, body, glue_weights(rest, landmarks[marks]), marks, rest, frame.scale)

    def rest(self) -> np.ndarray:
        rest = np.zeros((len(self.rest_xy), 3))
        rest[:, :2] = self.rest_xy
        return rest

    def follow(self, face: list[Point] | None, size: tuple[int, int]) -> np.ndarray:
        """Follow the face this frame (keep still if it is not in view) and
        carry the mesh along with it. Returns where the mesh rests now."""
        if face is not None and len(face) >= FACE_LANDMARKS:
            now = glued(*self.glue, _landmarks_px(face, size)[self.marks])
            moved = np.zeros((len(now), 3))
            moved[:, :2] = now - self.rest_xy
            self.body.carry(moved)
            self.rest_xy = now
        return self.rest()

    def hold(self, target: str, anchor_px: np.ndarray) -> Hold | None:
        """The points a pinch at `anchor_px` holds: firm in the middle of the
        pinch, gentle at its rim. None if it is not well inside the mesh."""
        rows, cols = self.patch.shape
        nearest = int(np.argmin(np.linalg.norm(self.rest_xy - anchor_px, axis=1)))
        row, col = divmod(nearest, cols)
        inside = abs(row - (rows - 1) / 2) <= JOIN_INSIDE * (rows - 1) / 2
        inside &= abs(col - (cols - 1) / 2) <= JOIN_INSIDE * (cols - 1) / 2
        if not inside:
            return None
        core_px = CORE_BY_TARGET.get(target, CORE_PER_FACE) * self.scale
        reach = np.linalg.norm(self.rest_xy - anchor_px, axis=1)
        free = ~self.body.pinned
        for other in self.holds.values():
            free[other.core] = False
        core = np.nonzero((reach <= core_px) & free)[0]
        if len(core) == 0:
            return None
        grip = np.clip(1.0 - (reach[core] / core_px - GRIP_FIRM) / (1.0 - GRIP_FIRM), 0.05, 1.0)
        return Hold(target, core, grip)

    def step(self, rest: np.ndarray, pulls: dict[str, np.ndarray]) -> None:
        """One frame of physics; each holding hand's points go to where they
        rest plus that hand's pull (pixels, and lift)."""
        holds = [(hand, hold) for hand, hold in self.holds.items() if hand in pulls]
        if not holds:
            self.body.step(rest, np.zeros(0, dtype=np.int64), np.zeros((0, 3)))
            return
        grabbed = np.concatenate([hold.core for _, hold in holds])
        targets = np.concatenate([rest[hold.core] + pulls[hand] for hand, hold in holds])
        self.body.step(rest, grabbed, targets, np.concatenate([hold.grip for _, hold in holds]))

    def mark_resting(self, mask: np.ndarray, scale: float) -> None:
        """Mark where the mesh rests on `mask` (a frame `scale` times the
        camera's): what is being stretched is never learned as background."""
        hull = cv2.convexHull(np.round(self.rest_xy * scale).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

    def leaves(self, face: list[Point], size: tuple[int, int], window: Window, visible: np.ndarray) -> np.ndarray:
        """Which of the visible skin can leave a gap behind it, over `window`:
        within the mesh's moving part (its pinned rim never moves), and only
        outside the face itself - an ear, say - where the room is behind it.
        Inside, a face's skin is one sheet: it never opens."""
        origin = np.array([window.x0, window.y0])
        inner = self.rest_xy[~_border(self.patch)] - origin
        inside = np.zeros(window.shape, dtype=np.uint8)
        cv2.fillConvexPoly(inside, cv2.convexHull(np.round(inner).astype(np.int32)), 1)
        cv2.fillConvexPoly(inside, _oval(face, size, origin), 0)
        return visible * inside


def face_visible(face: list[Point], labels: np.ndarray | None, bgr: np.ndarray, size: tuple[int, int],
                 window: Window) -> np.ndarray:
    """What of the face is drawn, over `window`, as it is now (0..1): all of
    it inside its outline - the mouth too, which stretches with its lips -
    and beyond it what the segmentation (or, without it, skin colour) calls
    skin: an ear, the neck. From a little inside the edge, where the pixels
    are pure skin."""
    if labels is not None:
        skin, _ = segmented_face_masks(labels, face, size, window)
    else:
        skin, _ = face_masks(bgr, face, window)
    cv2.fillConvexPoly(skin, _oval(face, size, np.array([window.x0, window.y0])), 255)
    ramp = np.clip((feathered(skin) - VISIBLE_FROM) / (VISIBLE_TO - VISIBLE_FROM), 0.0, 1.0)
    return (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)


def _oval(face: list[Point], size: tuple[int, int], origin: np.ndarray) -> np.ndarray:
    oval = np.array([[face[i][0] * size[0], face[i][1] * size[1]] for i in FACE_OVAL]) - origin
    return cv2.convexHull(np.round(oval).astype(np.int32))


def _border(patch: SkinPatch) -> np.ndarray:
    rows, cols = patch.shape
    grid = np.zeros((rows, cols), dtype=bool)
    grid[:BORDER_ROWS, :] = grid[-BORDER_ROWS:, :] = True
    grid[:, :BORDER_ROWS] = grid[:, -BORDER_ROWS:] = True
    return grid.ravel()
