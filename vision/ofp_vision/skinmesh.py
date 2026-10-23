"""The skin mesh: a fine grid of points over the face skin being pinched.

When a grab on the face starts, a square patch of skin around the pinch is
covered with a grid of a few thousand points. The grid is laid out in a frame
glued to the face: its origin is the face's centre, its x axis the eye line
and its unit the distance between the eyes. Each point remembers where it
sits in that frame, so when the head moves, turns or comes closer, every
point's resting place moves with it. Physics (`softbody.py`) then moves the
points away from where they rest. (A finger is a tube: `fingers.py`.)

The grid is triangulated for drawing, and joined by springs for the physics:
to its four neighbours (stretch), its four diagonals (shear), and the points
two steps away (bending - what makes skin fold smoothly instead of crumple).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .indices import FACE_EYE_OUTER_L, FACE_EYE_OUTER_R, FACE_OVAL
from .signals import Point


@dataclass(frozen=True)
class BodyFrame:
    """A coordinate frame glued to a body part, in image pixels."""

    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    scale: float

    def to_image(self, local: np.ndarray) -> np.ndarray:
        """(N, 2) frame coordinates -> (N, 2) image pixels."""
        return self.origin + self.scale * (local[:, :1] * self.x_axis + local[:, 1:2] * self.y_axis)

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """(N, 2) image pixels -> (N, 2) frame coordinates."""
        offset = points - self.origin
        return np.stack([offset @ self.x_axis, offset @ self.y_axis], axis=1) / self.scale


def _px(point: Point, size: tuple[int, int]) -> np.ndarray:
    return np.array([point[0] * size[0], point[1] * size[1]], dtype=np.float64)


def _frame(origin: np.ndarray, along: np.ndarray, scale: float) -> BodyFrame:
    x_axis = along / np.linalg.norm(along)
    return BodyFrame(origin, x_axis, np.array([-x_axis[1], x_axis[0]]), scale)


def face_frame(face: list[Point], size: tuple[int, int]) -> BodyFrame | None:
    """Origin at the face's centre, x along the eye line, unit = eyes apart."""
    right, left = _px(face[FACE_EYE_OUTER_R], size), _px(face[FACE_EYE_OUTER_L], size)
    apart = float(np.linalg.norm(left - right))
    if apart < 1.0:
        return None
    centre = np.mean([_px(face[i], size) for i in FACE_OVAL], axis=0)
    return _frame(centre, left - right, apart)


@dataclass(frozen=True)
class SkinPatch:
    """A square grid of points, in frame coordinates, with its triangles and
    springs. `shape` is (rows, cols); point (r, c) is index r * cols + c."""

    local: np.ndarray
    shape: tuple[int, int]
    triangles: np.ndarray
    edges: np.ndarray


def build_patch(centre: np.ndarray, half_size: float, spacing: float) -> SkinPatch:
    """A grid `spacing` apart covering `centre` +- `half_size` (frame units)."""
    steps = int(np.ceil(half_size / spacing))
    offsets = np.arange(-steps, steps + 1) * spacing
    count = len(offsets)
    xs, ys = np.meshgrid(offsets + centre[0], offsets + centre[1])
    local = np.stack([xs.ravel(), ys.ravel()], axis=1)
    index = np.arange(count * count).reshape(count, count)
    return SkinPatch(local, (count, count), _triangles(index), _springs(index))


def _triangles(index: np.ndarray) -> np.ndarray:
    a, b = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
    c, d = index[1:, :-1].ravel(), index[1:, 1:].ravel()
    return np.concatenate([np.stack([a, b, d], axis=1), np.stack([a, d, c], axis=1)]).astype(np.int32)


def _springs(index: np.ndarray) -> np.ndarray:
    pairs = [
        (index[:, :-1], index[:, 1:]),  # stretch, across
        (index[:-1, :], index[1:, :]),  # stretch, down
        (index[:-1, :-1], index[1:, 1:]),  # shear
        (index[:-1, 1:], index[1:, :-1]),  # shear
        (index[:, :-2], index[:, 2:]),  # bending, across
        (index[:-2, :], index[2:, :]),  # bending, down
    ]
    return np.concatenate([np.stack([a.ravel(), b.ravel()], axis=1) for a, b in pairs]).astype(np.int32)


@dataclass(frozen=True)
class Hold:
    """One hand's pinch on a mesh: the points it holds, and how firmly."""

    target: str
    core: np.ndarray
    grip: np.ndarray
