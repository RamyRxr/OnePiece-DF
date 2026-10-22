"""Tests for the skin mesh: the grid of points the GPU stretch is built on."""

from __future__ import annotations

import math

import numpy as np
import pytest
from face_fixture import H, W, landmarks

from ofp_vision.indices import FACE_EYE_OUTER_L, FACE_EYE_OUTER_R
from ofp_vision.skinmesh import BodyFrame, build_patch, face_frame

FACE = landmarks()


# --- the frame glued to the face ----------------------------------------------


def test_the_face_frame_follows_the_eye_line_and_spacing():
    frame = face_frame(FACE, (W, H))
    r = np.array([FACE[FACE_EYE_OUTER_R][0] * W, FACE[FACE_EYE_OUTER_R][1] * H])
    l = np.array([FACE[FACE_EYE_OUTER_L][0] * W, FACE[FACE_EYE_OUTER_L][1] * H])
    assert frame.scale == pytest.approx(np.linalg.norm(l - r))
    assert frame.x_axis == pytest.approx((l - r) / np.linalg.norm(l - r))
    assert float(frame.x_axis @ frame.y_axis) == pytest.approx(0.0, abs=1e-9)


def test_a_point_keeps_its_place_on_the_face_when_the_head_moves():
    """Shift, scale and tilt the face: a point glued to it moves with it."""
    still = BodyFrame(origin=np.array([600.0, 350.0]), x_axis=np.array([1.0, 0.0]), y_axis=np.array([0.0, 1.0]), scale=100.0)
    angle = math.radians(20)
    turned = BodyFrame(
        origin=np.array([650.0, 330.0]),
        x_axis=np.array([math.cos(angle), math.sin(angle)]),
        y_axis=np.array([-math.sin(angle), math.cos(angle)]),
        scale=130.0,
    )
    point = np.array([[0.4, 0.3]])
    on_still = still.to_image(point)
    assert still.to_local(on_still) == pytest.approx(point)
    on_turned = turned.to_image(point)
    # Same place relative to the eyes: 0.4 eye-distances along, 0.3 down.
    expected = turned.origin + 130.0 * (0.4 * turned.x_axis + 0.3 * turned.y_axis)
    assert on_turned[0] == pytest.approx(expected)


# --- the grid ----------------------------------------------------------------------


def test_the_patch_is_an_even_grid_around_its_centre():
    patch = build_patch(centre=np.array([0.5, 0.2]), half_size=0.6, spacing=0.05)
    rows, cols = patch.shape
    assert rows == cols == 25
    assert len(patch.local) == rows * cols
    assert patch.local.min(axis=0) == pytest.approx([-0.1, -0.4])
    assert patch.local.max(axis=0) == pytest.approx([1.1, 0.8])
    steps = np.diff(patch.local.reshape(rows, cols, 2)[0, :, 0])
    assert steps == pytest.approx(np.full(cols - 1, 0.05))


def test_every_cell_is_two_triangles():
    patch = build_patch(centre=np.zeros(2), half_size=0.2, spacing=0.1)
    rows, cols = patch.shape
    assert len(patch.triangles) == 2 * (rows - 1) * (cols - 1)
    assert patch.triangles.min() == 0 and patch.triangles.max() == rows * cols - 1


def test_springs_join_neighbours_diagonals_and_points_two_apart():
    """Stretch (4 neighbours), shear (diagonals), bending (two steps away)."""
    patch = build_patch(centre=np.zeros(2), half_size=0.2, spacing=0.1)
    rows, cols = patch.shape
    lengths = np.linalg.norm(patch.local[patch.edges[:, 0]] - patch.local[patch.edges[:, 1]], axis=1)
    kinds = sorted(set(np.round(lengths / 0.1, 3)))
    assert kinds == pytest.approx([1.0, math.sqrt(2), 2.0], abs=1e-3)
    stretch = 2 * rows * (cols - 1)
    shear = 2 * (rows - 1) * (cols - 1)
    bend = 2 * rows * (cols - 2)
    assert len(patch.edges) == stretch + shear + bend
