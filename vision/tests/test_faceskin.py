"""Tests for the face's skin mesh, glued to the face's own landmarks.

- glued, a point stays exactly where it is while the face is still
- the face moved, turned or scaled: every point goes exactly with it
- one landmark moving moves the points near it, and barely those far away
- the mesh follows a moving face with no lag, and a pinch holds its skin
"""

from __future__ import annotations

import math

import numpy as np
from face_fixture import CENTRE, H, W, landmarks

from ofp_vision.faceskin import FaceSession, glue_weights, glued
from ofp_vision.grab import face_target_points

SIZE = (W, H)


def scattered(count: int = 120) -> np.ndarray:
    return np.random.default_rng(5).uniform([100, 100], [500, 400], (count, 2))


def test_glued_points_stay_put_while_the_landmarks_do():
    marks = scattered()
    points = np.random.default_rng(6).uniform([150, 150], [450, 350], (300, 2))
    indices, weights = glue_weights(points, marks)
    assert np.abs(glued(indices, weights, marks) - points).max() < 1e-3


def test_the_face_moved_turned_and_scaled_takes_every_point_with_it():
    marks = scattered()
    points = np.random.default_rng(6).uniform([150, 150], [450, 350], (300, 2))
    indices, weights = glue_weights(points, marks)
    turn = np.array([[math.cos(0.4), -math.sin(0.4)], [math.sin(0.4), math.cos(0.4)]])
    moved = 1.3 * marks @ turn.T + [40.0, -25.0]
    assert np.abs(glued(indices, weights, moved) - (1.3 * points @ turn.T + [40.0, -25.0])).max() < 1e-2


def test_a_landmark_moving_carries_the_skin_near_it_not_far_away():
    marks = scattered()
    farthest = marks[int(np.argmax(np.linalg.norm(marks - marks[0], axis=1)))]
    points = np.array([marks[0] + [3.0, 0.0], farthest + [3.0, 0.0]])
    indices, weights = glue_weights(points, marks)
    moved = marks.copy()
    moved[0] += [20.0, 0.0]
    shift = glued(indices, weights, moved) - points
    assert shift[0, 0] > 10.0, "near it: most of the way"
    assert abs(shift[1, 0]) < 1.0, "far away: barely"


def px(point) -> np.ndarray:
    return np.array([point[0] * W, point[1] * H])


def test_the_mesh_follows_a_moving_face_with_no_lag():
    face = landmarks()
    session = FaceSession.start(face, SIZE, px(face_target_points(face)["cheekL"]))
    rest = None
    for step in range(1, 8):
        moved = landmarks(scale=1.0 + 0.02 * step, centre=(CENTRE[0] + 0.01 * step, CENTRE[1]))
        rest = session.follow(moved, SIZE)
        session.step(rest, {})
    assert np.abs(session.body.positions - rest).max() < 0.01


def test_a_pinch_holds_the_skin_under_it_and_pulls_it():
    face = landmarks()
    cheek = px(face_target_points(face)["cheekL"])
    session = FaceSession.start(face, SIZE, cheek)
    session.holds = {"Right": session.hold("cheekL", cheek)}
    rest = None
    for _ in range(20):
        rest = session.follow(face, SIZE)
        session.step(rest, {"Right": np.array([60.0, 0.0, 0.0])})
    under = int(np.argmin(np.linalg.norm(rest[:, :2] - cheek, axis=1)))
    assert np.abs(session.body.positions[under, :2] - (rest[under, :2] + [60.0, 0.0])).max() < 1.0
