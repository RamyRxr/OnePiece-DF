"""Tests for the skin physics: points with mass, springs, and tethers.

The behaviours the stretch needs from it, in the terms you'd use watching it:

- left alone, skin stays where it rests
- the pinched skin goes exactly where the fingers are
- the skin around follows, less and less with distance; far skin stays put
- pinned points (the eyes) never move
- it has weight: yank it and the surrounding skin lags a little, then catches up
- let go and it snaps back quickly and cleanly: no slime, no wobble
- lifted toward the camera, the skin around rises with it (a tent)
- springs that were cut (across the mouth) carry no pull

Fingers hold a patch of skin, not a point: the tests pinch the points within
15px of the centre, as the stretch does.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ofp_vision.skinmesh import build_patch
from ofp_vision.softbody import SoftBody

SPACING = 6.0  # pixels between points, about what a face at webcam distance gets
NOTHING = np.array([], dtype=np.int64)


def fastest(action, runs: int = 7) -> float:
    """Seconds the quickest of `runs` calls took. The fastest run is what
    the code costs; slower ones measure whatever else the machine was doing."""
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        action()
        best = min(best, time.perf_counter() - start)
    return best


def make_body(half: int = 30, cut=None) -> tuple[SoftBody, np.ndarray, np.ndarray]:
    """A (2*half+1)^2 grid, SPACING px apart, centred on the origin, and the
    pinched patch: the points within 15px of the centre."""
    patch = build_patch(np.zeros(2), half * SPACING, SPACING)
    rest = np.zeros((len(patch.local), 3))
    rest[:, :2] = patch.local
    edges = patch.edges if cut is None else patch.edges[~cut(patch.local, patch.edges)]
    body = SoftBody(rest, edges, pinned=np.zeros(len(rest), dtype=bool))
    core = np.nonzero(np.linalg.norm(patch.local, axis=1) <= 15.0)[0]
    return body, rest, core


def centre_of(rest: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(rest[:, :2], axis=1)))


def run(body: SoftBody, rest: np.ndarray, grabbed, target, frames: int) -> None:
    for _ in range(frames):
        body.step(rest, grabbed, target)


def distances(rest: np.ndarray) -> np.ndarray:
    return np.linalg.norm(rest[:, :2], axis=1)


def test_skin_left_alone_stays_where_it_rests():
    body, rest, _ = make_body()
    run(body, rest, NOTHING, np.zeros((0, 3)), 30)
    assert np.abs(body.positions - rest).max() < 1e-9


def test_the_pinched_skin_goes_exactly_where_the_fingers_are():
    body, rest, core = make_body()
    target = rest[core] + [60.0, 0.0, 0.0]
    run(body, rest, core, target, 5)
    assert body.positions[core] == pytest.approx(target)


def test_the_pull_fades_with_distance_and_far_skin_stays_put():
    body, rest, core = make_body()
    run(body, rest, core, rest[core] + [60.0, 0.0, 0.0], 60)
    moved = np.linalg.norm(body.positions - rest, axis=1)
    distance = distances(rest)
    near = moved[(distance > 20) & (distance < 26)].mean()
    middle = moved[(distance > 60) & (distance < 66)].mean()
    far = moved[(distance > 100) & (distance < 106)].mean()
    assert 60.0 > near > middle > far
    assert far < 3.0


def test_pinned_points_never_move():
    body, rest, core = make_body()
    beside = centre_of(rest) + 4
    pinned = np.zeros(len(rest), dtype=bool)
    pinned[beside] = True
    body = SoftBody(rest, body.edges, pinned=pinned)
    run(body, rest, core, rest[core] + [40.0, 0.0, 0.0], 30)
    assert body.positions[beside] == pytest.approx(rest[beside])


def test_it_has_weight_yanked_skin_lags_then_catches_up():
    """Skin six points behind the pinch is still on its way after a frame -
    but well on its way: weight, not goo."""
    body, rest, core = make_body()
    behind = centre_of(rest) - 6
    target = rest[core] + [60.0, 0.0, 0.0]
    body.step(rest, core, target)
    first_frame = body.positions[behind, 0] - rest[behind, 0]
    run(body, rest, core, target, 60)
    settled = body.positions[behind, 0] - rest[behind, 0]
    assert 0.5 * settled < first_frame < 0.9 * settled


def test_let_go_it_snaps_back_quickly_and_cleanly():
    """Not slime: back within 2 frames, at most a small flick past rest,
    still within a few frames."""
    body, rest, core = make_body()
    centre = centre_of(rest)
    run(body, rest, core, rest[core] + [60.0, 0.0, 0.0], 60)
    track = []
    for _ in range(10):
        body.step(rest, NOTHING, np.zeros((0, 3)))
        track.append(body.positions[centre, 0] - rest[centre, 0])
    assert 0.0 < track[0] < 0.4 * 60.0, "most of the way back in one frame, but not teleported"
    assert abs(track[1]) < 0.1 * 60.0, "back within two"
    assert min(track) > -0.05 * 60.0, "no wobble: at most a small flick past rest"
    assert body.settled(rest), "and still"


def test_lifted_toward_the_camera_the_skin_around_rises_into_a_tent():
    body, rest, core = make_body()
    run(body, rest, core, rest[core] + [0.0, 0.0, 50.0], 60)
    lift = body.positions[:, 2]
    distance = distances(rest)
    near = lift[(distance > 20) & (distance < 26)].mean()
    further = lift[(distance > 50) & (distance < 56)].mean()
    assert near > further > 0.0


def test_a_soft_grip_holds_the_centre_firmly_and_the_rim_loosely():
    """Fingers squeeze the middle of the pinch hardest: the rim of the
    pinched patch follows, but less than all the way."""
    body, rest, core = make_body()
    centre = centre_of(rest)
    distance = distances(rest)[core]
    grip = np.clip(1.0 - distance / 16.0, 0.1, 1.0)
    target = rest[core] + [60.0, 0.0, 0.0]
    run_gripped(body, rest, core, target, grip, 40)
    rim = core[np.argmax(distance)]
    assert body.positions[centre, 0] - rest[centre, 0] == pytest.approx(60.0)
    assert 0.0 < body.positions[rim, 0] - rest[rim, 0] < 59.0


def run_gripped(body, rest, grabbed, target, grip, frames):
    for _ in range(frames):
        body.step(rest, grabbed, target, grip)


def test_a_cut_carries_no_pull():
    """No springs cross x = 3 (a cut, like the mouth opening): pulling one
    side away leaves the other side where it rests."""

    def cut(local, edges):
        a, b = local[edges[:, 0], 0], local[edges[:, 1], 0]
        return (np.minimum(a, b) < 3.0) & (np.maximum(a, b) > 3.0)

    body, rest, core = make_body(cut=cut)
    left_core = core[rest[core, 0] < 3.0]
    run(body, rest, left_core, rest[left_core] + [-60.0, 0.0, 0.0], 60)
    other_side = rest[:, 0] > 3.0
    assert np.abs(body.positions[other_side] - rest[other_side]).max() < 1e-6


def test_a_frame_of_physics_fits_the_budget():
    body, rest, core = make_body(half=40)  # 6561 points, ~38k springs
    target = rest[core] + [80.0, 0.0, 20.0]
    body.step(rest, core, target)  # compile
    # ~4-8ms measured; the budget has headroom for a busy machine (the lab
    # running alongside) - it is here to catch a real slowdown, not noise.
    assert fastest(lambda: body.step(rest, core, target)) < 0.012


def test_a_new_grip_starts_from_where_its_skin_is():
    """One hand lets go as the other takes hold, the same number of points
    in each grip: the new grip starts from its own skin, not from where the
    old one was being pulled to."""
    body, rest, core = make_body()
    far = np.nonzero(np.linalg.norm(rest[:, :2] - [90.0, 0.0], axis=1) <= 15.0)[0][: len(core)]
    body.step(rest, core, rest[core] + [60.0, 0.0, 0.0])
    before = body.positions[far].copy()
    body.step(rest, far, before)
    neighbours = np.nonzero(np.linalg.norm(rest[:, :2] - [90.0, 0.0], axis=1) <= 30.0)[0]
    assert np.abs(body.positions[neighbours, :2] - rest[neighbours, :2]).max() < 3.0


def test_a_moving_body_part_carries_its_skin_without_lag():
    """Moving the head is not a pull on the skin: carried along with its
    resting places, every point stays exactly on them - no copy of the face
    sliding behind the real one, like a mask."""
    body, rest, _ = make_body()
    for frame in range(1, 6):
        moved = rest + [12.0 * frame, -5.0 * frame, 0.0]
        body.carry(moved - body_rest(rest, frame - 1))
        body.step(moved, NOTHING, np.zeros((0, 3)))
    assert np.abs(body.positions - moved).max() < 0.01
    assert body.settled(moved)


def body_rest(rest: np.ndarray, frame: int) -> np.ndarray:
    return rest + [12.0 * frame, -5.0 * frame, 0.0]
