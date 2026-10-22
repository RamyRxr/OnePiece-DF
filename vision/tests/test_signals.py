"""Synthetic-landmark tests for the stretch math.

These lock in the properties that matter:

* **Scale invariance** - multiply every landmark by 2 and no feature changes.
  This is what makes the signal survive the player leaning in or away.
* **Responsiveness** - move the cheekbone proxy outward and the ear channel
  goes up. Move a fingertip and the finger channel goes up.
* **Symmetry** - a perfectly symmetric face yields zero asymmetry; pulling
  one side yields a signed asymmetry.

No camera, no MediaPipe, no flakiness.
"""

from __future__ import annotations

import math

import pytest

from ofp_vision.indices import (
    FACE_CHEEK_BONE_L,
    FACE_CHEEK_BONE_R,
    FACE_CHIN,
    FACE_EYE_OUTER_L,
    FACE_EYE_OUTER_R,
    FACE_FOREHEAD,
    FACE_LIP_BOTTOM_INNER,
    FACE_LIP_TOP_INNER,
    FACE_MOUTH_L,
    FACE_MOUTH_R,
    FACE_NOSE_TIP,
    HAND_DIGITS,
    HAND_MIDDLE_MCP,
    HAND_WRIST,
    POSE_LEFT_ELBOW,
    POSE_LEFT_SHOULDER,
    POSE_LEFT_WRIST,
    POSE_RIGHT_ELBOW,
    POSE_RIGHT_SHOULDER,
    POSE_RIGHT_WRIST,
)
from ofp_vision.signals import (
    Frame,
    HandObservation,
    body_features,
    compute_features,
    face_features,
    hand_features,
    motion_score,
)

FACE_COUNT = 478
POSE_COUNT = 33
HAND_COUNT = 21


def make_face(overrides: dict | None = None) -> list[tuple[float, float, float]]:
    """A synthetic but anatomically plausible face mesh."""
    pts = [(0.0, 0.0, 0.0)] * FACE_COUNT
    base = {
        FACE_EYE_OUTER_R: (0.40, 0.40),
        FACE_EYE_OUTER_L: (0.60, 0.40),
        FACE_NOSE_TIP: (0.50, 0.50),
        FACE_CHIN: (0.50, 0.70),
        FACE_FOREHEAD: (0.50, 0.30),
        FACE_MOUTH_R: (0.45, 0.60),
        FACE_MOUTH_L: (0.55, 0.60),
        FACE_CHEEK_BONE_R: (0.30, 0.50),
        FACE_CHEEK_BONE_L: (0.70, 0.50),
        FACE_LIP_TOP_INNER: (0.50, 0.59),
        FACE_LIP_BOTTOM_INNER: (0.50, 0.61),
    }
    base.update(overrides or {})
    for index, (x, y) in base.items():
        pts[index] = (x, y, 0.0)
    return pts


def make_hand(tip_offset: tuple[float, float] = (0.08, -0.23)) -> list[tuple[float, float, float]]:
    pts = [(0.0, 0.0, 0.0)] * HAND_COUNT
    base = {
        HAND_WRIST: (0.50, 0.50),
        HAND_MIDDLE_MCP: (0.50, 0.40),
    }
    for digit, (mcp, pip, dip, tip) in HAND_DIGITS.items():
        base[mcp] = (0.52, 0.40)
        base[pip] = (0.54, 0.36)
        base[dip] = (0.56, 0.32)
        base[tip] = (0.50 + tip_offset[0], 0.50 + tip_offset[1])
    for index, (x, y) in base.items():
        pts[index] = (x, y, 0.0)
    return pts


def make_pose(overrides: dict | None = None) -> list[tuple[float, float, float]]:
    pts = [(0.0, 0.0, 0.0)] * POSE_COUNT
    base = {
        POSE_LEFT_SHOULDER: (0.40, 0.30),
        POSE_RIGHT_SHOULDER: (0.60, 0.30),
        POSE_LEFT_ELBOW: (0.35, 0.50),
        POSE_RIGHT_ELBOW: (0.65, 0.50),
        POSE_LEFT_WRIST: (0.30, 0.70),
        POSE_RIGHT_WRIST: (0.70, 0.70),
    }
    base.update(overrides or {})
    for index, (x, y) in base.items():
        pts[index] = (x, y, 0.0)
    return pts


# --- scale invariance -------------------------------------------------------


def test_face_features_are_scale_invariant():
    face = make_face()
    small = face_features(face, {})

    doubled = [(x * 2.0, y * 2.0, z) for (x, y, z) in face]
    large = face_features(doubled, {})

    assert set(small) == set(large)
    for key in small:
        assert small[key] == pytest.approx(large[key], rel=1e-9), key


def test_hand_features_are_scale_invariant():
    hand = HandObservation("Right", make_hand())
    small = hand_features(hand)
    doubled = HandObservation("Right", [(x * 3.0, y * 3.0, z) for (x, y, z) in hand.landmarks])
    large = hand_features(doubled)
    for key in small:
        assert small[key] == pytest.approx(large[key], rel=1e-9), key


def test_body_features_are_scale_invariant():
    pose = make_pose()
    small = body_features(pose)
    doubled = [(x * 2.0, y * 2.0, z) for (x, y, z) in pose]
    large = body_features(doubled)
    for key in small:
        assert small[key] == pytest.approx(large[key], rel=1e-9), key


# --- face -------------------------------------------------------------------


def test_symmetric_face_has_zero_asymmetry():
    features = face_features(make_face(), {})
    assert features["face.cheekAsym"] == pytest.approx(0.0, abs=1e-9)
    assert features["face.earAsym"] == pytest.approx(0.0, abs=1e-9)


def test_pulling_one_cheek_outward_raises_ear_and_asymmetry():
    neutral = face_features(make_face(), {})
    pulled = face_features(make_face({FACE_CHEEK_BONE_L: (0.78, 0.50)}), {})

    assert pulled["face.earL"] > neutral["face.earL"]
    assert pulled["face.earAsym"] > 0.0
    # the other side must not move
    assert pulled["face.earR"] == pytest.approx(neutral["face.earR"], rel=1e-9)


def test_cheek_feature_tracks_the_cheekbone():
    neutral = face_features(make_face(), {})
    pulled = face_features(make_face({FACE_CHEEK_BONE_R: (0.22, 0.50)}), {})
    assert pulled["face.cheekR"] > neutral["face.cheekR"]


def test_mouth_open_tracks_lip_separation():
    neutral = face_features(make_face(), {})
    open_mouth = face_features(
        make_face({FACE_LIP_BOTTOM_INNER: (0.50, 0.66)}), {}
    )
    assert open_mouth["face.mouthOpen"] > neutral["face.mouthOpen"]


def test_blendshapes_flow_through():
    features = face_features(make_face(), {"jawOpen": 0.42, "cheekSquintLeft": 0.13})
    assert features["face.jawOpen"] == pytest.approx(0.42)
    assert features["face.cheekSquintL"] == pytest.approx(0.13)


def test_missing_face_yields_no_features():
    assert face_features(None, {}) == {}


# --- hands ------------------------------------------------------------------


def test_finger_reach_and_ext_respond_to_a_pulled_fingertip():
    relaxed = hand_features(HandObservation("Right", make_hand(tip_offset=(0.08, -0.20))))
    pulled = hand_features(HandObservation("Right", make_hand(tip_offset=(0.14, -0.30))))

    assert pulled["hand.Right.index.reach"] > relaxed["hand.Right.index.reach"]
    assert pulled["hand.Right.index.ext"] > relaxed["hand.Right.index.ext"]
    assert pulled["hand.Right.meanExt"] > relaxed["hand.Right.meanExt"]


def test_hand_side_is_part_of_the_channel_name():
    features = hand_features(HandObservation("Left", make_hand()))
    assert "hand.Left.index.reach" in features
    assert "hand.Right.index.reach" not in features


def test_straightness_is_one_for_a_straight_finger():
    pts = [(0.0, 0.0, 0.0)] * HAND_COUNT
    pts[HAND_WRIST] = (0.50, 0.50)
    pts[HAND_MIDDLE_MCP] = (0.50, 0.40)
    mcp, pip, dip, tip = HAND_DIGITS["index"]
    pts[mcp] = (0.50, 0.45)
    pts[pip] = (0.50, 0.35)
    pts[dip] = (0.50, 0.30)
    pts[tip] = (0.50, 0.25)
    features = hand_features(HandObservation("Right", pts))
    assert features["hand.Right.index.straight"] == pytest.approx(1.0, abs=1e-6)


# --- body -------------------------------------------------------------------


def test_arm_reach_responds_to_an_extended_arm():
    neutral = body_features(make_pose())
    extended = body_features(make_pose({POSE_RIGHT_WRIST: (0.72, 0.80)}))
    assert extended["body.armReachR"] > neutral["body.armReachR"]
    assert extended["body.armExtendR"] > neutral["body.armExtendR"]
    # left arm untouched
    assert extended["body.armReachL"] == pytest.approx(neutral["body.armReachL"], rel=1e-9)


# --- top level --------------------------------------------------------------


def test_compute_features_omits_absent_sources():
    frame = Frame(
        t_ms=0.0,
        face=make_face(),
        hands=[HandObservation("Right", make_hand())],
        pose=None,
    )
    features = compute_features(frame)
    assert "face.cheekR" in features
    assert "hand.Right.index.reach" in features
    assert not any(k.startswith("body.") for k in features)


def test_compute_features_merges_both_hands():
    frame = Frame(
        t_ms=0.0,
        hands=[
            HandObservation("Left", make_hand()),
            HandObservation("Right", make_hand()),
        ],
    )
    features = compute_features(frame)
    assert "hand.Left.index.reach" in features
    assert "hand.Right.index.reach" in features


def test_features_are_finite_for_degenerate_geometry():
    collapsed = [(0.5, 0.5, 0.0)] * FACE_COUNT
    features = face_features(collapsed, {})
    for key, value in features.items():
        assert math.isfinite(value), key


# --- motion gating ----------------------------------------------------------


def test_motion_score_is_zero_for_a_still_scene():
    a = Frame(t_ms=0.0, face=make_face(), hands=[HandObservation("Right", make_hand())], pose=make_pose())
    b = Frame(t_ms=33.0, face=make_face(), hands=[HandObservation("Right", make_hand())], pose=make_pose())
    assert motion_score(a, b) == pytest.approx(0.0, abs=1e-12)


def test_motion_score_reports_real_movement():
    a = Frame(t_ms=0.0, face=make_face(), hands=[HandObservation("Right", make_hand())], pose=make_pose())
    moved_face = make_face()
    moved_face[FACE_NOSE_TIP] = (0.50 + 0.05, 0.50, 0.0)
    b = Frame(t_ms=33.0, face=moved_face, hands=[HandObservation("Right", make_hand())], pose=make_pose())
    assert motion_score(a, b) > 0.01


def test_motion_score_handles_a_missing_previous_frame():
    frame = Frame(t_ms=0.0, face=make_face())
    assert motion_score(None, frame) == 1.0


def test_motion_score_is_zero_when_nothing_is_detected():
    assert motion_score(Frame(t_ms=0.0), Frame(t_ms=33.0)) == 0.0
