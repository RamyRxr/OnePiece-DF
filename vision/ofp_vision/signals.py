"""Pure feature extraction: landmarks -> scale-invariant geometric features.

This module imports nothing but the standard library and `indices`. That is
deliberate: the stretch math is the risky part of this project, so it must be
testable against synthetic fixtures and replayable offline without a camera.

The central idea
----------------
A webcam gives you X and Y, but a "pull" is mostly a Z movement: grab your
cheek and tug sideways, and your hand comes toward the camera while the 2D
landmarks barely move. No amount of extra resolution fixes that.

The fix is to never measure an absolute distance. Measure a *ratio* against
something rigid that moves with you:

    feature = |a - b| / scale_reference

`scale_reference` is a rigid distance on the same body part (interocular
distance, palm length, shoulder span). Dividing by it cancels:
  - the player leaning toward or away from the camera
  - the player turning their head
  - the player moving their hand closer to the lens

What survives is the thing we actually care about: how much the tissue
between two landmarks has stretched relative to the body it sits on.

Two more normalisations make it work:
  - **Baseline**: every feature is measured relative to a calibration snapshot
    of the player's neutral pose. No two faces pull the same way.
  - **Asymmetry**: for bilateral features (cheek, ear) we also compute
    left/right - 1. A one-sided pull is literally a bilateral asymmetry, so
    this cancels scale effects in a single step.

Note on Z: MediaPipe does emit a z for each landmark, but it is in a
different, much weaker coordinate space than x/y and is not metric. Every
distance here is 2D on purpose - 2D is stable, z is a source of jitter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .indices import (
    BLENDSHAPE_CHEEK_SQUINT_L,
    BLENDSHAPE_CHEEK_SQUINT_R,
    BLENDSHAPE_JAW_OPEN,
    BLENDSHAPE_MOUTH_FUNNEL,
    BLENDSHAPE_MOUTH_PUCKER,
    BLENDSHAPE_SMILE_L,
    BLENDSHAPE_SMILE_R,
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
    POSE_LEFT_EAR,
    POSE_LEFT_SHOULDER,
    POSE_LEFT_WRIST,
    POSE_NOSE,
    POSE_RIGHT_ELBOW,
    POSE_RIGHT_EAR,
    POSE_RIGHT_SHOULDER,
    POSE_RIGHT_WRIST,
)

Point = tuple[float, float, float]


# --- Frame containers -------------------------------------------------------


@dataclass
class HandObservation:
    """One detected hand."""

    handedness: str  # "Left" or "Right", already corrected for mirroring
    landmarks: list[Point]

    @property
    def wrist(self) -> Point:
        return self.landmarks[HAND_WRIST]


@dataclass
class Frame:
    """Everything we know about one camera frame."""

    t_ms: float
    face: list[Point] | None = None
    blendshapes: dict[str, float] = field(default_factory=dict)
    hands: list[HandObservation] = field(default_factory=list)
    pose: list[Point] | None = None
    # The latest skin segmentation, if the model is present: a uint8 array of
    # labels (see skin.SEGMENT_*) at a fraction of the frame's resolution.
    # Kept untyped here so this module stays free of numpy.
    segmentation: object = None


# --- Geometry helpers -------------------------------------------------------


def _dist(a: Point, b: Point) -> float:
    """2D distance. x/y only - see module docstring on why z is excluded."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _vec(a: Point, b: Point) -> tuple[float, float]:
    return (b[0] - a[0], b[1] - a[1])


def _cos(a: tuple[float, float], b: tuple[float, float]) -> float:
    na = math.hypot(*a)
    nb = math.hypot(*b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return (a[0] * b[0] + a[1] * b[1]) / (na * nb)


def _safe_ratio(num: float, den: float) -> float:
    if abs(den) < 1e-9:
        return 1.0
    return num / den


# --- Scale references -------------------------------------------------------


def face_scale(face: list[Point] | None) -> float | None:
    """Interocular distance: the canonical scale for face features."""
    if face is None or len(face) <= FACE_EYE_OUTER_L:
        return None
    d = _dist(face[FACE_EYE_OUTER_R], face[FACE_EYE_OUTER_L])
    return d if d > 1e-6 else None


def face_height(face: list[Point] | None) -> float | None:
    """Chin-to-forehead distance, used for vertical face features."""
    if face is None or len(face) <= FACE_FOREHEAD:
        return None
    d = _dist(face[FACE_CHIN], face[FACE_FOREHEAD])
    return d if d > 1e-6 else None


def pose_scale(pose: list[Point] | None) -> float | None:
    """Shoulder span: the canonical scale for body features."""
    if pose is None or len(pose) <= POSE_RIGHT_SHOULDER:
        return None
    d = _dist(pose[POSE_LEFT_SHOULDER], pose[POSE_RIGHT_SHOULDER])
    return d if d > 1e-6 else None


def hand_scale(landmarks: list[Point]) -> float | None:
    """Palm length (wrist -> middle MCP).

    Preferred over the cross-wrist distance because it does not depend on the
    other hand being in frame, and because it is rigid.
    """
    if len(landmarks) <= HAND_MIDDLE_MCP:
        return None
    d = _dist(landmarks[HAND_WRIST], landmarks[HAND_MIDDLE_MCP])
    return d if d > 1e-6 else None


# --- Feature extraction -----------------------------------------------------


def face_features(face: list[Point] | None, blendshapes: dict[str, float]) -> dict[str, float]:
    """Geometric + blendshape features for the face."""
    out: dict[str, float] = {}
    s = face_scale(face)
    if s is None or face is None:
        return out

    h = face_height(face) or s

    # Horizontal extent of the cheek region: mouth corner to the widest
    # silhouette point on the same side.
    out["face.cheekR"] = _dist(face[FACE_MOUTH_R], face[FACE_CHEEK_BONE_R]) / s
    out["face.cheekL"] = _dist(face[FACE_MOUTH_L], face[FACE_CHEEK_BONE_L]) / s

    # "Ear" proxy. MediaPipe has no ear landmarks, so we use the cheekbone
    # silhouette point, which is where the ear physically attaches. Pulling
    # the ear pushes this point outward (and usually upward).
    out["face.earR"] = _dist(face[FACE_NOSE_TIP], face[FACE_CHEEK_BONE_R]) / s
    out["face.earL"] = _dist(face[FACE_NOSE_TIP], face[FACE_CHEEK_BONE_L]) / s

    # Vertical position of that point (smaller = pulled upward).
    out["face.earVertR"] = _dist(face[FACE_CHIN], face[FACE_CHEEK_BONE_R]) / h
    out["face.earVertL"] = _dist(face[FACE_CHIN], face[FACE_CHEEK_BONE_L]) / h

    # Mouth aperture.
    out["face.mouthOpen"] = _dist(face[FACE_LIP_TOP_INNER], face[FACE_LIP_BOTTOM_INNER]) / s

    # Bilateral asymmetry. A one-sided pull is a left/right imbalance, and
    # measuring the imbalance cancels scale effects completely.
    out["face.cheekAsym"] = _safe_ratio(out["face.cheekL"], out["face.cheekR"]) - 1.0
    out["face.earAsym"] = _safe_ratio(out["face.earL"], out["face.earR"]) - 1.0

    # --- tragion geometry ---
    #
    # These four are the raw material for the "pull" composite channels
    # (see config.COMPOSITE_CHANNELS). They are deliberately *not* finished
    # channels on their own: a single distance can only see the component of
    # the motion perpendicular to the line it measures, so each one alone is
    # blind to a whole family of pull directions.
    #
    # The chin and the forehead are used because they are bone. They do not
    # move when the player tugs their cheek or ear, so any change in these
    # distances is the tug itself.
    for side, bone in (("R", FACE_CHEEK_BONE_R), ("L", FACE_CHEEK_BONE_L)):
        out[f"face.tragChin{side}"] = _dist(face[bone], face[FACE_CHIN]) / s
        out[f"face.tragForehead{side}"] = _dist(face[bone], face[FACE_FOREHEAD]) / s

    # Blendshapes are already normalised 0..1 and cost us nothing extra.
    for key, name in (
        (BLENDSHAPE_CHEEK_SQUINT_R, "cheekSquintR"),
        (BLENDSHAPE_CHEEK_SQUINT_L, "cheekSquintL"),
        (BLENDSHAPE_SMILE_R, "smileR"),
        (BLENDSHAPE_SMILE_L, "smileL"),
        (BLENDSHAPE_JAW_OPEN, "jawOpen"),
        (BLENDSHAPE_MOUTH_PUCKER, "mouthPucker"),
        (BLENDSHAPE_MOUTH_FUNNEL, "mouthFunnel"),
    ):
        out[f"face.{name}"] = float(blendshapes.get(key, 0.0))

    return out


def hand_features(hand: HandObservation) -> dict[str, float]:
    """Geometric features for one hand."""
    out: dict[str, float] = {}
    lm = hand.landmarks
    s = hand_scale(lm)
    if s is None:
        return out

    side = hand.handedness
    exts: list[float] = []

    for digit, (mcp, pip, dip, tip) in HAND_DIGITS.items():
        # How far the fingertip is from the wrist, relative to palm length.
        # Responds to both straightening the finger and pulling it sideways.
        out[f"hand.{side}.{digit}.reach"] = _dist(lm[tip], lm[HAND_WRIST]) / s
        # Fingertip to its own knuckle: the cleanest "how stretched is this
        # finger" measure, since the knuckle moves with the hand.
        out[f"hand.{side}.{digit}.ext"] = _dist(lm[tip], lm[mcp]) / s
        # Straightness: 1.0 = perfectly straight finger, lower = curled.
        out[f"hand.{side}.{digit}.straight"] = _straightness(lm, mcp, pip, dip, tip)
        exts.append(out[f"hand.{side}.{digit}.ext"])

    out[f"hand.{side}.meanExt"] = sum(exts) / len(exts)
    out[f"hand.{side}.spread"] = (
        _dist(lm[HAND_DIGITS["index"][3]], lm[HAND_DIGITS["pinky"][3]]) / s
    )
    return out


def _straightness(lm: list[Point], mcp: int, pip: int, dip: int, tip: int) -> float:
    """Mean cosine between consecutive finger segments. 1.0 = straight."""
    v1 = _vec(lm[mcp], lm[pip])
    v2 = _vec(lm[pip], lm[dip])
    v3 = _vec(lm[dip], lm[tip])
    return (_cos(v1, v2) + _cos(v2, v3)) / 2.0


def body_features(pose: list[Point] | None) -> dict[str, float]:
    """Geometric features for arms and shoulders."""
    out: dict[str, float] = {}
    s = pose_scale(pose)
    if s is None or pose is None:
        return out

    out["body.armExtendR"] = _dist(pose[POSE_RIGHT_WRIST], pose[POSE_RIGHT_ELBOW]) / s
    out["body.armExtendL"] = _dist(pose[POSE_LEFT_WRIST], pose[POSE_LEFT_ELBOW]) / s
    out["body.armReachR"] = _dist(pose[POSE_RIGHT_WRIST], pose[POSE_RIGHT_SHOULDER]) / s
    out["body.armReachL"] = _dist(pose[POSE_LEFT_WRIST], pose[POSE_LEFT_SHOULDER]) / s
    out["body.shoulderRaiseR"] = _dist(pose[POSE_RIGHT_SHOULDER], pose[POSE_RIGHT_EAR]) / s
    out["body.shoulderRaiseL"] = _dist(pose[POSE_LEFT_SHOULDER], pose[POSE_LEFT_EAR]) / s
    return out


# --- Top level --------------------------------------------------------------


def compute_features(frame: Frame) -> dict[str, float]:
    """All features for a frame.

    A source that is not visible contributes no features at all, which is how
    downstream code knows to decay a channel rather than hold a stale value.
    """
    out: dict[str, float] = {}
    out.update(face_features(frame.face, frame.blendshapes))
    for hand in frame.hands:
        out.update(hand_features(hand))
    out.update(body_features(frame.pose))
    return out


def apply_baseline(
    features: dict[str, float], baseline: dict[str, float]
) -> dict[str, float]:
    """Turn absolute features into deviations from the calibration baseline."""
    return {
        name: value - baseline.get(name, value) if name in baseline else 0.0
        for name, value in features.items()
    }


# --- Motion -----------------------------------------------------------------


def motion_score(previous: Frame | None, current: Frame) -> float:
    """How much the landmarks moved since the previous frame.

    Returns the largest per-body-part displacement, in normalised image units
    (0..1 of frame width).

    Scored per group (face / each hand / pose) and then maximised, rather than
    averaged over every landmark. Averaging dilutes exactly the case we need
    to catch - a single finger moving 5% of the frame while the rest of the
    body holds still - down below the noise floor. Within a group we take the
    75th percentile so one twitchy landmark cannot spike the reading.

    This is what lets calibration reject samples taken while the player is
    still leaning in and settling. A stationary subject sits well under 0.004;
    deliberate movement is an order of magnitude above that.
    """
    if previous is None:
        return 1.0

    group_scores: list[float] = []

    if previous.face and current.face:
        if len(previous.face) > FACE_CHIN and len(current.face) > FACE_CHIN:
            group_scores.append(
                _percentile(
                    [
                        _dist(previous.face[i], current.face[i])
                        for i in (FACE_NOSE_TIP, FACE_EYE_OUTER_R, FACE_EYE_OUTER_L, FACE_CHIN)
                    ],
                    0.75,
                )
            )

    for cur_hand in current.hands:
        prev_hand = next(
            (h for h in previous.hands if h.handedness == cur_hand.handedness), None
        )
        if prev_hand is None:
            continue
        indices = (HAND_WRIST, HAND_DIGITS["index"][3], HAND_DIGITS["middle"][3])
        group_scores.append(
            _percentile(
                [_dist(prev_hand.landmarks[i], cur_hand.landmarks[i]) for i in indices],
                0.75,
            )
        )

    if previous.pose and current.pose:
        if len(previous.pose) > POSE_RIGHT_WRIST and len(current.pose) > POSE_RIGHT_WRIST:
            group_scores.append(
                _percentile(
                    [
                        _dist(previous.pose[i], current.pose[i])
                        for i in (
                            POSE_NOSE,
                            POSE_LEFT_SHOULDER,
                            POSE_RIGHT_SHOULDER,
                            POSE_LEFT_WRIST,
                            POSE_RIGHT_WRIST,
                        )
                    ],
                    0.75,
                )
            )

    if not group_scores:
        return 0.0

    return max(group_scores)


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of a list of floats."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight
