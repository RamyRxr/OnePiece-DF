"""Landmark index constants.

Kept in their own dependency-free module so `signals.py` stays pure and can be
unit-tested against synthetic landmark fixtures.

Index conventions
-----------------
MediaPipe's face model uses the subject's own left/right, but the raw pixel
image is NOT mirrored. So "Right" in these indices means the subject's right,
which appears on the LEFT of an unmirrored webcam image.

Handedness from the hand model is reported against the *image*, so it is the
opposite: an unmirrored image of a subject's right hand is labelled "Right"
only because MediaPipe assumes a selfie/mirrored view. `config.mirror_handedness`
exists to correct for this; see `signals.py` for where it is applied.
"""

from __future__ import annotations

# --- Face (MediaPipe Face Landmarker: 478 points = 468 mesh + 10 iris) -------

FACE_NOSE_TIP = 1
FACE_CHIN = 152
FACE_FOREHEAD = 10

# Outer eye corners. The gap between them is our canonical face scale.
FACE_EYE_OUTER_R = 33
FACE_EYE_OUTER_L = 263
FACE_EYE_INNER_R = 133
FACE_EYE_INNER_L = 362

# Iris centres (present in the 478-point model).
FACE_IRIS_R = 468
FACE_IRIS_L = 473

# Widest silhouette points of the face, roughly at the cheekbone / tragion.
# These are the closest thing MediaPipe has to an ear.
FACE_CHEEK_BONE_R = 234
FACE_CHEEK_BONE_L = 454

# Grab targets (see grab.py). As everywhere in this file, R = the subject's
# right. The mesh pairs its sides: a right-side index has a left-side twin.
FACE_CHEEK_R = 205
FACE_CHEEK_L = 425
# Lower jaw, between the ear and the chin.
FACE_JAW_R = 172
FACE_JAW_L = 397

# The face outline, forehead -> subject's left -> chin -> subject's right.
# Its hull is the skin the lab's warp is allowed to move.
FACE_OVAL: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
)

# The eye outline: corners and lid midpoints. A pinch never moves these.
FACE_EYE_POINTS: tuple[int, ...] = (33, 133, 159, 145, 263, 362, 386, 374)

# Midline points used to judge the lighting: forehead and chin centre.
FACE_FOREHEAD_CENTRE = 151
FACE_CHIN_CENTRE = 200

# Silhouette points level with the ears' middle (just below the cheekbone
# points). The ears sit just outside these.
FACE_EAR_LEVEL_R = 93
FACE_EAR_LEVEL_L = 323

# The inner edge of the lips, all the way round: the mouth opening. What is
# inside (teeth, tongue) is not skin and does not stretch.
FACE_INNER_LIPS: tuple[int, ...] = (
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95,
)

# The two halves of that edge (between the corners 78 and 308). The lips
# are joined only at the corners, so a pull on one never drags the other.
FACE_INNER_LIP_UPPER: tuple[int, ...] = (191, 80, 81, 82, 13, 312, 311, 310, 415)
FACE_INNER_LIP_LOWER: tuple[int, ...] = (324, 318, 402, 317, 14, 87, 178, 88, 95)

# Mouth corners.
FACE_MOUTH_R = 61
FACE_MOUTH_L = 291

# Inner / outer lips (vertical extent of the mouth opening).
FACE_LIP_TOP_INNER = 13
FACE_LIP_BOTTOM_INNER = 14
FACE_LIP_TOP_OUTER = 0
FACE_LIP_BOTTOM_OUTER = 17

# --- Hands (MediaPipe Hand Landmarker: 21 points) ---------------------------

HAND_WRIST = 0

# (mcp, pip, dip, tip) per digit, in landmark index order.
HAND_DIGITS: dict[str, tuple[int, int, int, int]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

HAND_DIGIT_ORDER = ("thumb", "index", "middle", "ring", "pinky")

# The part of each digit that is a finger - that bends, is grabbed and
# stretches - knuckle to tip. The thumb's first point is not its knuckle: it
# is deep in the palm, by the wrist.
HAND_FINGER_JOINTS: dict[str, tuple[int, ...]] = {
    digit: joints[1:] if digit == "thumb" else joints for digit, joints in HAND_DIGITS.items()
}

HAND_MIDDLE_MCP = HAND_DIGITS["middle"][0]

# The hand skeleton: bones as landmark pairs.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
# The palm outline: wrist, thumb base, and the four knuckles.
HAND_PALM: tuple[int, ...] = (0, 1, 2, 5, 9, 13, 17)

# --- Pose (MediaPipe Pose Landmarker: 33 points) ----------------------------

POSE_NOSE = 0
POSE_LEFT_EYE_INNER = 1
POSE_RIGHT_EYE_INNER = 2
POSE_LEFT_EAR = 7
POSE_RIGHT_EAR = 8
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16

# --- Blendshape names (Face Landmarker `output_face_blendshapes`) -----------

BLENDSHAPE_CHEEK_SQUINT_L = "cheekSquintLeft"
BLENDSHAPE_CHEEK_SQUINT_R = "cheekSquintRight"
BLENDSHAPE_SMILE_L = "mouthSmileLeft"
BLENDSHAPE_SMILE_R = "mouthSmileRight"
BLENDSHAPE_JAW_OPEN = "jawOpen"
BLENDSHAPE_MOUTH_PUCKER = "mouthPucker"
BLENDSHAPE_MOUTH_FUNNEL = "mouthFunnel"
BLENDSHAPE_BROW_INNER_UP = "browInnerUp"
