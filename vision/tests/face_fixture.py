"""A synthetic 1280x720 camera frame with a face in it, and its landmarks.

Built for the stretch effect's tests, so every region is recognisable:

- a round, skin-coloured face with a little skin texture;
- white teeth in an open mouth (the inner-lip outline around them);
- dark eyes;
- a background whose colour changes with every pixel (a blue-green
  gradient, nothing like skin), so any background pixel that moves shows up.

Positions are given in units of the frame's height, relative to the centre
of the face, and converted to MediaPipe's normalised coordinates.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ofp_vision.indices import (
    FACE_CHEEK_BONE_L,
    FACE_CHEEK_BONE_R,
    FACE_CHEEK_L,
    FACE_CHEEK_R,
    FACE_CHIN,
    FACE_CHIN_CENTRE,
    FACE_EAR_LEVEL_L,
    FACE_EAR_LEVEL_R,
    FACE_EYE_OUTER_L,
    FACE_EYE_OUTER_R,
    FACE_EYE_POINTS,
    FACE_FOREHEAD_CENTRE,
    FACE_INNER_LIPS,
    FACE_JAW_L,
    FACE_JAW_R,
    FACE_LIP_BOTTOM_OUTER,
    FACE_LIP_TOP_OUTER,
    FACE_MOUTH_L,
    FACE_MOUTH_R,
    FACE_NOSE_TIP,
    FACE_OVAL,
)

W, H = 1280, 720
CENTRE = (0.5, 0.5)
RADIUS = 0.2
SKIN = (120, 160, 215)
TEETH = (235, 240, 240)

EYE_R = (-0.07, -0.05)
EYE_L = (0.07, -0.05)
MOUTH = (0.0, 0.09)
MOUTH_HALF = (0.05, 0.018)


def norm(dx: float, dy: float, scale: float = 1.0, centre=CENTRE) -> tuple[float, float, float]:
    """A point `dx`, `dy` frame heights from the face centre, normalised.

    `scale` grows the face about its centre (2.0 = a face twice as close to
    the camera); `centre` moves it (y < 0.2 pushes the forehead off-frame).
    """
    return centre[0] + dx * scale * H / W, centre[1] + dy * scale, 0.0


def px(point) -> tuple[int, int]:
    return int(point[0] * W), int(point[1] * H)


def landmarks(scale: float = 1.0, centre=CENTRE) -> list[tuple[float, float, float]]:
    def at(dx: float, dy: float) -> tuple[float, float, float]:
        return norm(dx, dy, scale, centre)

    points = [at(0.0, 0.0)] * 478
    points = list(points)
    for i, index in enumerate(FACE_OVAL):
        angle = 2 * math.pi * i / len(FACE_OVAL)
        points[index] = at(RADIUS * math.sin(angle), -RADIUS * math.cos(angle))
    eyes = [EYE_R, EYE_L]
    for i, index in enumerate(FACE_EYE_POINTS):
        cx, cy = eyes[i // 4]
        offset = ((-0.02, 0.0), (0.02, 0.0), (0.0, -0.008), (0.0, 0.008))[i % 4]
        points[index] = at(cx + offset[0], cy + offset[1])
    for i, index in enumerate(FACE_INNER_LIPS):
        angle = 2 * math.pi * i / len(FACE_INNER_LIPS)
        points[index] = at(MOUTH[0] - MOUTH_HALF[0] * math.cos(angle), MOUTH[1] - MOUTH_HALF[1] * math.sin(angle))
    layout = {
        FACE_EYE_OUTER_R: (EYE_R[0] - 0.02, EYE_R[1]),
        FACE_EYE_OUTER_L: (EYE_L[0] + 0.02, EYE_L[1]),
        FACE_NOSE_TIP: (0.0, 0.02),
        FACE_CHEEK_R: (-0.1, 0.04),
        FACE_CHEEK_L: (0.1, 0.04),
        FACE_CHEEK_BONE_R: (-RADIUS, -0.02),
        FACE_CHEEK_BONE_L: (RADIUS, -0.02),
        FACE_EAR_LEVEL_R: (-RADIUS, 0.0),
        FACE_EAR_LEVEL_L: (RADIUS, 0.0),
        FACE_FOREHEAD_CENTRE: (0.0, -0.13),
        FACE_CHIN_CENTRE: (0.0, 0.15),
        FACE_CHIN: (0.0, RADIUS),
        FACE_JAW_R: (-0.14, 0.13),
        FACE_JAW_L: (0.14, 0.13),
        FACE_MOUTH_R: (-0.06, 0.09),
        FACE_MOUTH_L: (0.06, 0.09),
        FACE_LIP_TOP_OUTER: (0.0, 0.065),
        FACE_LIP_BOTTOM_OUTER: (0.0, 0.118),
    }
    for index, (dx, dy) in layout.items():
        points[index] = at(dx, dy)
    return points


def image(scale: float = 1.0, centre=CENTRE) -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W]
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[..., 0] = 60 + (xs % 180)
    frame[..., 1] = 60 + (ys % 180)
    frame[..., 2] = 50
    face = landmarks(scale, centre)
    middle = px(norm(0.0, 0.0, scale, centre))
    radius = int(RADIUS * scale * H)
    texture = np.random.default_rng(7).integers(-14, 15, (H, W, 1))
    skin = np.clip(np.array(SKIN) + texture, 0, 255).astype(np.uint8)
    inside = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(inside, middle, radius, 255, -1)
    frame[inside > 0] = skin[inside > 0]
    eye_axes = (int(0.02 * scale * H), int(0.008 * scale * H))
    for eye in (EYE_R, EYE_L):
        cv2.ellipse(frame, px(norm(*eye, scale, centre)), eye_axes, 0, 0, 360, (40, 30, 30), -1)
    mouth = np.array([px(face[i]) for i in FACE_INNER_LIPS], dtype=np.int32)
    teeth = np.clip(np.array(TEETH) + texture // 2, 0, 255).astype(np.uint8)
    opening = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(opening, [mouth], 255)
    frame[opening > 0] = teeth[opening > 0]
    return frame
