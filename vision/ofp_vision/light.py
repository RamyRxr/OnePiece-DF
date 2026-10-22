"""The room's light, judged from the face - for lighting stretched skin.

Both stretch effects light the pulled skin the way the room lights the face:
a face lit from its right has a brighter right cheek, and so does a strand
of skin pulled out of it.
"""

from __future__ import annotations

import numpy as np

from .indices import FACE_CHEEK_L, FACE_CHEEK_R, FACE_CHIN_CENTRE, FACE_FOREHEAD_CENTRE
from .signals import Point
from .skin import LUMA, interocular, patch_around

# Light comes from above by default; the face's own shading tilts it.
DEFAULT_LIGHT = (0.0, -0.35, 1.0)
LIGHT_SENSITIVITY = 2.5
# How far each frame's measurement moves the light (it changes slowly).
LIGHT_SMOOTHING = 0.15


def default_light() -> np.ndarray:
    """The light before anything is measured, as a unit vector."""
    return np.array(DEFAULT_LIGHT) / np.linalg.norm(DEFAULT_LIGHT)


def estimate_light(bgr: np.ndarray, face: list[Point]) -> np.ndarray | None:
    """Which way the light comes from, judged from the face's own shading.

    Returns a unit vector (x right, y down, z toward the camera) pointing at
    the light - or None if any of the spots it reads is off-frame. A brighter
    image-right cheek means light from the right; a brighter forehead than
    chin means light from above.
    """
    size = (bgr.shape[1], bgr.shape[0])
    radius = max(2, int(0.1 * interocular(face, size)))
    spots = (FACE_CHEEK_R, FACE_CHEEK_L, FACE_FOREHEAD_CENTRE, FACE_CHIN_CENTRE)
    patches = [patch_around(bgr, int(face[i][0] * size[0]), int(face[i][1] * size[1]), radius) for i in spots]
    if any(patch.size == 0 for patch in patches):
        return None
    left, right, top, bottom = (float(patch.reshape(-1, 3).mean(axis=0) @ LUMA) for patch in patches)
    lx = np.clip(LIGHT_SENSITIVITY * (right - left) / (right + left + 1.0), -0.85, 0.85)
    ly = np.clip(LIGHT_SENSITIVITY * (bottom - top) / (bottom + top + 1.0), -0.85, 0.85)
    light = np.array([lx + DEFAULT_LIGHT[0], ly + DEFAULT_LIGHT[1], DEFAULT_LIGHT[2]])
    return light / np.linalg.norm(light)


def followed_light(light: np.ndarray, bgr: np.ndarray, face: list[Point] | None) -> np.ndarray:
    """`light`, moved a step toward what the face shows now. Only call it
    between pinches: while one is held, the hand over the cheek would read
    as a shadow."""
    measured = estimate_light(bgr, face) if face is not None else None
    if measured is None:
        return light
    blended = (1.0 - LIGHT_SMOOTHING) * light + LIGHT_SMOOTHING * measured
    return blended / np.linalg.norm(blended)
