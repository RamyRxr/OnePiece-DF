"""Synthetic frame generator.

Lets the whole pipeline - capture -> infer -> features -> channels -> socket
- run without a camera. Useful for CI, for verifying the wire protocol, and
for testing the Unity client before any avatar work exists.

It also doubles as a reference for what a "pull" looks like in landmark space.
"""

from __future__ import annotations

import math

from .indices import (
    FACE_CHEEK_BONE_L,
    FACE_CHEEK_BONE_R,
    FACE_CHEEK_L,
    FACE_CHEEK_R,
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
from .signals import Frame, HandObservation

FACE_COUNT = 478
POSE_COUNT = 33
HAND_COUNT = 21


# The pinch-grab cycle, as phases of the period: the left hand arrives open
# at the left cheek, pinches, pulls outward, and lets go.
GRAB_ARRIVE = 0.30
GRAB_PINCH = 0.35
GRAB_RELEASE = 0.70
GRAB_LEAVE = 0.80
GRAB_PULL = 0.12  # in normalised image x, outward (subject's left = image +x)
CHEEK_L_POINT = (0.58, 0.56)


class SyntheticSource:
    """Emulates a player who periodically pulls their finger and their cheek,
    and pinch-grabs their left cheek and lets it go."""

    def __init__(self, period: float = 4.0) -> None:
        self.period = period
        self.t = 0.0

    def next_frame(self, dt: float) -> Frame:
        self.t += dt
        phase = (self.t % self.period) / self.period

        # 0.00-0.45 rest, 0.45-0.70 pulling, 0.70-0.80 release, 0.80-1.00 rest
        if 0.45 <= phase < 0.70:
            pull = (phase - 0.45) / 0.25
        else:
            pull = 0.0

        frame = Frame(t_ms=self.t * 1000.0)
        frame.face = self._face(pull)
        frame.blendshapes = {"jawOpen": 0.02, "cheekSquintLeft": 0.05 * pull}
        frame.hands = [HandObservation("Right", self._hand(pull))]
        if GRAB_ARRIVE <= phase < GRAB_LEAVE:
            frame.hands.append(HandObservation("Left", self._grab_hand(phase)))
        frame.pose = self._pose()
        return frame

    # --- pieces -------------------------------------------------------------

    def _face(self, pull: float) -> list[tuple[float, float, float]]:
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
            # The pull drags the left cheek silhouette point outward.
            FACE_CHEEK_BONE_L: (0.70 + 0.06 * pull, 0.50),
            FACE_CHEEK_R: (0.42, 0.56),
            FACE_CHEEK_L: CHEEK_L_POINT,
            FACE_LIP_TOP_INNER: (0.50, 0.59),
            FACE_LIP_BOTTOM_INNER: (0.50, 0.61),
        }
        for index, (x, y) in base.items():
            pts[index] = (x, y, 0.0)
        return pts

    def _hand(self, pull: float) -> list[tuple[float, float, float]]:
        pts = [(0.0, 0.0, 0.0)] * HAND_COUNT
        pts[HAND_WRIST] = (0.50, 0.50)
        pts[HAND_MIDDLE_MCP] = (0.50, 0.40)
        for mcp, pip, dip, tip in HAND_DIGITS.values():
            pts[mcp] = (0.52, 0.40)
            pts[pip] = (0.54, 0.36)
            pts[dip] = (0.56, 0.32)
            pts[tip] = (0.58, 0.27)
        # The index finger gets yanked outward during the pull.
        mcp, pip, dip, tip = HAND_DIGITS["index"]
        offset = 0.06 * pull
        pts[tip] = (0.58 + offset, 0.27 - 0.02 * pull)
        return pts

    def _grab_hand(self, phase: float) -> list[tuple[float, float, float]]:
        """Thumb and index tips straddle the pinch point; open = 0.6 palms apart."""
        pinched = GRAB_PINCH <= phase < GRAB_RELEASE
        progress = 0.0
        if phase >= GRAB_PINCH:
            span = GRAB_RELEASE - GRAB_PINCH
            progress = min(1.0, (phase - GRAB_PINCH) / span)
        x = CHEEK_L_POINT[0] + GRAB_PULL * progress
        y = CHEEK_L_POINT[1]
        palm = 0.10
        gap = (0.05 if pinched else 0.6) * palm
        pts = [(x, y + 0.10, 0.0)] * HAND_COUNT
        pts[HAND_WRIST] = (x, y + 0.15, 0.0)
        pts[HAND_MIDDLE_MCP] = (x, y + 0.05, 0.0)
        pts[HAND_DIGITS["thumb"][3]] = (x, y - gap / 2, 0.0)
        pts[HAND_DIGITS["index"][3]] = (x, y + gap / 2, 0.0)
        return pts

    def _pose(self) -> list[tuple[float, float, float]]:
        pts = [(0.0, 0.0, 0.0)] * POSE_COUNT
        base = {
            POSE_LEFT_SHOULDER: (0.40, 0.30),
            POSE_RIGHT_SHOULDER: (0.60, 0.30),
            POSE_LEFT_ELBOW: (0.35, 0.50),
            POSE_RIGHT_ELBOW: (0.65, 0.50),
            POSE_LEFT_WRIST: (0.30, 0.70),
            POSE_RIGHT_WRIST: (0.70, 0.70),
        }
        for index, (x, y) in base.items():
            pts[index] = (x, y, 0.0)
        return pts


def synthetic_pull_signal(t_seconds: float, period: float = 4.0) -> float:
    """The pull envelope the generator uses - handy for tests."""
    phase = (t_seconds % period) / period
    if 0.45 <= phase < 0.70:
        return (phase - 0.45) / 0.25
    return 0.0
