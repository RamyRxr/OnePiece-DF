"""The Signal Lab window.

This is the tuning surface. The camera fills most of the screen, with the
grabbable parts ringed on the player's face (any finger of the other hand is
grabbable too, anywhere along it). Underneath, a
panel holds the status and a live bar for every gameplay channel, so you can
see at a glance whether a pull moves the right number and by how much. If a
bar is flat while you pull, that channel is dead weight and gets cut.

    +------------------------------------------+
    |                                          |
    |            camera (scaled up)            |
    |                                          |
    +------------+-----------------------------+
    | status     | channel bars, 4 columns     |
    +------------+-----------------------------+
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .channels import ChannelState
from .config import GAMEPLAY_CHANNELS, GRAB_HANDS
from .grab import TARGET_CODES, GrabState, face_target_points
from .indices import HAND_CONNECTIONS
from .signals import Frame

# Landmark indices in the 478-point face mesh that form the brows, nose and
# lips. Drawing the full mesh is visual noise.
FACE_CONTOURS = [
    (70, 63, 105, 66, 107),
    (336, 296, 334, 293, 300),
    (168, 6, 197, 195, 5, 4),
    (98, 97, 2, 326, 327, 4),
    (0, 37, 39, 40, 185),
    (267, 269, 270, 409, 291),
]

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]

# Displayed range of a channel's bar. Values beyond it are clipped in the
# overlay only - the wire protocol carries the raw value.
BAR_RANGE = 0.25
GRAB_BAR_RANGE = {"stretch": 2.5, "stretchX": 2.5, "stretchY": 2.5, "dirX": 1.0, "dirY": 1.0, "active": 1.0}
# Each hand's pull vector and direction are left off the panel (no room for
# both hands'); the stretch and the target say what is happening.
PANEL_HIDDEN = frozenset(
    f"grab.{hand}.{field}" for hand in GRAB_HANDS for field in ("stretchX", "stretchY", "dirX", "dirY")
)

PANEL_HEIGHT = 190
STATUS_WIDTH = 380
CHANNEL_COLUMNS = 4
ROW_HEIGHT = 22
MARGIN = 14

# How long the RELEASED banner stays up. One frame (33ms) is unreadable.
RELEASE_BANNER_FRAMES = 20

PANEL_BG = (24, 24, 24)
TEXT = (235, 235, 235)
DIM = (130, 130, 130)
ACTIVE = (80, 220, 80)
TARGET_RING = (0, 200, 255)

TARGET_NAMES = {code: name for name, code in TARGET_CODES.items()}


def fit_video(frame_size: tuple[int, int], screen_size: tuple[int, int]) -> tuple[int, int]:
    """The largest video size that, with the panel under it, fits the screen."""
    frame_w, frame_h = frame_size
    max_w = screen_size[0] - 40
    max_h = screen_size[1] - PANEL_HEIGHT - 110  # title bar + taskbar
    scale = min(max_w / frame_w, max_h / frame_h)
    return max(320, int(frame_w * scale)), max(180, int(frame_h * scale))


class LabOverlay:
    """Renders the video, the landmarks, the grab and the bottom panel."""

    def __init__(self, video_size: tuple[int, int]) -> None:
        self.video_size = video_size
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        # Text on the video grows with it; the panel stays a fixed size.
        self._video_text = max(0.5, video_size[0] / 1280)
        # Each hand's RELEASED banner: (text, frames left).
        self._banners: dict[str, tuple[str, int]] = {}

    # --- video --------------------------------------------------------------

    @staticmethod
    def _to_px(lm, width: int, height: int, mirror: bool = False) -> tuple[int, int]:
        x = 1.0 - lm[0] if mirror else lm[0]
        return (int(x * width), int(lm[1] * height))

    def draw_frame(self, bgr, frame: Frame, mirror: bool = False) -> None:
        h, w = bgr.shape[:2]

        if frame.pose:
            for a, b in POSE_CONNECTIONS:
                if a < len(frame.pose) and b < len(frame.pose):
                    pa = self._to_px(frame.pose[a], w, h, mirror)
                    pb = self._to_px(frame.pose[b], w, h, mirror)
                    cv2.line(bgr, pa, pb, (200, 160, 0), 1, cv2.LINE_AA)

        if frame.face:
            for chain in FACE_CONTOURS:
                pts = [self._to_px(frame.face[i], w, h, mirror) for i in chain]
                for i in range(len(pts) - 1):
                    cv2.line(bgr, pts[i], pts[i + 1], (0, 220, 220), 1, cv2.LINE_AA)

        for hand in frame.hands:
            colour = (255, 120, 0) if hand.handedness == "Left" else (255, 0, 200)
            for a, b in HAND_CONNECTIONS:
                pa = self._to_px(hand.landmarks[a], w, h, mirror)
                pb = self._to_px(hand.landmarks[b], w, h, mirror)
                cv2.line(bgr, pa, pb, colour, 2, cv2.LINE_AA)

    def draw_targets(self, bgr, frame: Frame, mirror: bool = False) -> None:
        """Ring every grabbable face part: pinch on a ring to grab it. (The
        fingers are drawn already; each is grabbable along its length.)"""
        h, w = bgr.shape[:2]
        radius = max(6, int(9 * self._video_text))
        if frame.face:
            for point in face_target_points(frame.face).values():
                cv2.circle(bgr, self._to_px(point, w, h, mirror), radius, TARGET_RING, 1, cv2.LINE_AA)

    def draw_grabs(
        self, bgr, grabs: dict[str, GrabState], mirror: bool = False, snap_distance: float = 2.5
    ) -> None:
        """Each hand's grab, its text one under the other."""
        size = self._video_text
        x, y = int(16 * size), int(40 * size)
        for hand, grab in grabs.items():
            if self._draw_grab(bgr, hand, grab, (x, y), mirror, snap_distance):
                y += int(80 * size)

    def _draw_grab(self, bgr, hand: str, grab: GrabState, origin, mirror: bool, snap_distance: float) -> bool:
        """One hand's grab: a line from the grabbed skin to the fingers, and
        a meter - or its RELEASED banner. Returns whether it drew anything."""
        h, w = bgr.shape[:2]
        size = self._video_text
        x, y = origin

        if grab.just_released:
            text = f"{hand} RELEASED {grab.target}  peak {grab.release_peak:.2f}"
            self._banners[hand] = (text, RELEASE_BANNER_FRAMES)
        text, frames = self._banners.get(hand, ("", 0))
        if frames > 0 and not grab.active:
            self._banners[hand] = (text, frames - 1)
            self._outlined(bgr, text, (x, y), 0.9 * size, (255, 150, 0))
            return True
        if not grab.active or grab.anchor is None:
            return False

        self._outlined(bgr, f"{hand} GRABBING {grab.target}", (x, y), 0.9 * size, (0, 255, 200))
        y += int(34 * size)
        self._outlined(bgr, f"STRETCH {grab.stretch:.2f}", (x, y), 0.8 * size, (100, 255, 255))
        y += int(14 * size)
        # The meter is full at the snap distance, where the grab lets go.
        meter_w = int(260 * size)
        fill = int(meter_w * min(1.0, grab.stretch / snap_distance))
        cv2.rectangle(bgr, (x, y), (x + fill, y + int(14 * size)), (0, 255, 255), -1)
        cv2.rectangle(bgr, (x, y), (x + meter_w, y + int(14 * size)), (80, 80, 80), 1)

        pinch = (grab.anchor[0] + grab.pull[0], grab.anchor[1] + grab.pull[1])
        anchor_px = self._to_px(grab.anchor, w, h, mirror)
        pinch_px = self._to_px(pinch, w, h, mirror)
        cv2.line(bgr, anchor_px, pinch_px, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(bgr, anchor_px, 6, (0, 255, 0), -1)
        return True

    def _outlined(self, bgr, text: str, origin: tuple[int, int], scale: float, colour) -> None:
        """Text with a dark outline, readable on any background."""
        thickness = max(1, int(2 * scale))
        cv2.putText(bgr, text, origin, self._font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(bgr, text, origin, self._font, scale, colour, thickness, cv2.LINE_AA)

    # --- panel --------------------------------------------------------------

    def draw_status(self, panel, status_lines: list[str], calibrate_progress: float | None) -> None:
        y = 24
        for line in status_lines:
            cv2.putText(panel, line, (MARGIN, y), self._font, 0.45, TEXT, 1, cv2.LINE_AA)
            y += 20
        if calibrate_progress is not None:
            cv2.putText(panel, "CALIBRATING - hold still", (MARGIN, y + 4), self._font, 0.5,
                        (0, 220, 255), 1, cv2.LINE_AA)
            bar_w = int((STATUS_WIDTH - 2 * MARGIN) * calibrate_progress)
            cv2.rectangle(panel, (MARGIN, y + 12), (MARGIN + bar_w, y + 18), (0, 220, 255), -1)
        cv2.line(panel, (STATUS_WIDTH, 10), (STATUS_WIDTH, PANEL_HEIGHT - 10), (70, 70, 70), 1)

    def draw_channels(self, panel, channels: dict[str, ChannelState], activate_threshold: float) -> None:
        rows = [name for name in GAMEPLAY_CHANNELS if name in channels and name not in PANEL_HIDDEN]
        per_column = math.ceil(len(rows) / CHANNEL_COLUMNS)
        left = STATUS_WIDTH + MARGIN
        cell_w = (panel.shape[1] - left - MARGIN) // CHANNEL_COLUMNS
        for i, name in enumerate(rows):
            column, row = divmod(i, per_column)
            x = left + column * cell_w
            y = 22 + row * ROW_HEIGHT
            self._draw_channel(panel, channels[name], x, y, cell_w, activate_threshold)

    def _draw_channel(self, panel, state: ChannelState, x: int, y: int, cell_w: int, threshold: float) -> None:
        name = state.name
        short = (
            name.replace("hand.", "h.").replace("face.", "f.").replace("body.", "b.")
            .replace("Left.", "L.").replace("Right.", "R.")
        )
        cv2.putText(panel, short, (x, y + 4), self._font, 0.38, DIM if state.lost else TEXT, 1, cv2.LINE_AA)

        bar_left = x + int(cell_w * 0.48)
        bar_right = x + cell_w - 58
        value_x = bar_right + 6
        colour = ACTIVE if state.active else (150, 150, 150)

        grab_field = name.rsplit(".", 1)[1] if name.startswith("grab.") else None
        if grab_field == "target":
            label = TARGET_NAMES.get(int(round(state.value)), "-")
            cv2.putText(panel, label, (bar_left, y + 4), self._font, 0.4, colour, 1, cv2.LINE_AA)
            return

        full = GRAB_BAR_RANGE.get(grab_field, BAR_RANGE)
        width = bar_right - bar_left
        cv2.rectangle(panel, (bar_left, y - 3), (bar_right, y + 6), (50, 50, 50), -1)
        length = int(min(1.0, state.magnitude / full) * width)
        if length > 0 and not state.lost:
            cv2.rectangle(panel, (bar_left, y - 3), (bar_left + length, y + 6), colour, -1)
        if grab_field is None:
            mark = bar_left + int(min(1.0, threshold / full) * width)
            cv2.line(panel, (mark, y - 5), (mark, y + 8), (110, 110, 110), 1)
        cv2.putText(panel, f"{state.value:+.3f}", (value_x, y + 4), self._font, 0.35,
                    ACTIVE if state.active else DIM, 1, cv2.LINE_AA)

    # --- everything ---------------------------------------------------------

    def compose(
        self,
        video,
        frame: Frame,
        channels: dict[str, ChannelState],
        status_lines: list[str],
        calibrate_progress: float | None = None,
        activate_threshold: float = 0.035,
        mirror: bool = False,
        grabs: dict[str, GrabState] | None = None,
        snap_distance: float = 2.5,
        clean: bool = False,
    ) -> np.ndarray:
        """The whole window: scaled video on top, panel underneath.

        `clean` leaves the video alone - no landmarks, rings or grab line -
        so the stretch can be judged as it really looks.
        """
        if (video.shape[1], video.shape[0]) != self.video_size:
            video = cv2.resize(video, self.video_size, interpolation=cv2.INTER_LINEAR)
        else:
            video = video.copy()
        if not clean:
            self.draw_frame(video, frame, mirror=mirror)
            self.draw_targets(video, frame, mirror=mirror)
            if grabs:
                self.draw_grabs(video, grabs, mirror, snap_distance)

        panel = np.full((PANEL_HEIGHT, self.video_size[0], 3), PANEL_BG, dtype=np.uint8)
        self.draw_status(panel, status_lines, calibrate_progress)
        self.draw_channels(panel, channels, activate_threshold)
        return np.vstack([video, panel])
