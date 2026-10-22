"""Wire format shared with the Unity client.

Design rules
------------
* **Small.** The payload is ~60 numbers. It goes out at 30Hz over a local
  socket; there is no reason to ship video or landmarks to the game.
* **Dimensionless.** Values are ratios of a body-derived reference length, not
  pixels or metres. Unity multiplies by a per-channel gain to decide how many
  centimetres 0.1 of "cheek pull" becomes. Perception and game-feel stay
  decoupled: retuning the stretch length never requires touching the vision.
* **Explicit about missing data.** A channel whose source left the frame is
  flagged `lost` so the game can relax the avatar instead of freezing it.
"""

from __future__ import annotations

from .channels import ChannelManager, ChannelState, ReleaseEvent
from .config import GAMEPLAY_CHANNELS

PROTOCOL_VERSION = 3


def build_message(
    t_ms: float,
    channels: dict[str, ChannelState],
    events: list[ReleaseEvent],
    quality: dict,
    gameplay_only: bool = True,
) -> dict:
    """Assemble one state message."""
    names = GAMEPLAY_CHANNELS if gameplay_only else tuple(channels.keys())

    payload_channels: dict[str, dict] = {}
    for name in names:
        state = channels.get(name)
        if state is None:
            continue
        payload_channels[name] = {
            "v": round(state.value, 4),
            # Magnitude of the deviation from neutral. The avatar stretches
            # either way, so this is the number to drive the rig with; `v`
            # keeps the signed value for anything that cares about direction.
            "m": round(state.magnitude, 4),
            "a": 1 if state.active else 0,
            "l": 1 if state.lost else 0,
        }

    payload_events = [
        {
            "ch": event.channel,
            # Magnitude of the pull - "how hard was it yanked", always >= 0.
            "peak": round(event.peak, 4),
            # +1 / -1: the direction the excursion happened in.
            "d": 1 if event.signed_peak >= 0 else -1,
            "v": round(event.value, 4),
            "t": int(event.t_ms),
        }
        for event in events
    ]

    return {
        "v": PROTOCOL_VERSION,
        "t": int(t_ms),
        "ch": payload_channels,
        "ev": payload_events,
        "q": quality,
    }


def build_quality(
    has_face: bool,
    has_left_hand: bool,
    has_right_hand: bool,
    has_pose: bool,
    fps: float,
    calibrated: bool,
) -> dict:
    """Per-frame health flags, so the game can degrade gracefully."""
    return {
        "face": 1 if has_face else 0,
        "handL": 1 if has_left_hand else 0,
        "handR": 1 if has_right_hand else 0,
        "pose": 1 if has_pose else 0,
        "fps": round(fps, 1),
        "cal": 1 if calibrated else 0,
    }
