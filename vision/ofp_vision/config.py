"""Configuration for the vision service.

Everything tunable lives here so the tuning loop is: edit config -> reload ->
replay a recorded session -> compare. Nothing magic in the code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

VISION_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = VISION_DIR / "models"
RECORDS_DIR = VISION_DIR / "records"

MODEL_FILES = {
    "face": "face_landmarker.task",
    "hand": "hand_landmarker.task",
    "pose": "pose_landmarker.task",
    # Labels every pixel: background, hair, body skin, face skin, clothes,
    # other. Optional: the stretch effect uses it to move only real skin.
    "segment": "selfie_multiclass_256x256.tflite",
}
# Tracking needs these; the rest are optional extras.
REQUIRED_MODELS = ("face", "hand", "pose")

MODEL_URLS = {
    "face": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "segment": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
    ),
}


@dataclass
class CaptureConfig:
    """How to grab frames.

    `device_name` is matched case-insensitively as a substring against the
    DirectShow device list. Selecting by name is not optional: this machine has
    an "OBS Virtual Camera" that enumerates at a *lower* index than the real
    webcam and silently returns a frozen frame when OBS is not running.
    """

    device_name: str = "ACER HD User Facing"
    # 720p: sharp enough to fill the lab window. Inference costs about the
    # same as 640x480 (~44ms here) because MediaPipe resizes internally.
    width: int = 1280
    height: int = 720
    fps: int = 30
    # Used only if the name match fails.
    fallback_indices: tuple[int, ...] = (0, 1, 2)
    # Flip the preview horizontally so it behaves like a mirror.
    mirror: bool = True
    # MediaPipe reports handedness for an already-mirrored image. If we mirror
    # the preview ourselves, flip the labels back so "Left" means the user's left.
    mirror_handedness: bool = True
    # Substrings that mark a device as a virtual/virtual-camera we should avoid.
    reject_keywords: tuple[str, ...] = ("virtual", "obs", "manycam", "snap", "droidcam")


@dataclass
class FilterConfig:
    """OneEuroFilter parameters, per channel group.

    `beta` is responsiveness to speed: high beta = snappy but jittery (fingers),
    low beta = smooth but laggy (face).
    """

    face_min_cutoff: float = 1.0
    face_beta: float = 0.015
    hand_min_cutoff: float = 2.0
    hand_beta: float = 0.05
    body_min_cutoff: float = 1.2
    body_beta: float = 0.03


@dataclass
class DetectionConfig:
    """A channel becomes 'pulled' when its value exceeds the baseline by this
    much, and fires a 'release' event when it contracts faster than
    `release_velocity` while pulled.
    """

    activate_threshold: float = 0.035
    # A release fires when the value contracts by at least this many
    # activation thresholds per second. Expressed as a ratio rather than an
    # absolute value so it means the same thing for a cheek feature (range
    # ~0.05) and a finger reach feature (range ~3).
    release_velocity_ratio: float = 4.0
    release_threshold: float = 0.015
    # Below this the baseline is considered stale (e.g. user moved / camera changed).
    baseline_validity_seconds: float = 30.0
    # Median landmark displacement (normalised image units) above which the
    # player counts as "moving". Frames above this are rejected during
    # calibration so a settling player cannot poison the baseline.
    calibration_max_motion: float = 0.004
    # The face pull composites are built from the distance between the
    # tragion and the chin, and the chin swings when the jaw opens. A player
    # who opens their mouth mid-pull would otherwise look like a hard tug.
    # Once the jaw deviates from its calibrated aperture by more than this,
    # the face pull channels are suppressed.
    jaw_gate_feature: str = "face.jawOpen"
    jaw_gate_threshold: float = 0.10
    # A channel with no calibrated baseline (typically the hands, which are
    # rarely in view during a neutral-face calibration) learns one from the
    # median of its first this-many frames, and reads zero until then.
    # Without this the deviation is always 0 and the channel is dead.
    lazy_baseline_frames: int = 15


@dataclass
class GrabConfig:
    """Pinch-grab-pull, the main interaction (see `grab.py`).

    Distances are in units of the player's own body, so they hold for any
    face size and any distance from the camera: pinch gaps are fractions of
    palm length, everything on the face is a fraction of interocular distance.

    These are first guesses. Tune them against recordings with `tools.replay`.
    """

    # Thumb-tip to index-tip gap that starts a pinch...
    pinch_on: float = 0.15
    # ...and the larger gap that ends it. The gap between the two is
    # hysteresis: a pinch hovering near one threshold would otherwise flicker
    # on and off, and every flicker would drop the grab.
    pinch_off: float = 0.25
    # The pinch must close within this distance of a face target to grab it,
    # in interocular distances. Small, because the mouth, lips and nose sit
    # close together.
    grab_radius: float = 0.35
    # A finger is grabbed where the pinch lands on it, anywhere along it - but
    # only if the pinch is really on it: within this many palm lengths of its
    # centre line (a finger is ~0.2 palm wide). Further out is the air.
    finger_grab_radius: float = 0.15
    # If MediaPipe relabels the pinching hand mid-grab (or gives both hands
    # one label), a pinch this close to the grab's last one (normalised image
    # height) is taken to be the same pinch, and keeps its hand's label.
    relabel_distance: float = 0.08
    # A pinch only grabs in the first moments after it closes. Without this, a
    # hand that is already pinched grabs whatever it happens to pass over.
    grab_window_ms: float = 300.0
    # Pulling further than this snaps the grab, which counts as a release.
    snap_distance: float = 2.5
    # Hold the grab through tracking dropouts shorter than this. MediaPipe
    # loses a hand for a frame or two when it overlaps the face.
    lost_grace_ms: float = 200.0
    # OneEuroFilter on the pull vector, in interocular units. A yank moves
    # ~10 units/s, so beta = 1 lifts the cutoff to ~12Hz and the peak of a
    # fast pull survives; at rest (~0.1 units/s of jitter) it stays ~2Hz.
    smooth_min_cutoff: float = 2.0
    smooth_beta: float = 1.0


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    show_lab_window: bool = True
    # Run the three models every Nth frame (1 = every frame).
    inference_stride: int = 1
    record: bool = False
    record_path: Path | None = None


@dataclass
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    grab: GrabConfig = field(default_factory=GrabConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    models_dir: Path = MODELS_DIR
    baseline_path: Path = VISION_DIR / "baseline.json"

    def model_path(self, name: str) -> Path:
        return self.models_dir / MODEL_FILES[name]


# --- Channel groups ---------------------------------------------------------
#
# Channels are the units the Unity client can bind to an avatar body part.
# Everything listed here is emitted on the wire every frame.

CHANNEL_GROUPS: dict[str, tuple[str, ...]] = {
    "face": (
        "face.cheekR",
        "face.cheekL",
        "face.cheekAsym",
        "face.cheekSquintR",
        "face.cheekSquintL",
        "face.earR",
        "face.earL",
        "face.earAsym",
        "face.earVertR",
        "face.earVertL",
        "face.mouthOpen",
        "face.jawOpen",
        "face.mouthPucker",
        "face.mouthFunnel",
        "face.smileR",
        "face.smileL",
        # Raw tragion geometry - inputs to the pull composites below.
        "face.tragChinR",
        "face.tragChinL",
        "face.tragForeheadR",
        "face.tragForeheadL",
    ),
    "hand": tuple(
        f"hand.{side}.{digit}.{kind}"
        for side in ("Left", "Right")
        for digit in ("thumb", "index", "middle", "ring", "pinky")
        for kind in ("reach", "ext", "straight")
    )
    + tuple(f"hand.{side}.{kind}" for side in ("Left", "Right") for kind in ("meanExt", "spread")),
    "body": (
        "body.armExtendR",
        "body.armExtendL",
        "body.armReachR",
        "body.armReachL",
        "body.shoulderRaiseR",
        "body.shoulderRaiseL",
    ),
}

# --- Composite channels -----------------------------------------------------
#
# A composite is derived from other channels rather than measured directly.
#
# Why this exists: every face feature is a *distance* between two landmarks,
# and a distance only sees the component of the motion perpendicular to the
# line it spans. So each one is blind to a whole family of pull directions -
# measured across 36 directions on a real face, `face.cheekL` swings by a
# factor of 33 and `face.earL` by 73, with a minimum response so small that
# a real tug would never cross the activation threshold.
#
# The fix is to measure the tragion against two rigid anchors whose
# gradients are near-orthogonal (the chin and the forehead - both bone, and
# both far enough away to give a long lever), and take the norm of the two
# deviations. The result is a channel whose response is the same no matter
# which way the player tugs: max/min of 1.09 over the same 36 directions.
#
# Rules:
#   "norm" - sqrt of the sum of squares of the source deviations
#   "diff" - first source minus second source
COMPOSITE_CHANNELS: dict[str, tuple[str, tuple[str, ...]]] = {
    "face.pullL": ("norm", ("face.tragChinL", "face.tragForeheadL")),
    "face.pullR": ("norm", ("face.tragChinR", "face.tragForeheadR")),
    "face.pullAsym": ("diff", ("face.pullL", "face.pullR")),
}

# --- Grab channels ----------------------------------------------------------
#
# Produced by `grab.TwoHandGrabber`, not measured from landmarks: one set per
# pinching hand, so either hand - or both at once - can grab. They are
# already "how far from rest" (zero when nothing is grabbed), so the channel
# manager passes them through untouched: no baseline, no filter, no
# threshold-based release. Filtering `.target` in particular would blend one
# target's code into another's on the way.
#
#   grab.<hand>.active   1 while that hand holds a grab
#   grab.<hand>.stretch  pull length, in interocular distances (palm lengths on a finger)
#   grab.<hand>.stretchX/Y  the pull vector, same units. +x = player's right, +y = down
#   grab.<hand>.dirX/Y   unit vector of the pull
#   grab.<hand>.target   0 none, then the codes in grab.TARGET_CODES (face parts
#                        1-12, fingers 20-29)
# <hand> is the pinching hand: Left or Right.
GRAB_HANDS: tuple[str, ...] = ("Left", "Right")
GRAB_FIELDS: tuple[str, ...] = ("active", "stretch", "stretchX", "stretchY", "dirX", "dirY", "target")
GRAB_CHANNELS: tuple[str, ...] = tuple(f"grab.{hand}.{field}" for hand in GRAB_HANDS for field in GRAB_FIELDS)

# The subset the game is expected to bind to avatar parts. The rest are
# diagnostic and stay available but are not part of the gameplay contract.
#
# For the face, the pull composites come first: they are the only face
# channels that fire for a tug in *any* direction. The rest are kept because
# they localise the pull (which side, and how it is oriented) and because
# they are what the lab overlay shows while tuning.
GAMEPLAY_CHANNELS: tuple[str, ...] = GRAB_CHANNELS + (
    "face.pullL",
    "face.pullR",
    "face.pullAsym",
    "face.cheekR",
    "face.cheekL",
    "face.cheekAsym",
    "face.earR",
    "face.earL",
    "face.earAsym",
    "hand.Left.index.reach",
    "hand.Right.index.reach",
    "hand.Left.middle.reach",
    "hand.Right.middle.reach",
    "hand.Left.thumb.reach",
    "hand.Right.thumb.reach",
    "hand.Left.meanExt",
    "hand.Right.meanExt",
    "body.armExtendR",
    "body.armExtendL",
    "body.armReachR",
    "body.armReachL",
)

# Composite channels are appended after the measured ones: a composite may
# be built from other composites, and they have to resolve in order.
ALL_CHANNELS: tuple[str, ...] = (
    tuple(name for group in ("face", "hand", "body") for name in CHANNEL_GROUPS[group])
    + tuple(COMPOSITE_CHANNELS)
    + GRAB_CHANNELS
)


def group_of(channel: str) -> str:
    """Which filter group a channel belongs to (drives filter tuning)."""
    if channel.startswith("face."):
        return "face"
    if channel.startswith("hand."):
        return "hand"
    if channel.startswith("body."):
        return "body"
    if channel.startswith("grab."):
        return "hand"
    return "face"
