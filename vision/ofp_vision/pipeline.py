"""One frame through the pipeline: landmarks -> channels -> wire message.

    Frame -> compute_features ----------> calibration (until a baseline exists)
          \\                          \\-> ChannelManager -> message
           \\-> TwoHandGrabber -> grab.<hand>.* -/  + release events

Shared by the live server and the replay tool, so a replayed recording goes
through exactly the code the game sees. No camera, no MediaPipe, no socket:
the server feeds it frames and publishes what comes out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .baseline import BaselineRecorder
from .channels import ChannelManager, ReleaseEvent
from .config import GRAB_HANDS, Config
from .grab import GrabState, TwoHandGrabber
from .protocol import build_message, build_quality
from .signals import Frame, compute_features, motion_score

# The channel each hand's grab release is reported on.
GRAB_RELEASE_CHANNELS: dict[str, str] = {hand: f"grab.{hand}.stretch" for hand in GRAB_HANDS}


@dataclass(frozen=True)
class StepResult:
    message: dict
    # Each hand's grab, by hand ("Left", "Right"): none, one or both held.
    grabs: dict[str, GrabState] = field(default_factory=dict)
    # The new baseline, on the one frame where calibration finishes.
    calibrated: dict[str, float] | None = None
    # The frame with its hands' labels steadied - what the stretch effect
    # and the lab should draw from, so they agree with the grabs.
    frame: Frame | None = None


def grab_release_events(grabs: dict[str, GrabState], t_ms: float) -> list[ReleaseEvent]:
    """A grab that ends in an unpinch or a snap fires, like any other pull -
    each hand on its own channel."""
    return [
        ReleaseEvent(
            channel=GRAB_RELEASE_CHANNELS[hand],
            peak=grab.release_peak,
            value=0.0,
            t_ms=t_ms,
            signed_peak=grab.release_peak,
        )
        for hand, grab in grabs.items()
        if grab.just_released
    ]


class Pipeline:
    def __init__(
        self,
        config: Config,
        baseline: dict[str, float],
        recorder: BaselineRecorder | None = None,
    ) -> None:
        self.config = config
        self.manager = ChannelManager(config)
        self.manager.set_baseline(baseline)
        self.grabber = TwoHandGrabber(
            config.grab,
            mirror=config.capture.mirror,
            aspect=config.capture.width / config.capture.height,
        )
        self.recorder = recorder or BaselineRecorder(seconds=3.0, warmup=0.4)
        self.calibrating = not baseline
        self._previous: Frame | None = None

    def start_calibration(self, t_ms: float, reset: bool = False) -> None:
        """Begin a neutral-pose capture. `reset` also drops the old baseline."""
        if reset:
            self.manager.set_baseline({})
        self.recorder.start(t_ms)
        self.calibrating = True

    def calibration_progress(self, t_ms: float) -> float | None:
        return self.recorder.progress(t_ms) if self.calibrating else None

    def step(self, frame: Frame, fps: float) -> StepResult:
        # Steady hand labels first: the features, the grabs, the effect and
        # the lab must all agree which hand is which.
        frame = self.grabber.relabel(frame)
        features = compute_features(frame)
        grabs = self.grabber.update(frame, frame.t_ms)
        calibrated = None
        events: list[ReleaseEvent] = []

        if self.calibrating:
            calibrated = self._calibrate(features, frame)
        else:
            events = self.manager.update({**features, **self.grabber.features()}, frame.t_ms)
            events = events + grab_release_events(grabs, frame.t_ms)

        self._previous = frame
        quality = build_quality(
            has_face=frame.face is not None,
            has_left_hand=any(h.handedness == "Left" for h in frame.hands),
            has_right_hand=any(h.handedness == "Right" for h in frame.hands),
            has_pose=frame.pose is not None,
            fps=fps,
            calibrated=not self.calibrating,
        )
        message = build_message(frame.t_ms, self.manager.channels, events, quality)
        return StepResult(message=message, grabs=grabs, calibrated=calibrated, frame=frame)

    def _calibrate(self, features: dict[str, float], frame: Frame) -> dict[str, float] | None:
        """Feed one frame to the recorder. Returns the baseline when done.

        Grab features are left out on purpose: they are zero at rest by
        construction and pass through the channel manager unbaselined.
        """
        motion = motion_score(self._previous, frame)
        stable = motion <= self.config.detection.calibration_max_motion
        if not self.recorder.update(features, frame.t_ms, stable=stable):
            return None
        captured = self.recorder.result()
        self.manager.set_baseline(captured)
        self.calibrating = False
        return captured
