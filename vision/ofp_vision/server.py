"""The vision service: capture -> infer -> features -> channels -> stream.

Run it:

    python -m ofp_vision.server
    python -m ofp_vision.server --no-window --port 8765

Keyboard controls in the lab window:
    c   start calibration (3 second neutral capture)
    r   reset the baseline and recalibrate
    h   clean view: hide the landmarks and grab drawings
    v   start / stop recording a clip (see clips.py)
    q   quit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .baseline import load_baseline, save_baseline
from .capture import CameraSource
from .clips import ClipSwitch, frame_payload
from .config import Config
from .lab import PANEL_HEIGHT, LabOverlay, fit_video
from .landmarks import Tracker
from .pipeline import Pipeline, StepResult
from .signals import Frame
from .synthetic import SyntheticSource
from .warp import FaceWarper

log = logging.getLogger("ofp_vision")


# --- Streaming --------------------------------------------------------------


class StreamServer:
    """Broadcasts state messages to any connected client.

    Runs in its own thread with its own event loop so the capture loop never
    blocks on a slow or dead socket.
    """

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._outbox: queue.Queue[str] = queue.Queue(maxsize=3)
        self._clients: set = set()
        self._thread: threading.Thread | None = None
        self.clients_connected = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="ws-server")
        self._thread.start()

    def publish(self, message: dict) -> None:
        try:
            self._outbox.put_nowait(json.dumps(message, separators=(",", ":")))
        except queue.Full:
            # Drop the oldest frame rather than stall the capture loop.
            try:
                self._outbox.get_nowait()
                self._outbox.put_nowait(json.dumps(message, separators=(",", ":")))
            except queue.Empty:
                pass

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:  # pragma: no cover
            log.error("WebSocket server stopped: %s", exc)

    async def _main(self) -> None:
        from websockets.asyncio.server import serve

        async with serve(self._handler, self.host, self.port):
            log.info("Streaming on ws://%s:%d", self.host, self.port)
            await self._broadcast_loop()

    async def _handler(self, connection) -> None:
        self._clients.add(connection)
        log.info("Client connected (%d total)", len(self._clients))
        try:
            async for _ in connection:
                pass
        except Exception:
            pass
        finally:
            self._clients.discard(connection)
            log.info("Client disconnected (%d left)", len(self._clients))

    async def _broadcast_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            payload = await loop.run_in_executor(None, self._outbox.get)
            dead = []
            for client in list(self._clients):
                try:
                    await client.send(payload)
                except Exception:
                    dead.append(client)
            for client in dead:
                self._clients.discard(client)


# --- Recording --------------------------------------------------------------


class FrameRecorder:
    """Writes frames to JSONL so a session can be replayed offline."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")
        self.count = 0

    def write(self, frame: Frame) -> None:
        payload = frame_payload(frame)
        self._handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        # Flush every frame: a session killed mid-run should still be a valid
        # recording up to the last complete frame, not a truncated blob.
        self._handle.flush()
        self.count += 1

    def close(self) -> None:
        self._handle.close()



# --- Setup ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ofp-vision", description="Stretch perception service")
    parser.add_argument("--device", type=str, default=None, help="camera name substring")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--no-window", action="store_true", help="headless, no lab overlay")
    parser.add_argument("--record", type=str, default=None, help="write frames to this .jsonl")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop cleanly after N seconds (so a recording is flushed properly)",
    )
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--recalibrate", action="store_true", help="ignore any saved baseline")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="run on generated frames instead of a camera (pipeline/protocol check)",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def build_config(args: argparse.Namespace) -> Config:
    config = Config()
    if args.device:
        config.capture.device_name = args.device
    config.server.host = args.host
    config.server.port = args.port
    config.server.show_lab_window = not args.no_window
    if args.models_dir:
        config.models_dir = Path(args.models_dir)
    return config


@dataclass
class Inputs:
    """Where frames come from: a camera + MediaPipe, or the synthetic player."""

    camera: CameraSource
    tracker: Tracker | None = None
    synthetic: SyntheticSource | None = None
    # How long the last MediaPipe pass took (not counting the camera wait).
    last_infer_ms: float = 0.0

    def read(self, t_ms: float) -> tuple[Frame | None, np.ndarray | None]:
        """One frame of landmarks, and the camera image it came from."""
        if self.synthetic is not None:
            frame = self.synthetic.next_frame(1.0 / 30.0)
            frame.t_ms = t_ms
            return frame, None
        ok, bgr = self.camera.read()
        if not ok or bgr is None:
            return None, None
        started = time.perf_counter()
        frame = self.tracker.process(bgr, t_ms)
        self.last_infer_ms = (time.perf_counter() - started) * 1000.0
        return frame, bgr

    def close(self) -> None:
        if self.tracker is not None:
            self.tracker.close()
        self.camera.release()


def open_inputs(args: argparse.Namespace, config: Config) -> Inputs | int:
    """Open the camera and load the models. Returns an exit code on failure."""
    camera = CameraSource(config.capture)
    if args.synthetic:
        log.info("Running in SYNTHETIC mode - no camera, no MediaPipe")
        return Inputs(camera=camera, synthetic=SyntheticSource())
    if not camera.open():
        log.error(
            "Camera unavailable. If the message mentions privacy, enable "
            "Settings > Privacy & security > Camera > 'Let desktop apps access your camera'."
        )
        return 2
    try:
        tracker = Tracker(
            config.models_dir, delegate="cpu", swap_handedness=config.capture.mirror_handedness
        )
    except Exception as exc:
        log.error("%s", exc)
        camera.release()
        return 3
    return Inputs(camera=camera, tracker=tracker)


# --- Loop pieces ------------------------------------------------------------


class FrameClock:
    """Paces the loop to the camera rate and measures the achieved fps.

    Without pacing the loop outruns the camera (and, in synthetic mode, the
    CPU entirely) and starves the websocket thread of the GIL - the listener
    never gets a chance to bind.
    """

    def __init__(self, fps: int) -> None:
        self.start = time.monotonic()
        self._interval = 1.0 / max(fps, 1)
        self._next_at = self.start
        self._frames = 0
        self._measured_at = self.start
        self.fps = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self._next_at:
            time.sleep(self._next_at - now)
        else:
            self._next_at = now
        self._next_at += self._interval

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.start) * 1000.0

    def tick(self) -> None:
        self._frames += 1
        now = time.monotonic()
        if now - self._measured_at >= 0.5:
            self.fps = self._frames / (now - self._measured_at)
            self._frames = 0
            self._measured_at = now


def save_calibration(config: Config, pipeline: Pipeline, captured: dict[str, float]) -> None:
    save_baseline(
        config.baseline_path,
        captured,
        meta={"captured_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
    )
    log.info(
        "Calibration complete: %d baseline features (%d stable samples, %d rejected as moving)",
        len(captured),
        pipeline.recorder.accepted,
        pipeline.recorder.rejected,
    )


WINDOW_TITLE = "OnePiece-DF :: Signal Lab"
FALLBACK_SCREEN = (1280, 800)


def screen_size() -> tuple[int, int]:
    """The primary screen's size, so the window can fill it."""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:  # not Windows, or no display
        return FALLBACK_SCREEN


def make_effect():
    """The skin-physics stretch, drawn on the graphics card - or, where no
    OpenGL 3.3 context can be made, the CPU warp."""
    try:
        from .stretch import SkinStretcher

        effect = SkinStretcher()
        log.info("Stretch effect: skin physics on the graphics card")
        return effect
    except Exception as exc:
        log.warning("Graphics card unavailable (%s); using the CPU stretch effect", exc)
        return FaceWarper()


@dataclass
class Lab:
    """The Signal Lab window: warped, mirrored preview + overlay + keys."""

    config: Config
    warper: FaceWarper
    overlay: LabOverlay | None = None
    effect_failures: int = 0
    # Hide the landmarks and grab drawings (key h), to judge the effect.
    clean: bool = False
    # The REC line while a clip is being recorded (key v).
    recording: str | None = None

    def _stretched(self, bgr: np.ndarray, frame: Frame, result: StepResult) -> np.ndarray:
        """The camera image with the stretch drawn in - or, if the effect
        fails, the plain image: it is cosmetic, and must never stop the
        tracking and streaming."""
        try:
            self.warper.update(result.grabs, frame, (bgr.shape[1], bgr.shape[0]))
            return self.warper.warp(bgr)
        except Exception:
            self.effect_failures += 1
            if self.effect_failures == 1:
                log.exception("Stretch effect failed; showing the plain image")
            else:
                log.debug("Stretch effect failed again (%d times)", self.effect_failures)
            return bgr

    def _open_window(self, frame_size: tuple[int, int]) -> LabOverlay:
        """Size the window to the screen, once the frame size is known."""
        video_size = fit_video(frame_size, screen_size())
        overlay = LabOverlay(video_size)
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_TITLE, video_size[0], video_size[1] + PANEL_HEIGHT)
        log.info("Lab window: video %dx%d (camera %dx%d)", *video_size, *frame_size)
        return overlay

    def show(
        self,
        inputs: Inputs,
        frame: Frame,
        bgr: np.ndarray | None,
        result: StepResult,
        pipeline: Pipeline,
        stats: tuple[float, float],
    ) -> int:
        """Draw one frame. Returns the key pressed (0xFF for none)."""
        capture = self.config.capture
        if bgr is None:
            preview = np.zeros((capture.height, capture.width, 3), dtype=np.uint8)
        else:
            # Warp in raw camera space - the landmarks' space - then mirror.
            preview = self._stretched(bgr, frame, result)
            if capture.mirror:
                preview = cv2.flip(preview, 1)

        fps, infer_ms = stats
        opened = inputs.camera.opened
        status = [
            f"{opened.name if opened else 'synthetic'}   fps {fps:4.1f}   infer {infer_ms:3.0f}ms",
            f"face {'yes' if frame.face else 'NO'}   hands {len(frame.hands)}   "
            f"baseline {'CAPTURING' if pipeline.calibrating else 'READY'}",
            "PINCH a ring (ear, nose, jaw, cheek, mouth)",
            "or a finger of your other hand - one hand,",
            "or both at once. PULL, then OPEN to fire",
            self.recording or "c calibrate  r reset  h clean  v record  q quit",
        ]
        if self.overlay is None:
            self.overlay = self._open_window((preview.shape[1], preview.shape[0]))
        canvas = self.overlay.compose(
            preview,
            frame,
            pipeline.manager.channels,
            status,
            pipeline.calibration_progress(frame.t_ms),
            self.config.detection.activate_threshold,
            mirror=capture.mirror,
            grabs=result.grabs,
            snap_distance=self.config.grab.snap_distance,
            clean=self.clean,
        )
        cv2.imshow(WINDOW_TITLE, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("h"):
            self.clean = not self.clean
        return key


def handle_key(key: int, pipeline: Pipeline, t_ms: float) -> bool:
    """Apply a lab key. Returns False when the user asked to quit."""
    if key == ord("q"):
        return False
    if key == ord("c"):
        pipeline.start_calibration(t_ms)
        log.info("Manual calibration started")
    elif key == ord("r"):
        pipeline.start_calibration(t_ms, reset=True)
        log.info("Baseline reset")
    return True


def show_lab(
    lab: Lab,
    clips: ClipSwitch,
    inputs: Inputs,
    captured: tuple[Frame, np.ndarray | None],
    result: StepResult,
    pipeline: Pipeline,
    clock: FrameClock,
) -> bool:
    """Record the frame if a clip is running, draw the lab and apply its
    keys. Returns False when the user asked to quit."""
    frame, bgr = captured
    if bgr is not None:
        clips.write(bgr, frame)
    labelled = result.frame if result.frame is not None else frame
    lab.recording = clips.status()
    key = lab.show(inputs, labelled, bgr, result, pipeline, (clock.fps, inputs.last_infer_ms))
    if key == ord("v") and bgr is not None:
        clips.toggle((bgr.shape[1], bgr.shape[0]))
    return handle_key(key, pipeline, frame.t_ms)


# --- Main loop --------------------------------------------------------------


def run(args: argparse.Namespace, config: Config, inputs: Inputs) -> int:
    baseline = {} if args.recalibrate else load_baseline(config.baseline_path)
    pipeline = Pipeline(config, baseline)
    if pipeline.calibrating:
        log.info("No baseline found - starting calibration. Hold a neutral pose.")

    stream = StreamServer(config.server.host, config.server.port)
    stream.start()
    lab = Lab(config, make_effect()) if config.server.show_lab_window else None
    clips = ClipSwitch(config.clips_dir, config.capture.fps)
    recorder = FrameRecorder(Path(args.record)) if args.record else None
    clock = FrameClock(config.capture.fps)
    deadline = clock.start + args.duration if args.duration else None

    running = True
    try:
        while running:
            clock.wait()
            if deadline is not None and time.monotonic() >= deadline:
                log.info("Duration reached (%.1fs) - stopping", args.duration)
                running = False

            frame, bgr = inputs.read(clock.elapsed_ms())
            if frame is None:
                time.sleep(0.01)
                continue
            if bgr is not None:
                # The camera may not honour the requested 640x480.
                pipeline.grabber.set_aspect(bgr.shape[1] / bgr.shape[0])

            result = pipeline.step(frame, clock.fps)
            if result.calibrated is not None:
                save_calibration(config, pipeline, result.calibrated)
            if recorder:
                recorder.write(frame)
            stream.publish(result.message)

            if lab is not None:
                running = show_lab(lab, clips, inputs, (frame, bgr), result, pipeline, clock) and running
            clock.tick()
    except KeyboardInterrupt:
        pass
    finally:
        clips.stop()
        if lab is not None:
            cv2.destroyAllWindows()
        if recorder:
            recorder.close()
            log.info("Recorded %d frames to %s", recorder.count, recorder.path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    config = build_config(args)
    inputs = open_inputs(args, config)
    if isinstance(inputs, int):
        return inputs
    if args.duration:
        log.info(
            "Session runs %.1fs: ~3.5s of neutral pose for calibration, "
            "then perform the gesture until it stops.",
            args.duration,
        )
    try:
        return run(args, config, inputs)
    finally:
        inputs.close()


if __name__ == "__main__":
    raise SystemExit(main())
