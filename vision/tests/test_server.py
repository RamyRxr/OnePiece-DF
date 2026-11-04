"""Tests for the lab window's resilience.

The stretch effect is cosmetic. Whatever goes wrong inside it, the service
must keep tracking and streaming: the lab shows the plain picture instead.
"""

from __future__ import annotations

import cv2
from face_fixture import image, landmarks

from ofp_vision import server
from ofp_vision.capture import CameraSource
from ofp_vision.config import Config
from ofp_vision.grab import GrabState
from ofp_vision.pipeline import Pipeline, StepResult
from ofp_vision.signals import Frame
from ofp_vision.warp import FaceWarper


def test_the_lab_survives_a_failing_stretch_effect(monkeypatch):
    shown = []
    monkeypatch.setattr(cv2, "imshow", lambda name, frame: shown.append(frame))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: 0xFF)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *args, **kwargs: None)

    config = Config()
    warper = FaceWarper()

    def broken(bgr):
        raise RuntimeError("the effect broke")

    monkeypatch.setattr(warper, "warp", broken)
    lab = server.Lab(config, warper)
    inputs = server.Inputs(camera=CameraSource(config.capture))
    pipeline = Pipeline(config, {"face.cheekL": 1.0})
    frame = Frame(t_ms=0.0, face=landmarks())
    result = StepResult(message={}, grabs={"Left": GrabState(), "Right": GrabState()})

    for _ in range(3):
        key = lab.show(inputs, frame, image(), result, pipeline, (20.0, 40.0))

    assert key == 0xFF
    assert len(shown) == 3


# --- One devil fruit at a time ---------------------------------------------


class _CountingWarper:
    """Stands in for the stretch effect; counts what the lab asks of it."""

    def __init__(self) -> None:
        self.updates = 0
        self.warps = 0

    def update(self, grabs, frame, size) -> None:
        self.updates += 1

    def warp(self, bgr):
        self.warps += 1
        return bgr


class _CountingFire:
    """Stands in for Ace's fire; counts what the lab asks of it."""

    def __init__(self) -> None:
        self.updates = 0
        self.draws = 0
        self.resets = 0

    def update(self, frame, size) -> None:
        self.updates += 1

    def draw(self, bgr):
        self.draws += 1
        return bgr

    def reset(self) -> None:
        self.resets += 1

    def status_line(self) -> str:
        return "fire: test"


def _lab(monkeypatch):
    """A lab with counting effects, and no real window."""
    monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: 0xFF)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *args, **kwargs: None)
    config = Config()
    warper, fire = _CountingWarper(), _CountingFire()
    lab = server.Lab(config, warper)
    lab.fire = fire
    return lab, warper, fire


def _frame_and_result():
    frame = Frame(t_ms=0.0, face=landmarks())
    result = StepResult(message={}, grabs={"Left": GrabState(), "Right": GrabState()})
    return frame, result


def test_only_the_active_fruit_is_advanced_and_drawn(monkeypatch):
    lab, warper, fire = _lab(monkeypatch)
    frame, result = _frame_and_result()

    lab._stretched(image(), frame, result)
    assert (warper.updates, warper.warps) == (1, 1)
    assert (fire.updates, fire.draws) == (0, 0)

    lab.switch_fruit()
    lab._stretched(image(), frame, result)
    # The hidden rubber fruit is not even advanced: its state cannot drift.
    assert (warper.updates, warper.warps) == (1, 1)
    assert (fire.updates, fire.draws) == (1, 1)


def test_switching_away_from_fire_puts_it_out(monkeypatch):
    lab, _, fire = _lab(monkeypatch)
    lab.fruit = "fire"

    assert lab.switch_fruit() == "rubber"
    assert fire.resets == 1

    # Lighting up again must not re-reset what is already empty.
    assert lab.switch_fruit() == "fire"
    assert fire.resets == 1


def test_the_f_key_switches_the_fruit(monkeypatch):
    lab, _, _ = _lab(monkeypatch)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("f"))
    config = lab.config
    inputs = server.Inputs(camera=CameraSource(config.capture))
    pipeline = Pipeline(config, {"face.cheekL": 1.0})
    frame, result = _frame_and_result()

    assert lab.fruit == "rubber"
    key = lab.show(inputs, frame, image(), result, pipeline, (20.0, 40.0))

    assert key == ord("f")
    assert lab.fruit == "fire"


def test_fire_mode_hides_the_gum_gum_rings(monkeypatch):
    lab, _, _ = _lab(monkeypatch)
    drawn = []
    monkeypatch.setattr(server.LabOverlay, "draw_targets", lambda self, *a, **k: drawn.append("targets"))
    monkeypatch.setattr(server.LabOverlay, "draw_grabs", lambda self, *a, **k: drawn.append("grabs"))
    config = lab.config
    inputs = server.Inputs(camera=CameraSource(config.capture))
    pipeline = Pipeline(config, {"face.cheekL": 1.0})
    frame, result = _frame_and_result()

    lab.show(inputs, frame, image(), result, pipeline, (20.0, 40.0))
    assert drawn == ["targets", "grabs"]

    lab.switch_fruit()
    lab.show(inputs, frame, image(), result, pipeline, (20.0, 40.0))
    assert drawn == ["targets", "grabs"]  # the fire fruit draws neither
