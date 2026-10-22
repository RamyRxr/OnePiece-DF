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
