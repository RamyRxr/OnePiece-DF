"""Tests for the frame pipeline shared by the live server and the replay tool.

Driven by the synthetic source, so this is the whole chain short of the
camera and the socket: landmarks -> features -> calibration -> channels ->
grab -> wire message.
"""

from __future__ import annotations

from ofp_vision.baseline import BaselineRecorder
from ofp_vision.config import GRAB_CHANNELS, Config
from ofp_vision.pipeline import Pipeline
from ofp_vision.synthetic import SyntheticSource

FRAME_S = 1 / 30


def run(seconds: float, baseline: dict | None = None):
    pipeline = Pipeline(Config(), baseline or {}, BaselineRecorder(seconds=1.0, warmup=0.0))
    source = SyntheticSource()
    results = []
    for _ in range(int(seconds / FRAME_S)):
        frame = source.next_frame(FRAME_S)
        results.append(pipeline.step(frame, fps=30.0))
    return pipeline, results


def test_it_calibrates_first_and_reports_the_baseline_once():
    _, results = run(3.0)
    captured = [r.calibrated for r in results if r.calibrated is not None]
    assert len(captured) == 1
    assert "face.cheekL" in captured[0]
    assert results[0].message["q"]["cal"] == 0
    assert results[-1].message["q"]["cal"] == 1


def test_grab_features_are_not_part_of_the_baseline():
    _, results = run(3.0)
    baseline = next(r.calibrated for r in results if r.calibrated is not None)
    assert not any(name in baseline for name in GRAB_CHANNELS)


def test_the_synthetic_grab_fires_a_release_event():
    _, results = run(10.0)
    # The synthetic player grabs with their left hand.
    events = [e for r in results for e in r.message["ev"] if e["ch"] == "grab.Left.stretch"]
    assert events, "the synthetic player grabs and releases their cheek every cycle"
    assert all(e["peak"] > 0.2 for e in events)


def test_the_message_carries_the_grab_while_held():
    _, results = run(10.0)
    held = [r for r in results if r.message["ch"]["grab.Left.active"]["v"] == 1.0]
    assert held
    assert any(r.message["ch"]["grab.Left.stretch"]["v"] > 0.2 for r in held)
    assert all(r.message["ch"]["grab.Left.target"]["v"] == 1.0 for r in held)
    assert all(r.grabs["Left"].active for r in held)
    assert not any(r.message["ch"]["grab.Right.active"]["v"] for r in results), "the right hand never pinches"


def test_recalibrating_with_reset_clears_the_baseline():
    pipeline, _ = run(3.0)
    pipeline.start_calibration(t_ms=10_000.0, reset=True)
    assert pipeline.calibrating is True
    assert pipeline.manager.baseline == {}
