"""Tests for the signal conditioning layer."""

from __future__ import annotations

import random

import pytest

from ofp_vision.filters import OneEuroFilter, Slope


def test_filter_reduces_jitter():
    """A noisy constant signal should come out much smoother."""
    rng = random.Random(7)
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)

    raw = [1.0 + rng.gauss(0, 0.05) for _ in range(200)]
    smoothed = [filt(v, i / 30.0) for i, v in enumerate(raw)]

    def spread(values):
        return max(values) - min(values)

    assert spread(smoothed) < spread(raw) * 0.5


def test_filter_tracks_a_step_promptly():
    """beta > 0 should let the filter reach a moving target quickly."""
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    t = 0.0
    for _ in range(10):
        filt(0.0, t)
        t += 1 / 30.0
    for _ in range(6):
        filt(1.0, t)
        t += 1 / 30.0
    assert filt.value == pytest.approx(1.0, abs=0.15)


def test_filter_output_stays_within_the_input_range():
    filt = OneEuroFilter(min_cutoff=2.0, beta=0.1)
    values = [filt(v, i / 30.0) for i, v in enumerate([0.2, 0.4, 0.1, 0.9, 0.3])]
    assert min(values) >= 0.1 - 1e-6
    assert max(values) <= 0.9 + 1e-6


def test_slope_reports_units_per_second():
    slope = Slope()
    assert slope(0.0, 0.0) == 0.0
    assert slope(1.0, 0.5) == pytest.approx(2.0)
    assert slope(2.0, 1.0) == pytest.approx(2.0)


def test_slope_handles_duplicate_timestamps():
    slope = Slope()
    slope(0.0, 0.0)
    value = slope(1.0, 0.0)
    assert value == pytest.approx(0.0) or value == pytest.approx(1e3, rel=1e-2)


def test_filter_reset_clears_state():
    filt = OneEuroFilter()
    filt(1.0, 0.0)
    filt.reset()
    assert filt.value is None
