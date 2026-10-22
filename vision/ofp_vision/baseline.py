"""Calibration: capture a neutral baseline and persist it.

Every feature is measured relative to this snapshot, so the numbers mean
"how much more than your neutral pose", not "how much in absolute image
units". This is what makes the same thresholds work for a player with a wide
face and a player with a narrow one.

The median is used rather than the mean: a single blink, a cough or a hand
crossing the frame during calibration would otherwise poison a feature for
the whole session.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

log = logging.getLogger(__name__)

# A feature must be visible in at least this fraction of calibration frames to
# be trusted. Below that we refuse to set a baseline for it.
MIN_PRESENCE = 0.6


class BaselineRecorder:
    """Accumulates features over a calibration window.

    Only *stable* samples are accepted. A player who is still leaning in and
    settling would otherwise poison the baseline for the whole session, and
    every channel would read as permanently offset.
    """

    def __init__(self, seconds: float = 3.0, warmup: float = 0.5, min_samples: int = 20) -> None:
        self.seconds = seconds
        self.warmup = warmup
        self.min_samples = min_samples
        self.reset()

    def reset(self) -> None:
        self._samples: dict[str, list[float]] = {}
        self._started_at: float | None = None
        self._done = False
        self.accepted = 0
        self.rejected = 0

    @property
    def done(self) -> bool:
        return self._done

    def start(self, t: float) -> None:
        self.reset()
        self._started_at = t

    def progress(self, t: float) -> float:
        """How far through warmup + capture we are, 0..1."""
        if self._done:
            return 1.0
        if self._started_at is None:
            return 0.0
        elapsed = (t - self._started_at) / 1000.0
        return min(1.0, max(0.0, elapsed / (self.warmup + self.seconds)))

    def update(self, features: dict[str, float], t: float, stable: bool = True) -> bool:
        """Feed a frame. Returns True when the baseline is ready."""
        if self._done:
            return True
        if self._started_at is None:
            self.start(t)
        elapsed = (t - self._started_at) / 1000.0
        if elapsed < self.warmup:
            return False
        if elapsed >= self.warmup + self.seconds:
            self._done = True
            return True

        if not stable:
            self.rejected += 1
            return False

        self.accepted += 1
        for name, value in features.items():
            self._samples.setdefault(name, []).append(value)
        return False

    def result(self) -> dict[str, float]:
        """Median per feature. Features that were too rarely visible are dropped."""
        out: dict[str, float] = {}
        for name, values in self._samples.items():
            if len(values) < 3:
                continue
            out[name] = float(statistics.median(values))
        return out


def load_baseline(path: Path) -> dict[str, float]:
    """Load a saved baseline, or an empty dict if there isn't one."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        features = data.get("features", data)
        if not isinstance(features, dict):
            return {}
        return {str(k): float(v) for k, v in features.items()}
    except Exception as exc:
        log.warning("Could not read baseline %s: %s", path, exc)
        return {}


def save_baseline(path: Path, features: dict[str, float], meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "meta": meta or {}, "features": features}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    log.info("Baseline saved to %s (%d features)", path, len(features))
