"""Replay a recorded session offline and report how each channel behaved.

This is the tool that makes tuning fast. Record 30 seconds of yourself
pulling your cheek, replay it, and get a table telling you which channels
actually moved, by how much, and which ones are dead weight.

    python -m ofp_vision.tools.replay records/pull-cheek.jsonl
    python -m ofp_vision.tools.replay records/pull-cheek.jsonl --top 30
    python -m ofp_vision.tools.replay records/pull-cheek.jsonl --csv out.csv
    python -m ofp_vision.tools.replay records/pull-cheek.jsonl --channel face.cheekL --watch

A channel whose dynamic range (p95 - p5) is near zero is not measuring
anything, and should be cut from the gameplay channel list rather than
shipped as noise.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from ..baseline import BaselineRecorder
from ..config import Config
from ..pipeline import GRAB_RELEASE_CHANNELS, Pipeline
from ..signals import Frame, HandObservation


def load_frames(path: Path) -> list[Frame]:
    frames: list[Frame] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            hands = [
                HandObservation(
                    handedness=h["h"],
                    landmarks=[tuple(p) for p in h["lm"]],
                )
                for h in data.get("hands", [])
            ]
            frames.append(
                Frame(
                    t_ms=float(data["t"]),
                    face=[tuple(p) for p in data["face"]] if data.get("face") else None,
                    blendshapes=data.get("bs", {}) or {},
                    hands=hands,
                    pose=[tuple(p) for p in data["pose"]] if data.get("pose") else None,
                )
            )
    return frames


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--calib-seconds", type=float, default=3.0)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--channel", type=str, default=None, help="only this channel")
    parser.add_argument("--csv", type=Path, default=None, help="write the time series")
    parser.add_argument("--watch", action="store_true", help="print a live table")
    parser.add_argument("--min-range", type=float, default=0.0, help="hide channels below this")
    args = parser.parse_args()

    if not args.record.exists():
        print(f"record not found: {args.record}", file=sys.stderr)
        return 1

    frames = load_frames(args.record)
    if not frames:
        print("record is empty", file=sys.stderr)
        return 1

    duration = (frames[-1].t_ms - frames[0].t_ms) / 1000.0
    print(f"loaded {len(frames)} frames, {duration:.1f}s")

    # --- run the session through the same pipeline as the live server ---
    #
    # The opening window calibrates (moving frames rejected, as live), then
    # every later frame is streamed through the channels and the grab.
    config = Config()
    recorder = BaselineRecorder(seconds=args.calib_seconds, warmup=0.0)
    pipeline = Pipeline(config, {}, recorder)

    series: dict[str, list[float]] = {}
    times: list[float] = []
    active_frames = 0
    events: list[dict] = []

    for frame in frames:
        result = pipeline.step(frame, fps=0.0)
        if result.calibrated is not None:
            print(
                f"baseline: {len(result.calibrated)} features from the first "
                f"{args.calib_seconds}s ({recorder.accepted} stable, "
                f"{recorder.rejected} rejected as moving)"
            )
        if pipeline.calibrating:
            continue
        events += result.message["ev"]
        times.append(frame.t_ms)
        if any(state.active for state in pipeline.manager.channels.values()):
            active_frames += 1
        for name, state in pipeline.manager.channels.items():
            series.setdefault(name, []).append(state.value)

    if not times:
        print("the recording ended before calibration finished", file=sys.stderr)
        return 1

    grab_events = [e for e in events if e["ch"] in GRAB_RELEASE_CHANNELS.values()]
    print(f"release events: {len(events)} ({len(grab_events)} from grabs)")
    for event in grab_events:
        grab = event["ch"].rsplit(".", 1)[0]
        print(f"  {grab} released at {event['t']:>7}ms  peak stretch {event['peak']:.3f}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        names = sorted(series)
        with args.csv.open("w", encoding="utf-8") as handle:
            handle.write("t_ms," + ",".join(names) + "\n")
            for i, t in enumerate(times):
                handle.write(f"{t:.1f}," + ",".join(f"{series[n][i]:.5f}" for n in names) + "\n")
        print(f"wrote {args.csv}")

    if args.watch and args.channel:
        for t, value in zip(times, series[args.channel]):
            bar = "#" * int(min(60, abs(value) * 200))
            sign = "+" if value >= 0 else "-"
            print(f"{t:>9.1f}ms {sign}{abs(value):.4f} {bar}")

    # --- report ---
    stats = []
    for name, values in series.items():
        if len(values) < 3:
            continue
        p5 = percentile(values, 0.05)
        p95 = percentile(values, 0.95)
        stats.append(
            {
                "channel": name,
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
                "p5": p5,
                "p95": p95,
                "range": p95 - p5,
                "peak": max(values, key=abs),
            }
        )

    stats.sort(key=lambda s: s["range"], reverse=True)

    if args.channel:
        stats = [s for s in stats if s["channel"] == args.channel]

    print(f"\nframes with an active channel: {active_frames}/{len(times)} "
          f"({100*active_frames/len(times):.0f}%)\n")

    header = f"{'channel':<34}{'range':>9}{'mean':>9}{'std':>9}{'p5':>9}{'p95':>9}{'peak':>9}"
    print(header)
    print("-" * len(header))
    shown = 0
    for s in stats:
        if shown >= args.top:
            break
        if s["range"] < args.min_range:
            continue
        shown += 1
        print(
            f"{s['channel']:<34}{s['range']:>9.4f}{s['mean']:>9.4f}"
            f"{s['std']:>9.4f}{s['p5']:>9.4f}{s['p95']:>9.4f}{s['peak']:>9.4f}"
        )

    dead = [s["channel"] for s in stats if s["range"] < 0.005]
    if dead:
        print(f"\n{len(dead)} channels with no measurable response (range < 0.005):")
        for name in dead[:40]:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
