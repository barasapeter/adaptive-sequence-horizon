#!/usr/bin/env python3
"""Train on an append-only binary sequence and forecast from its latest head."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

from adaptive_horizon import OnlineAdaptiveHorizon, parse_binary_sequence


def load_sequence(path):
    return parse_binary_sequence(Path(path).read_text(encoding="utf-8"))


def model_options(args):
    return {
        "n_min": args.n_min,
        "n_max": args.n_max,
        "horizon": args.horizon,
        "performance_memory": args.memory,
        "prior_strength": args.prior_strength,
        "min_resolved_per_n": args.min_resolved,
        "decay": args.decay,
        "min_lift": args.min_lift,
        "ensemble_size": args.ensemble_size,
        "baseline_memory": args.baseline_memory,
    }


def build_learner(sequence, args):
    learner = OnlineAdaptiveHorizon(**model_options(args))
    learner.extend(sequence)
    return learner


def format_window(sequence, n, width=60):
    text = "".join(map(str, sequence[-n:]))
    return text if len(text) <= width else "..." + text[-(width - 3):]


def print_picked_combinations(sequence, forecast, horizon):
    bit = forecast.suggested_target
    label = "Candidate" if forecast.abstained else "Pick"
    print()
    print("Picked window combinations:")
    print("-" * 62)
    print(f"{'N':>4}  {'Observed window':<42}  {label:>9}")
    print("-" * 62)
    for n in forecast.contributing_ns:
        print(f"{n:>4}  {format_window(sequence, n, 42):<42}  {bit:>9}")
    print("-" * 62)
    horizon_label = "candidate" if forecast.abstained else "picked"
    print(
        f"Combined forecast: {horizon_label} bit {bit} -> "
        f"appears within the next {horizon} unseen observations"
    )


def print_forecast(learner, args, update_kind="loaded"):
    sequence = learner.history
    print()
    print(f"Head forecast ({update_kind}) — {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 72)
    print(f"Observations available:     {len(sequence)}")
    print(f"Current head index:         {len(sequence) - 1}")
    unseen_start = len(sequence)
    unseen_end = unseen_start + args.horizon - 1
    print(f"Unseen forecast positions:  {unseen_start} through {unseen_end}")
    print(f"Future horizon:             next {args.horizon} unseen observations")

    if len(sequence) < args.n_min:
        print(f"Status:                     WAITING")
        print(f"Reason:                     need {args.n_min - len(sequence)} more observation(s)")
        return

    forecast = learner.forecast(force_pick=args.force_pick)
    print(f"Representative N:           {forecast.window_n}")
    print(f"Observed window:            {format_window(sequence, forecast.window_n)}")
    print(
        "Contributing N values:      "
        + ", ".join(map(str, forecast.contributing_ns))
    )
    print(f"Ending bit / full streak:   {forecast.features.ending_bit} / {forecast.features.ending_streak_length}")
    print(f"Local transition rate:      {forecast.features.transition_rate:.2%}")
    print(f"Detected drift:             {forecast.drift_score:.2%}")
    print(f"Estimated P(hit):           {forecast.estimated_probability:.4%}")
    print(f"Marginal baseline P(hit):   {forecast.baseline_probability:.4%}")
    print(f"Estimated conditional lift: {forecast.lift:+.4%}")
    print(f"Context confidence:         {forecast.confidence:.2%}")

    if forecast.abstained:
        print("Status:                     NO SIGNAL")
        print(f"Best candidate bit:         {forecast.suggested_target} (not actionable)")
        print(f"Reason:                     {forecast.reason}")
    else:
        print("Status:                     PREDICTION")
        print(f"Pick bit:                   {forecast.target}")
        print(
            f"Prediction:                 bit {forecast.target} appears at least once "
            f"in the next {args.horizon} observations"
        )
        if args.force_pick and forecast.reason.startswith("forced pick"):
            print(f"Warning:                    {forecast.reason}")

    print_picked_combinations(sequence, forecast, args.horizon)


def watch_file(path, learner, args):
    current = list(learner.history)
    print(f"Watching {Path(path).resolve()} every {args.interval:g}s. Press Ctrl+C to stop.")
    while True:
        time.sleep(args.interval)
        try:
            latest = load_sequence(path)
        except (OSError, ValueError) as error:
            print(f"Watch warning: {error}")
            continue
        if latest == current:
            continue
        if len(latest) >= len(current) and latest[:len(current)] == current:
            added = latest[len(current):]
            learner.extend(added)
            update_kind = f"{len(added)} appended"
        else:
            learner = build_learner(latest, args)
            update_kind = "file rewritten; model rebuilt"
        current = list(latest)
        print_forecast(learner, args, update_kind)


def main():
    parser = argparse.ArgumentParser(
        description="Forecast at the newest observation in a growing binary file."
    )
    parser.add_argument("file")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=30)
    parser.add_argument("--memory", type=int, default=150)
    parser.add_argument("--baseline-memory", type=int, default=300)
    parser.add_argument("--prior-strength", type=float, default=20.0)
    parser.add_argument("--min-resolved", type=int, default=12)
    parser.add_argument("--decay", type=float, default=0.97)
    parser.add_argument("--min-lift", type=float, default=0.015)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--force-pick", action="store_true",
                        help="Always choose the best bit, even without a validated signal.")
    parser.add_argument("--watch", action="store_true",
                        help="Keep watching the file and forecast after appended observations.")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Watch polling interval in seconds (default: 2).")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be greater than zero")

    try:
        sequence = load_sequence(args.file)
        learner = build_learner(sequence, args)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print_forecast(learner, args)
    if args.watch:
        try:
            watch_file(args.file, learner, args)
        except KeyboardInterrupt:
            print("\nWatch stopped.")


if __name__ == "__main__":
    main()
