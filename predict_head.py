#!/usr/bin/env python3
"""Train on an append-only binary sequence and forecast from its latest head."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

from adaptive_horizon import (
    OnlineAdaptiveHorizon,
    parse_binary_sequence,
    walk_forward_backtest,
)


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
    return text if len(text) <= width else "..." + text[-(width - 3) :]


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


def historical_rows(sequence, args):
    return walk_forward_backtest(
        sequence=sequence,
        n_min=args.n_min,
        n_max=args.n_max,
        horizon=args.horizon,
        performance_memory=args.memory,
        prior_strength=args.prior_strength,
        min_resolved_per_n=args.min_resolved,
        warmup=args.warmup,
        step=args.horizon if args.non_overlap else 1,
        decay=args.decay,
        min_lift=args.min_lift,
        ensemble_size=args.ensemble_size,
        baseline_memory=args.baseline_memory,
    )


def summarize_historical_performance(rows):
    predictions = [row for row in rows if not row.abstained]
    hits = sum(row.hit for row in predictions)

    def target_summary(target):
        selected = [row for row in predictions if row.target == target]
        target_hits = sum(row.hit for row in selected)
        return len(selected), target_hits

    picked_zero, zero_hits = target_summary(0)
    picked_one, one_hits = target_summary(1)
    turnout_zero = sum(0 in row.actual_future for row in predictions)
    turnout_one = sum(1 in row.actual_future for row in predictions)
    return {
        "opportunities": len(rows),
        "predictions": len(predictions),
        "no_signal": len(rows) - len(predictions),
        "hits": hits,
        "misses": len(predictions) - hits,
        "picked_zero": picked_zero,
        "zero_hits": zero_hits,
        "picked_one": picked_one,
        "one_hits": one_hits,
        "turnout_zero": turnout_zero,
        "turnout_one": turnout_one,
    }


def _rate(hits, total):
    return f"{hits / total:.2%}" if total else "n/a"


def print_historical_performance(rows):
    result = summarize_historical_performance(rows)
    predicted = result["predictions"]
    print()
    print("Previous prediction performance:")
    print("-" * 62)
    print(f"Resolved opportunities:      {result['opportunities']}")
    print(f"Predictions made:            {predicted}")
    print(f"ABSTAIN / abstained:       {result['no_signal']}")
    print(f"Coverage:                    {_rate(predicted, result['opportunities'])}")
    print(f"Hits:                        {result['hits']}")
    print(f"Misses:                      {result['misses']}")
    print(f"Overall success rate:        {_rate(result['hits'], predicted)}")
    print(
        f"When bit 0 was picked:       {result['zero_hits']}/{result['picked_zero']} "
        f"hits ({_rate(result['zero_hits'], result['picked_zero'])})"
    )
    print(
        f"When bit 1 was picked:       {result['one_hits']}/{result['picked_one']} "
        f"hits ({_rate(result['one_hits'], result['picked_one'])})"
    )
    print("Actual horizon turnouts on the same predicted rows:")
    print(
        f"  0 appeared:                {result['turnout_zero']}/{predicted} "
        f"({_rate(result['turnout_zero'], predicted)})"
    )
    print(
        f"  1 appeared:                {result['turnout_one']}/{predicted} "
        f"({_rate(result['turnout_one'], predicted)})"
    )
    print("-" * 62)


def print_history_continuation(sequence, rows, forecast, args):
    if args.show_predictions == 0:
        return
    displayed = rows if args.show_predictions < 0 else rows[-args.show_predictions :]
    print()
    print(f"Resolved predictions and head continuation ({len(displayed)} historical):")
    print("-" * 108)
    print(
        f"{'At t':>6} {'N':>3}  {'Observed window':<30} {'Pick':>6} "
        f"{'P(hit)':>8} {'Lift':>8} {'Horizon reveal':<15} Result"
    )
    print("-" * 108)
    for row in displayed:
        window = sequence[row.index - row.window_n + 1 : row.index + 1]
        window_text = "".join(map(str, window))
        if len(window_text) > 30:
            window_text = "..." + window_text[-27:]
        reveal = " ".join(map(str, row.actual_future))
        if row.abstained:
            pick, probability, lift, result = "-", "-", "-", "ABSTAIN"
        else:
            pick = str(row.target)
            probability = f"{row.probability:.3f}"
            lift = f"{row.lift:+.3f}"
            result = "HIT" if row.hit else "MISS"
        print(
            f"{row.index:>6} {row.window_n:>3}  {window_text:<30} {pick:>6} "
            f"{probability:>8} {lift:>8} {reveal:<15} {result}"
        )

    print("." * 108)
    pick = forecast.suggested_target if forecast.abstained else forecast.target
    result = "ABSTAIN" if forecast.abstained else "PENDING"
    unseen = " ".join("?" for _ in range(args.horizon))
    print(
        f"{'HEAD':>6} {forecast.window_n:>3}  "
        f"{format_window(sequence, forecast.window_n, 30):<30} {pick:>6} "
        f"{forecast.estimated_probability:>8.3f} {forecast.lift:>+8.3f} "
        f"{unseen:<15} {result}"
    )
    print("-" * 108)
    print(
        "Resolved rows reveal their outcomes; HEAD points beyond the dataset and remains pending."
    )


def print_forecast(learner, args, update_kind="loaded"):
    sequence = learner.history
    print()
    print(
        f"Head forecast ({update_kind}) — {datetime.now().isoformat(timespec='seconds')}"
    )
    print("=" * 72)
    print(f"Observations available:     {len(sequence)}")
    print(f"Current head index:         {len(sequence) - 1}")
    unseen_start = len(sequence)
    unseen_end = unseen_start + args.horizon - 1
    print(f"Unseen forecast positions:  {unseen_start} through {unseen_end}")
    print(f"Future horizon:             next {args.horizon} unseen observations")

    if len(sequence) < args.n_min:
        print(f"Status:                     WAITING")
        print(
            f"Reason:                     need {args.n_min - len(sequence)} more observation(s)"
        )
        return

    forecast = learner.forecast(force_pick=args.force_pick)
    rows = historical_rows(sequence, args)
    print_historical_performance(rows)
    print_history_continuation(sequence, rows, forecast, args)
    print()
    print("Current head decision:")
    print(f"Representative N:           {forecast.window_n}")
    print(f"Observed window:            {format_window(sequence, forecast.window_n)}")
    print(
        "Contributing N values:      " + ", ".join(map(str, forecast.contributing_ns))
    )
    print(
        f"Ending bit / full streak:   {forecast.features.ending_bit} / {forecast.features.ending_streak_length}"
    )
    print(f"Local transition rate:      {forecast.features.transition_rate:.2%}")
    print(f"Detected drift:             {forecast.drift_score:.2%}")
    print(f"Estimated P(hit):           {forecast.estimated_probability:.4%}")
    print(f"Marginal baseline P(hit):   {forecast.baseline_probability:.4%}")
    print(f"Estimated conditional lift: {forecast.lift:+.4%}")
    print(f"Context confidence:         {forecast.confidence:.2%}")

    if forecast.abstained:
        print("Status:                     ABSTAIN")
        print(
            f"Best candidate bit:         {forecast.suggested_target} (not actionable)"
        )
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
    print(
        f"Watching {Path(path).resolve()} every {args.interval:g}s. Press Ctrl+C to stop."
    )
    while True:
        time.sleep(args.interval)
        try:
            latest = load_sequence(path)
        except (OSError, ValueError) as error:
            print(f"Watch warning: {error}")
            continue
        if latest == current:
            continue
        if len(latest) >= len(current) and latest[: len(current)] == current:
            added = latest[len(current) :]
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
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--show-predictions",
        type=int,
        default=20,
        metavar="COUNT",
        help="Recent resolved predictions to show; -1 shows all, 0 hides them.",
    )
    parser.add_argument(
        "--non-overlap",
        action="store_true",
        help="Show conservative historical rows every H observations.",
    )
    parser.add_argument(
        "--force-pick",
        action="store_true",
        help="Always choose the best bit, even without a validated signal.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep watching the file and forecast after appended observations.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Watch polling interval in seconds (default: 2).",
    )
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
