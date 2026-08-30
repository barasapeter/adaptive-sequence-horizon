#!/usr/bin/env python3
"""
run_experiments.py

Research experiment runner for Adaptive Horizon.

Examples:
    python run_experiments.py data.txt
    python run_experiments.py data.txt --horizon 5
    python run_experiments.py data.txt --horizon 4 --non-overlap
    python run_experiments.py data.txt --scan-horizons 1 8
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

from adaptive_horizon import (
    parse_binary_sequence,
    walk_forward_backtest,
)


def load_sequence(path: str):
    text = Path(path).read_text(encoding="utf-8")
    return parse_binary_sequence(text)


def constant_target_rate(sequence, target, horizon, warmup, step):
    hits = 0
    total = 0

    for t in range(warmup, len(sequence) - horizon, step):
        future = sequence[t + 1 : t + 1 + horizon]
        hits += int(target in future)
        total += 1

    return hits / total if total else float("nan")


def summarize(rows):
    total = len(rows)
    hits = sum(r.hit for r in rows)

    brier = mean(
        (r.probability - r.hit) ** 2
        for r in rows
    ) if rows else float("nan")

    n_counts = Counter(r.window_n for r in rows)

    return {
        "predictions": total,
        "hits": hits,
        "misses": total - hits,
        "hit_rate": hits / total if total else float("nan"),
        "mean_probability": mean(r.probability for r in rows)
            if rows else float("nan"),
        "brier_score": brier,
        "n_counts": n_counts,
    }


def print_prediction_trace(sequence, rows, limit):
    """Show how each displayed forecast was made and resolved."""
    if limit == 0 or not rows:
        return

    displayed = rows if limit < 0 else rows[-limit:]

    print()
    if len(displayed) < len(rows):
        print(
            f"Prediction trace (last {len(displayed)} of {len(rows)}; "
            "use --show-predictions -1 for all):"
        )
    else:
        print(f"Prediction trace ({len(displayed)} predictions):")

    print("-" * 78)
    print(
        f"{'At t':>6}  {'N':>3}  {'Observed window':<30}  "
        f"{'Pick':>4}  {'Horizon reveal':<16}  Result"
    )
    print("-" * 78)

    for row in displayed:
        window = sequence[row.index - row.window_n + 1 : row.index + 1]
        window_text = "".join(map(str, window))
        if len(window_text) > 30:
            window_text = "..." + window_text[-27:]

        future_text = " ".join(map(str, row.actual_future))
        result = "HIT" if row.hit else "MISS"

        print(
            f"{row.index:>6}  {row.window_n:>3}  {window_text:<30}  "
            f"{row.target:>4}  {future_text:<16}  {result}"
        )

    print("-" * 78)
    print("HIT = chosen bit appeared in the horizon reveal; MISS = it did not.")


def print_result(sequence, horizon, args):
    step = horizon if args.non_overlap else 1

    rows = walk_forward_backtest(
        sequence=sequence,
        n_min=args.n_min,
        n_max=args.n_max,
        horizon=horizon,
        performance_memory=args.memory,
        prior_strength=args.prior_strength,
        min_resolved_per_n=args.min_resolved,
        warmup=args.warmup,
        step=step,
    )

    result = summarize(rows)

    baseline_0 = constant_target_rate(
        sequence,
        target=0,
        horizon=horizon,
        warmup=args.warmup,
        step=step,
    )

    baseline_1 = constant_target_rate(
        sequence,
        target=1,
        horizon=horizon,
        warmup=args.warmup,
        step=step,
    )

    print()
    print(f"Horizon H = {horizon}")
    print("=" * 48)
    print(f"Predictions:              {result['predictions']}")
    print(f"Hits:                     {result['hits']}")
    print(f"Misses:                   {result['misses']}")
    print(f"Adaptive hit rate:        {result['hit_rate']:.4%}")
    print(f"Always target 0:          {baseline_0:.4%}")
    print(f"Always target 1:          {baseline_1:.4%}")
    print(f"Mean predicted P(hit):    {result['mean_probability']:.4%}")
    print(f"Brier score:              {result['brier_score']:.6f}")

    print()
    print("Most selected N values:")
    for n, count in result["n_counts"].most_common(12):
        print(f"  N={n:>2}  {count:>5} selections")

    print_prediction_trace(sequence, rows, args.show_predictions)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("file")
    parser.add_argument("--horizon", type=int, default=4)

    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=30)
    parser.add_argument("--memory", type=int, default=75)
    parser.add_argument("--prior-strength", type=float, default=25.0)
    parser.add_argument("--min-resolved", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--show-predictions",
        type=int,
        default=20,
        metavar="COUNT",
        help=(
            "Number of recent prediction traces to display "
            "(default: 20; 0 hides them; -1 shows all)."
        ),
    )

    parser.add_argument(
        "--non-overlap",
        action="store_true",
        help="Evaluate only every H observations.",
    )

    parser.add_argument(
        "--scan-horizons",
        nargs=2,
        type=int,
        metavar=("MIN_H", "MAX_H"),
        help="Run the same experiment across a range of horizons.",
    )

    args = parser.parse_args()

    sequence = load_sequence(args.file)

    print(f"Loaded observations: {len(sequence)}")
    print(f"Zero count:          {sequence.count(0)}")
    print(f"One count:           {sequence.count(1)}")

    if args.scan_horizons:
        h_min, h_max = args.scan_horizons

        for horizon in range(h_min, h_max + 1):
            print_result(sequence, horizon, args)
    else:
        print_result(sequence, args.horizon, args)


if __name__ == "__main__":
    main()
