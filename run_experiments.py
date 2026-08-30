#!/usr/bin/env python3
"""Command-line research runner for Adaptive Horizon."""

from __future__ import annotations

import argparse
from collections import Counter
from math import log
from pathlib import Path
import random
from statistics import mean

from adaptive_horizon import parse_binary_sequence, walk_forward_backtest


def load_sequence(path):
    return parse_binary_sequence(Path(path).read_text(encoding="utf-8"))


def _log_loss(probability, outcome):
    probability = min(1.0 - 1e-12, max(1e-12, probability))
    return -(outcome * log(probability) + (1 - outcome) * log(1 - probability))


def evaluation_rows(rows, sequence_length, test_fraction):
    if not 0 < test_fraction <= 1:
        raise ValueError("test_fraction must be in (0, 1]")
    start = int(sequence_length * (1.0 - test_fraction))
    selected = [row for row in rows if row.index >= start]
    return selected or rows


def summarize(rows):
    forecasts = [row for row in rows if not row.abstained]
    hits = sum(row.hit for row in forecasts)
    total = len(forecasts)
    opportunities = len(rows)

    if not forecasts:
        return {
            "opportunities": opportunities,
            "predictions": 0,
            "abstentions": opportunities,
            "coverage": 0.0,
            "hits": 0,
            "misses": 0,
            "hit_rate": float("nan"),
            "mean_probability": float("nan"),
            "mean_lift": float("nan"),
            "brier_score": float("nan"),
            "baseline_brier": float("nan"),
            "brier_skill": float("nan"),
            "log_loss": float("nan"),
            "baseline_log_loss": float("nan"),
            "n_counts": Counter(),
        }

    brier = mean((row.probability - row.hit) ** 2 for row in forecasts)
    baseline_brier = mean(
        (row.baseline_probability - row.hit) ** 2 for row in forecasts
    )
    return {
        "opportunities": opportunities,
        "predictions": total,
        "abstentions": opportunities - total,
        "coverage": total / opportunities if opportunities else 0.0,
        "hits": hits,
        "misses": total - hits,
        "hit_rate": hits / total,
        "mean_probability": mean(row.probability for row in forecasts),
        "mean_lift": mean(row.lift for row in forecasts),
        "brier_score": brier,
        "baseline_brier": baseline_brier,
        "brier_skill": 1.0 - brier / baseline_brier
            if baseline_brier > 0 else float("nan"),
        "log_loss": mean(_log_loss(row.probability, row.hit) for row in forecasts),
        "baseline_log_loss": mean(
            _log_loss(row.baseline_probability, row.hit) for row in forecasts
        ),
        "n_counts": Counter(row.window_n for row in forecasts),
    }


def constant_target_rate(rows, target):
    futures = [row.actual_future for row in rows]
    return mean(target in future for future in futures) if futures else float("nan")


def block_bootstrap_lift_interval(rows, horizon, samples, seed):
    """CI for mean outcome-minus-baseline using contiguous blocks."""
    forecasts = [row for row in rows if not row.abstained]
    values = [row.hit - row.baseline_probability for row in forecasts]
    if samples <= 0 or len(values) < 2:
        return None

    rng = random.Random(seed)
    block = max(1, horizon)
    estimates = []
    for _ in range(samples):
        draw = []
        while len(draw) < len(values):
            start = rng.randrange(len(values))
            draw.extend(values[(start + offset) % len(values)] for offset in range(block))
        estimates.append(mean(draw[:len(values)]))
    estimates.sort()
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    return low, high


def calibration_bins(rows):
    forecasts = [row for row in rows if not row.abstained]
    bins = [[] for _ in range(5)]
    for row in forecasts:
        bins[min(4, int(row.probability * 5))].append(row)
    return [
        (index, len(group), mean(row.probability for row in group),
         mean(row.hit for row in group))
        for index, group in enumerate(bins) if group
    ]


def print_prediction_trace(sequence, rows, limit):
    if limit == 0 or not rows:
        return
    displayed = rows if limit < 0 else rows[-limit:]
    print()
    label = f"last {len(displayed)} of {len(rows)}" if len(displayed) < len(rows) else str(len(rows))
    print(f"Prediction trace ({label} opportunities):")
    print("-" * 108)
    print(
        f"{'At t':>6} {'N':>3}  {'Observed window':<30} {'Pick':>6} "
        f"{'P(hit)':>8} {'Lift':>8} {'Reveal':<15} Result"
    )
    print("-" * 108)
    for row in displayed:
        window = sequence[row.index - row.window_n + 1:row.index + 1]
        window_text = "".join(map(str, window))
        if len(window_text) > 30:
            window_text = "..." + window_text[-27:]
        future_text = " ".join(map(str, row.actual_future))
        if row.abstained:
            pick, probability, lift, result = "-", "-", "-", "NO SIGNAL"
        else:
            pick = str(row.target)
            probability = f"{row.probability:.3f}"
            lift = f"{row.lift:+.3f}"
            result = "HIT" if row.hit else "MISS"
        print(
            f"{row.index:>6} {row.window_n:>3}  {window_text:<30} {pick:>6} "
            f"{probability:>8} {lift:>8} {future_text:<15} {result}"
        )
    print("-" * 108)
    print("Lift is predicted P(hit) minus the target-specific marginal baseline.")


def backtest_kwargs(args, horizon):
    return {
        "n_min": args.n_min,
        "n_max": args.n_max,
        "horizon": horizon,
        "performance_memory": args.memory,
        "prior_strength": args.prior_strength,
        "min_resolved_per_n": args.min_resolved,
        "warmup": args.warmup,
        "step": horizon if args.non_overlap else 1,
        "decay": args.decay,
        "min_lift": args.min_lift,
        "ensemble_size": args.ensemble_size,
        "baseline_memory": args.baseline_memory,
    }


def shuffled_null(sequence, horizon, args, real_skill):
    if args.null_runs <= 0 or real_skill != real_skill:
        return None
    rng = random.Random(args.seed)
    skills = []
    for _ in range(args.null_runs):
        shuffled = list(sequence)
        rng.shuffle(shuffled)
        rows = walk_forward_backtest(shuffled, **backtest_kwargs(args, horizon))
        rows = evaluation_rows(rows, len(shuffled), args.test_fraction)
        skill = summarize(rows)["brier_skill"]
        if skill == skill:
            skills.append(skill)
    if not skills:
        return None
    exceedances = sum(skill >= real_skill for skill in skills)
    # Plus-one correction avoids a misleading zero p-value for finite runs.
    p_value = (exceedances + 1) / (len(skills) + 1)
    return mean(skills), max(skills), p_value


def print_result(sequence, horizon, args):
    all_rows = walk_forward_backtest(sequence, **backtest_kwargs(args, horizon))
    rows = evaluation_rows(all_rows, len(sequence), args.test_fraction)
    result = summarize(rows)
    forecast_rows = [row for row in rows if not row.abstained]

    print()
    print(f"Horizon H = {horizon}")
    print("=" * 58)
    print(f"Chronological test fraction: {args.test_fraction:.1%}")
    print(f"Test opportunities:          {result['opportunities']}")
    print(f"Predictions made:            {result['predictions']}")
    print(f"Abstentions / no signal:     {result['abstentions']}")
    print(f"Coverage:                    {result['coverage']:.2%}")
    print(f"Hits:                        {result['hits']}")
    print(f"Misses:                      {result['misses']}")
    print(f"Adaptive hit rate:           {result['hit_rate']:.4%}")
    print(f"Always target 0 (same rows): {constant_target_rate(forecast_rows, 0):.4%}")
    print(f"Always target 1 (same rows): {constant_target_rate(forecast_rows, 1):.4%}")
    print(f"Mean predicted P(hit):       {result['mean_probability']:.4%}")
    print(f"Mean predicted lift:         {result['mean_lift']:+.4%}")
    print(f"Brier score:                 {result['brier_score']:.6f}")
    print(f"Marginal baseline Brier:     {result['baseline_brier']:.6f}")
    print(f"Brier skill vs baseline:     {result['brier_skill']:+.4%}")
    print(f"Log loss:                    {result['log_loss']:.6f}")
    print(f"Marginal baseline log loss:  {result['baseline_log_loss']:.6f}")

    interval = block_bootstrap_lift_interval(
        rows, horizon, args.bootstrap_samples, args.seed
    )
    if interval:
        print(f"Observed lift 95% block CI:  [{interval[0]:+.4%}, {interval[1]:+.4%}]")

    print()
    print("Most selected representative N values:")
    for n, count in result["n_counts"].most_common(12):
        print(f"  N={n:>2}  {count:>5} predictions")

    print()
    print("Calibration (predicted vs observed):")
    for index, count, predicted, observed in calibration_bins(rows):
        low, high = index * 20, (index + 1) * 20
        print(f"  {low:>2}-{high:<3}%  n={count:>5}  predicted={predicted:.3%}  observed={observed:.3%}")

    null = shuffled_null(sequence, horizon, args, result["brier_skill"])
    if null:
        null_mean, null_best, p_value = null
        print()
        print(f"Shuffled-null runs:          {args.null_runs}")
        print(f"Mean shuffled Brier skill:   {null_mean:+.4%}")
        print(f"Best shuffled Brier skill:   {null_best:+.4%}")
        print(f"Empirical null p-value:      {p_value:.4f}")

    print_prediction_trace(sequence, rows, args.show_predictions)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate contextual local-state forecasts without look-ahead."
    )
    parser.add_argument("file")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=30)
    parser.add_argument("--memory", type=int, default=150)
    parser.add_argument("--baseline-memory", type=int, default=300)
    parser.add_argument("--prior-strength", type=float, default=20.0)
    parser.add_argument("--min-resolved", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--decay", type=float, default=0.97)
    parser.add_argument("--min-lift", type=float, default=0.015)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--null-runs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--show-predictions", type=int, default=20, metavar="COUNT")
    parser.add_argument(
        "--non-overlap", action="store_true",
        help="Evaluate every H observations for a conservative trace.",
    )
    parser.add_argument(
        "--scan-horizons", nargs=2, type=int, metavar=("MIN_H", "MAX_H")
    )
    args = parser.parse_args()
    sequence = load_sequence(args.file)
    if len(sequence) <= args.horizon + args.n_max:
        parser.error("The sequence is too short for n-max plus the horizon.")

    print(f"Loaded observations: {len(sequence)}")
    print(f"Zero count:          {sequence.count(0)}")
    print(f"One count:           {sequence.count(1)}")
    print("Mode:                non-overlapping" if args.non_overlap else "Mode:                overlapping")

    if args.scan_horizons:
        h_min, h_max = args.scan_horizons
        if h_min < 1 or h_max < h_min:
            parser.error("scan horizon range must satisfy 1 <= MIN_H <= MAX_H")
        for horizon in range(h_min, h_max + 1):
            print_result(sequence, horizon, args)
    else:
        print_result(sequence, args.horizon, args)


if __name__ == "__main__":
    main()
