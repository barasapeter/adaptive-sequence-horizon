#!/usr/bin/env python3
"""
adaptive_horizon.py

Adaptive Horizon
Peter Barasa

Experimental adaptive local-context predictor for binary sequences.

Binary convention:
    0 and 1

Forecast objective:
    Select a target bit X and estimate whether X will appear at least once
    within the next H observations.

At every historical point, future observations remain hidden until the
prediction is committed and later resolved.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence, Tuple
import re


@dataclass
class Forecast:
    index: int
    window_n: int
    target: int
    estimated_probability: float
    raw_recent_probability: float
    recent_samples: int
    zero_count: int
    one_count: int
    local_bias: float
    transitions: int
    ending_bit: int
    ending_streak_length: int


@dataclass
class PendingPrediction:
    made_at: int
    resolve_at: int
    window_n: int
    target: int


@dataclass
class BacktestRow:
    index: int
    window_n: int
    target: int
    probability: float
    actual_future: Tuple[int, ...]
    hit: int


def parse_binary_sequence(text: str) -> List[int]:
    """
    Parse either native 0/1 data or legacy L/P data.

    Mapping for legacy data:
        L -> 0
        P -> 1
    """
    text = text.upper()

    if "L" in text or "P" in text:
        chars = re.findall(r"[LP]", text)
        if not chars:
            raise ValueError("No binary observations found.")
        return [0 if c == "L" else 1 for c in chars]

    bits = re.findall(r"[01]", text)
    if not bits:
        raise ValueError("No binary observations found.")
    return [int(x) for x in bits]


def opposite(bit: int) -> int:
    return 1 - bit


def local_target(window: Sequence[int]) -> int:
    """
    Initial target-selection rule.

    If zeros dominate locally, target 1.
    If ones dominate locally, target 0.
    If tied, target the opposite of the most recent bit.
    """
    zeros = window.count(0)
    ones = len(window) - zeros

    if zeros > ones:
        return 1
    if ones > zeros:
        return 0
    return opposite(window[-1])


def count_transitions(window: Sequence[int]) -> int:
    return sum(a != b for a, b in zip(window, window[1:]))


def ending_streak(window: Sequence[int]) -> Tuple[int, int]:
    bit = window[-1]
    length = 1

    for x in reversed(window[:-1]):
        if x == bit:
            length += 1
        else:
            break

    return bit, length


def beta_shrunk_probability(
    hits: int,
    trials: int,
    prior_mean: float,
    prior_strength: float,
) -> float:
    alpha = prior_mean * prior_strength
    beta = (1.0 - prior_mean) * prior_strength
    return (hits + alpha) / (trials + alpha + beta)


class AdaptiveHorizon:
    """
    Adaptive window selector.

    For each candidate N:
        1. Build a local window of length N.
        2. Select a candidate target bit.
        3. Evaluate that N using recent resolved historical outcomes.
        4. Apply Bayesian shrinkage.
        5. Select the best current N.
    """

    def __init__(
        self,
        n_min: int = 2,
        n_max: int = 30,
        horizon: int = 4,
        performance_memory: int = 75,
        prior_strength: float = 25.0,
        min_resolved_per_n: int = 10,
    ):
        if n_min < 1:
            raise ValueError("n_min must be >= 1")
        if n_max < n_min:
            raise ValueError("n_max must be >= n_min")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        self.n_min = n_min
        self.n_max = n_max
        self.horizon = horizon
        self.performance_memory = performance_memory
        self.prior_strength = prior_strength
        self.min_resolved_per_n = min_resolved_per_n

        self.recent: Dict[int, Deque[int]] = {
            n: deque(maxlen=performance_memory)
            for n in range(n_min, n_max + 1)
        }

    def baseline_probability(
        self,
        history: Sequence[int],
        target: int,
    ) -> float:
        """
        Independence-style marginal baseline:
            P(target appears at least once in H)
        """
        if not history:
            p = 0.5
        else:
            p = history.count(target) / len(history)

        return 1.0 - (1.0 - p) ** self.horizon

    def score_window(
        self,
        n: int,
        history: Sequence[int],
        target: int,
    ) -> Tuple[float, float, int]:
        outcomes = self.recent[n]
        trials = len(outcomes)
        hits = sum(outcomes)

        baseline = self.baseline_probability(history, target)

        if trials == 0:
            return baseline, baseline, 0

        raw = hits / trials

        shrunk = beta_shrunk_probability(
            hits=hits,
            trials=trials,
            prior_mean=baseline,
            prior_strength=self.prior_strength,
        )

        if trials < self.min_resolved_per_n:
            weight = trials / self.min_resolved_per_n
            shrunk = weight * shrunk + (1.0 - weight) * baseline

        return shrunk, raw, trials

    def forecast(self, history: Sequence[int]) -> Forecast:
        if len(history) < self.n_min:
            raise ValueError("Not enough history.")

        best = None

        for n in range(
            self.n_min,
            min(self.n_max, len(history)) + 1,
        ):
            window = list(history[-n:])
            target = local_target(window)

            score, raw, samples = self.score_window(
                n=n,
                history=history,
                target=target,
            )

            key = (score, samples, -n)

            if best is None or key > best[0]:
                zeros = window.count(0)
                ones = n - zeros
                ending_bit, streak_length = ending_streak(window)

                best = (
                    key,
                    Forecast(
                        index=len(history) - 1,
                        window_n=n,
                        target=target,
                        estimated_probability=score,
                        raw_recent_probability=raw,
                        recent_samples=samples,
                        zero_count=zeros,
                        one_count=ones,
                        local_bias=(ones - zeros) / n,
                        transitions=count_transitions(window),
                        ending_bit=ending_bit,
                        ending_streak_length=streak_length,
                    ),
                )

        return best[1]

    def update(self, n: int, hit: int) -> None:
        self.recent[n].append(int(bool(hit)))


def walk_forward_backtest(
    sequence: Sequence[int],
    n_min: int = 2,
    n_max: int = 30,
    horizon: int = 4,
    performance_memory: int = 75,
    prior_strength: float = 25.0,
    min_resolved_per_n: int = 10,
    warmup: int = 100,
    step: int = 1,
) -> List[BacktestRow]:
    """
    Strict walk-forward backtest.

    A historical prediction is added to model memory only after its complete
    future horizon has become observable.
    """
    seq = list(sequence)

    model = AdaptiveHorizon(
        n_min=n_min,
        n_max=n_max,
        horizon=horizon,
        performance_memory=performance_memory,
        prior_strength=prior_strength,
        min_resolved_per_n=min_resolved_per_n,
    )

    pending: List[PendingPrediction] = []
    rows: List[BacktestRow] = []

    start = max(n_max - 1, 0)

    for t in range(start, len(seq) - horizon):

        unresolved = []

        for item in pending:
            if item.resolve_at <= t:
                future = seq[
                    item.made_at + 1 :
                    item.made_at + 1 + horizon
                ]
                hit = int(item.target in future)
                model.update(item.window_n, hit)
            else:
                unresolved.append(item)

        pending = unresolved

        history = seq[: t + 1]

        for n in range(
            n_min,
            min(n_max, len(history)) + 1,
        ):
            target = local_target(history[-n:])

            pending.append(
                PendingPrediction(
                    made_at=t,
                    resolve_at=t + horizon,
                    window_n=n,
                    target=target,
                )
            )

        if t < warmup:
            continue

        if (t - warmup) % step != 0:
            continue

        forecast = model.forecast(history)

        future = tuple(
            seq[t + 1 : t + 1 + horizon]
        )

        hit = int(forecast.target in future)

        rows.append(
            BacktestRow(
                index=t,
                window_n=forecast.window_n,
                target=forecast.target,
                probability=forecast.estimated_probability,
                actual_future=future,
                hit=hit,
            )
        )

    return rows
