#!/usr/bin/env python3
"""Leakage-safe adaptive local-state forecasting for binary sequences."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import exp, log2
import re
from typing import Deque, Dict, Hashable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class StateFeatures:
    """Features known at forecast time for one candidate window length."""

    window_n: int
    zero_count: int
    one_count: int
    local_bias: float
    transition_rate: float
    ending_bit: int
    ending_streak_length: int
    longest_streak: int
    suffix_code: int
    suffix_length: int


@dataclass
class CandidateScore:
    window_n: int
    target: int
    probability: float
    baseline_probability: float
    lift: float
    effective_samples: float
    reliability: float
    features: StateFeatures


@dataclass
class Forecast:
    index: int
    window_n: int
    target: Optional[int]
    estimated_probability: float
    baseline_probability: float
    lift: float
    effective_samples: float
    confidence: float
    contributing_ns: Tuple[int, ...]
    abstained: bool
    reason: str
    drift_score: float
    features: StateFeatures


@dataclass
class PendingCandidate:
    window_n: int
    target: int
    state_keys: Tuple[Hashable, ...]


@dataclass
class PendingPrediction:
    made_at: int
    resolve_at: int
    candidates: Tuple[PendingCandidate, ...]


@dataclass
class BacktestRow:
    index: int
    window_n: int
    target: Optional[int]
    probability: float
    baseline_probability: float
    lift: float
    confidence: float
    contributing_ns: Tuple[int, ...]
    actual_future: Tuple[int, ...]
    hit: Optional[int]
    abstained: bool
    reason: str
    drift_score: float


def parse_binary_sequence(text: str) -> List[int]:
    """Parse native 0/1 data or legacy L/P data (L=0, P=1)."""
    text = text.upper()
    if "L" in text or "P" in text:
        chars = re.findall(r"[LP]", text)
        if not chars:
            raise ValueError("No binary observations found.")
        return [0 if char == "L" else 1 for char in chars]

    bits = re.findall(r"[01]", text)
    if not bits:
        raise ValueError("No binary observations found.")
    return [int(bit) for bit in bits]


def count_transitions(window: Sequence[int]) -> int:
    return sum(left != right for left, right in zip(window, window[1:]))


def ending_streak(sequence: Sequence[int]) -> Tuple[int, int]:
    if not sequence:
        raise ValueError("A streak requires at least one observation.")
    bit = sequence[-1]
    length = 1
    for value in reversed(sequence[:-1]):
        if value != bit:
            break
        length += 1
    return bit, length


def longest_streak(window: Sequence[int]) -> int:
    if not window:
        return 0
    longest = current = 1
    for left, right in zip(window, window[1:]):
        current = current + 1 if left == right else 1
        longest = max(longest, current)
    return longest


def streak_bucket(length: int) -> int:
    """Log bucket with no scan over the streak, safe for very long runs."""
    return 0 if length <= 0 else int(log2(length))


def state_features(
    history: Sequence[int],
    n: int,
    ending_info: Optional[Tuple[int, int]] = None,
) -> StateFeatures:
    if n < 1 or len(history) < n:
        raise ValueError("Window length must fit inside history.")

    window = history[-n:]
    zeros = window.count(0)
    ones = n - zeros
    end_bit, full_streak = ending_info or ending_streak(history)
    suffix_length = min(6, n)
    suffix_code = 0
    for bit in window[-suffix_length:]:
        suffix_code = (suffix_code << 1) | bit

    return StateFeatures(
        window_n=n,
        zero_count=zeros,
        one_count=ones,
        local_bias=(ones - zeros) / n,
        transition_rate=count_transitions(window) / max(1, n - 1),
        ending_bit=end_bit,
        ending_streak_length=full_streak,
        longest_streak=longest_streak(window),
        suffix_code=suffix_code,
        suffix_length=suffix_length,
    )


def _bucket(value: float, bins: int) -> int:
    return max(0, min(bins, int(round(value * bins))))


def context_keys(features: StateFeatures, target: int) -> Tuple[Hashable, ...]:
    """Multi-resolution state keys, from specific to broad."""
    ones_ratio = features.one_count / features.window_n
    fine_bias = _bucket(ones_ratio, 8)
    fine_transitions = _bucket(features.transition_rate, 8)
    coarse_bias = _bucket(ones_ratio, 4)
    coarse_transitions = _bucket(features.transition_rate, 4)
    run = streak_bucket(features.ending_streak_length)
    longest = streak_bucket(features.longest_streak)
    n = features.window_n

    return (
        (
            "suffix", n, target, fine_bias, fine_transitions,
            features.ending_bit, run, features.suffix_length,
            features.suffix_code,
        ),
        ("shape", n, target, fine_bias, fine_transitions,
         features.ending_bit, run, longest),
        ("coarse", n, target, coarse_bias, coarse_transitions,
         features.ending_bit, run),
        ("run", n, target, features.ending_bit, run),
        ("window", n, target),
    )


def beta_shrunk_probability(
    hits: float,
    trials: float,
    prior_mean: float,
    prior_strength: float,
) -> float:
    alpha = prior_mean * prior_strength
    beta = (1.0 - prior_mean) * prior_strength
    return (hits + alpha) / (trials + alpha + beta)


class AdaptiveHorizon:
    """Online contextual, dual-target, multi-window ensemble."""

    def __init__(
        self,
        n_min: int = 2,
        n_max: int = 30,
        horizon: int = 4,
        performance_memory: int = 150,
        prior_strength: float = 20.0,
        min_resolved_per_n: int = 12,
        decay: float = 0.97,
        min_lift: float = 0.015,
        ensemble_size: int = 5,
        baseline_memory: int = 300,
    ):
        if n_min < 1:
            raise ValueError("n_min must be >= 1")
        if n_max < n_min:
            raise ValueError("n_max must be >= n_min")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if performance_memory < 1 or baseline_memory < 1:
            raise ValueError("memory values must be >= 1")
        if not 0 < decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1")

        self.n_min = n_min
        self.n_max = n_max
        self.horizon = horizon
        self.performance_memory = performance_memory
        self.prior_strength = prior_strength
        self.min_resolved_per_n = min_resolved_per_n
        self.decay = decay
        self.min_lift = min_lift
        self.ensemble_size = ensemble_size
        self.baseline_memory = baseline_memory
        self.outcomes: Dict[Hashable, Deque[int]] = defaultdict(
            lambda: deque(maxlen=performance_memory)
        )

    def baseline_probability(self, history: Sequence[int], target: int) -> float:
        """Recent marginal baseline with beta smoothing, converted to H-event."""
        recent = history[-self.baseline_memory:]
        target_count = recent.count(target)
        # Beta(1, 1) prevents zero/one probabilities during constant streaks.
        marginal = (target_count + 1.0) / (len(recent) + 2.0)
        return 1.0 - (1.0 - marginal) ** self.horizon

    def drift_score(self, history: Sequence[int]) -> float:
        """Difference between adjacent recent means, scaled to [0, 1]."""
        width = min(50, len(history) // 2)
        if width < 5:
            return 0.0
        old = history[-2 * width:-width]
        new = history[-width:]
        return min(1.0, abs(sum(new) / width - sum(old) / width))

    def _weighted_stats(self, values: Sequence[int], drift: float) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        # Drift accelerates forgetting; constant regimes remain numerically stable.
        decay = max(0.80, self.decay - 0.12 * drift)
        weight = 1.0
        hits = trials = 0.0
        for outcome in reversed(values):
            hits += weight * outcome
            trials += weight
            weight *= decay
        return hits, trials

    def score_candidate(
        self,
        history: Sequence[int],
        features: StateFeatures,
        target: int,
        drift: float,
    ) -> CandidateScore:
        baseline = self.baseline_probability(history, target)
        estimates = []

        # Specific levels receive more weight when they have evidence. Broad
        # levels provide graceful backoff for rare states and very long streaks.
        specificity = (1.8, 1.5, 1.2, 1.0, 0.8)
        for key, importance in zip(context_keys(features, target), specificity):
            hits, trials = self._weighted_stats(self.outcomes[key], drift)
            if trials <= 0:
                continue
            probability = beta_shrunk_probability(
                hits, trials, baseline, self.prior_strength
            )
            evidence = trials / (trials + self.min_resolved_per_n)
            estimates.append((probability, evidence * importance, trials))

        if estimates:
            total_weight = sum(weight for _, weight, _ in estimates)
            probability = sum(p * weight for p, weight, _ in estimates) / total_weight
            effective_samples = max(samples for _, _, samples in estimates)
        else:
            probability = baseline
            effective_samples = 0.0

        lift = probability - baseline
        reliability = effective_samples / (
            effective_samples + self.min_resolved_per_n
        )
        return CandidateScore(
            window_n=features.window_n,
            target=target,
            probability=probability,
            baseline_probability=baseline,
            lift=lift,
            effective_samples=effective_samples,
            reliability=reliability,
            features=features,
        )

    def forecast(self, history: Sequence[int]) -> Forecast:
        if len(history) < self.n_min:
            raise ValueError("Not enough history.")

        drift = self.drift_score(history)
        ending_info = ending_streak(history)
        candidates: List[CandidateScore] = []
        for n in range(self.n_min, min(self.n_max, len(history)) + 1):
            features = state_features(history, n, ending_info)
            for target in (0, 1):
                candidates.append(
                    self.score_candidate(history, features, target, drift)
                )

        target_ensembles = {}
        for target in (0, 1):
            ranked = sorted(
                (item for item in candidates if item.target == target),
                key=lambda item: (item.lift, item.reliability, -item.window_n),
                reverse=True,
            )[: self.ensemble_size]
            weights = [max(0.05, item.reliability) for item in ranked]
            total_weight = sum(weights)
            probability = sum(
                item.probability * weight
                for item, weight in zip(ranked, weights)
            ) / total_weight
            baseline = sum(
                item.baseline_probability * weight
                for item, weight in zip(ranked, weights)
            ) / total_weight
            confidence = sum(
                item.reliability * weight
                for item, weight in zip(ranked, weights)
            ) / total_weight
            target_ensembles[target] = (
                probability - baseline,
                probability,
                baseline,
                confidence,
                ranked,
            )

        target = max(
            (0, 1),
            key=lambda bit: (
                target_ensembles[bit][0],
                target_ensembles[bit][3],
                target_ensembles[bit][1],
                -bit,
            ),
        )
        lift, probability, baseline, confidence, ranked = target_ensembles[target]
        representative = ranked[0]
        enough_evidence = representative.effective_samples >= self.min_resolved_per_n
        abstained = not enough_evidence or lift < self.min_lift
        if not enough_evidence:
            reason = "insufficient contextual evidence"
        elif lift < self.min_lift:
            reason = "estimated lift below threshold"
        else:
            reason = "signal"

        return Forecast(
            index=len(history) - 1,
            window_n=representative.window_n,
            target=None if abstained else target,
            estimated_probability=probability,
            baseline_probability=baseline,
            lift=lift,
            effective_samples=representative.effective_samples,
            confidence=confidence,
            contributing_ns=tuple(item.window_n for item in ranked),
            abstained=abstained,
            reason=reason,
            drift_score=drift,
            features=representative.features,
        )

    def pending_candidates(self, history: Sequence[int]) -> Tuple[PendingCandidate, ...]:
        result = []
        ending_info = ending_streak(history)
        for n in range(self.n_min, min(self.n_max, len(history)) + 1):
            features = state_features(history, n, ending_info)
            for target in (0, 1):
                result.append(
                    PendingCandidate(n, target, context_keys(features, target))
                )
        return tuple(result)

    def update(self, candidate: PendingCandidate, hit: int) -> None:
        outcome = int(bool(hit))
        for key in candidate.state_keys:
            self.outcomes[key].append(outcome)


def walk_forward_backtest(
    sequence: Sequence[int],
    n_min: int = 2,
    n_max: int = 30,
    horizon: int = 4,
    performance_memory: int = 150,
    prior_strength: float = 20.0,
    min_resolved_per_n: int = 12,
    warmup: int = 100,
    step: int = 1,
    decay: float = 0.97,
    min_lift: float = 0.015,
    ensemble_size: int = 5,
    baseline_memory: int = 300,
) -> List[BacktestRow]:
    """Strict walk-forward evaluation with delayed outcome resolution."""
    seq = list(sequence)
    if any(value not in (0, 1) for value in seq):
        raise ValueError("Sequence values must be 0 or 1.")
    if step < 1:
        raise ValueError("step must be >= 1")

    model = AdaptiveHorizon(
        n_min=n_min,
        n_max=n_max,
        horizon=horizon,
        performance_memory=performance_memory,
        prior_strength=prior_strength,
        min_resolved_per_n=min_resolved_per_n,
        decay=decay,
        min_lift=min_lift,
        ensemble_size=ensemble_size,
        baseline_memory=baseline_memory,
    )
    pending: List[PendingPrediction] = []
    rows: List[BacktestRow] = []
    start = max(n_max - 1, 0)
    history = seq[:start]

    for t in range(start, len(seq) - horizon):
        history.append(seq[t])
        unresolved = []
        for item in pending:
            if item.resolve_at <= t:
                future = seq[item.made_at + 1:item.made_at + 1 + horizon]
                for candidate in item.candidates:
                    model.update(candidate, int(candidate.target in future))
            else:
                unresolved.append(item)
        pending = unresolved

        pending.append(
            PendingPrediction(
                made_at=t,
                resolve_at=t + horizon,
                candidates=model.pending_candidates(history),
            )
        )

        if t < warmup or (t - warmup) % step != 0:
            continue

        forecast = model.forecast(history)
        future = tuple(seq[t + 1:t + 1 + horizon])
        hit = None if forecast.abstained else int(forecast.target in future)
        rows.append(
            BacktestRow(
                index=t,
                window_n=forecast.window_n,
                target=forecast.target,
                probability=forecast.estimated_probability,
                baseline_probability=forecast.baseline_probability,
                lift=forecast.lift,
                confidence=forecast.confidence,
                contributing_ns=forecast.contributing_ns,
                actual_future=future,
                hit=hit,
                abstained=forecast.abstained,
                reason=forecast.reason,
                drift_score=forecast.drift_score,
            )
        )

    return rows
