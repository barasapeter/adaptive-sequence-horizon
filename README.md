# Adaptive Horizon

Run:
```pwsh
python run_experiments.py dataset/trail-bb4f6830.txt `
  --horizon 5 `
  --n-min 2 `
  --n-max 30 `
  --memory 75
```

```shell
python run_experiments.py dataset/trail-bb4f6830.txt \
    --horizon 5 \
    --n-min 2 \
    --n-max 30 \
    --memory 75
```

**Adaptive Horizon** is an experimental algorithm for discovering and evaluating local predictive structure in binary sequences.

The project explores a simple question:

> Given the most recent portion of a binary sequence, can an algorithm adaptively determine which local context is currently informative and estimate the probability that a selected target bit will appear at least once within a future horizon?

Rather than assuming that an entire sequence follows one fixed pattern, Adaptive Horizon focuses on **local behavior**.

The central hypothesis is that a sequence may appear approximately random when viewed globally while still contain temporary local regimes where certain patterns, streak structures, transition rates, or binary imbalances provide useful information about the near future.

## The Problem

Consider a binary sequence:

```text
0, 1, 0, 0, 1, 1, 1, 0, 1, 0, ...
```

At some point in the sequence, only the observations up to the current position are known.

The algorithm must choose:

* a local sample size `N`
* a target bit `X` (`0` or `1`)
* a future horizon `H`

For example:

```text
Current local sample:

0 1 1 0 0 1 1 1 0 0
                  ↑
               current
```

Suppose the algorithm selects:

```text
N = 10
X = 1
H = 4
```

The prediction is not necessarily that the **next bit** will be `1`.

Instead, the event being predicted is:

> **Will `1` appear at least once somewhere within the next four observations?**

If the hidden future is:

```text
0 0 1 0
```

the prediction succeeds because `1` appeared at least once.

If the hidden future is:

```text
0 0 0 0
```

the prediction fails because `1` was completely absent from the horizon.

Formally:

```text
success = X ∈ {S[t+1], ..., S[t+H]}
```

The corresponding probability of interest is:

```text
P(X appears at least once within the next H observations | current local state)
```

An equivalent and often useful formulation is:

```text
1 - P(X is completely absent from the next H observations | current local state)
```

## Why Local Context?

A global binary sequence may contain approximately balanced frequencies while still move through different local structures.

For example:

```text
0000001111
```

and

```text
0101010100
```

can contain similar numbers of `0`s and `1`s, but their internal structures are very different.

Useful local characteristics may include:

* frequency of `0` and `1`
* local binary imbalance
* transition frequency
* ending streak
* streak length
* longest streak
* recent concentration of a bit
* alternating behavior
* continuation behavior
* reversal behavior

Adaptive Horizon therefore does not assume that one fixed sample size is appropriate for the entire sequence.

The useful amount of history may itself change over time.

## Adaptive Sample Size

Instead of fixing:

```text
N = 10
```

the algorithm searches across candidate local windows:

```text
N ∈ {2, 3, 4, ..., N_max}
```

At each prediction point, different values of `N` represent different views of the current sequence.

For example:

```text
N = 3    → very short-term state
N = 8    → short local regime
N = 15   → medium local regime
N = 30   → broader recent regime
```

The objective is to determine which local scale currently contains the most useful historical information.

This means `N` is part of the prediction problem rather than simply a configuration constant.

## Adaptive Target Selection

The initial implementation uses local binary imbalance as a simple target-selection rule.

If a window contains more `0`s than `1`s, the candidate target is `1`.

If a window contains more `1`s than `0`s, the candidate target is `0`.

For example:

```text
Window:
0 0 1 0 0 1 0 0 1 0

0 count = 7
1 count = 3

Candidate target:
X = 1
```

This provides a simple local mean-reversion hypothesis that can be tested objectively.

However, this is only the starting model.

A more complete Adaptive Horizon implementation should learn both:

```text
N = appropriate local context
```

and:

```text
X = appropriate target for that local state
```

from historical observations.

The target should therefore eventually be selected by comparing:

```text
P(0 appears within H | local state)
```

against:

```text
P(1 appears within H | local state)
```

and selecting the bit with the stronger historically validated probability.

## Local State

A local window can be represented by more than its raw sequence.

For a window of size `N`, a state representation may contain:

```text
State(N) = {
    window_size,
    zero_count,
    one_count,
    local_bias,
    transition_count,
    transition_rate,
    ending_bit,
    ending_streak_length,
    longest_zero_streak,
    longest_one_streak,
    recent_bit_frequencies
}
```

This allows structurally similar local patterns to be compared even when their exact sequences are not identical.

For example, two 15-bit sequences may both represent:

```text
moderate 0 bias
high transition rate
ending in 11
short maximum streaks
```

even though the literal 15-bit strings differ.

This is important because exact long patterns become increasingly rare as `N` grows.

## Prediction Horizon

The horizon `H` is also an important part of the experiment.

For:

```text
H = 1
```

the problem becomes ordinary next-bit prediction.

For:

```text
H = 4
```

the question becomes:

> Will target `X` occur at least once in the next four observations?

Different horizons can be evaluated:

```text
H = 1
H = 2
H = 3
H = 4
H = 5
...
```

A local pattern may contain little information about the immediately following bit while still changing the probability that a bit occurs somewhere within a slightly longer horizon.

For this reason, both `N` and `H` should be treated as experimental variables.

## Walk-Forward Evaluation

Avoiding future information is a fundamental requirement of this project.

At historical position `t`, the algorithm may only use:

```text
S[0:t]
```

The future:

```text
S[t+1:t+H]
```

must remain hidden while the prediction is generated.

The process is:

```text
1. Observe history up to t
2. Analyze candidate local windows
3. Select N
4. Select target X ∈ {0, 1}
5. Estimate P(X appears within H)
6. Record the prediction
7. Reveal the next H observations
8. Score the prediction
9. Move forward
10. Repeat
```

A prediction is considered successful when:

```text
X appears at least once within the next H observations
```

Otherwise it is unsuccessful.

Historical predictions must not influence the learner until their complete future horizon has become observable.

This prevents look-ahead leakage.

## Overlapping and Non-Overlapping Evaluation

Two evaluation modes are useful.

### Overlapping

Generate a prediction at every possible position.

For `H = 4`:

```text
Prediction 1 → positions 101-104
Prediction 2 → positions 102-105
Prediction 3 → positions 103-106
```

This provides the maximum number of observations but neighboring predictions share future outcomes.

### Non-Overlapping

Advance by the complete horizon:

```text
Prediction 1 → positions 101-104
Prediction 2 → positions 105-108
Prediction 3 → positions 109-112
```

This produces fewer observations but provides a stricter evaluation because future blocks do not overlap.

Both should be reported.

## Baselines

A high success rate does not automatically imply predictive information.

With a balanced binary sequence and horizon `H = 4`, even a fixed target has a high probability of occurring at least once:

```text
P(X appears within 4)
= 1 - P(X absent for all 4)
```

Under an independent 50/50 process:

```text
P(X appears within 4)
= 1 - (0.5)^4
= 0.9375
= 93.75%
```

For this reason, Adaptive Horizon must always be compared against simple baselines.

Useful baselines include:

```text
Always select 0
Always select 1
Global majority bit
Local majority bit
Local minority bit
Random target selection
```

The objective is not merely to achieve a high raw success rate.

The important question is:

> Does adaptive local-state selection perform better than appropriate simple baselines on future observations?

## Local Regime Changes

One of the primary ideas being investigated is **non-stationary local behavior**.

The useful window may change:

```text
N = 5
```

during one section of the sequence and later become:

```text
N = 17
```

or:

```text
N = 9
```

Likewise, a local structure that previously favored `0` may later favor `1`.

The algorithm should therefore adapt to recent resolved evidence rather than assuming that relationships discovered earlier remain permanently valid.

This introduces the idea of a rolling performance memory.

For example:

```text
performance_memory = 75
```

means the algorithm gives greater relevance to the behavior of recently resolved local predictions rather than the entire historical sequence.

## Current Algorithm

The initial implementation contains:

* adaptive candidate window sizes
* local binary bias measurement
* local target selection
* rolling performance history
* Bayesian probability shrinkage
* configurable prediction horizon
* strict chronological backtesting
* overlapping evaluation
* non-overlapping evaluation
* live prediction from the end of a supplied sequence

The current implementation should be treated as an experimental baseline rather than the final model.

## Planned Direction

The next stage is to make target selection fully dependent on local state.

Instead of using only:

```text
local majority → select minority bit
```

the algorithm should estimate:

```text
P(0 appears within H | State(N))
```

and:

```text
P(1 appears within H | State(N))
```

for multiple candidate values of `N`.

The eventual decision becomes:

```text
(N*, X*) =
argmax P(X appears within H | State(N))
```

where:

```text
X ∈ {0, 1}
```

subject to sufficient historical support and forward validation.

Potential extensions include:

* transition-density regimes
* streak-state modeling
* similarity-based local-state retrieval
* adaptive horizon selection
* probability calibration
* Bayesian state estimation
* confidence intervals
* minimum-support requirements
* regime-change detection
* recency weighting
* out-of-sample validation
* nested walk-forward model selection

## Research Questions

Adaptive Horizon is intended to investigate questions such as:

1. Can a globally noisy binary sequence contain locally useful predictive structure?
2. Is the optimal historical window size dynamic?
3. Which local features provide information beyond simple `0`/`1` frequency?
4. Do streaks contain continuation or reversal information?
5. Does transition density identify different local regimes?
6. Can the algorithm determine when to favor `0` versus `1`?
7. Which prediction horizons contain the strongest measurable local signal?
8. Do discovered relationships survive genuinely unseen data?
9. Can predicted probabilities be calibrated reliably?
10. Can regime changes be detected quickly enough to remain useful?

## Philosophy

This project is deliberately empirical.

A pattern is not considered useful merely because it looks convincing.

A candidate relationship should:

1. occur often enough to measure,
2. be discovered without future information,
3. produce measurable probability differences,
4. survive chronological validation,
5. outperform appropriate baselines,
6. continue working on previously unseen observations.

The goal is therefore not to prove that a sequence is predictable.

The goal is to determine **whether measurable local predictive structure exists, how long that structure persists, which local scale reveals it, and whether an adaptive algorithm can use that information without seeing the future.**

## Status

**Experimental / Research**

Adaptive Horizon is currently a standalone binary sequence-analysis experiment.

The implementation and methodology are expected to evolve as additional datasets, local-state representations, adaptive window-selection methods, and validation techniques are tested.
