# Adaptive Horizon

Adaptive Horizon is a leakage-safe, online experiment for finding temporary
predictive structure in binary sequences. It asks whether `0` or `1` will
appear at least once in the next `H` observations, conditional on the local
state visible at prediction time.

The model is deliberately allowed to say **NO SIGNAL**. A high raw hit rate is
not evidence of skill when `H > 1`; for balanced independent bits, the chance
that either chosen bit appears within the horizon is already `1 - 0.5^H`.
The primary diagnostics are therefore lift, Brier skill, log loss,
calibration, coverage, confidence intervals, and shuffled-null results.

## Quick start

PowerShell:

```powershell
python run_experiments.py dataset/trail-bb4f6830.txt `
  --horizon 5 `
  --n-min 2 `
  --n-max 30 `
  --memory 150 `
  --non-overlap
```

Bash:

```bash
python run_experiments.py dataset/trail-bb4f6830.txt \
  --horizon 5 \
  --n-min 2 \
  --n-max 30 \
  --memory 150 \
  --non-overlap
```

Use `--show-predictions -1` to print every test opportunity, `0` to hide the
trace, or a positive number to show that many recent opportunities.

## Forecast event

At position `t`, the model sees only:

```text
S[0:t]
```

It may pick `X = 0`, pick `X = 1`, or abstain. A prediction is a hit when:

```text
X appears in S[t+1:t+H+1]
```

For example:

```text
observed state     pick       hidden horizon      result
0010011100           1            0 0 1 0           HIT
0010011100           1            0 0 0 0           MISS
```

The complete horizon stays hidden until it has resolved. Only then can its
outcome enter model memory.

## Implemented approach

### 1. Multi-scale local states

For every candidate `N` from `n_min` through `n_max`, the latest `N`
observations are represented by:

- zero and one counts;
- normalized local bias;
- transition rate;
- ending bit;
- full terminal streak length;
- longest streak inside the window; and
- an exact suffix of up to six bits.

The state is indexed at several resolutions: exact suffix, detailed shape,
coarse shape, run state, and window-only. Specific matches carry more weight;
broader levels provide backoff when a state is rare.

This is contextual learning. Outcomes from a high-transition alternating
window are no longer treated as equivalent to outcomes from a constant window
merely because both used the same `N`.

### 2. Both targets are learned

The old fixed “pick the local minority” rule has been removed. For every `N`,
the model separately estimates:

```text
P(0 appears within H | current state, N)
P(1 appears within H | current state, N)
```

It evaluates both targets from resolved historical outcomes. Target selection
is based on conditional improvement over that target's recent marginal
baseline, not on local majority or minority alone.

### 3. Bayesian backoff and recent evidence

Sparse contextual estimates are shrunk toward a recent target-specific
marginal baseline. Evidence is exponentially decayed so recent outcomes matter
more than old outcomes. When adjacent recent segments have different means,
the measured drift score accelerates forgetting.

Marginal probabilities use beta smoothing. They never become exactly zero or
one, including when all observations seen so far are identical.

### 4. Ensemble across `N`

The model does not trust a single lucky window length. For each target it
ranks candidate windows by conditional lift and reliability, then combines the
best `--ensemble-size` estimates. The trace reports a representative `N`; the
forecast itself is an ensemble and records all contributing values.

The selected target is the ensemble with the strongest estimated lift over
its own baseline.

### 5. Abstention

A forecast becomes **NO SIGNAL** when:

- the best context lacks `--min-resolved` effective observations; or
- estimated lift is below `--min-lift`.

This prevents forced guesses from being presented as discovered structure.
Coverage reports the fraction of opportunities on which a forecast was made.

## Long streak handling

Long sequences such as:

```text
000000000000000000000000...
111111111111111111111111...
```

are supported explicitly:

- terminal streak length is measured across the entire visible history, not
  clipped to `N`;
- streak lengths use logarithmic buckets, so very long runs remain distinct
  without creating one state per possible length;
- the terminal streak is calculated once per forecast time, rather than once
  per candidate window;
- beta smoothing prevents impossible `0.0` or `1.0` priors; and
- broad run/window backoff remains available when an exact streak state has
  not occurred before.

A constant sequence may still produce **NO SIGNAL** under normal thresholds.
That is intentional: always choosing its dominant bit already has an extremely
strong marginal baseline, so repeating that baseline is not conditional lift.

## Walk-forward evaluation

The implementation follows this order at every historical position:

```text
1. Resolve only predictions whose complete H observations are now visible.
2. Add those resolved candidate outcomes to model memory.
3. Construct every current local state from history through t only.
4. Record state keys for later delayed learning.
5. Produce a forecast or NO SIGNAL.
6. Reveal the future only to score the saved backtest row.
```

Pending learning records retain the state that existed when they were made.
They are never reconstructed using later information.

By default, metrics are reported only for the final chronological 25% of the
sequence (`--test-fraction 0.25`). The learner remains online: after each test
forecast fully resolves, that past outcome may train later forecasts. This is
prequential evaluation, not random train/test splitting.

`--non-overlap` evaluates every `H` positions and is the preferred conservative
report because overlapping horizons share observations. Overlapping mode is
available for higher-resolution exploratory traces.

## Reported metrics

- **Coverage**: predictions divided by all eligible opportunities.
- **Hit rate**: hits divided by predictions, excluding abstentions.
- **Mean predicted lift**: forecast probability minus recent marginal
  probability. This is an estimate, not realized performance.
- **Brier score**: squared probability error; lower is better.
- **Brier skill**: improvement over the target-specific marginal forecast;
  positive is better.
- **Log loss**: probability loss that strongly penalizes confident mistakes.
- **Calibration bins**: predicted probabilities versus observed frequencies.
- **Observed lift block CI**: a block-bootstrap interval for
  `hit - marginal_probability`, using blocks of at least `H`.
- **Constant-target rates**: always-zero and always-one results on exactly the
  rows where the adaptive model chose to predict.

Example trace:

```text
  At t   N  Observed window                  Pick   P(hit)     Lift Reveal          Result
------------------------------------------------------------------------------------------------------------
   124   8  01001110                            1    0.821   +0.047 0 0 1 0         HIT
   128   5  11111                               -        -        - 1 1 1 1         NO SIGNAL
```

## Shuffled-null validation

The entire adaptive search can find lucky structure even in random data. Use:

```powershell
python run_experiments.py dataset/trail-bb4f6830.txt `
  --horizon 5 `
  --non-overlap `
  --null-runs 100
```

Each null run shuffles the sequence and reruns the complete walk-forward
selection procedure. The report compares real Brier skill with the null skill
distribution and prints a plus-one-corrected empirical p-value. Null runs can
be slow, so the default is zero; use many runs before interpreting the value.

## Important options

| Option | Default | Meaning |
|---|---:|---|
| `--horizon` | `4` | Future event horizon `H` |
| `--n-min` / `--n-max` | `2` / `30` | Candidate local windows |
| `--memory` | `150` | Maximum outcomes retained per context |
| `--baseline-memory` | `300` | Recent observations for marginal priors |
| `--prior-strength` | `20` | Strength of Bayesian shrinkage |
| `--min-resolved` | `12` | Evidence needed before issuing a signal |
| `--decay` | `0.97` | Base exponential outcome decay |
| `--min-lift` | `0.015` | Minimum conditional improvement to predict |
| `--ensemble-size` | `5` | Window estimates combined per target |
| `--test-fraction` | `0.25` | Final chronological fraction reported |
| `--bootstrap-samples` | `500` | Block-bootstrap repetitions; `0` disables |
| `--null-runs` | `0` | Full shuffled-sequence runs |
| `--non-overlap` | off | Evaluate every `H` observations |
| `--scan-horizons A B` | off | Exploratory scan from `A` through `B` |

Scanning many horizons is exploratory multiple testing. A horizon selected by
a scan must be confirmed on a later untouched sequence or time period.

## Tests

Run the standard-library test suite:

```powershell
python -m unittest discover -s tests -v
```

The tests cover native and legacy parsing, abstention without evidence,
invalid input, long terminal streak measurement, and walk-forward behavior on
long all-zero and all-one sequences.

## Remaining assumptions and limitations

The implementation is a research model, not proof that a dataset is
predictable. It still assumes that the chosen feature representation groups
usefully similar states, recent outcomes have some persistence, exponential
decay is appropriate, and the event “appears at least once within `H`” is the
right objective.

State bins, `N`, `H`, decay, memory, lift threshold, and ensemble size are all
research choices. Tuning them repeatedly on the same final period invalidates
that period as a test. Strong evidence requires replication on later data,
positive skill rather than merely high hit rate, sensible calibration,
confidence intervals excluding zero, and performance exceeding shuffled-null
runs.
