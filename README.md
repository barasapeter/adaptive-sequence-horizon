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

## Prediction beyond the dataset head

`run_experiments.py` remains the historical testing tool. To feed one of the
same dataset files into the trained historical learner and predict beyond its
last available entry, use the separate head predictor:

```powershell
python predict_head.py dataset/trail-bb4f6830.txt `
  --horizon 5 `
  --n-min 2 `
  --n-max 30 `
  --non-overlap `
  --show-predictions 20
```

It treats the file as the complete information currently available. It replays
that history without look-ahead, learns only from horizons that resolve inside
the dataset, and forecasts beyond the final entry. The next `H` positions do
not exist in the input and remain unseen.

The command first prints recent resolved historical predictions with their
horizon reveals and `HIT`/`MISS` results. It then appends the current `HEAD`
row. Its reveal contains question marks and remains `PENDING` because those
observations lie beyond the supplied dataset.

The output includes:

- the current head index;
- representative and contributing `N` values;
- the exact observed window;
- ending bit and full streak length;
- drift and transition measurements;
- selected bit, predicted probability, marginal baseline, and lift; and
- either `PREDICTION` or `NO SIGNAL`.

If the same dataset file later receives more observations, it can optionally be
watched in place:

```powershell
python predict_head.py dataset/trail-bb4f6830.txt --horizon 5 --watch --interval 2
```

On a pure append, only new observations are fed to the in-memory learner. If
earlier data changes or the file is truncated, the watcher detects that it is
no longer append-only and safely rebuilds the learner from the new contents.

By default the live command refuses to present a weak candidate as a signal.
If an external process absolutely requires a `0` or `1` on every invocation,
use:

```powershell
python predict_head.py dataset/trail-bb4f6830.txt --horizon 5 --force-pick
```

Forced output is explicitly marked with a warning when the normal evidence or
lift requirement was not met. It is the best available candidate, not a
validated signal. With fewer than `--n-min` observations, no forecast is
mathematically available even in forced mode.

The live learner retains only recent contextual outcome deques plus the latest
`H` unresolved state records. A 1,000-observation file is replayed directly;
after that, append-mode memory does not grow with the full number of historical
forecast records.

### Understanding `predict_head.py` output

Example:

```text
Head forecast (loaded)
========================================================================
Observations available:     959
Current head index:         958
Unseen forecast positions:  959 through 963
Future horizon:             next 5 unseen observations

Previous prediction performance:
--------------------------------------------------------------
Resolved opportunities:      171
Predictions made:            52
No signal / abstained:       119
Coverage:                    30.41%
Hits:                        49
Misses:                      3
Overall success rate:        94.23%
When bit 0 was picked:       21/23 hits (91.30%)
When bit 1 was picked:       28/29 hits (96.55%)
Actual horizon turnouts on the same predicted rows:
  0 appeared:                48/52 (92.31%)
  1 appeared:                50/52 (96.15%)
--------------------------------------------------------------

Resolved predictions and head continuation (3 historical):
------------------------------------------------------------------------------------------------------------
  At t   N  Observed window                  Pick   P(hit)     Lift Horizon reveal  Result
------------------------------------------------------------------------------------------------------------
   940   4  1101                                1    0.975   +0.020 1 1 1 1 0       HIT
   945   5  11110                               -        -        - 1 0 1 1 0       NO SIGNAL
   950   6  010110                              -        -        - 0 1 1 1 0       NO SIGNAL
............................................................................................................
  HEAD   3  001                                 1    0.976   +0.017 ? ? ? ? ?       PENDING
------------------------------------------------------------------------------------------------------------

Current head decision:
Representative N:           3
Observed window:            001
Contributing N values:      3, 11, 15, 17, 13
Ending bit / full streak:   1 / 1
Local transition rate:      50.00%
Detected drift:             6.00%
Estimated P(hit):           97.6447%
Marginal baseline P(hit):   95.9547%
Estimated conditional lift: +1.6900%
Context confidence:         69.06%
Status:                     PREDICTION
Pick bit:                   1
Prediction:                 bit 1 appears at least once in the next 5 observations

Picked window combinations:
--------------------------------------------------------------
   N  Observed window                                  Pick
--------------------------------------------------------------
   3  001                                                 1
  11  11001110001                                         1
  15  101011001110001                                     1
  17  11101011001110001                                   1
  13  1011001110001                                       1
--------------------------------------------------------------
Combined forecast: picked bit 1 -> appears within the next 5 unseen observations
```

`Representative N` is the strongest individual contextual window for display.
`Contributing N values` are the windows actually combined into the target
ensemble. The prediction is therefore not based only on the representative
window.

The `Picked window combinations` table shows the actual suffix used by every
contributing `N` and the common bit selected by their combined target model.
When normal signal requirements are not met, the final column is labeled
`Candidate` and the combined line is explicitly non-actionable.

`Previous prediction performance` uses all eligible resolved historical rows,
not only the few rows printed in the trace. It separates picks of `0` and `1`
and compares them with the actual horizon turnout rates on exactly the same
predicted rows. A horizon can contain both bits, so the two turnout percentages
do not need to sum to 100%. Coverage is shown because success rate excludes
`NO SIGNAL` opportunities and should not be interpreted without knowing how
often the model chose to predict.

`Estimated P(hit)` is the probability that the picked bit appears at least
once in the next `H` entries. `Marginal baseline P(hit)` is the corresponding
recent frequency-based probability without local-state conditioning. Their
difference is `Estimated conditional lift`.

`Context confidence` measures the amount of resolved, recency-weighted
evidence available for the contributing contexts. It is not the probability
that the forecast is correct.

When the status is `NO SIGNAL`, `Best candidate bit` is printed for diagnosis
but should not be treated as a prediction. `--force-pick` promotes that best
candidate to an operational pick and prints a warning if normal requirements
were not satisfied.

### `predict_head.py` options

| Option | Default | Meaning |
|---|---:|---|
| `file` | required | Growing text file containing `0`/`1` or legacy `L`/`P` observations |
| `--horizon` | `4` | Number of future observations in which the bit must appear |
| `--n-min` / `--n-max` | `2` / `30` | Candidate local window sizes |
| `--memory` | `150` | Resolved outcomes retained per contextual state |
| `--baseline-memory` | `300` | Recent observations used by marginal baselines |
| `--prior-strength` | `20` | Bayesian shrinkage toward the marginal baseline |
| `--min-resolved` | `12` | Effective contextual evidence required for a normal signal |
| `--decay` | `0.97` | Recency decay applied to resolved outcomes |
| `--min-lift` | `0.015` | Minimum improvement over baseline required to predict |
| `--ensemble-size` | `5` | Number of candidate windows combined per target |
| `--warmup` | `100` | Historical position before displayed predictions begin |
| `--show-predictions` | `20` | Recent resolved rows before `HEAD`; `-1` all, `0` none |
| `--non-overlap` | off | Display conservative historical rows every `H` positions |
| `--watch` | off | Continue watching the file for changes |
| `--interval` | `2` | Seconds between file checks in watch mode |
| `--force-pick` | off | Return the best bit even when evidence requirements fail |

### Live operating sequence

For each newly appended observation, `predict_head.py` performs:

```text
1. Append the new 0 or 1 to visible history.
2. Resolve old state records whose complete H-step future is now visible.
3. Add only those resolved outcomes to contextual memory.
4. Record the new head state for resolution after H more observations.
5. Determine the local-state ensemble and preferred target.
6. Emit PREDICTION or NO SIGNAL for the future beyond the current head.
```

Stopping and restarting watch mode is safe. The file is replayed chronologically
on startup, reconstructing the same resolved contextual memory before the new
head forecast is produced.

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
............................................................................................................
  HEAD   3  001                                  1    0.976   +0.017 ? ? ? ? ?       PENDING
```

The final `HEAD` row is a continuation of the resolved historical trace. Its
observed window and picked bit are known, but its horizon lies beyond the final
dataset entry. Question marks represent those unseen observations, and
`PENDING` cannot become `HIT` or `MISS` until the next `H` values arrive. The
head ensemble combinations are printed immediately below the trace. If the
head does not meet evidence requirements, the row says `NO SIGNAL` instead.

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
