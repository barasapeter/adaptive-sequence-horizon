"""
head_predictor.py

Focuses entirely on the HEAD regime (the most recent / still-open regime
at the end of the trail) and answers one question:

    "Will a chosen bit appear at least once in the next horizon (n=5)?"

Approach: run several independent, simple tactics (each a different way
of reading the trail - global stats, recent-window stats, momentum/EWMA,
adaptive local pattern matching, first-order Markov, current-regime
majority). Combine them with a FIXED-SHARE weighted-experts algorithm
(a variant of the classic Hedge/Weighted-Majority online-learning
algorithm, designed specifically for switching/regime environments):
each tactic's voting weight grows when it's been right and shrinks when
it's been wrong, with a small "share" redistributed back toward the
average each step so the ensemble can adapt if a previously-weak tactic
becomes strong again after a new regime starts.

The whole ensemble is itself backtested walk-forward on the trail before
being pointed at the live, unresolved head regime.
"""
import math
from collections import Counter

HORIZON = 5
WARMUP = 50
ETA = 0.5          # learning rate for weight updates
ALPHA_SHARE = 0.02 # fixed-share mixing back toward uniform each step
DEFAULT_PATH = 'trail-c70cd348.txt'


def load_sequence(path):
    with open(path) as f:
        raw = f.read()
    return [x.strip() for x in raw.split(',') if x.strip()]


# ---------------- individual tactics (all causal: history only) ----------------
def tactic_global(history):
    return Counter(history).most_common(1)[0][0]

def tactic_recent(history, window=100):
    w = history[-window:] if len(history) > window else history
    return Counter(w).most_common(1)[0][0]

def tactic_ewma(history, halflife=30):
    alpha = 1 - 2 ** (-1 / halflife)
    p = 0.5
    for s in history:
        p = alpha * (1 if s == 'P' else 0) + (1 - alpha) * p
    return 'P' if p >= 0.5 else 'L'

def tactic_markov1(history):
    if len(history) < 2:
        return tactic_global(history)
    last = history[-1]
    followers = Counter()
    for i in range(len(history) - 1):
        if history[i] == last:
            followers[history[i + 1]] += 1
    if not followers:
        return tactic_global(history)
    return followers.most_common(1)[0][0]

def page_hinkley(seq, symbol='P', delta=0.005, lam=6.0, min_window=20):
    x = [1 if s == symbol else 0 for s in seq]
    n = len(x)
    if n == 0:
        return []
    m_min, ph = 0.0, 0.0
    changepoints, window_start = [], 0
    running_sum, count = x[0], 1
    for i in range(1, n):
        count += 1
        running_sum += x[i]
        mean = running_sum / count
        ph += (x[i] - mean - delta)
        m_min = min(m_min, ph)
        if (ph - m_min) > lam and (i - window_start) >= min_window:
            changepoints.append(i)
            window_start, running_sum, count, ph, m_min = i, x[i], 1, 0.0, 0.0
    return changepoints

def tactic_regime_majority(history):
    cps = page_hinkley(history)
    start = cps[-1] if cps else 0
    return Counter(history[start:]).most_common(1)[0][0]

def zscore_skew(count_majority, total):
    if total == 0:
        return 0.0
    p_hat = count_majority / total
    return (p_hat - 0.5) / math.sqrt(0.25 / total)

def tactic_ngram(history, k_max=8, k_min=1, min_support=6, z_threshold=1.0):
    n = len(history)
    for k in range(min(k_max, n), k_min - 1, -1):
        context = tuple(history[n - k:n])
        followers = Counter()
        for i in range(n - k):
            if tuple(history[i:i + k]) == context:
                followers[history[i + k]] += 1
        total = sum(followers.values())
        if total >= min_support:
            maj_sym, maj_count = followers.most_common(1)[0]
            if abs(zscore_skew(maj_count, total)) >= z_threshold:
                return maj_sym
    return tactic_global(history)


TACTICS = [
    ("global_majority",   tactic_global),
    ("recent100",         lambda h: tactic_recent(h, 100)),
    ("ewma30",            lambda h: tactic_ewma(h, 30)),
    ("ngram_backoff",     tactic_ngram),
    ("markov1",           tactic_markov1),
    ("regime_majority",   tactic_regime_majority),
]


# ---------------- fixed-share weighted-experts backtest ----------------
def run_ensemble(seq, tactics=TACTICS, horizon=HORIZON, warmup=WARMUP,
                  eta=ETA, alpha_share=ALPHA_SHARE):
    n = len(seq)
    k = len(tactics)
    weights = [1.0 / k] * k
    ensemble_hits = 0
    scored = 0
    tactic_hits = [0] * k

    for t in range(warmup, n - horizon):
        history = seq[:t]
        preds = [fn(history) for _, fn in tactics]

        vote_P = sum(w for w, p in zip(weights, preds) if p == 'P')
        vote_L = sum(w for w, p in zip(weights, preds) if p == 'L')
        ensemble_bit = 'P' if vote_P >= vote_L else 'L'

        window = seq[t:t + horizon]
        ensemble_hit = ensemble_bit in window
        ensemble_hits += ensemble_hit
        scored += 1

        # update weights: reward tactics whose OWN pick hit
        new_weights = []
        for i, p in enumerate(preds):
            hit = p in window
            tactic_hits[i] += hit
            w = weights[i] * math.exp(eta if hit else -eta)
            new_weights.append(w)
        total = sum(new_weights)
        new_weights = [w / total for w in new_weights]
        # fixed-share: blend back toward uniform so the ensemble can adapt
        # if a currently-weak tactic becomes strong again in a new regime
        weights = [(1 - alpha_share) * w + alpha_share * (1 / k) for w in new_weights]

    return {
        "weights": weights,
        "ensemble_hit_rate": ensemble_hits / scored if scored else None,
        "tactic_hit_rates": [h / scored for h in tactic_hits] if scored else None,
        "scored": scored,
    }


def trivial_hit_prob(p_bit, horizon=HORIZON):
    return 1 - (1 - p_bit) ** horizon


def main(path):
    seq = load_sequence(path)
    n = len(seq)
    p_L, p_P = seq.count('L') / n, seq.count('P') / n

    result = run_ensemble(seq)
    weights = result["weights"]

    print(f"Sequence length: {n}  (base rates L={p_L:.3f}, P={p_P:.3f})")
    print(f"Trivial baseline hit-prob (chance alone): "
          f"L={trivial_hit_prob(p_L):.3f}, P={trivial_hit_prob(p_P):.3f}")

    print(f"\nEnsemble backtest over {result['scored']} walk-forward predictions:")
    print(f"  Ensemble (fixed-share weighted vote) hit rate: {result['ensemble_hit_rate']:.4f}")
    print(f"  Individual tactic hit rates:")
    for (name, _), hr, w in zip(TACTICS, result["tactic_hit_rates"], weights):
        print(f"    {name:<16} hit_rate={hr:.4f}   final_weight={w:.3f}")

    # ---- HEAD REGIME ANALYSIS ----
    cps = page_hinkley(seq)
    head_start = cps[-1] if cps else 0
    head = seq[head_start:]
    print(f"\n--- HEAD REGIME: seq[{head_start}:{n}] = {''.join(head)} "
          f"(length {len(head)}) ---")
    head_counts = Counter(head)
    print(f"  Internal composition: {dict(head_counts)}")

    # each tactic's live prediction, using the FULL trail as history
    preds = [(name, fn(seq)) for name, fn in TACTICS]
    print(f"\n  Tactic votes (using full trail as history):")
    for (name, bit), w in zip(preds, weights):
        print(f"    {name:<16} picks '{bit}'   (weight {w:.3f})")

    vote_P = sum(w for (_, p), w in zip(preds, weights) if p == 'P')
    vote_L = sum(w for (_, p), w in zip(preds, weights) if p == 'L')
    final_bit = 'P' if vote_P >= vote_L else 'L'
    confidence = max(vote_P, vote_L) / (vote_P + vote_L)

    # supporting evidence: how did regimes with the SAME majority bit as
    # the head resolve historically? (small-sample corroboration, not a vote)
    regimes_bounds = [0] + cps + [n]
    analogs = []
    for a, b in zip(regimes_bounds, regimes_bounds[1:]):
        if b == n:
            continue  # skip the head itself
        maj = Counter(seq[a:b]).most_common(1)[0][0]
        if maj == final_bit:
            fut = seq[b:b + HORIZON]
            if len(fut) == HORIZON:
                analogs.append(final_bit in fut)
    if analogs:
        print(f"\n  Corroborating evidence: {sum(analogs)}/{len(analogs)} historical regimes "
              f"whose majority was also '{final_bit}' saw '{final_bit}' appear in their "
              f"following horizon ({sum(analogs)/len(analogs):.1%}).")

    print(f"\n=== FINAL HEAD PREDICTION ===")
    print(f"Selected bit: {final_bit}")
    print(f"Ensemble vote share: {confidence:.1%}  (P weight={vote_P:.3f}, L weight={vote_L:.3f})")
    print(f"Question: does '{final_bit}' appear at least once in the next {HORIZON} outcomes?")
    print(f"Outcome: ?   <-- unresolved, verify once new data arrives")
    print(f"(For reference: trivial chance-alone hit-probability for this bit "
          f"= {trivial_hit_prob(p_P if final_bit=='P' else p_L):.3f}; "
          f"this ensemble's own backtested hit rate = {result['ensemble_hit_rate']:.3f})")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)
