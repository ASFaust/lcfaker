"""Common normalisation and plain descriptors of learning curves, shared by all analyses.

Normalisation (per task), so that sources with different metrics can be compared:

    z = (y - best) / (ref - best)

where `best` is the lowest finite value anywhere in the task and `ref` is the 90th percentile of the configs'
first values, i.e. a bad starting level (TaskSet: the loss at initialisation; classifiers: about chance level).
So z = 0 is the best value seen in the task and z = 1 is a bad start. Non-finite values (diverged runs) and values
above CAP are set to CAP; replicas that are NaN everywhere are padding and stay NaN.

Curve classes (per config, on its replica-mean curve, smoothed with a short moving average `s` so that noise does
not decide the class; final = mean of the last 10% of the unsmoothed curve):
    diverged  : any point was non-finite or above CAP, or it ends clearly worse than it started (final > s[0] + 0.1)
    dead      : never gets meaningfully below the bad level: min(s) > DEAD (0.9)
    turns     : improves, then gets clearly worse again: final - min(s) > max(0.05, 0.2 * (s[0] - min(s)))
                (overfitting or late instability)
    improving : everything else
"""
from __future__ import annotations

import warnings

import numpy as np

CAP = 2.0
FLAT = 0.05
DEAD = 0.9
CLASSES = ("improving", "dead", "turns", "diverged")


def normalise(y: np.ndarray) -> tuple[np.ndarray, dict]:
    """y [n, R, T] or [n, T] (lower is better) -> z of the same shape, and the reference values used."""
    fin = np.isfinite(y)
    best = np.min(y[fin]) if fin.any() else 0.0
    first = y[..., 0]
    first = first[np.isfinite(first)]
    start = np.quantile(first, 0.9) if first.size else np.nan
    if not np.isfinite(start) or start - best <= 1e-12:
        start = np.max(y[fin]) if fin.any() else best + 1.0
    scale = max(start - best, 1e-12)
    with np.errstate(all="ignore"):
        z = (y - best) / scale
    pad = np.isnan(y).all(-1, keepdims=True)          # a replica that does not exist (NaN padding): stays NaN
    bad = (~np.isfinite(z) | (z > CAP)) & ~pad
    z = np.where(bad, CAP, z).astype(np.float32)
    z[np.broadcast_to(pad, z.shape)] = np.nan
    return z, {"best": float(best), "start": float(start), "bad_frac": float(bad.sum() / max(1, (~pad).sum() * y.shape[-1]))}


def mean_curve(z: np.ndarray) -> np.ndarray:
    """[n, R, T] -> [n, T], mean over present replicas (NaN padding ignored)."""
    if z.ndim == 2:
        return z
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(z, axis=1)


def smooth(m: np.ndarray) -> np.ndarray:
    """Centred moving average over k = max(3, T // 25) points (shrinking window at the ends), [n, T]."""
    T = m.shape[1]
    k = max(3, T // 25) | 1
    h = k // 2
    c = np.concatenate([np.zeros((len(m), 1)), np.cumsum(m, 1, dtype=np.float64)], 1)
    lo = np.clip(np.arange(T) - h, 0, T)
    hi = np.clip(np.arange(T) + h + 1, 0, T)
    return ((c[:, hi] - c[:, lo]) / (hi - lo)).astype(np.float32)


def final_value(m: np.ndarray) -> np.ndarray:
    k = max(1, m.shape[1] // 10)
    return m[:, -k:].mean(1)


def classify(m: np.ndarray) -> np.ndarray:
    """m [n, T] normalised mean curves -> class index into CLASSES, [n]."""
    s = smooth(m)
    start, final, low = s[:, 0], final_value(m), s.min(1)
    gain = start - low
    div = (final > start + 0.1) | (m >= CAP).any(1)
    dead = low > DEAD
    turns = final - low > np.maximum(0.05, 0.2 * gain)
    c = np.zeros(len(m), int)
    c[turns] = 2
    c[dead] = 1
    c[div] = 3
    return c


def time_to_fraction(m: np.ndarray, frac: float) -> np.ndarray:
    """Fraction of the budget (0 = first point, 1 = last) at which a curve's smoothed version first gets `frac` of
    its total gain s[0] - min(s). NaN for curves with gain < FLAT."""
    s = smooth(m)
    start, low = s[:, 0], s.min(1)
    gain = start - low
    hit = s <= (start - frac * gain)[:, None] + 1e-9
    out = hit.argmax(1) / (m.shape[1] - 1)
    out[gain < FLAT] = np.nan
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def task_stats(m: np.ndarray) -> dict:
    """Task-level descriptors of normalised mean curves m [n, T]."""
    c = classify(m)
    T = m.shape[1]
    out = {f"frac_{k}": float((c == i).mean()) for i, k in enumerate(CLASSES)}
    live = c != 1
    fin = final_value(m)
    out["final_q10"], out["final_q50"], out["final_q90"] = (float(q) for q in np.quantile(fin, [0.1, 0.5, 0.9]))
    out["final_spread"] = float(np.quantile(fin, 0.9) - np.quantile(fin, 0.1))
    t50 = time_to_fraction(m, 0.5)
    t90 = time_to_fraction(m, 0.9)
    out["t50_median"] = float(np.nanmedian(t50)) if np.isfinite(t50).any() else np.nan
    out["t90_median"] = float(np.nanmedian(t90)) if np.isfinite(t90).any() else np.nan
    # how well early performance predicts final performance (what freeze-thaw exploits): rank correlation between
    # the value at 10% / 30% of the budget and the final value, over all configs and over live configs only
    for f in (0.1, 0.3):
        k = max(1, int(round(f * (T - 1))))
        out[f"rank_early{int(f * 100)}"] = spearman(m[:, k], fin)
        out[f"rank_early{int(f * 100)}_live"] = spearman(m[live, k], fin[live]) if live.sum() > 10 else np.nan
    # among the best 10% at the end, which fraction was already in the best 10% at 10% of the budget
    k = max(1, int(round(0.1 * (T - 1))))
    top_end = fin <= np.quantile(fin, 0.1)
    top_early = m[:, k] <= np.quantile(m[:, k], 0.1)
    out["top10_kept"] = float((top_end & top_early).sum() / max(1, top_end.sum()))
    return out
