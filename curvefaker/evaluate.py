"""Measure 2 (README): do generated tasks look like real ones? Task descriptors, coverage and precision.

A task is summarised by a descriptor vector computed from its curves y [n, R, T] (lower is better):
    the atlas statistics of curvefaker.curves.task_stats on the normalised curves of ONE seed (replica 0: what an
    optimiser sees, and comparable across sources and fakers with different numbers of seeds), resampled to G points,
    plus noise: median over configs of the robust step-to-step noise of one seed (MAD of first differences of
    replica 0 / sqrt 2, in z units; log10),
    plus landscape_rho (needs the configs X): how predictable the final value is from the config. Gradient-boosted
    trees trained on 200 random configs, Spearman rank correlation with the true final value on up to 500 others.

Coverage: a real task is *covered* if its nearest generated task (Euclidean distance on descriptors standardised by
the real tasks' mean and sd) is closer than its nearest other real task would typically be (the median real-real
nearest-neighbour distance). Precision: the same with roles swapped (generated tasks that resemble some real task).
Use equal numbers of real and generated tasks, since both depend on the counts.
"""
from __future__ import annotations

import numpy as np

from curvefaker import curves

G = 50
DESCRIPTORS = ["frac_dead", "frac_diverged", "frac_turns", "final_q50", "final_spread", "t50_median", "t90_median",
               "rank_early10", "rank_early30_live", "top10_kept", "log_noise", "landscape_rho"]


def resample(m: np.ndarray, g: int = G) -> np.ndarray:
    T = m.shape[-1]
    if T == g:
        return m
    src, dst = np.linspace(0, 1, T), np.linspace(0, 1, g)
    flat = m.reshape(-1, T)
    out = np.stack([np.interp(dst, src, r) if np.isfinite(r).all() else np.full(g, np.nan) for r in flat])
    return out.reshape(*m.shape[:-1], g)


def landscape_rho(X: np.ndarray, final: np.ndarray, seed: int = 0, n_train: int = 200, n_test: int = 500) -> float:
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    n = len(final)
    ntr = min(n_train, n // 2)
    perm = rng.permutation(n)
    tr, te = perm[:ntr], perm[ntr:ntr + n_test]
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=5, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4)
    m.fit(X[tr], final[tr])
    return curves.spearman(m.predict(X[te]), final[te])


def descriptors(y: np.ndarray, X: np.ndarray | None = None) -> dict:
    if y.ndim == 3:
        y = y[:, :1]
    else:
        y = y[:, None]
    z, _ = curves.normalise(y)
    m = resample(curves.mean_curve(z))
    ok_m = np.isfinite(m).all(1)
    m = m[ok_m]
    d = curves.task_stats(m)
    d["landscape_rho"] = landscape_rho(X[ok_m], curves.final_value(m)) if X is not None else np.nan
    r0 = z[:, 0]
    ok = np.isfinite(r0).all(1) & (r0 < curves.CAP).all(1)
    dif = np.diff(r0[ok], axis=1)
    if len(dif):
        mad = np.median(np.abs(dif - np.median(dif, 1, keepdims=True)), 1) * 1.4826 / np.sqrt(2)
        d["log_noise"] = float(np.log10(np.median(mad) + 1e-6))
    else:
        d["log_noise"] = np.nan
    return {k: d[k] for k in DESCRIPTORS}


def _matrix(rows):
    return np.array([[r[k] for k in DESCRIPTORS] for r in rows], dtype=float)


def coverage(real: list[dict], gen: list[dict]) -> dict:
    R, Gm = _matrix(real), _matrix(gen)
    mu, sd = np.nanmean(R, 0), np.nanstd(R, 0) + 1e-9
    R, Gm = (R - mu) / sd, (Gm - mu) / sd
    R, Gm = np.nan_to_num(R), np.nan_to_num(Gm)
    drr = np.linalg.norm(R[:, None] - R[None], axis=-1)
    np.fill_diagonal(drr, np.inf)
    thr = np.median(drr.min(1))
    drg = np.linalg.norm(R[:, None] - Gm[None], axis=-1)
    return {"coverage": float((drg.min(1) <= thr).mean()), "precision": float((drg.min(0) <= thr).mean()),
            "covered": drg.min(1) <= thr, "nn_real_to_gen": drg.min(1), "threshold": float(thr)}


def per_descriptor_gap(real: list[dict], gen: list[dict]) -> dict:
    """1-D Wasserstein distance per descriptor, in units of the real tasks' sd."""
    from scipy.stats import wasserstein_distance
    R, Gm = _matrix(real), _matrix(gen)
    out = {}
    for j, k in enumerate(DESCRIPTORS):
        a, b = R[:, j][np.isfinite(R[:, j])], Gm[:, j][np.isfinite(Gm[:, j])]
        out[k] = float(wasserstein_distance(a, b) / (a.std() + 1e-9)) if len(a) and len(b) else np.nan
    return out
