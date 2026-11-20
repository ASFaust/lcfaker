"""Which loss scale should curves be modelled on? (fit split only)

Note: a logistic in log-time with a floor, L + A / (1 + exp((log t - c) / w)), is exactly a broken power law
L + A / (1 + (t / tau)^(1/w)): plateau, then (y - L) ~ t^(-1/w). The shape family is shared; the open question is
the scale on which per-config warps and noise act.

Scales: lin       v = y
        logex_m   v = log(y - L), L = task minimum - m * task range (m = 0.02, 0.1)

E1 noise scaling: per task, residuals r = y - smooth(y) of live curves; robust sd in bins of the smoothed excess
   e = s - L (L with m = 0.02); slope k of log sd on log e (k ~ 0: additive noise, k ~ 1: multiplicative).
E3 partial-curve prediction: per task, the shape-invariant model (curvefaker/sim.py) is fitted on the transformed
   full curves of 80% of the live curves; for each held-out curve its warps are fitted on the first 20% / 40% of the
   budget with the template fixed, and the final value (mean of the last 10%) is predicted and transformed back.
   Scores (held-out curves of a task): rank correlation of predicted vs true final values; median absolute error /
   task range. Baseline: the current value (smoothed last observed value).

    PYTHONPATH=. python scripts/representation_study.py
"""
from __future__ import annotations

import os
from multiprocessing import get_context

import numpy as np
import pandas as pd

from curvefaker import curves, data, sim
from curvefaker.paths import REPO

PER_SOURCE = {"taskset": 160, "jahs": 40, "quicktune": 40, "nasbench201": 3}
SCALES = {"lin": None, "logex_0.02": 0.02, "logex_0.1": 0.1}
FRACS = (0.2, 0.4)


def _transform(y, L, m):
    return y if m is None else np.log(np.maximum(y - L, 1e-12))


def _inverse(v, L, m):
    return v if m is None else L + np.exp(v)


def fit_warps(f, Y, u, x_range):
    """Grid search over (beta, gamma) with alpha, a > 0 in closed form, template fixed. Y [N, t], u [t]."""
    N = len(Y)
    best = np.full(N, np.inf)
    out = np.zeros((N, 4))
    for bb in sim.BETAS:
        for gg in sim.GAMMAS:
            x = np.clip((u - bb) * np.exp(gg), *x_range)
            fx = sim.weights(x) @ f
            A = np.stack([np.ones_like(fx), fx], 1)
            coef, *_ = np.linalg.lstsq(A, Y.T, rcond=None)
            coef[1] = np.maximum(coef[1], 1e-3)
            sse = ((A @ coef - Y.T) ** 2).sum(0)
            b = sse < best
            best[b] = sse[b]
            out[b] = np.stack([coef[0, b], coef[1, b], np.full(b.sum(), bb), np.full(b.sum(), gg)], 1)
    return out


def _task(tid):
    rng = np.random.default_rng(abs(hash(tid)) % 2**32)
    t = data.load(tid)
    y = t.y[:, 0].astype(float)
    y = y[np.isfinite(y).all(1)]
    if t.source == "taskset":
        y = y[:, 1:]                        # drop step 0 (before training): log-time needs t > 0
    if len(y) < 60:
        return None
    T = y.shape[1]
    rngt = np.quantile(y[:, 0], 0.9) - y.min()
    if rngt <= 0:
        return None
    s = curves.smooth(y)
    live = (s[:, 0] - s.min(1) >= 0.1 * rngt) & (s.max(1) <= y.min() + 2 * rngt)   # improving, not diverged
    y, s = y[live], s[live]
    if len(y) < 60:
        return None
    if len(y) > 300:
        k = rng.choice(len(y), 300, replace=False)
        y, s = y[k], s[k]
    out = {"task": tid, "source": t.source, "T": T, "n": len(y)}
    # E1 noise scaling
    L = y.min() - 0.02 * rngt
    r = (y - s)[:, 2:-2].ravel()
    e = (s - L)[:, 2:-2].ravel()
    qs = np.quantile(e, np.linspace(0, 1, 9))
    xs, ys = [], []
    for a, b in zip(qs[:-1], qs[1:]):
        m = (e >= a) & (e <= b)
        if m.sum() > 50:
            rr = r[m]
            xs.append(np.log(np.median(e[m])))
            ys.append(np.log(np.median(np.abs(rr - np.median(rr))) * 1.4826 + 1e-12))
    out["noise_slope"] = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 4 else np.nan
    # E3 partial-curve prediction
    u = np.log(np.arange(1, T + 1) / T)
    perm = rng.permutation(len(y))
    ntr = int(0.8 * len(y))
    tr, te = perm[:ntr], perm[ntr:]
    fin = y[te, -max(1, T // 10):].mean(1)
    for frac in FRACS:
        k = max(3, int(round(frac * T)))
        cur = s[te, k - 1]
        out[f"base_rho_{frac}"] = curves.spearman(cur, fin)
        out[f"base_mae_{frac}"] = float(np.median(np.abs(cur - fin)) / rngt)
    for name, m in SCALES.items():
        L = y.min() - (m or 0) * rngt
        v = _transform(y, L, m)
        fit = sim.fit_task(v[tr])
        xr = sim.covered_range(fit["beta"], fit["gamma"], u)
        for frac in FRACS:
            k = max(3, int(round(frac * T)))
            W = fit_warps(fit["f"], v[te, :k], u[:k], xr)
            pv = sim.evaluate(fit["f"], W[:, 0], W[:, 1], W[:, 2], W[:, 3], u, xr)
            pred = _inverse(pv, L, m)[:, -max(1, T // 10):].mean(1)
            out[f"{name}_rho_{frac}"] = curves.spearman(pred, fin)
            out[f"{name}_mae_{frac}"] = float(np.median(np.abs(pred - fin)) / rngt)
        # in-sample fit quality of the full curves, measured in linear units / task range
        back = _inverse(fit["fitted"], L, m)
        out[f"{name}_fit_rmse"] = float(np.sqrt(np.median((back - y[tr]) ** 2)) / rngt)
    return out


def main():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    rng = np.random.default_rng(1)
    ids = []
    for src, k in PER_SOURCE.items():
        a = data.ids("fit", [src])
        ids += list(rng.choice(a, min(k, len(a)), replace=False))
    with get_context("spawn").Pool(60) as pool:
        rows = [r for r in pool.map(_task, ids, chunksize=1) if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "representation_study.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["noise_slope"] + [c for c in df.columns if "_rho_" in c or "_mae_" in c or "fit_rmse" in c]
    print(df.groupby("source")[cols].median().round(3).T.to_string())
    print("tasks:", df.groupby("source").size().to_dict())


if __name__ == "__main__":
    main()
