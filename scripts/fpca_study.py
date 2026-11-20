"""Low-rank structure of a task's curves, per loss scale (fit split only).

Per task (live, non-diverged curves, one seed; TaskSet step 0 dropped), on scale v (lin / logex_0.02):
    V = mean curve + scores @ components (PCA over configs, time points as features)
E4a: share of within-task variance explained by 1, 2, 3 components (in-sample, 80% of curves).
E4b: partial-curve prediction: components fitted on 80% of curves; for held-out curves the k scores (k = 1..3) are
     estimated by ridge least squares on the first 20% / 40% of the budget; the final value (mean of the last 10%)
     is predicted and transformed back. Scores: rank correlation with true finals, median abs error / task range;
     baseline: current smoothed value (as in scripts/representation_study.py).

    PYTHONPATH=. python scripts/fpca_study.py
"""
from __future__ import annotations

import os
import sys
from multiprocessing import get_context

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from representation_study import PER_SOURCE, _inverse, _transform  # noqa: E402
from curvefaker import curves, data  # noqa: E402
from curvefaker.paths import REPO  # noqa: E402

SCALES = {"lin": None, "logex": 0.02}


def _task(tid):
    rng = np.random.default_rng(abs(hash(tid)) % 2**32)
    t = data.load(tid)
    y = t.y[:, 0].astype(float)
    y = y[np.isfinite(y).all(1)]
    if t.source == "taskset":
        y = y[:, 1:]
    if len(y) < 60:
        return None
    T = y.shape[1]
    rngt = np.quantile(y[:, 0], 0.9) - y.min()
    if rngt <= 0:
        return None
    s = curves.smooth(y)
    live = (s[:, 0] - s.min(1) >= 0.1 * rngt) & (s.max(1) <= y.min() + 2 * rngt)
    y, s = y[live], s[live]
    if len(y) < 60:
        return None
    if len(y) > 300:
        k = rng.choice(len(y), 300, replace=False)
        y, s = y[k], s[k]
    out = {"task": tid, "source": t.source}
    perm = rng.permutation(len(y))
    ntr = int(0.8 * len(y))
    tr, te = perm[:ntr], perm[ntr:]
    fin = y[te, -max(1, T // 10):].mean(1)
    for frac in (0.2, 0.4):
        kk = max(3, int(round(frac * T)))
        out[f"base_rho_{frac}"] = curves.spearman(s[te, kk - 1], fin)
    for name, m in SCALES.items():
        L = y.min() - (m or 0) * rngt
        v = _transform(y, L, m)
        mu = v[tr].mean(0)
        U, S, Vt = np.linalg.svd(v[tr] - mu, full_matrices=False)
        ev = S ** 2 / (S ** 2).sum()
        for k in (1, 2, 3):
            out[f"{name}_ev{k}"] = float(ev[:k].sum())
        for frac in (0.2, 0.4):
            kk = max(3, int(round(frac * T)))
            for k in (1, 2, 3):
                C = Vt[:k] * S[:k, None] / np.sqrt(ntr)          # components scaled to unit-variance scores
                A = C[:, :kk].T                                     # [kk, k]
                lam = 1e-1 * np.trace(A.T @ A) / k / kk + 1e-9
                sc = np.linalg.solve(A.T @ A + lam * np.eye(k), A.T @ (v[te, :kk] - mu[:kk]).T).T
                pv = mu + sc @ C
                pred = _inverse(pv, L, m)[:, -max(1, T // 10):].mean(1)
                out[f"{name}_k{k}_rho_{frac}"] = curves.spearman(pred, fin)
                out[f"{name}_k{k}_mae_{frac}"] = float(np.median(np.abs(pred - fin)) / rngt)
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
    df.to_csv(REPO / "results" / "fpca_study.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.groupby("source")[[c for c in df.columns if c not in ("task", "source")]].median().round(3).T.to_string())


if __name__ == "__main__":
    main()
