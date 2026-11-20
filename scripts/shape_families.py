"""Analysis S1: which parametric families explain real training curves? (curvefaker/families.py)

Curves: live curves (improving or turning, curvefaker.curves.classify) of one seed, normalised z, native length.
Per source: fit split (analysis) and select split (replication), up to ~3000 curves per source.
Per curve and family:
    interp : fit on all points except every 5th, RMSE on every 5th            (shape)
    extrap : fit on the first 70% of points, RMSE on the last 30%              (asymptotic behaviour)
    bic    : fit on all points, BIC = n log(SSE / n) + p log n                 (which family a curve prefers)
Errors are divided by the curve's own noise sd (MAD of first differences / sqrt 2): about 1 = as good as noise
allows for white Gaussian noise; heavy tails raise it. Reference: ref_neighbour = error of predicting each held-out
point by the mean of its two neighbours (white noise: about 1.22). Output: results/shape_families.csv (per source x family), results/shape_families_curves.npz (per curve).

    CUDA_VISIBLE_DEVICES=<one gpu> PYTHONPATH=. python scripts/shape_families.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from curvefaker import curves, data, families
from curvefaker.paths import REPO

RNG = np.random.default_rng(10)
PLAN = {   # source -> (split, max tasks, curves per task)
    "taskset": [("fit", 300, 10), ("select", 150, 10)],
    "nasbench201": [("fit", 3, 600)],
    "quicktune": [("fit", 86, 20)],
    "jahs": [("fit", 81, 25)],
    "fcnet": [("select", 4, 600)],
    "bbomix": [("select", 35, 60)],
}


def collect(src, split, n_tasks, per_task):
    ids = data.ids(split, [src])
    if len(ids) > n_tasks:
        ids = list(RNG.choice(ids, n_tasks, replace=False))
    Y, meta = [], []
    for tid in ids:
        t = data.load(tid)
        y = t.y[:, 0]
        if src == "quicktune":
            y = y[np.isfinite(y[:, -1])]
        z, _ = curves.normalise(y)
        ok = np.isfinite(z).all(1)
        c = np.full(len(z), 3)
        c[ok] = curves.classify(z[ok])
        live = np.where((c == 0) | (c == 2))[0]
        pick = RNG.choice(live, min(per_task, len(live)), replace=False) if len(live) else []
        Y += list(z[pick])
        meta += [(tid, int(c[i]), int(i)) for i in pick]
    return np.array(Y, np.float32), meta


def main():
    rows, per_curve = [], {}
    for src, parts in PLAN.items():
        for split, n_tasks, per_task in parts:
            t0 = time.time()
            Y, meta = collect(src, split, n_tasks, per_task)
            T = Y.shape[1]
            sd = np.maximum(families.noise_sd(Y), 1e-4)
            idx = np.arange(T)
            m_int = idx % 5 != 2
            m_ext = idx < int(0.7 * T)
            m_all = np.ones(T, bool)
            res = {}
            for fam, p in families.FAMILIES.items():
                fi, _ = families.fit(fam, Y, m_int)
                fe, _ = families.fit(fam, Y, m_ext)
                fa, sse = families.fit(fam, Y, m_all)
                r_int = np.sqrt(((fi - Y)[:, ~m_int] ** 2).mean(1)) / sd
                r_ext = np.sqrt(((fe - Y)[:, ~m_ext] ** 2).mean(1)) / sd
                bic = T * np.log(np.maximum(sse, 1e-12) / T) + p * np.log(T)
                res[fam] = (r_int, r_ext, bic)
            # nonparametric reference: each held-out interior point predicted by the mean of its two neighbours
            k = np.where(~m_int)[0]
            k = k[(k > 0) & (k < T - 1)]
            ref = np.sqrt(((Y[:, k] - 0.5 * (Y[:, k - 1] + Y[:, k + 1])) ** 2).mean(1)) / sd
            fams = list(families.FAMILIES)
            B = np.stack([res[f][2] for f in fams], 1)
            best = np.array(fams)[np.nanargmin(B, 1)]
            best_int = np.min(np.stack([res[f][0] for f in fams], 1), 1)
            for fam in fams:
                r_int, r_ext, _ = res[fam]
                rows.append({"source": src, "split": split, "family": fam, "params": families.FAMILIES[fam],
                             "n_curves": len(Y), "T": T,
                             "interp_med": float(np.nanmedian(r_int)), "interp_q90": float(np.nanquantile(r_int, 0.9)),
                             "extrap_med": float(np.nanmedian(r_ext)), "extrap_q90": float(np.nanquantile(r_ext, 0.9)),
                             "bic_best_share": float((best == fam).mean()),
                             "ref_neighbour_med": float(np.nanmedian(ref))})
            key = f"{src}/{split}"
            per_curve[key + "/best"] = best
            per_curve[key + "/best_interp"] = best_int
            per_curve[key + "/noise_sd"] = sd
            for fam in fams:
                per_curve[f"{key}/interp/{fam}"] = res[fam][0]
                per_curve[f"{key}/extrap/{fam}"] = res[fam][1]
            per_curve[key + "/task"] = np.array([m[0] for m in meta])
            print(f"{key}: {len(Y)} curves, T={T}, {time.time() - t0:.0f}s; best by BIC:",
                  pd.Series(best).value_counts(normalize=True).round(2).to_dict(),
                  f"| no family within 1.5x noise (interp): {(best_int > 1.5).mean():.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "shape_families.csv", index=False)
    np.savez(REPO / "results" / "shape_families_curves.npz", **per_curve)


if __name__ == "__main__":
    main()
