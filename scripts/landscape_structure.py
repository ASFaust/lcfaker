"""Analyses H1 (landscape modality) and H5 (where dead / diverged configs lie).

Per task (one seed, final value = mean of the last 10% of the normalised curve; configs as in Task.X(), at most 1000):
H1, on the k-nearest-neighbour graph of configs (k = 10, Euclidean in the [0, 1] encoding):
    local_optima      share of configs better than all their k neighbours; also for randomly shuffled final values
                      (ratio < 1: smoother than random; ~1: as rugged as noise)
    best_basin        share of configs from which greedy descent on the graph ends at the global best
    near_best         share of configs within 0.05 (z) of the best final value
H5, for tasks with a learning-rate hp (log scale, rank-transformed within the task to [0, 1]):
    median learning-rate rank of dead, diverged and the best 10% of configs
    sharpness of the divergence boundary: AUC of predicting diverged from the learning rate alone
    gap: learning-rate rank of the best config minus the 5th percentile rank of diverged configs (negative: the
    best config sits below the divergence boundary)

    PYTHONPATH=. python scripts/landscape_structure.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from curvefaker import curves, data
from curvefaker.paths import REPO

RNG = np.random.default_rng(16)
PLAN = {"taskset": ("fit", 200), "nasbench201": ("fit", 3), "quicktune": ("fit", 86), "jahs": ("fit", 81),
        "fcnet": ("select", 4), "bbomix": ("select", 35)}
LR = {"taskset": "learning_rate", "fcnet": "init_lr", "bbomix": "learning_rate", "quicktune": "lr",
      "jahs": "LearningRate"}


def graph_stats(X, f, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    idx = nn.kneighbors(X, return_distance=False)[:, 1:]
    lo = (f[:, None] < f[idx]).all(1).mean()
    fs = RNG.permutation(f)
    lo_s = (fs[:, None] < fs[idx]).all(1).mean()
    best = int(np.argmin(f))
    # greedy descent: move to the best neighbour while it improves
    nxt = np.where(f[idx].min(1) < f, idx[np.arange(len(f)), f[idx].argmin(1)], np.arange(len(f)))
    end = np.arange(len(f))
    for _ in range(len(f)):
        new = nxt[end]
        if (new == end).all():
            break
        end = new
    return lo, lo_s, float((end == best).mean())


def main():
    rows = []
    for src, (split, n_tasks) in PLAN.items():
        ids = data.ids(split, [src])
        if len(ids) > n_tasks:
            ids = list(RNG.choice(ids, n_tasks, replace=False))
        for tid in ids:
            t = data.load(tid)
            y = t.y[:, 0]
            X = t.X()
            keep = np.arange(len(y))
            if src == "quicktune":
                keep = keep[np.isfinite(y[:, -1])]
            if len(keep) > 1000:
                keep = RNG.choice(keep, 1000, replace=False)
            y, X = y[keep], X[keep]
            z, _ = curves.normalise(y)
            ok = np.isfinite(z).all(1)
            c = np.full(len(z), 3)
            c[ok] = curves.classify(z[ok])
            f = curves.final_value(np.nan_to_num(z, nan=curves.CAP))
            if X.shape[1] == 0 or len(f) < 50:
                continue
            lo, lo_s, basin = graph_stats(X, f + 1e-9 * RNG.standard_normal(len(f)))
            r = {"source": src, "task": tid, "n": len(f), "d": X.shape[1], "local_optima": lo,
                 "local_optima_shuffled": lo_s, "best_basin": basin,
                 "near_best": float((f <= f.min() + 0.05).mean())}
            lrn = LR.get(src)
            if lrn and lrn in t.hps:
                lr = np.log(t.hps[lrn].to_numpy(float)[keep])
                rk = (np.argsort(np.argsort(lr)) + 0.5) / len(lr)
                top = f <= np.quantile(f, 0.1)
                r["lr_rank_dead"] = float(np.median(rk[c == 1])) if (c == 1).sum() >= 5 else np.nan
                r["lr_rank_div"] = float(np.median(rk[c == 3])) if (c == 3).sum() >= 5 else np.nan
                r["lr_rank_top10"] = float(np.median(rk[top]))
                if 5 <= (c == 3).sum() <= len(c) - 5:
                    r["auc_div_from_lr"] = float(roc_auc_score(c == 3, rk))
                    r["gap_best_to_div"] = float(rk[np.argmin(f)] - np.quantile(rk[c == 3], 0.05))
                if 5 <= (c == 1).sum() <= len(c) - 5:
                    r["auc_dead_from_low_lr"] = float(roc_auc_score(c == 1, -rk))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "landscape_structure.csv", index=False)
    df["optima_ratio"] = df.local_optima / df.local_optima_shuffled
    cols = ["d", "optima_ratio", "local_optima", "best_basin", "near_best", "lr_rank_dead", "lr_rank_div",
            "lr_rank_top10", "auc_div_from_lr", "auc_dead_from_low_lr", "gap_best_to_div"]
    pd.set_option("display.width", 250)
    print(df.groupby("source")[cols].median().round(3).to_string())
    print("\ntasks with lr stats:", df.groupby("source").lr_rank_top10.count().to_dict())


if __name__ == "__main__":
    main()
