"""Analysis H2: is the learning rate a time warp? (TaskSet adam1p: the learning rate is the only hyperparameter)

Within a task, live curves (gain >= 0.1) sorted by learning rate; each curve is paired with the curve k = n/10
places further (a clearly higher learning rate). Curve B is aligned to curve A by
    none     : no change
    level    : a vertical offset (closed form)
    time     : a shift d in log-time, B(u) vs A(u + d), on the overlapping points (at least half of them)
    both     : time shift and vertical offset
and the RMSE (z units) after alignment is compared. If the learning rate is a time warp, "time" removes most of
the difference and the fitted shift d grows with the log learning-rate ratio (slope 1: a pure cumulative-learning-
rate clock with a constant schedule).

    PYTHONPATH=. python scripts/lr_timewarp.py [adam1p|adam4p]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from curvefaker import curves, data
from curvefaker.paths import REPO

RNG = np.random.default_rng(13)
SHIFTS = np.linspace(-4, 4, 161)


def align(a, b, u):
    """Best RMSE of b against a under the four alignments; and the best time shift."""
    out = {"none": np.sqrt(((a - b) ** 2).mean())}
    c = (a - b).mean()
    out["level"] = np.sqrt(((a - b - c) ** 2).mean())
    best_t, best_b, dt, db = np.inf, np.inf, 0.0, 0.0
    for d in SHIFTS:
        v = u + d
        ok = (v >= u[0]) & (v <= u[-1])
        if ok.sum() < len(u) // 2:
            continue
        a_s = np.interp(v[ok], u, a)                  # A evaluated at u + d
        r = a_s - b[ok]
        e_t = np.sqrt((r ** 2).mean())
        e_b = np.sqrt(((r - r.mean()) ** 2).mean())
        if e_t < best_t:
            best_t, dt = e_t, d
        if e_b < best_b:
            best_b, db = e_b, d
    out["time"], out["both"], out["shift_time"], out["shift_both"] = best_t, best_b, dt, db
    return out


def main(opt="adam1p", n_tasks=400):
    ids = [i for i in data.ids("fit", ["taskset"]) if i.endswith("/" + opt)]
    ids = list(RNG.choice(ids, min(n_tasks, len(ids)), replace=False))
    rows = []
    for tid in ids:
        t = data.load(tid)
        z, _ = curves.normalise(t.y[:, 0])
        ok = np.isfinite(z).all(1)
        c = np.full(len(z), 3)
        c[ok] = curves.classify(z[ok])
        s = curves.smooth(np.nan_to_num(z, nan=curves.CAP))
        live = np.where(((c == 0) | (c == 2)) & (s[:, 0] - s.min(1) >= 0.1))[0]
        if len(live) < 30:
            continue
        lr = t.hps["learning_rate"].to_numpy(float)[live]
        o = np.argsort(lr)
        live, lr = live[o], lr[o]
        k = max(1, len(live) // 10)
        T = s.shape[1]
        u = np.log(np.arange(1, T + 1) / T)
        for i in range(0, len(live) - k, max(1, (len(live) - k) // 20)):
            a, b = s[live[i]], s[live[i + k]]
            r = align(a, b, u)
            rows.append({"task": tid, "dlog_lr": float(np.log(lr[i + k] / lr[i])), **r})
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / f"lr_timewarp_{opt}.csv", index=False)
    rel = {m: float(np.median(df[m] / df["none"])) for m in ("level", "time", "both")}
    print(f"{opt}: {df.task.nunique()} tasks, {len(df)} pairs; median RMSE after alignment / before:",
          {k: round(v, 3) for k, v in rel.items()})
    print("share of pairs where time shift beats level shift:", round(float((df["time"] < df["level"]).mean()), 3))
    # does the shift follow the learning-rate ratio? (positive shift: the higher-lr curve B is ahead in log-time)
    sl = np.polyfit(df["dlog_lr"], df["shift_time"], 1)[0]
    print("median fitted log-time shift:", round(float(df["shift_time"].median()), 2),
          "| median log lr ratio:", round(float(df["dlog_lr"].median()), 2),
          "| slope shift ~ dlog_lr:", round(float(sl), 2),
          "| rank corr:", round(curves.spearman(df["dlog_lr"].to_numpy(), df["shift_time"].to_numpy()), 2))


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["adam1p"]))
