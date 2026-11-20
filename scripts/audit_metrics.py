"""Audit of what the curves are (docs/data_semantics.md): metric type, value range, time axis, schedule
hyperparameters, and how often validation curves overfit. Fit and select splits only (loaders refuse held-out).

Overfitting, per live curve (one seed; smoothed with curves.smooth on the raw values):
    min index m < 0.9 T, and rise = s[-1] - s[m] >= max(0.1 * (s[0] - s[m]), 3 * noise sd)
    noise sd = robust sd of step differences / sqrt(2)
For TaskSet (training curves available): of overfitting curves, share whose training loss at the end is below its
value at m (training still improving while validation worsens = classic overfitting) vs. not (instability).

    PYTHONPATH=. python scripts/audit_metrics.py
"""
from __future__ import annotations

import os
from multiprocessing import get_context

import numpy as np
import pandas as pd

from curvefaker import curves, data
from curvefaker.paths import REPO

PER_SOURCE = {"taskset": 400, "jahs": 81, "quicktune": 86, "nasbench201": 3, "fcnet": 4, "bbomix": 35}


def _one(tid):
    t = data.load(tid)
    y = t.y[:, 0].astype(float)
    ok = np.isfinite(y).all(1)
    y = y[ok]
    if len(y) < 20:
        return None
    if len(y) > 400:
        idx = np.random.default_rng(0).choice(len(y), 400, replace=False)
        y = y[idx]
    else:
        idx = np.arange(len(y))
    T = y.shape[1]
    s = curves.smooth(y)
    m = s.argmin(1)
    drop = s[:, 0] - s[np.arange(len(s)), m]
    rng_task = np.nanquantile(y[:, 0], 0.9) - np.nanmin(y)
    live = drop >= 0.1 * max(rng_task, 1e-12)
    d = np.diff(y, axis=1)
    sd = np.median(np.abs(d - np.median(d, 1, keepdims=True)), 1) * 1.4826 / np.sqrt(2)
    rise = s[:, -1] - s[np.arange(len(s)), m]
    over = live & (m < 0.9 * T) & (rise >= np.maximum(0.1 * drop, 3 * sd))
    out = {"task": tid, "source": t.source, "metric": t.metric, "T": T, "step0": float(t.steps[0]),
           "min": float(np.nanmin(t.y)), "neg": bool(np.nanmin(t.y) <= 0), "n_live": int(live.sum()),
           "share_over": float(over[live].mean()) if live.any() else np.nan,
           "over_rise_rel": float(np.median(rise[over] / drop[over])) if over.any() else np.nan,
           "hps": ",".join(map(str, t.hps.columns))}
    if t.source == "taskset" and "train" in t.meta:
        tr = t.meta["train"][:, 0][np.where(ok)[0][idx]].astype(float)
        trs = curves.smooth(np.nan_to_num(tr, nan=np.nanmax(tr)))
        if over.any():
            still = trs[over, -1] < trs[np.where(over)[0], m[over]]
            out["over_train_improving"] = float(still.mean())
        # generalisation gap at the end vs at the start (valid - train), live curves
        g0 = (y[live, 0] - tr[live, 0])
        g1 = (y[live, -1] - tr[live, -1])
        out["gap_growth"] = float(np.nanmedian(g1 - g0) / max(rng_task, 1e-12)) if live.any() else np.nan
    return out


def main():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    rng = np.random.default_rng(0)
    ids = []
    for src, k in PER_SOURCE.items():
        a = data.ids(None, [src])
        ids += list(rng.choice(a, min(k, len(a)), replace=False))
    with get_context("spawn").Pool(48) as pool:
        rows = [r for r in pool.map(_one, ids, chunksize=2) if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "audit_metrics.csv", index=False)
    df["family"] = [t.split("/")[1].split("_")[0] if s == "taskset" else s for t, s in zip(df.task, df.source)]
    agg = {"tasks": ("task", "size"), "neg_share": ("neg", "mean"), "share_over_median": ("share_over", "median"),
           "share_over_mean": ("share_over", "mean"), "tasks_over_10pct": ("share_over", lambda x: (x > 0.1).mean()),
           "rise_rel": ("over_rise_rel", "median")}
    if "over_train_improving" in df:
        agg["train_improving"] = ("over_train_improving", "median")
        agg["gap_growth"] = ("gap_growth", "median")
    pd.set_option("display.width", 200)
    print(df.groupby("source").agg(**agg).round(3).to_string())
    print()
    ts = df[df.source == "taskset"]
    print(ts.groupby("family").agg(**agg).round(3).sort_values("tasks", ascending=False).head(20).to_string())


if __name__ == "__main__":
    main()
