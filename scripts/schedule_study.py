"""Does schedule-aware time explain TaskSet curves better? (fit split only)

Per live curve of TaskSet tasks with schedules (adam6p, adam8p, nadamw), on the log-excess scale v = log(y - L)
(L = task minimum - 0.02 * task range), three broken-power-law models, fitted by least squares on v:
    M_t     y = L + d + A / (1 + (t  / tau)^a)                 time = training step
    M_s1    y = L + d + A / (1 + (S1 / tau)^a)                 time = cumulative learning rate (optimiser progress)
    M_s1s2  y = L + d + A / (1 + (S1 / tau)^a) - C * S2 / T    plus the annealing term (Tissue et al. 2024)
(d >= 0: the config's own floor above the task floor; C >= 0.)
Scores per curve: in-sample RMSE on v; final value predicted from the first 40% of the budget (the model knows the
config's schedule) -> rank correlation over the task's held-out curves and median abs error / task range.
Curves are grouped by how much the learning rate decays: lr multiplier at the end < 0.5 ("decayed") vs >= 0.9
("constant-ish").

    PYTHONPATH=. python scripts/schedule_study.py
"""
from __future__ import annotations

import os
from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from curvefaker import curves, data, schedules
from curvefaker.paths import REPO


def _model(p, x, s2, T, L, kind):
    d, logA, logtau, loga = p[:4]
    y = L + np.exp(d) + np.exp(logA) / (1 + (x / np.exp(logtau)) ** np.exp(loga))
    if kind == "s1s2":
        y = y - np.exp(p[4]) * s2 / T
    return y


def _fit(v, x, s2, T, L, kind, k):
    """Fit on the first k points (v on log-excess scale). Returns predicted y over the full curve."""
    y0 = np.exp(v[0]) + L
    p0 = [np.log(max(np.exp(v[k - 1]) * 0.3, 1e-6)), np.log(max(y0 - L, 1e-6)), np.log(np.median(x[x > 0])), 0.0]
    if kind == "s1s2":
        p0.append(np.log(1e-3))

    def res(p):
        yh = _model(p, x[:k], s2[:k], T, L, kind)
        return np.log(np.maximum(yh - L, 1e-12)) - v[:k]

    try:
        r = least_squares(res, p0, method="trf", max_nfev=400)
        return _model(r.x, x, s2, T, L, kind), float(np.sqrt(np.mean(r.fun ** 2)))
    except Exception:
        return None, np.nan


def _task(tid):
    t = data.load(tid)
    opt = tid.split("/")[-1]
    y = t.y[:, 0].astype(float)
    keep = np.isfinite(y).all(1)
    y, hps = y[keep][:, 1:], t.hps[keep].reset_index(drop=True)
    steps = t.steps[1:].astype(float)
    T = steps[-1]
    rngt = np.quantile(y[:, 0], 0.9) - y.min()
    if rngt <= 0 or len(y) < 60:
        return []
    s = curves.smooth(y)
    live = np.where((s[:, 0] - s.min(1) >= 0.1 * rngt) & (s.max(1) <= y.min() + 2 * rngt))[0]
    rng = np.random.default_rng(abs(hash(tid)) % 2**32)
    live = rng.choice(live, min(80, len(live)), replace=False) if len(live) else live
    L = y.min() - 0.02 * rngt
    k = int(round(0.4 * len(steps)))
    rows = []
    for i in live:
        hp = hps.iloc[i].to_dict()
        s1, s2 = schedules.areas(opt, hp, steps)
        endmult = float(schedules.lr_multiplier(opt, hp, np.array([T]))[0])
        v = np.log(np.maximum(y[i] - L, 1e-12))
        fin = y[i, -5:].mean()
        r = {"task": tid, "opt": opt, "end_mult": endmult, "fin": fin, "cur": s[i, k - 1], "range": rngt}
        for kind, x in (("t", steps), ("s1", np.maximum(s1, 1e-9)), ("s1s2", np.maximum(s1, 1e-9))):
            full, rmse = _fit(v, x, s2, T, L, kind, len(v))
            part, _ = _fit(v, x, s2, T, L, kind, k)
            r[f"{kind}_rmse"] = rmse
            r[f"{kind}_pred"] = float(part[-5:].mean()) if part is not None else np.nan
        rows.append(r)
    return rows


def main():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    rng = np.random.default_rng(2)
    ids = [i for i in data.ids("fit", ["taskset"]) if i.split("/")[-1] in ("adam6p", "adam8p", "nadamw")]
    ids = list(rng.choice(ids, 90, replace=False))
    with get_context("spawn").Pool(60) as pool:
        rows = [r for rs in pool.map(_task, ids, chunksize=1) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "schedule_study.csv", index=False)
    df["group"] = np.where(df.end_mult < 0.5, "decayed", np.where(df.end_mult >= 0.9, "constant-ish", "mild"))
    out = []
    for (g, opt), d in df.groupby(["group", "opt"]):
        r = {"group": g, "opt": opt, "curves": len(d)}
        for kind in ("t", "s1", "s1s2"):
            r[f"rmse_{kind}"] = d[f"{kind}_rmse"].median()
            rho = []
            for _, dt in d.groupby("task"):
                if len(dt) >= 8:
                    rho.append(curves.spearman(dt[f"{kind}_pred"].fillna(dt.cur).to_numpy(), dt.fin.to_numpy()))
            r[f"rho_{kind}"] = np.nanmedian(rho) if rho else np.nan
            r[f"mae_{kind}"] = ((d[f"{kind}_pred"] - d.fin).abs() / d.range).median()
        rho = [curves.spearman(dt.cur.to_numpy(), dt.fin.to_numpy()) for _, dt in d.groupby("task") if len(dt) >= 8]
        r["rho_current"] = np.nanmedian(rho) if rho else np.nan
        out.append(r)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
