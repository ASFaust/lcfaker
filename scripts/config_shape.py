"""Analysis H4: does the config determine the curve's shape beyond its final value?

Per task: live curves (gain >= 0.1, one seed), progress curve in log-time and the shifted-S template fit of
scripts/shape_template.py -> location m (speed) and log steepness k; plus u50 (log-time at half progress). Targets
are predicted with gradient-boosted trees, 5-fold cross-validation within the task, from
    final : the final value only (normalised z)
    config: the config only (Task.X())
    both  : config + final value
R^2 (cross-validated) per target and input, median over tasks. If "both" clearly beats "final", configs shape the
curve beyond its level; if "config" ~ "both", the final value adds nothing beyond the config.

    PYTHONPATH=. python scripts/config_shape.py
"""
from __future__ import annotations

import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, __import__("os").path.dirname(__file__))
from shape_clusters import progress_curves  # noqa: E402
from shape_template import fit_template  # noqa: E402

from curvefaker import curves, data  # noqa: E402
from curvefaker.paths import REPO  # noqa: E402

RNG = np.random.default_rng(17)
G = 40
U = np.linspace(np.log(0.02), 0, G)
PLAN = {"taskset": ("fit", 120), "jahs": ("fit", 40), "quicktune": ("fit", 40), "fcnet": ("select", 4),
        "bbomix": ("select", 35)}


def r2_cv(X, y):
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10, verbose=-1,
                          n_jobs=8)
    p = cross_val_predict(m, X, y, cv=5)
    return 1 - ((y - p) ** 2).mean() / y.var()


def main():
    warnings.filterwarnings("ignore")
    rows = []
    for src, (split, n_tasks) in PLAN.items():
        ids = data.ids(split, [src])
        if src == "taskset":   # spaces with several hps
            ids = [i for i in ids if i.split("/")[-1] in ("adam4p", "adam6p", "adam8p", "nadamw")]
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
            s = curves.smooth(np.nan_to_num(z, nan=curves.CAP))
            live = ((c == 0) | (c == 2)) & (s[:, 0] - s.min(1) >= 0.1)
            if live.sum() < 80 or X.shape[1] == 0:
                continue
            P = progress_curves(z[live], "log")
            rmse, par, _ = fit_template(P, U, np.linspace(-5, 1.5, 40), np.exp(np.linspace(np.log(0.3), np.log(40), 15)))
            fin = curves.final_value(z[live])
            tg = {"m (speed)": par["m"], "log k": np.log(par["k"]), "u50": np.array([U[np.argmax(r >= 0.5)] for r in P])}
            Xl = X[live]
            for name, target in tg.items():
                if target.std() == 0:
                    continue
                rows.append({"source": src, "task": tid, "target": name, "n": int(live.sum()),
                             "final": r2_cv(fin[:, None], target), "config": r2_cv(Xl, target),
                             "both": r2_cv(np.column_stack([Xl, fin]), target)})
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "config_shape.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["target", "source"])[["final", "config", "both"]].median().round(2).to_string())
    print("\ntasks:", df.groupby("source").task.nunique().to_dict())


if __name__ == "__main__":
    main()
