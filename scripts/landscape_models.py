"""Question A, part 1: how do hyperparameters determine final performance within a task?

Per task: target = final normalised loss (mean over seeds of the last 10% of the curve, z units, capped at 2),
replaced by its normal score (rank among the task's sampled configs mapped to a standard normal quantile): a
monotone change that keeps the order and stops a few dead / diverged configs from dominating squared errors.
Train on 200 random configs (QuickTune / small tasks: half), test on up to 500 others. Models:
    const     : training mean (floor)
    rf        : random forest, 300 trees, one-hot configs
    lgbm      : gradient-boosted trees
    gp_full   : GP with a product kernel over hyperparameters (all interactions)        [curvefaker/landscape.py]
    gp_add    : GP with a sum of 1-D kernels (no interactions)
    gp_addfull: both
Metrics on the test configs: R^2 and Spearman rank correlation (what matters to an optimiser: the order).
Descriptors from gp_addfull: interaction_share, eff_hps, top_hp_share, noise_share.
The same pipeline runs on ifBO tasks (configs uniform in [0, 1]^d, d = 1..10).

    PYTHONPATH=. python scripts/landscape_models.py      # CPU, parallel over tasks
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from multiprocessing import get_context

import numpy as np
import pandas as pd

from curvefaker import curves, data
from curvefaker.paths import REPO

N_TRAIN, N_TEST = 200, 500
SAMPLE = {"taskset": 60, "nasbench201": 3, "fcnet": 4, "bbomix": 20, "quicktune": 40, "jahs": 27}


def final_target(y):
    z, _ = curves.normalise(y)
    return np.clip(curves.final_value(curves.mean_curve(z)), 0, curves.CAP)


def run_task(job):
    import torch
    from sklearn.ensemble import RandomForestRegressor
    from curvefaker.landscape import LandscapeGP
    torch.set_num_threads(2)
    warnings.filterwarnings("ignore")
    src, tid, Xg, is_cat, Xoh, y, seed = job
    from scipy.stats import norm, rankdata
    y = norm.ppf(rankdata(y) / (len(y) + 1))      # normal scores: rank -> standard normal quantile
    rng = np.random.default_rng(seed)
    n = len(y)
    ntr = min(N_TRAIN, n // 2)
    perm = rng.permutation(n)
    tr, te = perm[:ntr], perm[ntr:ntr + N_TEST]
    out = {"source": src, "task": tid, "n_train": ntr, "n_test": len(te), "n_hps": Xg.shape[1],
           "y_std": float(y.std())}

    def score(name, pred):
        yt = y[te]
        out[f"r2[{name}]"] = float(1 - ((yt - pred) ** 2).mean() / max(yt.var(), 1e-12))
        out[f"rho[{name}]"] = curves.spearman(pred, yt) if pred.std() > 0 else 0.0

    score("const", np.full(len(te), y[tr].mean()))
    rf = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=2, random_state=0).fit(Xoh[tr], y[tr])
    score("rf", rf.predict(Xoh[te]))
    import lightgbm as lgb
    gb = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=5,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2)
    gb.fit(Xoh[tr], y[tr])
    score("lgbm", gb.predict(Xoh[te]))
    Xt = torch.tensor(Xg, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    for kind in ("full", "additive", "add+full"):
        try:
            gp = LandscapeGP(Xg.shape[1], is_cat, kind).fit(Xt[tr], yt[tr])
            score(f"gp_{kind}", gp.predict(Xt[te]).numpy())
            if kind == "add+full":
                out.update(gp.describe())
        except Exception as e:   # numerical failure: report, don't crash the run
            out[f"err[gp_{kind}]"] = str(e)[:80]
    return out


def real_jobs(rng):
    jobs = []
    for src, k in SAMPLE.items():
        ids = data.ids(sources=[src])
        ids = list(rng.choice(ids, min(k, len(ids)), replace=False))
        for tid in ids:
            t = data.load(tid)
            y = t.y
            keep = np.ones(len(y), bool)
            if src == "quicktune":
                keep = np.isfinite(y[:, 0, -1])
            idx = np.where(keep)[0]
            if len(idx) > N_TRAIN + N_TEST:
                idx = rng.choice(idx, N_TRAIN + N_TEST, replace=False)
            from curvefaker.landscape import encode
            hps = t.hps.iloc[idx].reset_index(drop=True)
            Xg, is_cat = encode(hps, t.space)
            Xoh = t.X()[idx]
            jobs.append((src, tid, Xg, is_cat, Xoh, final_target(y[idx]), int(rng.integers(1 << 30))))
    return jobs


def ifbo_jobs(rng, n_tasks=60, T=100):
    import torch
    sys.path.insert(0, str(REPO.parent / "rogue-dagger"))
    from roguedagger.prior import PriorTasks
    g = torch.Generator().manual_seed(2)
    n = N_TRAIN + N_TEST
    p = PriorTasks(n_tasks, 10, "cpu", g)
    d = torch.randint(1, 11, (n_tasks,), generator=g)
    p.resample(torch.arange(n_tasks), d)
    x = torch.rand(n_tasks, n, 10, generator=g)
    y = p.curves(x, torch.full((n_tasks,), T), T).numpy()
    jobs = []
    for e in range(n_tasks):
        de = int(d[e])
        X = x[e, :, :de].numpy().astype(np.float64)
        jobs.append(("ifbo", f"ifbo/{e}", X, np.zeros(de, bool), X, final_target(y[e]), int(rng.integers(1 << 30))))
    return jobs


def main():
    rng = np.random.default_rng(3)
    t0 = time.time()
    jobs = real_jobs(rng) + ifbo_jobs(rng)
    print(f"{len(jobs)} tasks prepared in {time.time() - t0:.0f}s", flush=True)
    with get_context("spawn").Pool(min(64, os.cpu_count() // 2)) as pool:
        rows = []
        for k, r in enumerate(pool.imap_unordered(run_task, jobs)):
            rows.append(r)
            if k % 20 == 0:
                print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "landscape_tasks.csv", index=False)
    models = ["const", "rf", "lgbm", "gp_full", "gp_additive", "gp_add+full"]
    order = ["taskset", "nasbench201", "fcnet", "bbomix", "quicktune", "jahs", "ifbo"]
    for metric in ("r2", "rho"):
        s = df.groupby("source")[[f"{metric}[{m}]" for m in models]].median().round(2).reindex(order)
        print(f"\nmedian test {metric}\n", s.to_string())
    desc = df.groupby("source")[["n_hps", "interaction_share", "eff_hps", "top_hp_share", "noise_share"]].median()
    print("\n", desc.round(2).reindex(order).to_string())


if __name__ == "__main__":
    main()
