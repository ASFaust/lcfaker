"""Question C: which noise model explains single learning curves best, per source? (curvefaker/noise.py)

Per source: up to 40 tasks x 30 configs (at most 300 curves when T > 100), one seed each, normalised z (curves that hit the divergence cap are left
out). Curves are split into halves A / B by task (by config for sources with < 8 tasks). For each noise model:
choose the global shape parameters on A (grid search), freeze them, re-fit the per-curve ones on B, report B's log
likelihood per point relative to white noise. Extra statistics on B's estimated noise (white model):
    spike_frac : share of noise values beyond 5 robust standard deviations (MAD-based); Gaussian: ~6e-7
    shared     : share of noise variance common to all configs of a task at the same step (same validation set)
    sigma_med  : median per-curve noise sd (in z units: fraction of the task's start-to-best range)
dLL[m] is the held-out log likelihood per point of model m minus that of white noise (paired over curves, se = its
standard error); "best" is the highest among the models that beat white by more than 3 standard errors.

    CUDA_VISIBLE_DEVICES=<one gpu> PYTHONPATH=. python scripts/noise_models.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch

from curvefaker import curves, data, noise
from curvefaker.paths import REPO

RNG = np.random.default_rng(1)
DEV = "cuda"
N_TASKS, N_CFG = 40, 30
MAX_LONG = 300               # curve cap for sources with T > 100


def sample_source(src):
    ids = data.ids(sources=[src])
    if src == "taskset":
        ids = list(RNG.choice(ids, N_TASKS, replace=False))
    elif len(ids) > N_TASKS:
        ids = list(RNG.choice(ids, N_TASKS, replace=False))
    per_task = N_CFG if len(ids) >= 8 else 400 // len(ids)
    Y, tid = [], []
    for k, i in enumerate(ids):
        t = data.load(i)
        y = t.y[:, 0]
        if src == "quicktune":
            y = y[np.isfinite(y[:, -1])]
        z, _ = curves.normalise(y)
        ok = np.isfinite(z).all(1) & (z < curves.CAP).all(1)
        idx = np.where(ok)[0]
        idx = RNG.choice(idx, min(per_task, len(idx)), replace=False)
        Y.append(z[idx])
        tid += [k] * len(idx)
    Y, tid = np.concatenate(Y), np.array(tid)
    if Y.shape[1] > 100 and len(Y) > MAX_LONG:          # GP cost grows with T^3: fewer curves for long curves
        keep = np.sort(RNG.choice(len(Y), MAX_LONG, replace=False))
        Y, tid = Y[keep], tid[keep]
    return Y, tid, len(ids) >= 8


def sample_ifbo(n_tasks=40, n_cfg=30, T=100):
    sys.path.insert(0, str(REPO.parent / "rogue-dagger"))
    from roguedagger.prior import PriorTasks
    g = torch.Generator(device=DEV).manual_seed(1)
    p = PriorTasks(n_tasks, 10, DEV, g)
    d = torch.randint(1, 11, (n_tasks,), device=DEV, generator=g)
    p.resample(torch.arange(n_tasks, device=DEV), d)
    x = torch.rand(n_tasks, n_cfg, 10, device=DEV, generator=g)
    y = p.curves(x, torch.full((n_tasks,), T, device=DEV), T).cpu().numpy()
    Y = np.concatenate([curves.normalise(y[e])[0] for e in range(n_tasks)])
    return Y, np.repeat(np.arange(n_tasks), n_cfg), True


def residual_stats(res, tid):
    mad = np.median(np.abs(res - np.median(res, 1, keepdims=True)), 1, keepdims=True) * 1.4826
    spike = float((np.abs(res) > 5 * np.maximum(mad, 1e-9)).mean())
    shared = []
    for k in np.unique(tid):
        E = res[tid == k]
        if len(E) < 5:
            continue
        n = len(E)
        tot = E.var(1).mean()
        col = E.mean(0).var()
        shared.append(max(0.0, (col - tot / n) / tot) if tot > 0 else np.nan)
    return spike, float(np.nanmedian(shared))


def main():
    sources = sys.argv[1:] or ["taskset", "nasbench201", "fcnet", "bbomix", "quicktune", "jahs", "ifbo"]
    rows = []
    for src in sources:
        t0 = time.time()
        Y, tid, by_task = sample_ifbo() if src == "ifbo" else sample_source(src)
        if by_task:
            tasks = np.unique(tid)
            RNG.shuffle(tasks)
            inA = np.isin(tid, tasks[: len(tasks) // 2])
        else:
            inA = RNG.random(len(Y)) < 0.5
        yA = torch.tensor(Y[inA], dtype=torch.float64, device=DEV)
        yB = torch.tensor(Y[~inA], dtype=torch.float64, device=DEV)
        row = {"source": src, "curves_A": int(inA.sum()), "curves_B": int((~inA).sum()), "T": Y.shape[1]}
        ll, llc = {}, {}
        for m in noise.MODELS:
            fr, _ = noise.select_global(yA, m)
            gpB, llB = noise.fit(yB, m, frozen=fr)
            llc[m] = llB.cpu().numpy() / Y.shape[1]
            ll[m] = float(llc[m].mean())
            if "ar1" in m:
                row[f"rho[{m}]"] = round(float(gpB.rho()), 3)
            if "phase" in m:
                row[f"beta[{m}]"] = round(fr["beta"], 2)
            if m == "white":
                with torch.no_grad():
                    res = gpB.residuals(yB).cpu().numpy()
                row["sigma_med"] = float(np.median(np.exp(gpB.log_sig.detach().cpu().numpy())))
                row["spike_frac"], row["shared"] = residual_stats(res, tid[~inA])
        se = {}
        for m in noise.MODELS:   # paired over held-out curves: mean and standard error of the difference to white
            d = llc[m] - llc["white"]
            row[f"dLL[{m}]"] = round(float(d.mean()), 4)
            se[m] = float(d.std() / np.sqrt(len(d))) if m != "white" else 0.0
            row[f"se[{m}]"] = round(se[m], 4)
        # best = highest held-out likelihood among models that beat white by more than 3 standard errors
        ok = [m for m in noise.MODELS if m == "white" or ll[m] - ll["white"] > 3 * se[m]]
        row["best"] = max(ok, key=ll.get)
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
              f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = REPO / "results" / "noise_models.csv"
    if out.exists() and sys.argv[1:]:
        old = pd.read_csv(out)
        df = pd.concat([old[~old.source.isin(df.source)], df])
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
