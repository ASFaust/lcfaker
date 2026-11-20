"""Measure 2 for fakers: coverage / precision of the real *select* tasks, and per-descriptor gaps (curvefaker/evaluate.py).

Generation is repeated with 3 seeds (mean, and sd of coverage / precision). Real: 150 TaskSet select tasks, all FCNet and bbomix tasks (at most 1000 configs each).
Fakers:
    ifbo     : the ifBO prior (1-10 hps, 1000 uniform configs, 51 points)
    real-fit : real tasks of the *fit* split, as if they were generated (reference: what real but different task
               families reach)
    hgp-v0   : curvefaker/faker_hgp.py (hierarchical task bootstrap + GP landscapes + PCA curve basis)
    hgp-v1   : v0 with the real fit task's own curves instead of a Gaussian over PCA scores
    hgp-v2   : v1 + level warp z -> z^gamma per task
    hgp-v3   : v2 + time warp u -> u^a per task
    hgp-v4   : v2 with dead / turns / diverged shares (and curves) each drawn from a different fit task
    others   : registered in FAKERS as functions (n_tasks, rng) -> list of (y [n, R, T], X [n, d])

    PYTHONPATH=. python scripts/evaluate_fakers.py [faker ...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from curvefaker import data, evaluate
from curvefaker.paths import REPO

RNG = np.random.default_rng(5)
MAXN = 1000
SEEDS = (6, 7, 8)


def load_y(tid):
    t = data.load(tid)
    y, X = t.y, t.X()
    keep = np.arange(len(y))
    if t.source == "quicktune":
        keep = keep[np.isfinite(y[:, 0, -1])]
    if len(keep) > MAXN:
        keep = RNG.choice(keep, MAXN, replace=False)
    return y[keep], X[keep], t.source


def real_select():
    ts = data.ids("select", ["taskset"])
    ids = list(RNG.choice(ts, 150, replace=False)) + data.ids("select", ["fcnet", "bbomix"])
    return [load_y(i) for i in ids]


def faker_real_fit(n, rng):
    ids = data.ids("fit")
    return [load_y(i)[:2] for i in rng.choice(ids, n, replace=False)]


def faker_ifbo(n, rng, T=51, n_cfg=MAXN):
    import torch
    sys.path.insert(0, str(REPO.parent / "rogue-dagger"))
    from roguedagger.prior import PriorTasks
    g = torch.Generator().manual_seed(int(rng.integers(1 << 30)))
    out = []
    for s in range(0, n, 20):
        k = min(20, n - s)
        p = PriorTasks(k, 10, "cpu", g)
        d = torch.randint(1, 11, (k,), generator=g)
        p.resample(torch.arange(k), d)
        x = torch.rand(k, n_cfg, 10, generator=g)
        y = p.curves(x, torch.full((k,), T), T).numpy()
        out += [(y[e], x[e, :, :int(d[e])].numpy()) for e in range(k)]
    return out


def faker_hgp_v0(n, rng):
    from curvefaker import faker_hgp
    return faker_hgp.faker(n, rng)


def faker_hgp_v1(n, rng):
    from curvefaker import faker_hgp
    return faker_hgp.faker_v1(n, rng)


def faker_hgp_v2(n, rng):
    from curvefaker import faker_hgp
    return faker_hgp.faker_v1(n, rng, warp_level=True)


def faker_hgp_v3(n, rng):
    from curvefaker import faker_hgp
    return faker_hgp.faker_v1(n, rng, warp_level=True, warp_time=True)


def faker_hgp_v4(n, rng):
    from curvefaker import faker_hgp
    return faker_hgp.faker_v4(n, rng)


FAKERS = {"hgp-v4": faker_hgp_v4, "ifbo": faker_ifbo, "real-fit": faker_real_fit, "hgp-v0": faker_hgp_v0, "hgp-v1": faker_hgp_v1,
          "hgp-v2": faker_hgp_v2, "hgp-v3": faker_hgp_v3}


def main():
    names = sys.argv[1:] or list(FAKERS)
    real = real_select()
    rd = [evaluate.descriptors(y, X) for y, X, _ in real]
    src = np.array([s for _, _, s in real])
    rows, gaps = [], []
    for name in names:
        reps, greps = [], []
        for seed in SEEDS:   # repeated generation: mean and sd over seeds
            ys = FAKERS[name](len(real), np.random.default_rng(seed))
            gd = [evaluate.descriptors(y, X) for y, X in ys]
            cov = evaluate.coverage(rd, gd)
            r = {"coverage": cov["coverage"], "precision": cov["precision"]}
            for s in np.unique(src):
                r[f"cov[{s}]"] = float(cov["covered"][src == s].mean())
            reps.append(r)
            greps.append(evaluate.per_descriptor_gap(rd, gd))
        R = pd.DataFrame(reps)
        row = {"faker": name, **R.mean().to_dict(), **{f"sd_{k}": v for k, v in R.std().items()
                                                        if k in ("coverage", "precision")}}
        rows.append(row)
        gaps.append({"faker": name, **pd.DataFrame(greps).mean().to_dict()})
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    df, gp = pd.DataFrame(rows).round(3), pd.DataFrame(gaps).round(2)
    print(df.to_string(index=False), "\n\nper-descriptor gap (Wasserstein / real sd)\n", gp.to_string(index=False))
    out = REPO / "results" / "faker_eval.csv"
    if out.exists() and sys.argv[1:]:
        old = pd.read_csv(out)
        df = pd.concat([old[~old.faker.isin(df.faker)], df])
    df.to_csv(out, index=False)
    gout = REPO / "results" / "faker_gaps.csv"
    if gout.exists() and sys.argv[1:]:
        old = pd.read_csv(gout)
        gp = pd.concat([old[~old.faker.isin(gp.faker)], gp])
    gp.to_csv(gout, index=False)


if __name__ == "__main__":
    main()
