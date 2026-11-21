"""Measure 2 for fakers: coverage / precision of the real *select* tasks, and per-descriptor gaps (curvefaker/evaluate.py).

novelty: median distance of generated tasks to the nearest of 600 fit tasks, relative to the same for the real
select tasks (~0: copies of fit tasks, ~1: as new as an unseen real family).
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
    hgp-v1w  : v1 with records weighted like the fit split (diagnostic: task mixture vs landscape)
    hgp-v5   : v1w with landscape predictability matched to each record's real task (nugget)
    setvae   : curvefaker/setvae.py (learned hierarchical VAE over curve sets + v5 landscapes)
    setvae-v2: setvae with the task latent drawn from a mixture fitted to fit tasks' posteriors
    records-raw: diagnostic, the real tasks behind v1w's records, raw
    comp     : curvefaker/faker_comp.py (learned live curves anchored to a fit record + explicit dead / diverged)
    comp-v2  : comp with config latents from the record's own posterior (latent-space copula)
    swt-v1   : curvefaker/faker_swt.py (shape-warp-template: template in log-time + config-driven warps)
    swt-v2   : swt-v1 with speed as its own smooth function of the config (nearest real warp to a target)
    swt-v3   : swt-v2 with a rough, spatially coherent unpredictable landscape part instead of white noise
    swt-v4   : swt-v3 with the noise level from step-to-step differences instead of fit residuals
    swt-v5   : swt-v4 with records fitted on all live curves of a task (no ~5x re-use of warp vectors)
    swt-v6   : swt-v5 with the final-value coordinate weighted 4x in the warp matching
    swt-v7   : swt-v6, warps constructed (final value and beta exact quantiles of the targets, alpha solved)
    swt-v8   : swt-v7, rough landscape part on the final-value field only (not on speed)
    swt-v9   : swt-v8, constructed curves that undershoot the task's lowest fitted real curve fall back to matched vectors
    swt-v10  : swt-v9, landscape field calibrated to the live configs' predictability of the real task
    swt-v11  : swt-v10, dead level relative to the generated task's own best and first-value 90th percentile
    bpl-v1   : broken power laws in schedule-aware progress time on the log-excess scale, overfitting, raw units
    bpl-v2   : bpl-v1 with joint real parameter vectors (nearest to field targets), floor offset solved
    bpl-v3   : bpl-v2, learning-rate dimension over the live configs' range
    bpl-v4   : bpl-v3, real vectors matched also on the config's progress budget (schedule, learning rate)
    bpl-v5   : bpl-v4, second slow stage: floor offset decays as x^-b (records refitted with 5 parameters)
    bpl-v6   : bpl-v5, noise level and time profile from second differences
    bpl-v7   : bpl-v5 (not v6), relative speed constructed from the speed field, tau from the config's progress budget
    bpl-v8   : bpl-v5, tau solved so the time to 50% progress hits the speed field's quantile of real values
    bpl-v9   : bpl-v8, start-consistent curves (amplitude and floor solved from start level and final target)
    bpl-v10  : bpl-v5, targets include the record's real learning-rate relation; matching on log learning rate
    bpl-v11  : bpl-v10, overfitting only among live configs, curve held at the onset and rising from there
    bpl-v12  : bpl-v10 (not v11), start-consistent TaskSet curves (amplitude and floor from start level and final)
    bpl-v13  : bpl-v1 (copula) + the record's real learning-rate relation as conditional mean of each score
    bpl-v14  : bpl-v13, floor offset solved so the final hits a quantile of the record's real finals
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


def faker_hgp_v1w(n, rng):
    from curvefaker import faker_hgp
    return faker_hgp.faker_v1w(n, rng)


def faker_hgp_v5(n, rng):
    from curvefaker import faker_hgp
    return faker_hgp.faker_v5(n, rng)


def faker_setvae(n, rng):
    from curvefaker import setvae
    return setvae.faker(n, rng)


def faker_setvae_v2(n, rng):
    from curvefaker import setvae
    return setvae.faker_v2(n, rng)


def faker_records_raw(n, rng):
    """Diagnostic: the real tasks behind v1w's records, drawn with v1w's weights, loaded raw (no resampling)."""
    from curvefaker import faker_hgp as F
    P = dict(np.load(F.PATH_V1))
    src = np.array([t.split("/")[0] for t in P["task"]])
    counts = {s: len(data.ids("fit", [s])) for s in np.unique(src)}
    w = np.array([counts[s] / (src == s).sum() for s in src])
    w /= w.sum()
    return [load_y(str(P["task"][int(rng.choice(len(w), p=w))]))[:2] for _ in range(n)]


def faker_comp(n, rng):
    from curvefaker import faker_comp
    return faker_comp.faker(n, rng)


def faker_comp_v2(n, rng):
    from curvefaker import faker_comp
    return faker_comp.faker_v2(n, rng)


def faker_swt_v1(n, rng):
    from curvefaker import faker_swt
    return faker_swt.faker(n, rng, version=1)


def faker_swt_v2(n, rng):
    from curvefaker import faker_swt
    return faker_swt.faker(n, rng, version=2)


def faker_swt_v3(n, rng):
    from curvefaker import faker_swt
    return faker_swt.faker(n, rng, version=3)


def faker_swt_v4(n, rng):
    from curvefaker import faker_swt
    return faker_swt.faker(n, rng, version=4)


def faker_swt_v5(n, rng):
    from curvefaker import faker_swt
    return faker_swt.faker(n, rng, version=5)


def faker_swt_v6(n, rng):
    from curvefaker import faker_swt
    return faker_swt.faker(n, rng, version=6)


def faker_swt_v7(n, rng):
    from curvefaker import faker_swt
    return faker_swt.faker(n, rng, version=7)


def faker_swt_v8(n, rng):
    from curvefaker import faker_swt
    return faker_swt.faker(n, rng, version=8)


def faker_swt_v9(n, rng):
    from curvefaker import faker_swt
    return faker_swt.faker(n, rng, version=9)


def faker_swt_v10(n, rng):
    from curvefaker import faker_swt
    return faker_swt.faker(n, rng, version=10)


def faker_bpl_v14(n, rng):
    from curvefaker import faker_bpl
    return faker_bpl.faker(n, rng, version=14)


def faker_bpl_v13(n, rng):
    from curvefaker import faker_bpl
    return faker_bpl.faker(n, rng, version=13)


def faker_bpl_v12(n, rng):
    from curvefaker import faker_bpl
    return faker_bpl.faker(n, rng, version=12)


def faker_bpl_v11(n, rng):
    from curvefaker import faker_bpl
    return faker_bpl.faker(n, rng, version=11)


def faker_bpl_v10(n, rng):
    from curvefaker import faker_bpl
    return faker_bpl.faker(n, rng, version=10)


def faker_bpl_v9(n, rng):
    from curvefaker import faker_bpl
    return faker_bpl.faker(n, rng, version=9)


def faker_bpl_v8(n, rng):
    from curvefaker import faker_bpl
    return faker_bpl.faker(n, rng, version=8)


def faker_bpl_v7(n, rng):
    from curvefaker import faker_bpl
    return faker_bpl.faker(n, rng, version=7)


def faker_bpl_v6(n, rng):
    from curvefaker import faker_bpl
    return faker_bpl.faker(n, rng, version=6)


def faker_bpl_v5(n, rng):
    from curvefaker import faker_bpl
    return faker_bpl.faker(n, rng, version=5)


def faker_bpl_v4(n, rng):
    from curvefaker import faker_bpl
    return faker_bpl.faker(n, rng, version=4)


def faker_bpl_v3(n, rng):
    from curvefaker import faker_bpl
    return faker_bpl.faker(n, rng, version=3)


def faker_bpl_v2(n, rng):
    from curvefaker import faker_bpl
    return faker_bpl.faker(n, rng, version=2)


def faker_bpl_v1(n, rng):
    from curvefaker import faker_bpl
    return faker_bpl.faker(n, rng, version=1)


def faker_swt_v11(n, rng):
    from curvefaker import faker_swt
    return faker_swt.faker(n, rng, version=11)


FAKERS = {"bpl-v14": faker_bpl_v14, "bpl-v13": faker_bpl_v13, "bpl-v12": faker_bpl_v12, "bpl-v11": faker_bpl_v11, "bpl-v10": faker_bpl_v10, "bpl-v9": faker_bpl_v9, "bpl-v8": faker_bpl_v8, "bpl-v7": faker_bpl_v7, "bpl-v6": faker_bpl_v6, "bpl-v5": faker_bpl_v5, "bpl-v4": faker_bpl_v4, "bpl-v3": faker_bpl_v3, "bpl-v2": faker_bpl_v2, "bpl-v1": faker_bpl_v1, "swt-v11": faker_swt_v11, "swt-v10": faker_swt_v10, "swt-v9": faker_swt_v9, "swt-v8": faker_swt_v8, "swt-v7": faker_swt_v7, "swt-v6": faker_swt_v6, "swt-v5": faker_swt_v5, "swt-v4": faker_swt_v4, "swt-v3": faker_swt_v3, "swt-v2": faker_swt_v2, "swt-v1": faker_swt_v1, "comp-v2": faker_comp_v2, "comp": faker_comp, "records-raw": faker_records_raw, "setvae-v2": faker_setvae_v2, "setvae": faker_setvae, "hgp-v5": faker_hgp_v5, "hgp-v1w": faker_hgp_v1w, "hgp-v4": faker_hgp_v4, "ifbo": faker_ifbo, "real-fit": faker_real_fit, "hgp-v0": faker_hgp_v0, "hgp-v1": faker_hgp_v1,
          "hgp-v2": faker_hgp_v2, "hgp-v3": faker_hgp_v3}


def fit_pool(n=600):
    """Descriptors of n random fit tasks (cached): the reference for novelty."""
    path = REPO / "results" / "fit_descriptors.csv"
    if path.exists():
        return pd.read_csv(path).to_dict("records")
    rng = np.random.default_rng(9)
    ids = rng.choice(data.ids("fit"), n, replace=False)
    rows = [evaluate.descriptors(*load_y(i)[:2]) for i in ids]
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


def novelty(real_d, gen_d, pool):
    """Median distance of generated tasks to their nearest fit task, divided by the same for the real select tasks
    (standardised like coverage). ~0: copies of fit tasks; ~1: as new as an unseen real task family."""
    R, Gm, F = evaluate._matrix(real_d), evaluate._matrix(gen_d), evaluate._matrix(pool)
    mu, sd = np.nanmean(R, 0), np.nanstd(R, 0) + 1e-9
    R, Gm, F = (np.nan_to_num((A - mu) / sd) for A in (R, Gm, F))
    nn = lambda A: np.linalg.norm(A[:, None] - F[None], axis=-1).min(1)
    return float(np.median(nn(Gm)) / np.median(nn(R)))


def main():
    names = sys.argv[1:] or list(FAKERS)
    real = real_select()
    rd = [evaluate.descriptors(y, X) for y, X, _ in real]
    src = np.array([s for _, _, s in real])
    pool = fit_pool()
    rows, gaps = [], []
    for name in names:
        reps, greps = [], []
        for seed in SEEDS:   # repeated generation: mean and sd over seeds
            ys = FAKERS[name](len(real), np.random.default_rng(seed))
            gd = [evaluate.descriptors(y, X) for y, X in ys]
            cov = evaluate.coverage(rd, gd)
            r = {"coverage": cov["coverage"], "precision": cov["precision"], "novelty": novelty(rd, gd, pool)}
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
