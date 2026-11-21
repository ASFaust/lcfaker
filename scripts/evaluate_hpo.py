"""Measure 3: do existing HPO algorithms behave on generated tasks as they do on real ones? (curvefaker/hpo.py)

Real: the same select tasks as scripts/evaluate_fakers.py (one seed, at most 1000 configs). Generated: equal
numbers per faker. Reported per algorithm and budget (in full trainings): mean normalised regret, and per faker
    mae       : mean absolute difference of its mean regrets from the real ones, over algorithms and budgets
    gain_hb   : mean over budgets of regret(random) - regret(hyperband), i.e. how much Hyperband helps;
    gain_ft   : the same for the freeze-thaw heuristic. Compare with the real gains.
    se_gain_* : standard error of the gain over tasks (the gain varies a lot from task to task: about 0.01 for
                ~190 tasks, so differences below ~0.03 are not meaningful).

    PYTHONPATH=. python scripts/evaluate_hpo.py [faker ...]
"""
from __future__ import annotations

import os
import sys
from multiprocessing import get_context

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import evaluate_fakers as EF  # noqa: E402
from curvefaker import hpo  # noqa: E402
from curvefaker.paths import REPO  # noqa: E402

GEN_SEED = int(os.environ.get("HPO_SEED", 6))   # generation seed; results of other seeds are stored as name@seed


def _eval(y):
    return hpo.evaluate_task(y)


def summarise(rows):
    """Mean regrets, plus per-task gains (random - hyperband / freezethaw, mean over budgets) with standard errors."""
    df = pd.DataFrame(rows)
    out = df.mean()
    for alg, key in (("hyperband", "gain_hb"), ("freezethaw", "gain_ft")):
        g = np.mean([df[f"random@{b}"] - df[f"{alg}@{b}"] for b in hpo.BUDGETS], 0)
        out[f"se_{key}"] = float(g.std() / np.sqrt(len(g)))
    return out


def main():
    names = sys.argv[1:] or ["ifbo", "real-fit", "hgp-v1", "hgp-v2"]
    real = [y for y, _, _ in EF.real_select()]
    with get_context("spawn").Pool(32) as pool:
        res = {"real": summarise(pool.map(_eval, real))}
        for name in names:
            ys = [y for y, _ in EF.FAKERS[name](len(real), np.random.default_rng(GEN_SEED))]
            res[name if GEN_SEED == 6 else f"{name}@{GEN_SEED}"] = summarise(pool.map(_eval, ys))
            print(name, "done", flush=True)
    df = pd.DataFrame(res).T
    out = pd.DataFrame(index=df.index)
    se_cols = [c for c in df.columns if c.startswith("se_")]
    for name in df.index:
        out.loc[name, "mae"] = float((df.loc[name].drop(se_cols) - df.loc["real"].drop(se_cols)).abs().mean())
        for c in se_cols:
            out.loc[name, c] = df.loc[name, c]
        for alg, key in (("hyperband", "gain_hb"), ("freezethaw", "gain_ft")):
            out.loc[name, key] = float(np.mean([df.loc[name, f"random@{b}"] - df.loc[name, f"{alg}@{b}"]
                                                for b in hpo.BUDGETS]))
    for frame, fname in ((df, "hpo_regret.csv"), (out, "hpo_summary.csv")):   # merge with earlier runs
        path = REPO / "results" / fname
        if path.exists():
            old = pd.read_csv(path, index_col=0)
            frame = pd.concat([old[~old.index.isin(frame.index)], frame])
        frame.round(3).to_csv(path)
    print(df.round(2).to_string(), "\n\n", out.round(3).to_string())


if __name__ == "__main__":
    main()
