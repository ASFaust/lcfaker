"""Measure 3: do existing HPO algorithms behave on generated tasks as they do on real ones? (curvefaker/hpo.py)

Real: the same select tasks as scripts/evaluate_fakers.py (one seed, at most 1000 configs). Generated: equal
numbers per faker. Reported per algorithm and budget (in full trainings): mean normalised regret, and per faker
    mae       : mean absolute difference of its mean regrets from the real ones, over algorithms and budgets
    gain_hb   : mean over budgets of regret(random) - regret(hyperband), i.e. how much Hyperband helps;
    gain_ft   : the same for the freeze-thaw heuristic. Compare with the real gains.

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


def _eval(y):
    return hpo.evaluate_task(y)


def summarise(rows):
    df = pd.DataFrame(rows)
    return df.mean()


def main():
    names = sys.argv[1:] or ["ifbo", "real-fit", "hgp-v1", "hgp-v2"]
    real = [y for y, _, _ in EF.real_select()]
    with get_context("spawn").Pool(32) as pool:
        res = {"real": summarise(pool.map(_eval, real))}
        for name in names:
            ys = [y for y, _ in EF.FAKERS[name](len(real), np.random.default_rng(6))]
            res[name] = summarise(pool.map(_eval, ys))
            print(name, "done", flush=True)
    df = pd.DataFrame(res).T
    out = pd.DataFrame(index=df.index)
    for name in df.index:
        out.loc[name, "mae"] = float((df.loc[name] - df.loc["real"]).abs().mean())
        for alg, key in (("hyperband", "gain_hb"), ("freezethaw", "gain_ft")):
            out.loc[name, key] = float(np.mean([df.loc[name, f"random@{b}"] - df.loc[name, f"{alg}@{b}"]
                                                for b in hpo.BUDGETS]))
    df.round(3).to_csv(REPO / "results" / "hpo_regret.csv")
    out.round(3).to_csv(REPO / "results" / "hpo_summary.csv")
    print(df.round(2).to_string(), "\n\n", out.round(3).to_string())


if __name__ == "__main__":
    main()
