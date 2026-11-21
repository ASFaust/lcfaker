"""Readable comparison of fakers (markdown): one reference column (real select tasks), the raw statistic per faker,
a "better" column, and the best faker per row in bold.

    PYTHONPATH=. python scripts/compare_table.py faker [faker ...] > results/compare.md

Needs results/axes_real.json and results/axes_<faker>.json (scripts/evaluate_axes.py), results/hpo_summary.csv.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from curvefaker.paths import REPO

# (section, label, key, better): better = "closer" (to real), "lower", "higher"
ROWS = [
    ("Curves", "time to 50% of a curve's progress (log budget fraction)", "B_median_u50", "closer"),
    ("Curves", "progress made in the last 20% of the budget", "B_median_late", "closer"),
    ("Curves", "share of steps where the smoothed curve rises", "B_median_rise_share", "closer"),
    ("Curves", "share of curves that rise after their minimum (overfitting / instability)", "B_share_over", "closer"),
    ("Curves", "final value (normalised)", "B_median_final", "closer"),
    ("Curves", "classifier telling real from generated curves (AUC; 0.5 = cannot)", "B_c2st_auc", "lower"),
    ("Curves", "generated curves that look real (precision)", "B_precision", "higher"),
    ("Curves", "real curves the faker can produce (recall)", "B_recall", "higher"),
    ("Noise", "excess kurtosis of residuals (heavy tails)", "C_median_kurtosis", "closer"),
    ("Noise", "residual spread, last third / first third", "C_median_late_vs_early_sd", "closer"),
    ("Landscape", "final value predictable from config (rank corr., 200 configs)", "A_median_rho200", "closer"),
    ("Landscape", "share of configs within 0.05 of the best", "A_median_near_best", "closer"),
    ("Landscape", "ruggedness: local optima vs random (1 = random)", "A_median_optima_ratio", "closer"),
    ("Landscape", "curve speed predictable from config (R²)", "A_median_shape_from_config", "closer"),
    ("HPO", "Hyperband gain over random search", "gain_hb", "closer"),
    ("HPO", "freeze-thaw gain over random search", "gain_ft", "closer"),
    ("HPO", "mean abs. difference of regrets from real", "mae", "lower"),
]


def main():
    names = sys.argv[1:]
    real = json.load(open(REPO / "results" / "axes_real.json"))
    h = pd.read_csv(REPO / "results" / "hpo_summary.csv", index_col=0)
    vals = {}
    for n in names:
        a = json.load(open(REPO / "results" / f"axes_{n}.json"))
        if n in h.index:
            a.update({k: float(h.loc[n, k]) for k in ("gain_hb", "gain_ft", "mae")})
        vals[n] = a
    real.update({k: float(h.loc["real", k]) for k in ("gain_hb", "gain_ft", "mae")})
    print("| | statistic | **real (reference)** | " + " | ".join(names) + " | better |")
    print("|---|---|---|" + "---|" * len(names) + "---|")
    for sec, label, key, better in ROWS:
        r = real.get(key, np.nan)
        v = [vals[n].get(key, np.nan) for n in names]
        v = [np.nan if x is None else x for x in v]
        if better == "closer":
            score = [abs(x - r) if np.isfinite(x) else np.inf for x in v]
        elif better == "lower":
            score = [x if np.isfinite(x) else np.inf for x in v]
        else:
            score = [-x if np.isfinite(x) else np.inf for x in v]
        best = int(np.argmin(score)) if np.isfinite(min(score)) else -1
        fmt = lambda x: "–" if not np.isfinite(x) else (f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}")
        cells = [(f"**{fmt(x)}**" if j == best else fmt(x)) for j, x in enumerate(v)]
        rr = "–" if key in ("B_c2st_auc", "B_precision", "B_recall") else fmt(r)
        hint = {"closer": "closer to real", "lower": "lower", "higher": "higher"}[better]
        print(f"| {sec} | {label} | {rr} | " + " | ".join(cells) + f" | {hint} |")


if __name__ == "__main__":
    main()
