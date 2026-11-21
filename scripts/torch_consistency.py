"""release/consistency.md: the GPU prior (lcfaker.torch_prior, faker "bpl-torch") against generate() ("bpl-v16")
on the paper's statistics (select split, 189 held-back real tasks; HPO: mean over generation seeds 6, 7, 8).

    PYTHONPATH=. python scripts/evaluate_axes.py bpl-torch; HPO_SEED={6,7,8} python scripts/evaluate_hpo.py bpl-torch
    PYTHONPATH=. python scripts/torch_consistency.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from compare_table import ROWS  # noqa: E402
from curvefaker.paths import REPO  # noqa: E402

NAMES = [("bpl-v16", "generate()"), ("bpl-torch", "GPU prior (torch_prior)")]


def main():
    h = pd.read_csv(REPO / "results" / "hpo_summary.csv", index_col=0)
    mean = lambda k: h.loc[[r for r in (k, f"{k}@7", f"{k}@8") if r in h.index], ["gain_hb", "gain_ft", "mae"]].mean()
    real = json.load(open(REPO / "results" / "axes_real.json"))
    real.update(mean("real").to_dict())
    vals = {}
    for k, _ in NAMES:
        a = json.load(open(REPO / "results" / f"axes_{k}.json"))
        a.update(mean(k).to_dict())
        vals[k] = a
    fmt = lambda x: "–" if x is None or not np.isfinite(x) else (f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}")
    L = ["# Consistency: GPU prior vs generate()", "",
         "Both are bpl-v16. Statistics of the paper on the select split (189 held-back real tasks, medians over",
         "tasks; HPO rows: mean over three generation seeds). The reference is real data; the two right columns",
         "should agree with each other.", "",
         "| | statistic | **real (reference)** | " + " | ".join(n for _, n in NAMES) + " | better |",
         "|---|---|---|---|---|---|"]
    for sec, label, key, better in ROWS:
        r = real.get(key, np.nan)
        rr = "–" if key in ("B_c2st_auc", "B_precision", "B_recall") else fmt(r)
        hint = {"closer": "closer to real", "lower": "lower", "higher": "higher"}[better]
        L.append(f"| {sec} | {label} | {rr} | " + " | ".join(fmt(vals[k].get(key, np.nan)) for k, _ in NAMES)
                 + f" | {hint} |")
    L += ["", "Differences: the GPU prior's landscapes are somewhat smoother (fewer local optima, larger near-best",
          "region, more predictable speed) and its median final value is lower. Likely causes: random Fourier",
          "features approximate the rough landscape part with 256 features, and class shares are thresholds set on",
          "512 reference configs instead of exact counts per task. HPO behaviour agrees within seed variation."]
    (REPO / "release" / "consistency.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
