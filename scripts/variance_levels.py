"""Analysis T1: which curve properties are task-level and which config-level?

On the live curves of scripts/shape_clusters.py and their template fits (scripts/shape_template.py): shape
location m (speed, log-time), steepness log k, half-progress time u50, late share (progress in the last 20%), and
the template misfit. Variance is split into: between sources, between tasks within a source, within tasks
(between configs). Plain grouped-mean decomposition (descriptive, unbalanced groups).

    PYTHONPATH=. python scripts/variance_levels.py
"""
import numpy as np
import pandas as pd

from curvefaker.paths import REPO

Z = np.load(REPO / "results" / "shape_clusters_curves.npz", allow_pickle=True)
F = np.load(REPO / "results" / "shape_template_fits.npz")
G = 40
grid = np.linspace(np.log(0.02), 0, G)
P = Z["P_log"]
df = pd.DataFrame({"source": Z["src"], "task": Z["task"],
                   "m (location)": np.clip(F["m"], -5, 1.5), "log k (steepness)": np.log(F["k"]),
                   "u50": [grid[np.argmax(r >= 0.5)] for r in P],
                   "late share": np.clip([1 - np.interp(0.8, np.exp(grid), r) for r in P], -0.2, 1),
                   "template misfit": F["rmse_log"]})
rows = []
for col in ["m (location)", "log k (steepness)", "u50", "late share", "template misfit"]:
    x = df[col]
    tot = x.var()
    src_mean = df.groupby("source")[col].transform("mean")
    task_mean = df.groupby("task")[col].transform("mean")
    rows.append({"feature": col, "between sources": ((src_mean - x.mean()) ** 2).mean() / tot,
                 "between tasks (within source)": ((task_mean - src_mean) ** 2).mean() / tot,
                 "within tasks (configs)": ((x - task_mean) ** 2).mean() / tot})
    # per source: task vs config share
    for s, g in df.groupby("source"):
        tm = g.groupby("task")[col].transform("mean")
        rows.append({"feature": col, "source": s, "between tasks (within source)": ((tm - g[col].mean()) ** 2).mean() / g[col].var(),
                     "within tasks (configs)": ((g[col] - tm) ** 2).mean() / g[col].var()})
out = pd.DataFrame(rows)
out.to_csv(REPO / "results" / "variance_levels.csv", index=False)
pd.set_option("display.width", 200)
print(out[out.source.isna()].drop(columns="source").round(2).to_string(index=False))
print(out[out.source.notna()].pivot(index="feature", columns="source", values="between tasks (within source)").round(2).to_string())
