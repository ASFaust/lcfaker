"""Data atlas: what kinds of learning curves exist per source, compared with the ifBO prior.

For a sample of tasks per source: normalise (curvefaker.curves), classify each config's mean curve, compute task
statistics. Writes results/atlas_tasks.csv (one row per task), results/atlas_summary.md (median over tasks per
source) and figures in docs/figures/.

    CUDA_VISIBLE_DEVICES=<one gpu> PYTHONPATH=. python scripts/atlas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from curvefaker import curves, data
from curvefaker.paths import REPO

RNG = np.random.default_rng(0)
MAX_CONFIGS = 1000
OUT = REPO / "results"
FIG = REPO / "docs" / "figures"
COLORS = {"improving": "#2a78d6", "dead": "#eb6834", "turns": "#1baf7a", "diverged": "#eda100"}


def real_tasks():
    """(source label, split, task id) for the sampled tasks."""
    out = []
    ts = data.ids(sources=["taskset"])
    for split in ("fit", "select"):
        pool = [i for i in ts if data.split_of(data.family_of(i)) == split]
        out += [("taskset", split, i) for i in RNG.choice(pool, 150, replace=False)]
    for src in ("nasbench201", "fcnet", "bbomix", "quicktune", "jahs"):
        for i in data.ids(sources=[src]):
            out.append((src, data.split_of(data.family_of(i)), i))
    return out


def task_curves(t: data.Task) -> np.ndarray:
    """Normalised replica-mean curves [n, T] of one task (configs subsampled, QuickTune: full-length runs only)."""
    y = t.y
    if t.source == "quicktune":
        y = y[np.isfinite(y[:, 0, -1])]
    if len(y) > MAX_CONFIGS:
        y = y[RNG.choice(len(y), MAX_CONFIGS, replace=False)]
    z, _ = curves.normalise(y)
    return curves.mean_curve(z)


def ifbo_tasks(n_tasks=300, n_cfg=1000, T=100, d_max=10):
    import torch
    sys.path.insert(0, str(REPO.parent / "rogue-dagger"))
    from roguedagger.prior import PriorTasks
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev).manual_seed(0)
    out = []
    chunk = 25
    for s in range(0, n_tasks, chunk):
        p = PriorTasks(chunk, d_max, dev, g)
        d = torch.randint(1, d_max + 1, (chunk,), device=dev, generator=g)
        p.resample(torch.arange(chunk, device=dev), d)
        x = torch.rand(chunk, n_cfg, d_max, device=dev, generator=g)
        y = p.curves(x, torch.full((chunk,), T, device=dev), T).cpu().numpy()
        for e in range(chunk):
            z, _ = curves.normalise(y[e])
            out.append(("ifbo", "-", f"ifbo/{s + e}", z))
    return out


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.astype(object).values)]
    return "\n".join([head, sep] + body) + "\n"


def plot_gallery(examples: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    srcs = list(examples)
    fig, axes = plt.subplots(len(srcs), 4, figsize=(13, 2.1 * len(srcs)), sharey=True)
    for r, src in enumerate(srcs):
        for c, (tid, m) in enumerate(examples[src][:4]):
            ax = axes[r, c]
            cls = curves.classify(m)
            idx = RNG.choice(len(m), min(40, len(m)), replace=False)
            x = np.linspace(0, 1, m.shape[1])
            for i in idx:
                ax.plot(x, m[i], lw=0.9, alpha=0.75, color=COLORS[curves.CLASSES[cls[i]]])
            ax.set_ylim(-0.1, 2.05)
            ax.set_title(tid if len(tid) < 48 else tid[:45] + "...", fontsize=7, color="#444")
            ax.tick_params(labelsize=7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if c == 0:
                ax.set_ylabel(src, fontsize=9)
    handles = [plt.Line2D([], [], color=v, lw=2, label=k) for k, v in COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.supxlabel("fraction of training budget", fontsize=9)
    fig.supylabel("normalised loss z (0 = best in task, 1 = typical start, 2 = cap)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows, examples = [], {}
    items = real_tasks()
    for k, (src, split, tid) in enumerate(items):
        t = data.load(tid)
        m = task_curves(t)
        st = curves.task_stats(m)
        rows.append({"source": src, "split": split, "task": tid, "n": len(m), "T": m.shape[1], **st})
        examples.setdefault(src, []).append((tid, m))
        if k % 50 == 0:
            print(f"{k}/{len(items)} {tid}", flush=True)
    for src, split, tid, m in ifbo_tasks():
        rows.append({"source": src, "split": split, "task": tid, "n": len(m), "T": m.shape[1],
                     **curves.task_stats(m)})
        examples.setdefault(src, [])
        if len(examples[src]) < 4:
            examples[src].append((tid, m))
    for src, ex in examples.items():   # 4 example tasks per source, spread over its task list
        examples[src] = [ex[i] for i in np.unique(np.linspace(0, len(ex) - 1, 4).round().astype(int))]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "atlas_tasks.csv", index=False)
    cols = ["frac_improving", "frac_dead", "frac_turns", "frac_diverged", "final_spread", "t50_median",
            "t90_median", "rank_early10", "rank_early10_live", "rank_early30_live", "top10_kept"]
    summ = df.groupby("source")[cols].median().round(2)
    summ.insert(0, "tasks", df.groupby("source").size())
    order = ["taskset", "nasbench201", "fcnet", "bbomix", "quicktune", "jahs", "ifbo"]
    summ = summ.loc[[s for s in order if s in summ.index]]
    (OUT / "atlas_summary.md").write_text(md_table(summ))
    print(summ.to_string())
    plot_gallery(examples, FIG / "atlas_gallery.png")


if __name__ == "__main__":
    main()
