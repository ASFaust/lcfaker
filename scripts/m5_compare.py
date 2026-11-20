"""Seed-aware comparison of M5 results (port of rogue-dagger's scripts/compare_seeds.py), per source.

    PYTHONPATH=. python scripts/m5_compare.py ifbo=wf03-n0-t0.25,wf03s1-n0-t0.25,wf03s2-n0-t0.25 \
        noisefix=wf03noise-n0-t0.25,... --ref ifbo [--split select]

Per group, the per-task mean final regret is averaged over its models (training seeds). CI: two-level bootstrap
(tasks; models within a group). Reported for all tasks and per source (taskset, fcnet, bbomix, ...).
"""
import argparse

import numpy as np

from curvefaker.paths import REPO


def load(split, tag):
    z = np.load(REPO / "runs" / "m5" / split / f"{tag}.npz")
    return dict(zip(map(str, z["tasks"]), z["regret"][:, :, -1].mean(1)))


def report(name, tasks, M, ref, boot, rng):
    T = len(tasks)
    print(f"\n{name}: {T} tasks")
    for n, x in M.items():
        print(f"  {n:12s} models {len(x)}  mean {x.mean():.4f}  per model " + " ".join(f"{v:.4f}" for v in x.mean(1)))
    R = M[ref]
    for n, x in M.items():
        if n == ref:
            continue
        point = (R.mean(0) - x.mean(0)).mean()
        b = np.empty(boot)
        for k in range(boot):
            ti = rng.integers(0, T, T)
            b[k] = R[rng.integers(0, len(R), len(R))][:, ti].mean() - x[rng.integers(0, len(x), len(x))][:, ti].mean()
        lo, hi = np.percentile(b, [2.5, 97.5])
        wins = int(((R.mean(0) - x.mean(0)) > 0).sum())
        print(f"  {n} vs {ref}: better on {wins}/{T}, gain {point:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", help="name=tag1,tag2,...")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--split", default="select")
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    G = {}
    for g in a.groups:
        name, tags = g.split("=", 1)
        G[name] = [load(a.split, t) for t in tags.split(",")]
    tasks = sorted(set.intersection(*[set(m) for ms in G.values() for m in ms]))
    rng = np.random.default_rng(0)
    M = {n: np.array([[m[t] for t in tasks] for m in ms]) for n, ms in G.items()}
    report("all", tasks, M, a.ref, a.boot, rng)
    for src in sorted({t.split("/")[0] for t in tasks}):
        idx = [i for i, t in enumerate(tasks) if t.startswith(src + "/")]
        report(src, [tasks[i] for i in idx], {n: x[:, idx] for n, x in M.items()}, a.ref, a.boot, rng)


if __name__ == "__main__":
    main()
