"""M1 (curve statistics) and M4 (coverage): allowed real tasks vs ifBO-prior twins, per source.

Every real task gets `--twins` prior tasks with the same number of configs, hp dimension (capped at 10, the
prior's maximum) and curve length. Several prior variants run on identical seeds, so their twins differ only in
the varied knob. Default variants: ifbo (the original prior) and noisefix (noise x 0.25, smoothed over 1 point:
the rogue-dagger variant that was *worse* downstream, i.e. the meta-check pair).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python scripts/m1.py --split fit \
        [--sources taskset fcnet ...] [--per-source 300] [--twins 2] [--tag name]

Writes runs/m1/<tag>.json (per-task statistics) and prints the tables.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from curvefaker import data
from curvefaker.measures import compare, coverage, curves, ifbo_twins, replica_stats, stats
from curvefaker.paths import REPO

VARIANTS = {"ifbo": (1.0, 0.0), "noisefix": (0.25, 1.0)}
COVER_KEYS = ["rank_corr@0.1", "flip@0.1", "final_spread", "frac_near_best", "frac_flat", "frac_diverge",
              "improve_late", "frac_overfit", "noise", "frac_up_steps", "incr_autocorr", "cfg_smooth"]


def pick(ids, n, seed=0):
    if n is None or len(ids) <= n:
        return ids
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(ids, n, replace=False).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="fit", choices=["fit", "select", "all"])
    ap.add_argument("--sources", nargs="+", default=list(data.SOURCES))
    ap.add_argument("--per-source", type=int, default=300)
    ap.add_argument("--twins", type=int, default=2)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--replica", default="first", choices=["first", "mean"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or f"{a.split}_{a.replica}"
    out = {"args": vars(a), "real": {}, "fake": {v: {} for v in a.variants}}
    rng = np.random.default_rng(0)
    t0 = time.time()
    for src in a.sources:
        I = pick(data.ids(None if a.split == "all" else a.split, [src]), a.per_source)
        if not I:
            continue
        R, shapes, names = [], [], []
        for i in I:
            t = data.load(i)
            Y, X = curves(t, replica=a.replica, rng=np.random.default_rng(0))
            if len(Y) < 20 or Y.shape[1] < 7:
                continue
            s = stats(Y, X, rng)
            s.update(replica_stats(t))
            s["family"] = t.family
            R.append(s)
            shapes.append((Y.shape[0], X.shape[1], Y.shape[1]))
            names.append(i)
        out["real"][src] = dict(zip(names, R))
        print(f"\n=== {src} ({a.split}): {len(R)} real tasks, {len(set(r['family'] for r in R))} families, "
              f"{time.time() - t0:.0f}s", flush=True)
        S = {}
        for v in a.variants:
            ns, sm = VARIANTS[v]
            S[v] = [stats(Y, X, rng) for _, X, Y in ifbo_twins(shapes, a.twins, a.device, 0, ns, sm)]
            out["fake"][v][src] = S[v]
        rows = {v: {r[0]: r for r in compare(R, S[v])} for v in a.variants}
        keys = [r[0] for r in compare(R, S[a.variants[0]])]
        head = "statistic".ljust(16) + "real".rjust(8) + "".join(f"{v:>10}{'KS':>6}" for v in a.variants)
        print(head)
        for key in keys:
            line = key.ljust(16) + f"{rows[a.variants[0]][key][2]:8.3f}"
            for v in a.variants:
                _, D, _, fm = rows[v][key]
                line += f"{fm:10.3f}{D:6.2f}"
            print(line)
        print("KS sum (shared stats)".ljust(24) + "".join(
            f"{v}: {np.nansum([rows[v][k][1] for k in keys if k in COVER_KEYS]):.2f}  " for v in a.variants))
        for v in a.variants:
            c = coverage(R, S[v], COVER_KEYS)
            print(f"coverage {v}: recall {c['recall']:.2f} precision {c['precision']:.2f} "
                  f"({c['n_real']} real, {c['n_fake']} fake, {len(c['keys'])} stats)")
            out.setdefault("coverage", {}).setdefault(v, {})[src] = c
        for key in ("luck_ratio", "seed_rank_corr"):
            vals = np.array([r.get(key, np.nan) for r in R], float)
            if np.isfinite(vals).any():
                q = np.nanquantile(vals, [0.25, 0.5, 0.75])
                print(f"real {key}: median {q[1]:.3f} [IQR {q[0]:.3f}, {q[2]:.3f}]")
    p = REPO / "runs" / "m1" / f"{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
