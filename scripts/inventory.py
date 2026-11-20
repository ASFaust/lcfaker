"""Inventory of the allowed data: one markdown row per (source, family). Used for docs/data.md.

    PYTHONPATH=. python scripts/inventory.py > docs/inventory.md
"""
import collections
import sys

import numpy as np

from curvefaker import data


def main():
    rows = collections.defaultdict(list)
    for src in sys.argv[1:] or data.SOURCES:
        for i in data.ids(sources=[src]):
            fam = data.family_of(i)
            if src == "taskset":    # reading every file is slow; shapes are fixed (1000 configs, 5 replicas, 51 pts)
                opt = i.split("/")[-1]
                d = {"adam1p": 1, "adam4p": 4, "adam6p": 6, "adam8p": 8, "nadamw": 10}[opt]
                rows[(src, fam)].append((1000, 5, 51, d))
            else:
                t = data.load(i)
                rows[(src, fam)].append((t.n, t.y.shape[1], t.T, len(t.space)))
    print("| source | family | split | tasks | configs/task | replicas | T | hps |")
    print("|---|---|---|---|---|---|---|---|")
    for (src, fam), r in sorted(rows.items()):
        a = np.array(r)
        rng = lambda c: f"{a[:, c].min()}" if a[:, c].min() == a[:, c].max() else f"{a[:, c].min()}-{a[:, c].max()}"
        print(f"| {src} | {fam} | {data.split_of(fam)} | {len(r)} | {rng(0)} | {rng(1)} | {rng(2)} | {rng(3)} |")


if __name__ == "__main__":
    main()
