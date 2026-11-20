"""Are dead (flat) configs a region of config space, or per-config bad luck? (docs/avenues.md, section 2A)

Per task: label each config flat (own range < 2% of the task's, as in measures.stats); 5-fold cross-validated AUC
of a random forest predicting the label from the config. AUC near 0.5: flat configs are scattered (luck); near 1:
a region (e.g. learning rate too high). Also: which hp matters most (forest importance).

    PYTHONPATH=. python scripts/flat_clustering.py [--split fit] [--sources ...] [--per-source 60]
"""
import argparse
import collections

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from curvefaker import data
from curvefaker.measures import curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="fit")
    ap.add_argument("--sources", nargs="+", default=list(data.SOURCES))
    ap.add_argument("--per-source", type=int, default=60)
    a = ap.parse_args()
    for src in a.sources:
        I = data.ids(None if a.split == "all" else a.split, [src])
        if not I:
            continue
        rng = np.random.default_rng(0)
        I = sorted(rng.choice(I, min(a.per_source, len(I)), replace=False))
        aucs, fracs, top = [], [], collections.Counter()
        for i in I:
            t = data.load(i)
            Y, X = curves(t, rng=np.random.default_rng(0), cat="onehot")
            lo, hi = np.quantile(Y, 0.01), np.quantile(Y, 0.99)
            flat = (Y.max(1) - Y.min(1)) < 0.02 * max(hi - lo, 1e-9)
            fracs.append(flat.mean())
            if flat.sum() < 10 or (~flat).sum() < 10 or X.shape[1] == 0:
                continue
            clf = RandomForestClassifier(200, min_samples_leaf=3, n_jobs=8, random_state=0)
            p = cross_val_predict(clf, X, flat, cv=5, method="predict_proba")[:, 1]
            aucs.append(roc_auc_score(flat, p))
            clf.fit(X, flat)
            names = [n for n, s in t.space.items()]   # approximate: one-hot expands categoricals
            j = int(np.argmax(clf.feature_importances_))
            top[names[j] if X.shape[1] == len(names) else f"col{j}"] += 1
        q = lambda v: np.quantile(v, [0.25, 0.5, 0.75]) if len(v) else [np.nan] * 3
        fa, au = q(fracs), q(aucs)
        print(f"{src:12s} tasks {len(I):4d}  flat frac median {fa[1]:.2f} [{fa[0]:.2f},{fa[2]:.2f}]  "
              f"tasks with >=10 flat: {len(aucs):4d}  AUC median {au[1]:.2f} [{au[0]:.2f},{au[2]:.2f}]  "
              f"top hp: {top.most_common(3)}", flush=True)


if __name__ == "__main__":
    main()
