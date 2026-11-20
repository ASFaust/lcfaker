"""Split the converted JAHS-Bench-201 files (data/jahs/<dataset>.npz, from convert_jahs.py) into one npz per task.

A task is one (dataset, N, W, Resolution): the architecture size and image resolution are fixed, the search space is
the cell operations Op1-6, Activation, LearningRate, WeightDecay and TrivialAugment (Optimizer is always SGD).

Only runs trained for the full 200 epochs are kept; the raw data also has runs stopped at lower budgets (1/10 or
50/100/150 epochs), which are separate runs, not prefixes. The raw data repeats some configs: exact copies of a run
(identical curve) are dropped, distinct runs of the same config become replicas (at most MAX_R, NaN-padded).

    python scripts/prepare_jahs.py        # writes data/jahs/tasks/<dataset>/N<N>_W<W>_R<Res>.npz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from curvefaker.paths import DATA  # noqa: E402

DATASETS = ["cifar10", "colorectal_histology", "fashion_mnist"]
HP = ["Activation", "Op1", "Op2", "Op3", "Op4", "Op5", "Op6", "LearningRate", "WeightDecay", "TrivialAugment"]
MAX_R = 5


def main():
    for ds in DATASETS:
        z = np.load(DATA / "jahs" / f"{ds}.npz")
        acc = z["valid_acc"]
        full = np.isfinite(acc).all(1)
        hp = pd.DataFrame({h: z[f"hp_{h}"][full] for h in ["N", "W", "Resolution"] + HP})
        y = (100.0 - acc[full]).astype(np.float32)                 # validation error (%), lower is better
        yt = (100.0 - z["test_acc"][full]).astype(np.float32)
        out = DATA / "jahs" / "tasks" / ds
        out.mkdir(parents=True, exist_ok=True)
        for (n, w, r), g in hp.groupby(["N", "W", "Resolution"]):
            idx = g.index.to_numpy()
            key = pd.util.hash_pandas_object(g[HP].round(10), index=False).to_numpy()
            configs, reps, n_copies = {}, {}, 0
            for i, k in zip(idx, key):
                lst = reps.setdefault(k, [])
                if any(np.array_equal(y[i], y[j]) for j in lst):
                    n_copies += 1
                    continue
                if len(lst) < MAX_R:
                    lst.append(i)
                configs.setdefault(k, i)
            keys = list(configs)
            R = max(len(reps[k]) for k in keys)
            Y = np.full((len(keys), R, y.shape[1]), np.nan, np.float32)
            for a, k in enumerate(keys):
                Y[a, :len(reps[k])] = y[reps[k]]
            first = np.array([configs[k] for k in keys])
            Yt = yt[first]
            hps = {f"hp_{h}": hp.loc[first, h].to_numpy() for h in HP}
            hps["hp_Activation"] = hps["hp_Activation"].astype(str)
            name = f"N{int(n)}_W{int(w)}_R{r:g}"
            np.savez_compressed(out / f"{name}.npz", y=Y, y_test=Yt, epochs=z["epochs"], **hps)
            nrep = np.isfinite(Y[:, :, -1]).sum(1)
            print(f"{ds}/{name}: {len(keys)} configs, replicas {np.bincount(nrep)[1:].tolist()}, "
                  f"{n_copies} exact copies dropped", flush=True)


if __name__ == "__main__":
    main()
