"""Convert JAHS-Bench-201 raw metric data (raw.pkl.gz, one row per run x epoch) to one compact npz per dataset.

The pickle is the official distribution format; unpickling runs code, so run this with an isolated interpreter
from outside the data directory (python -I). Standalone on purpose (no curvefaker import under -I); JAHS is not
on the held-out list.

    python -I scripts/convert_jahs.py data/jahs/raw/<dataset>/raw.pkl.gz data/jahs/<dataset>.npz

Output arrays, one row per run (taskid, model_idx): hp columns (N, W, Resolution, Activation, Op1-6,
LearningRate, Optimizer, TrivialAugment, WeightDecay), valid_acc / test_acc / train_loss [runs, max_epoch]
(NaN-padded), epochs [max_epoch].
"""
import sys
import time

import numpy as np
import pandas as pd

HP = ["N", "W", "Resolution", "Activation", "Op1", "Op2", "Op3", "Op4", "Op5", "Op6", "LearningRate", "Optimizer",
      "TrivialAugment", "WeightDecay"]


def main(src, dst):
    t0 = time.time()
    df = pd.read_pickle(src)
    print(f"loaded {df.shape} in {time.time() - t0:.0f}s", flush=True)
    mc = df["model_config"]
    first = mc.groupby(level=["taskid", "model_idx"]).first()
    out = {f"hp_{h}": first[h].to_numpy() for h in HP}
    for h in ("Activation", "Optimizer"):
        out[f"hp_{h}"] = out[f"hp_{h}"].astype(str)
    out["hp_TrivialAugment"] = out["hp_TrivialAugment"].astype(bool)
    for key, col in (("valid_acc", ("valid", "acc")), ("test_acc", ("test", "acc")), ("train_loss", ("train", "loss"))):
        w = df[col].astype(np.float32).unstack("Epoch")
        w = w.reindex(first.index)
        out[key] = w.to_numpy(np.float32)
        out["epochs"] = w.columns.to_numpy()
        del w
    out["run_id"] = np.array([f"{a}_{b}" for a, b in first.index])
    np.savez_compressed(dst, **out)
    print(f"wrote {dst}: {len(first)} runs, {out['valid_acc'].shape[1]} epochs, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
