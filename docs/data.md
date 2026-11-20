# Data inventory (2026-11-20)

All data here is **allowed** (README "Rule 1"; the held-out list is `heldout.yaml`, enforced by
`curvefaker/heldout.py` in every loader and downloader). The internal fit/select split is `splits.yaml`, by family.

## Where it lives

`$CURVEFAKER_DATA`, default `curve-faker/data/` (git-ignored, about 37 GB). `/shared/datasets` was the intended
home, but it is mounted read-only in the agent container, so it couldn't be written there. To move it there later:
copy `data/` to `/shared/datasets/learning-curves/` and set `CURVEFAKER_DATA=/shared/datasets/learning-curves`.

| Folder | Source | Size | Fetched with |
|---|---|---|---|
| `taskset/` | gs://gresearch/task_set_data: all 1149 allowed tasks x 5 optimizer families (adam1p/4p/6p/8p, nadamw), all 5 replicas and 4 splits, float32 | 16 GB | `scripts/download_taskset.py` |
| `synetune/nasbench201`, `fcnet`, `bbomix_*` | syne-tune blackbox repository (Hugging Face `synetune/blackbox-repository`), as published | 3.7 GB | `scripts/download_synetune.py` |
| `quicktune/` | QuickTune meta-dataset (micro / mini / extended), github.com/sebastianpinedaar/hpo-data | 0.25 GB | curl, see README of releaunifreiburg/QuickTune |
| `jahs/raw/` | JAHS-Bench-201 raw metric data (`raw.pkl.gz` per dataset), ml.informatik.uni-freiburg.de | 13.9 GB | curl; **not converted yet**, see below |

## What is loadable (`curvefaker.data`)

| source | family | split | tasks | configs/task | replicas | T | hps |
|---|---|---|---|---|---|---|---|
| bbomix | bbomix | select | 35 | 1000 | 3 | 300 | 8-11 |
| fcnet | fcnet | select | 4 | 62208 | 4 | 100 | 9 |
| nasbench201 | nasbench201 | fit | 3 | 15625 | 3 | 200 | 6 (categorical) |
| quicktune | quicktune | fit | 86 | 129-327 | 1 | 50 | 32 |
| taskset | Associative, Copy, char, conv, rnn | select | 1970 | 1000 | 5 | 51 | 1-10 |
| taskset | mlp, losg, quadratic, maf, nvp, word, TwoD, Fixed* (non-text), Imagenet32x30, Mnist | fit | 3775 | 1000 | 5 | 51 | 1-10 |

A TaskSet task is one (task, optimizer family) pair, so each TaskSet problem appears 5 times with different
search spaces: 1 hp (adam1p: learning rate) to 10 (nadamw: incl. warm-up, schedule shape, a boolean).
Full per-family table: `runs/inventory.md` (`scripts/inventory.py`).

Notes per source:
- **TaskSet**: y = valid1 loss per replica; y_test = valid2. Losses can blow up (up to 1e34) or be NaN; the
  normalisation (`measures.curves`) maps those to the task's median initial loss, as rogue-dagger does.
- **NAS-Bench-201**: validation error, 200 epochs, 3 seeds, 6 categorical cell operations, no numeric hps.
- **FCNet**: validation MSE, 100 epochs, 4 seeds; the full grid of 62208 configs (subsampled to 1000 per use).
- **bbomix**: autoencoders on omics data; y = validation reconstruction loss, 300 epochs, 3 seeds. 4 model
  variants x 2 data collections.
- **QuickTune**: fine-tuning of 24 pretrained timm models on Meta-Album datasets; y = 100 - top-1 accuracy.
  Runs are cut at different lengths (time budget); measures use only full-length runs.

## Open: JAHS-Bench-201

The raw files are downloaded (13.9 GB), but they are pandas pickles, and unpickling downloaded data executes code.
The agent's permission check blocked converting them, so they are untouched. `scripts/convert_jahs.py` (run with
`python -I`) would turn each into a compact npz. Plan: one task per (dataset, N, W, Resolution), up to 27 per
dataset, each searching over cell operations, learning rate, weight decay, activation, TrivialAugment and
optimizer, with 200-epoch curves. It is listed under `fit` in `splits.yaml`, but no loader exists yet.

## Not fetched (yet)

- **More curve sources:** NAS-Bench-ASR, TransNAS-Bench-101 (NAS; categorical spaces), YAHPO's nb301 (a surrogate).
- **No curves:** syne-tune's `tabrepo_*`, `hpob_*`, and YAHPO's `rbv2_*` (fidelity = train size or repetitions).
  Useful only for final-performance landscapes (avenue A).
- **LCDB 1.1:** sample-wise curves; for statistics that don't need epochs.
