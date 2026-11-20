# curve-faker

**The question: how do we best model learning curves?**

A learning-curve faker generates synthetic HPO tasks. Each task is a search space plus, for every config in it, a
learning curve. Fakers are what freeze-thaw HPO methods get trained on: FT-PFN is trained on the ifBO prior, which
is one such faker. This repo looks for a better one by studying real learning curves and building models of them
from first principles. It does not patch ifBO.

## Rules

1. **Held-out benchmarks are untouchable** (details below). Nothing may be fitted, tuned or selected on LCBench,
   PD1 or TaskSet `FixedTextRNNClassification`. Loaders refuse them.
2. **No HPO models are trained in this repo.** Never train FT-PFN, rogue-dagger policies or any other HPO method
   on a faker here. That happens at the very end, in a separate project, once a faker is finished. Models trained
   *here* are **models of learning curves**: generative curve models, config → curve maps, noise models.
3. **Fakers are judged with statistics on real data and with existing HPO algorithms** (random search, ASHA /
   Hyperband, BOHB, simple freeze-thaw heuristics, and so on). Existing FT-PFN and rogue-dagger checkpoints may be
   used, but they are biased toward ifBO-like data, since they were trained on it.
4. **One GPU only** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, one device in `CUDA_VISIBLE_DEVICES`). It is for training curve
   models and running evaluations.
5. **ifBO is a baseline, not a starting point.** Compare every idea against it, but don't build by adding parts to
   it.
6. **One change at a time within a model family**, and every claim backed by a measurement.
7. **Report in plain language.** Define every statistic and every name you introduce. The reader should be able to
   follow progress without having read the code.

## Why the earlier attempt was razed (2026-11-20)

The first overnight session built good infrastructure (kept, see below). After that it went wrong in two ways:

- It added small parts to the ifBO prior (a region of configs that never learn; noise randomised per task) instead
  of asking afresh how curves should be modelled.
- It judged those patches by training new rogue-dagger policies on them (about 9 GPU-hours per seed, 3 seeds each).
  That violates rule 2, and it did not work either: on the allowed real tasks, the policies trained on ifBO and on
  the noise-fixed ifBO scored the same within noise, so this test can't distinguish even a known pair.

What a good overnight session looks like here: several *different* model families, fitted to real data and
compared head to head in tables, with a plain-language conclusion about what explains real curves best.

## What is already here

- **Data** (`data/`, git-ignored, about 32 GB; inventory in `docs/data.md`, `docs/inventory.md`):
  - TaskSet: all 1149 allowed tasks × 5 optimizer families, 1000 configs each, 5 seeds, 51 points per curve;
  - NAS-Bench-201: 3 tasks, 200 epochs;
  - FCNet: 4 tasks, 100 epochs;
  - bbomix: 35 autoencoder tasks, 300 epochs;
  - QuickTune: 86 fine-tuning tasks, 50 epochs.
  - JAHS-Bench-201 is downloaded but not converted: it ships as pickles, and unpickling downloaded data needs the
    user's approval.
- **Loaders**: `curvefaker/data.py`. `data.ids(split, sources)` and `data.load(id)` return a `Task` with the
  hyperparameters (a DataFrame plus a space description), the curves `y [configs, seeds, T]` (lower is better), the
  metric name, and `X()`, the configs scaled to [0, 1].
- **Held-out guard**: `heldout.yaml` + `curvefaker/heldout.py`, used by every loader and downloader. Tests are in
  `tests/`.
- **Internal split** (`splits.yaml`, by task family, so whole families are held back):
  - **fit**: most TaskSet families, NAS-Bench-201, QuickTune (and JAHS);
  - **select**: TaskSet {char, conv, rnn, Associative, Copy}, FCNet, bbomix.
  
  Fit models on *fit* and compare them on *select*.
- **Download scripts** in `scripts/`. The ifBO prior itself (batched, for baselines) is
  `../rogue-dagger/roguedagger/prior.py`.

## Facts about the real data found so far

These come from measurements on the allowed data (the code was razed, the facts stand; re-measure if in doubt):

- **Flat curves are common and clustered.** 3-37% of configs per source never improve. ifBO has almost none.
  Whether a config is flat is predictable from its hyperparameters (random-forest AUC 0.79-0.99). So dead configs
  form a region of config space, not random bad luck.
- **Final performance varies a lot between configs**: the spread of final losses is 0.84-0.99 (normalised) on most
  sources vs about 0.45 on ifBO. Real tasks also show far less overfitting and divergence than ifBO.
- **Noise is source-specific.** Measured as step-to-step changes of a single seed's curve:
  - TaskSet: mildly positively autocorrelated (+0.2);
  - NAS-Bench-201, FCNet, QuickTune, bbomix: negatively autocorrelated (−0.1 to −0.5), close to white;
  - noise size 0.015-0.05 of the loss range, vs 0.04-0.08 on ifBO.
- **Seed luck is small**: across-seed variance of the final loss is 0.1-5% of the across-config variance. Config
  rankings agree between seeds (rank correlation 0.96-0.99).
- **Matching summary statistics is a misleading target.** Rogue-dagger's noise fix moved curve statistics closer
  to real data and made the trained HPO policy worse on the benchmarks. Summed per-statistic distances preferred
  that worse prior on most sources. Coverage, i.e. whether each real task looks like *some* generated task
  (nearest-neighbour recall), preferred the better one. One data point, but it says: value breadth, not just
  closeness on averages.

## The research programme

A faker has to answer three modelling questions. Treat each as its own model comparison on real data.

**A. How do hyperparameters shape the curve?** Candidates (not exhaustive, and not built on ifBO):
- Gaussian processes with a separate length-scale per hyperparameter (controls how smooth and how many
  dimensions matter);
- sparse additive models: a few 1-D and 2-D effects, with sampled importances, as fANOVA finds on real tasks;
- landscapes with structure: a smooth part, plus dead regions, plus a basin of near-best configs;
- hierarchical models learned from the real tasks, sampled for new tasks;
- warped real landscapes: a real task's config → performance map under random monotone warps and rotations;
- ifBO's random Bayesian neural net, as the baseline.

Also: does a config move only the final level, or also the curve's shape (speed, plateau, overfitting)?

**B. What do curves look like?**
- parametric families fitted to real curves (which basis shapes are actually needed?);
- non-parametric curves (e.g. Gaussian processes in log-time);
- learned generative curve models (VAE, normalising flow, diffusion, autoregressive transformer), unconditional,
  then conditioned on a latent that an A-model sets from the config. The GPU is for this.

**C. What is the noise?** White, autocorrelated, shared by all configs of a task (same validation set), dependent
on the loss level or the training phase, occasional spikes. Fit per source; the facts above show sources differ.

Also task-level: how many configs are dead, how spread the final values are, how often curves cross (what
freeze-thaw exploits), curve length and metric type (bounded error vs unbounded loss).

**How to compare models** (all on the *select* split, after fitting on *fit*):
1. **Held-out fit.** How well does the model explain real tasks it hasn't seen: likelihood, or the error of
   predicting held-back curves / configs. This is the main, cheap, direct measure for A, B and C.
2. **Statistics and coverage.** Generated tasks vs real ones: per-statistic distances, and above all coverage
   (does every real task have look-alikes among the generated ones?).
3. **Existing HPO algorithms.** Run random search, ASHA etc. on real tasks and on generated tasks. Do the runs look
   alike (configs started, fidelity reached, regret over time)? Do the algorithms rank the same way?

**Suggested first steps for a fresh session**
1. Look at the data. Plot curves per source, and describe what kinds of curves exist and how often.
2. Pick the noise question (C) as a warm-up. Fit 3-4 noise models per source, and compare their held-out
   likelihood.
3. Then the config → final-performance question (A). Fit 3-4 model families, compare held-out prediction, and
   compare against ifBO's map.
4. Then curve shape (B), including a learned generative model.
5. Combine the winners into a faker, then compare it with ifBO using measures 2 and 3.

Write the narrative in `docs/journal.md` (newest at the bottom), and commit and push to `origin main` regularly
(never `data/`).

## Held-out data (Rule 1, in full)

FT-PFN (ifBO, ICML'24) and rogue-dagger are evaluated on the ifBO benchmarks. Nothing here may be fitted,
calibrated, tuned or selected on them, including summary statistics and HPO-run statistics computed from them.
Only the benchmark *results* are held out. The datasets the benchmark models were trained on (CIFAR-10, MNIST, …)
may be used, e.g. to train small networks and record their curves.

| Family | What is held out | Why the whole family |
|---|---|---|
| **LCBench** | All 35 datasets, all 2000 configs of `data_2k`, and any surrogate trained on LCBench (e.g. YAHPO's `lcbench`). | The other 1000 configs are the same tasks. |
| **PD1** | All of PD1 (every dataset × model × batch size, phases 0 and 1, incl. `max_pooling_cnn`), and surrogates built on it (mf-prior-bench). | Every combination shares a dataset and model with a benchmark task. |
| **TaskSet** | The whole `FixedTextRNNClassification` family, for every optimizer family. | The curves under other optimizers would leak task difficulty and curve shape. |

The machine-readable list is `heldout.yaml`. Benchmark task names are in `../rogue-dagger` (the ifBO benchmark
loaders there).

## Environment

`source /shared/work/venv/bin/activate`; run from the repo root with `PYTHONPATH=.`. Data lives in `data/`, or
wherever `CURVEFAKER_DATA` points (`/shared/datasets` is read-only in the agent container).
