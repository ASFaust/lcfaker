# curve-faker

A general learning-curve faker: a generator of synthetic HPO tasks (a search space, and for each config a learning
curve) to train and evaluate freeze-thaw HPO methods on. The ifBO prior that FT-PFN is trained on is one such faker.
This repo asks what a better one would look like and how to tell it is better.

This is a sister repo of [`../rogue-dagger`](../rogue-dagger), which trains a freeze-thaw policy on the ifBO prior.
The work there on the prior grew too large to be a side track (rogue-dagger `docs/journal.md`, 2026-11-15/16 entries,
and `docs/future-ideas/prior-sim2real.md`). What we know so far:

- The ifBO prior's curves are measurably unrealistic. On 240 TaskSet dev tasks, its noise is 3-4× too large and
  white (real noise is autocorrelated), there is too much overfitting and divergence (this comes from the basis-curve
  shapes, not the noise), there are no dead configs and no cluster of near-best configs. Hyperparameters predict the
  final loss far more on the prior than on TaskSet (config smoothness 0.15 vs 0.01).
- Yet a scorer trained on the prior judges real states about as well as prior states. The exception is comparing
  unseen configs on TaskSet.
- **Fixing the clearest single discrepancy made things worse.** Noise calibrated to real curves (× 0.25, smoothed)
  gave a worse policy on all three benchmarks (3 seeds each, at every deploy temperature). Matching curve statistics
  is therefore not a reliable target on its own, and that is the main reason this repo starts with the quality
  measures, not with a new generator.

## Rule 1: held-out data

FT-PFN (ifBO, ICML'24) and rogue-dagger are evaluated on the ifBO benchmarks. **Nothing in this repo may be fitted,
calibrated, tuned or selected on them**, including summary statistics and HPO-run statistics computed from them. They
are used once, at the end, to test a method trained on a finished faker.

Held out, in full:

| Family | What is held out | Why the whole family |
|---|---|---|
| **LCBench** | All 35 datasets, all 2000 configs of `data_2k` (the benchmark uses the first 1000 per dataset). Also any surrogate trained on LCBench, e.g. YAHPO Gym's `lcbench` scenario. | The other 1000 configs are the same tasks. |
| **PD1** | All of PD1: every dataset × model × batch-size combination in phase 0 and phase 1, including the `max_pooling_cnn` runs that ifBO skipped. Also surrogates built on PD1 (mf-prior-bench's PD1 benchmarks). | Every combination not in the benchmark shares a dataset and model with one that is. |
| **TaskSet** | The whole `FixedTextRNNClassification` family, for every optimizer family (not just `adam4p`/`adam8p`). | The curves of the same tasks under other optimizers would leak task difficulty and curve shape. |

The benchmark tasks themselves, for reference (as processed in rogue-dagger):

- **TaskSet** (12 tasks × {`adam4p_wide_grid_1k`, `adam8p_wide_grid_1k`} = 24):
  `FixedTextRNNClassification_imdb_` + `patch128_LSTM128_avg_bs64`, `patch128_LSTM128_bs64`,
  `patch128_LSTM128_embed128_bs64`, `patch32_GRU128_bs128`, `patch32_GRU64_avg_bs128`,
  `patch32_IRNN64_relu_avg_bs128`, `patch32_IRNN64_relu_last_bs128`, `patch32_LSTM128_E128_bs128`,
  `patch32_LSTM128_bs128`, `patch32_VRNN128_tanh_bs128`, `patch32_VRNN64_relu_avg_bs128`,
  `patch32_VRNN64_tanh_avg_bs128`.
- **PD1** (16 dataset/model/batch-size tasks, 28 variants with the curve subsamplings `-1/-2/-5/-10`):
  cifar10 and cifar100 wide-resnet {256, 2048}; svhn wide-resnet {256, 1024}; fashion-mnist and mnist simple-cnn
  {256, 2048}; imagenet resnet {256, 512, 1024} × {1, 2, 5, 10}; translate-wmt xformer-translate 64 × {1, 2, 5, 10};
  lm1b transformer 2048; uniref50 transformer 128.
- **LCBench** (35): APSFailure, Amazon_employee_access, Australian, Fashion-MNIST, KDDCup09_appetency, MiniBooNE,
  adult, airlines, albert, bank-marketing, blood-transfusion-service-center, car, christine, cnae-9, connect-4,
  covertype, credit-g, dionis, fabert, helena, higgs, jannis, jasmine, jungle_chess_2pcs_raw_endgame_complete, kc1,
  kr-vs-kp, mfeat-factors, nomao, numerai28.6, phoneme, segment, shuttle, sylvine, vehicle, volkert.

Allowed for fitting:

- The rest of TaskSet: about 1150 tasks outside `FixedTextRNNClassification`, any optimizer family. 240 of them
  (120 tasks × `adam4p`/`adam8p`) are already downloaded in `../rogue-dagger/data/taskset_dev`
  (`../rogue-dagger/scripts/taskset_dev_tasks.txt`).
- Other epoch-curve data: NAS-Bench-201, HPOBench FCNet, the Quick-Tune meta-dataset, JAHS-Bench-201 (a surrogate,
  so it carries its own model bias), YAHPO Gym scenarios other than `lcbench`.
- LCDB 1.1, only for statistics that don't need epoch curves (its curves are over training-set size).
- Curves we generate ourselves by training small models (see avenue B5).

The allowed data also gets an internal split: fit on one part, select on another. Otherwise the faker overfits the
dev tasks the way a model overfits a validation set it is tuned on. Split by task family (e.g. hold out whole TaskSet
families), not by task.

Mechanics, to set up before any data is loaded: one machine-readable held-out list, and loaders that refuse
anything on it.

## Rule 2: one change at a time, and the measures come first

The noise result shows that a plausible fix can hurt. Every faker change gets the quality measures below, and claims
about downstream methods need the seed-aware comparison from rogue-dagger (seed noise there is up to 0.017 / 0.013 /
0.005 regret on TaskSet / PD1 / LCBench).

## First item: brainstorm the avenues from scratch

The ifBO prior is a good starting point, not the only design. A faker has to make choices on two main axes, and a
few smaller ones. The lists below seed the brainstorm; they are not a plan yet.

### A. How hyperparameters influence the curve

ifBO: per task, a random Bayesian neural network maps the config in [0, 1]^d to 23 latents, and each latent becomes
a curve parameter through an empirical CDF.

Is that defensible? Questions to check against real data:
- **Effective dimension.** On real tasks a few hyperparameters (often the learning rate) carry most of the variance
  (fANOVA). Does a random BNN produce that, or does every input matter about equally?
- **Smoothness.** The measured config smoothness is 0.15 on the prior vs 0.01 on TaskSet. Real tasks may be
  smooth in a few directions and noisy in the rest.
- **Structure.** Optima in the interior vs at the edges of the box. Cliffs: regions that diverge or never learn
  (learning rate too high), which the prior lacks entirely. Interactions such as learning rate × batch size.
- **Which latents a config moves.** Does a config move the final level only, or also the shape (speed, overfitting,
  noise)? Real data can answer this.

Alternatives:
- **A1** Gaussian-process draws with automatic relevance determination (ARD) length-scales: explicit control
  over smoothness and over how many dimensions matter.
- **A2** Additive or low-order functional ANOVA: a sum of 1-D and 2-D effects with sampled importances, matched to
  fANOVA statistics from real tasks.
- **A3** Structured mixtures: a smooth landscape, plus "dead" regions with a sampled shape and size, plus a basin
  of near-best configs.
- **A4** Meta-learned maps: fit a hierarchical model to the allowed real tasks and sample new tasks from its task
  distribution.
- **A5** Remap real landscapes: take a real task's config → performance ranks and apply random monotone warps and
  rotations of the config space, plus random dimension subsets.

**Categorical and discrete hyperparameters.** Discretising a continuous dimension gives ordinal integers for free.
A categorical can be emulated by taking a random continuous point per category. The open question is whether that
is enough: in real tasks a categorical (optimizer, activation) can change the curve's *shape*, not only its level,
and conditional spaces (parameters that exist only for some choices) have no continuous analogue. Test before
building: compare categorical effects in allowed data (e.g. TaskSet's optimizer families on the same task) with what
discretised continuous dimensions produce. The current benchmarks are all numeric, so this matters for generality,
not for the benchmark numbers.

### B. The shape of the curves

ifBO: a weighted sum of 4 parametric basis curves (pow4, exp, log, hill), an optional break point, and i.i.d.
Gaussian noise.

Alternatives:
- **B1** A richer parametric family. Add learning-rate-schedule drops, warm-up, plateaus then sudden learning,
  divergence to NaN or blow-up, dead curves, overfitting U-shapes with realistic frequency, and correlated noise.
  Interpretable, and each addition is a separate ablation.
- **B2** Non-parametric. GP draws in log-time with a monotone (or mostly monotone) transform; shape is controlled
  through kernels.
- **B3** A generative model trained on real curves: VAE, normalising flow, diffusion, or an autoregressive
  transformer over time-normalised curves. Unconditional generation of curves is the easy part.
  **Conditioning on the config is the hard part**: real tasks have few configs each, so learning how a config
  changes a curve gets little signal. An early attempt (learning-faker, July 2026, before ifBO) fit curves well
  per run but never learned the config dependence, though it didn't go far. One option is a hybrid: a generative
  model of curve *shapes* from a latent `z`, and an avenue-A map from config to `z`.
- **B4** Resample real curves: real curves with random time warps, monotone transforms of the loss, and new
  assignments to configs through a smooth map. This is the cheapest "real" generator, but its diversity is
  limited by the dev data.
- **B5** Fake by doing: actually train many small networks (MLPs and small CNNs/RNNs on varied small datasets) and
  record their curves, the way TabPFN's prior samples real-ish data-generating processes. This is expensive per
  task but has every real phenomenon (schedules, divergence, overfitting, categorical effects) for free. It can be
  a dataset in its own right, or a source of statistics for other avenues.

### C. Smaller axes (all task-level)

- **Noise model**: per-point vs autocorrelated noise; evaluation noise shared by all configs of a task (same
  validation set).
- **Time axis**: steps per nominal training, fidelity cap vs continuation past `t = 1`, epochs vs steps,
  subsampled curves (the PD1 jump variants).
- **Metric type**: bounded error rates that saturate vs unbounded losses.
- **Task difficulty and spread**: how much hyperparameters matter at all on a task (spread of final losses, fraction
  near the best).
- **Cross-config structure**: how often curves cross and when. This is what freeze-thaw exploits (early-to-final
  rank correlation), so it may matter more than any single curve's look.

Brainstorm output: a list of avenues (combinations of A × B × C), each with its premise, cost, and which measure would
show it working. Then a ranked plan, explored one avenue at a time.

## Quality measures

What "better faker" means needs to be settled before any faker is built. Candidates, from cheap to decisive:

- **M1: curve statistics, real vs fake**, on matched twins (same d, T, budget). Examples: early-late rank
  correlation by fraction of training, crossing rate, final-loss spread, fraction near best, noise size and
  autocorrelation, divergence, overfitting, config smoothness. Compared per statistic with KS or Wasserstein
  distances. Code to start from: `../rogue-dagger/scripts/curve_stats.py`, `twins.py`. Known weakness: the noise fix
  improved M1 and still hurt downstream.
- **M2: HPO-run statistics, real vs fake** (user's idea). Run HPO methods on allowed real tasks and on matched fake
  twins, and compare how the runs *look*:
  - how many configs get started;
  - the distribution of fidelity reached per run, and how often runs are thawed;
  - when the incumbent changes, and how often;
  - the regret-over-time curve, and final regret.
  
  Two refinements:
  - **Use several methods, including prior-free ones** (random search, Hyperband/ASHA, TPE, BOHB, DyHPO, rogue-dagger
    policies), not only FT-PFN. FT-PFN is trained on the ifBO prior, so on ifBO data it is in-distribution by
    construction, and its statistics would favour fakers close to ifBO.
  - **Method-ranking agreement.** Do the methods rank the same way on fake twins as on real tasks (Spearman ρ, as in
    rogue-dagger's twin experiment)? This measures what a faker is ultimately for: telling good HPO methods from bad
    ones.
- **M3: classifier two-sample test**, at curve level and at task level (a set transformer over all curves of a
  task). Accuracy near 0.5 means indistinguishable; feature attributions show what gives the fake away. Normalise
  first, so trivial tells (value range, step grid) don't win.
- **M4: coverage vs precision.** For a prior, missing a real phenomenon is probably worse than adding unrealistic
  extras: a PFN trained on a wider prior only loses some sharpness. Measure whether each real dev task lies inside
  the fake task distribution (nearest-neighbour or density-based precision/recall), not just distance between
  averages.
- **M5: downstream, the final arbiter.** Train a method on the faker (rogue-dagger `wide-focus03`, and possibly a
  small FT-PFN-style model), then evaluate it on held-out *allowed* tasks. The three benchmarks come only at the very
  end. This is expensive and noisy: 3 seeds minimum.

**Validate the measures against each other before trusting them.** We already have a pair of priors with a known
downstream ordering: the ifBO prior vs the noise-fixed prior (worse on the benchmarks, 3 seeds). A cheap measure that
prefers the noise-fixed prior is misleading for exactly the decision we need it for. Any further prior variant with
M5 results adds a data point to that check.

## Next steps

1. **Brainstorm session** on avenues A × B × C, written to `docs/avenues.md`.
2. **Data inventory and held-out guard**: the machine-readable held-out list, loaders, and the internal split of
   the allowed data.
3. **Measurement harness**: M1 and M2 on allowed real data vs the ifBO prior, then the meta-check on ifBO vs the
   noise-fixed prior.
4. Only then: the first new faker.

## Environment

Same as rogue-dagger: `source /shared/work/venv/bin/activate`, GPUs 3-6 with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, wandb
for training runs. The batched ifBO prior is `../rogue-dagger/roguedagger/prior.py`, and FT-PFN is vendored in
`../rogue-dagger/third_party/ifbo`.
