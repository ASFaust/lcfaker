# Journal

The narrative record: what was tried, why, what was learned, which decisions followed. Newest entries at the bottom.

## 2026-11-20: Restart

The first session's work was razed, except the data, its loaders, the held-out guard and the split. It had patched
the ifBO prior and trained rogue-dagger policies to test the patches; both were against the project's intent (see
README, "Why the earlier attempt was razed"). The real-data facts it found are listed in the README. Its code is
in the git history (commits up to `5cbee54`), if a detail is ever needed.

## 2026-11-20: JAHS-Bench-201 added

With the user's approval the JAHS pickles were converted (`scripts/convert_jahs.py`, `scripts/prepare_jahs.py`,
details in `docs/data.md`). 81 tasks, one per (dataset, depth N, width W, resolution), 200 epochs, split *fit*.
Two surprises in the raw data: separate runs stopped at lower budgets (dropped), and repeated configs (exact copies
dropped; distinct runs of one config kept as replicas). fashion_mnist has only 123-407 distinct configs per task.

## 2026-11-20: Data atlas (step 1)

**Question.** What kinds of learning curves exist, per source, and how does the ifBO prior compare?

**Method** (`scripts/atlas.py`, `curvefaker/curves.py`; per-task numbers in `results/atlas_tasks.csv`).
300 TaskSet tasks (150 *fit*, 150 *select*, random), every task of the other sources, and 300 ifBO tasks (1-10
hyperparameters, 1000 random configs, 100 points). At most 1000 configs per task. Each task is normalised:

- **z = (y − best) / (ref − best)**, where *best* is the lowest value anywhere in the task and *ref* is the 90th
  percentile of the configs' first values (a bad starting level: the loss at initialisation, or about chance level
  for classifiers). z = 0 is the task's best, z = 1 a bad start; diverged values are capped at 2.

Each config's curve (mean over seeds) is put in one class. The class is decided on a lightly smoothed curve, so
that noise does not decide it:

- **dead**: never gets below z = 0.9, i.e. never makes 10% of the possible progress;
- **diverged**: blows up (non-finite or z > 2) or ends 0.1 worse than it started;
- **turns**: improves, then gets clearly worse again (overfitting or late instability);
- **improving**: the rest.

Task statistics (the table shows the median over the tasks of a source):

- **final_spread**: 90th minus 10th percentile of the configs' final z. How different good and bad configs are.
- **t50 / t90**: fraction of the budget a curve needs for 50% / 90% of its total improvement (median over configs).
  Small = fast early learning, large = learning happens late.
- **rank_early10 / 30**: rank correlation between z at 10% (30%) of the budget and the final z. How well an early
  look predicts the end, which is what freeze-thaw methods exploit. *_live*: dead configs left out.
- **top10_kept**: of the configs that end in the best 10%, the share already in the best 10% at 10% of the budget.

| source | tasks | improving | dead | turns | diverged | final_spread | t50 | t90 | rank_early10 | rank_early10_live | rank_early30_live | top10_kept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| taskset | 300 | 0.60 | 0.19 | 0.01 | 0.15 | 1.21 | 0.06 | 0.30 | 0.89 | 0.87 | 0.94 | 0.65 |
| nasbench201 | 3 | 0.98 | 0.02 | 0.00 | 0.00 | 0.32 | 0.25 | 0.78 | 0.79 | 0.77 | 0.84 | 0.47 |
| fcnet | 4 | 0.76 | 0.06 | 0.04 | 0.10 | 0.91 | 0.10 | 0.50 | 0.93 | 0.92 | 0.96 | 0.76 |
| bbomix | 35 | 0.93 | 0.00 | 0.02 | 0.04 | 0.37 | 0.03 | 0.16 | 0.95 | 0.95 | 0.98 | 0.77 |
| quicktune | 86 | 0.58 | 0.38 | 0.02 | 0.00 | 0.80 | 0.18 | 0.49 | 0.80 | 0.72 | 0.87 | 0.62 |
| jahs | 81 | 0.96 | 0.03 | 0.00 | 0.00 | 0.27 | 0.09 | 0.60 | 0.65 | 0.61 | 0.61 | 0.30 |
| ifbo | 300 | 0.51 | 0.02 | 0.10 | 0.12 | 0.63 | 0.11 | 0.45 | 0.68 | 0.66 | 0.84 | 0.50 |

![gallery](figures/atlas_gallery.png)

**What the curves look like** (gallery: 40 random configs from 4 tasks per source, coloured by class):

- **TaskSet**: a fast drop in the first few percent of training, then slow refinement. Dead configs sit exactly at
  the initial loss; diverged ones jump to the cap abruptly, often mid-training (a step, not a drift).
- **NAS-Bench-201 and JAHS** (cosine learning-rate schedules): a large share of the improvement comes at the
  *end*, when the learning rate decays (t90 = 0.6-0.8). Some NAS-Bench-201 curves even rise first and fall late.
  Here curves cross most, and the early look predicts least (JAHS: rank correlation 0.61 at 30% of the budget,
  and only 30% of the final top-10% are in the top 10% early on).
- **FCNet**: very noisy; bad configs bounce around high values rather than sitting flat.
- **bbomix**: converges very fast (t50 = 0.03), then plateaus; occasional single-epoch spikes.
- **QuickTune**: 38% of fine-tuning runs are dead (sit at chance), the highest share of any source.
- **ifBO**: curves often start low already (the first point is not a common bad start), much more "turns"
  (10% vs 0-4% on real data) and noise large relative to the progress made; in some tasks noise is all there is
  (third ifBO panel). Dead configs are rare (2%) vs up to 38% on real sources.

**Conclusions for the faker.**

1. The time profile is source-dependent and driven by the learning-rate schedule: "fast drop then plateau"
   (TaskSet, bbomix) vs "slow, with late improvement" (cosine schedules). A curve model needs both; ifBO's
   saturating basis curves have no late drop.
2. Dead configs (0-38%) and abrupt divergence (up to 15%) are separate phenomena from slow learners and need
   their own model component.
3. How predictive early performance is varies a lot (rank correlation at 10%: 0.65-0.95). This is the property
   freeze-thaw methods depend on, so the faker should cover this range rather than one value.

Caveats: three NAS-Bench-201 and four FCNet tasks are few; the class thresholds (0.9, 0.05, 0.1) are judgment
calls, chosen by looking at the gallery, not tuned on anything held out.

## 2026-11-20: Config → final performance (question A, first pass)

**Question.** Within a task, how do hyperparameters determine the final loss? How predictable is it from a few
hundred configs, does it need interactions, and how many hyperparameters matter? Same questions for ifBO.

**Method** (`scripts/landscape_models.py`, `curvefaker/landscape.py`; per task: `results/landscape_tasks.csv`).
60 TaskSet tasks, 3 NAS-Bench-201, 4 FCNet, 20 bbomix, 40 QuickTune, 27 JAHS, 60 ifBO (1-10 hps). Target: the
final normalised loss, replaced by its **normal score** (rank mapped to a standard-normal quantile). This keeps the
order and stops the few dead or diverged configs from dominating squared errors; without it, the GPs overfit
(checked on one task). Train on 200 random configs (QuickTune: about 100), test on up to 500 others.

- **const**: predict the training mean (floor).
- **rf / lgbm**: random forest / gradient-boosted trees on one-hot configs.
- **gp_full**: Gaussian process with a product kernel over hyperparameters, so any interaction is possible
  (Matern-5/2 per numeric hp, "same or different" kernel per categorical hp).
- **gp_additive**: sum of one kernel per hp, so no interactions.
- **gp_add+full**: both. **interaction_share** = variance of the product part / total.
- **eff_hps**: from the additive part, the "effective number" of hps that matter (exp of the entropy of their
  variance shares; 1 = one hp does everything, 6 = six matter equally). **top_hp_share**: share of the most
  important hp.
- Score: **Spearman rank correlation** between predicted and true test values. The order is what matters to an
  optimiser.

| source | hps | const | rf | lgbm | gp_full | gp_additive | gp_add+full | full − additive | eff_hps | top_hp_share |
|---|---|---|---|---|---|---|---|---|---|---|
| taskset | 6 | 0.00 | 0.86 | 0.84 | 0.83 | 0.59 | 0.82 | 0.19 | 1.8 | 0.83 |
| nasbench201 | 6 | 0.00 | 0.78 | 0.85 | 0.86 | 0.81 | 0.83 | 0.06 | 4.6 | 0.32 |
| fcnet | 9 | 0.03 | 0.92 | 0.94 | 0.95 | 0.88 | 0.95 | 0.08 | 3.3 | 0.51 |
| bbomix | 8 | 0.00 | 0.98 | 0.98 | 0.99 | 0.98 | 0.99 | 0.00 | 2.5 | 0.71 |
| quicktune | 32 | 0.00 | 0.29 | 0.24 | 0.12 | 0.20 | 0.19 | −0.11 | 7.7 | 0.29 |
| jahs | 10 | 0.00 | 0.70 | 0.75 | 0.76 | 0.51 | 0.72 | 0.19 | 5.1 | 0.39 |
| ifbo | 4 | 0.00 | 0.82 | 0.78 | 0.90 | 0.55 | 0.90 | 0.21 | 2.0 | 0.67 |

(median over tasks; "full − additive" is the median per-task difference of rank correlation.)

TaskSet by optimizer family (number of hps): adam1p (1) 0.99; adam4p (4), adam6p (6), adam8p (8), nadamw (10):
0.77-0.82 for gp_full, with the additive model 0.21-0.37 worse. One hp (the learning rate, presumably) carries
60-86% of the additive variance.

**Findings.**

1. **Predictability ranges from almost perfect to almost none**: bbomix 0.99, FCNet 0.95, TaskSet 0.83, JAHS 0.76,
   QuickTune 0.12-0.29 (32 hps, about 100 configs, and the pretrained model, a 24-way categorical, presumably
   dominates). A faker should cover this range. ifBO sits at the easy end (0.90 with 4 hps on average; 0.6-0.85 at
   6-10 hps).
2. **Interactions matter where one hp dominates** (TaskSet, JAHS: additive loses about 0.2). Near-additive
   landscapes exist too (bbomix, NAS-Bench-201, FCNet). ifBO's random networks are strongly non-additive at every
   dimension ≥ 3.
3. **Few hps matter on TaskSet** (eff_hps 1.8); 3-5 matter on FCNet, JAHS and NAS-Bench-201.
4. Trees and GPs are close; the GP is better on smooth numeric spaces, trees on QuickTune.

interaction_share (median per source): bbomix 0.04, FCNet 0.29, NAS-Bench-201 0.30, QuickTune 0.32, ifBO 0.43,
TaskSet 0.50, JAHS 0.60. It orders the sources like the rank-correlation gap, but per task the two correlate only
0.35 (Spearman). So it is a coarse descriptor; the gap measures the predictive effect directly.

Caveat: eff_hps depends on the number of hps.

## 2026-11-20: Curve shapes (question B, first pass)

**Question.** Which curve families can represent real curves, and which extrapolate from the start of a curve?

**Method** (`scripts/curve_shapes.py`; `results/curve_shapes*.csv`). Replica-mean normalised curves, resampled to
50 points evenly over the budget; diverged curves left out. Learned models are fitted on curves of the *fit* split
(about 22k curves) and evaluated on the *select* split (TaskSet select families, FCNet, bbomix).

- **pow3** y = c + a·u^(−b), **exp3** y = c + a·exp(−b u): classic 3-parameter saturating curves, fitted per curve.
- **pca-k**: the mean curve plus k principal components of the fit curves (probabilistic PCA: the latent is
  shrunk toward 0 by its prior when only part of the curve is seen).
- **vae-k**: a variational autoencoder with a k-dimensional latent (2 hidden layers of 256), latent fitted per
  curve under its N(0, I) prior.
- **knn**: average final value of the 50 fit curves most similar on the observed part. **last**: last observed
  value.

Scores: **rmse_full**: error of the best fit to the whole curve (z units; the noise level is 0.003-0.05).
**mae_final**: see the first 20% of the curve, predict the final value, mean absolute error. **rank_final**:
per task, rank correlation of predicted vs true final values (median over tasks).

Held-out select split (all three sources):

| model | rmse_full | mae_final | rank_final |
|---|---|---|---|
| pow3 | 0.046 | 0.076 | 0.87 |
| exp3 | 0.040 | 0.083 | 0.89 |
| pca-2 / 4 / 8 | 0.059 / 0.032 / 0.022 | 0.199 / 0.105 / 0.094 | 0.77 / 0.89 / 0.92 |
| vae-2 / 4 / 8 | 0.045 / 0.029 / 0.028 | 0.084 / 0.075 / 0.073 | 0.89 / 0.86 / 0.88 |
| knn | – | 0.068 | 0.90 |
| last | – | 0.074 | **0.94** |

Leave-one-source-out within *fit* (sources with cosine schedules and fine-tuning):

| model | rmse JAHS / NB201 / QT | mae_final JAHS / NB201 / QT | rank_final JAHS / NB201 / QT |
|---|---|---|---|
| pow3 | 0.069 / 0.081 / 0.049 | 0.170 / 0.313 / 0.120 | 0.50 / 0.83 / 0.86 |
| exp3 | 0.062 / 0.062 / 0.038 | 0.215 / 0.388 / 0.147 | 0.63 / 0.86 / 0.84 |
| pca-8 | 0.042 / 0.023 / 0.022 | 0.148 / 0.275 / 0.121 | 0.55 / 0.83 / 0.86 |
| vae-8 | 0.047 / 0.025 / 0.028 | 0.160 / 0.277 / 0.110 | 0.48 / 0.80 / 0.84 |
| knn | – | **0.119** / 0.255 / **0.098** | 0.58 / 0.86 / 0.86 |
| last | – | 0.210 / 0.378 / 0.115 | 0.65 / 0.84 / 0.88 |

**Findings.**

1. **A few learned shapes represent real curves better than the classic parametric families.** Four components
   already beat the 3-parameter power law and exponential, and they transfer to sources not seen in fitting. On
   late-drop sources the gap is large (NAS-Bench-201: 0.023 vs 0.062-0.081). Saturating curves cannot make the
   late, schedule-driven drop.
2. **The VAE gives nothing over PCA at k ≥ 4.** The space of curve shapes is close to linear and low-dimensional
   (about 4-8 dimensions).
3. **For ranking configs from 20% of their curve, nothing beats the last observed value** on any source, and on
   JAHS nothing gets far above 0.65. Shape models do improve the *level* forecast where curves still fall a lot
   late (knn: JAHS 0.21 → 0.12). For a faker this means the crossing pattern (how much the order changes after the
   early phase) is a property of tasks to model explicitly. It does not follow from a good mean shape.

Caveats: the extrapolation priors are pooled over all sources, not conditioned on the task. A task-aware model
(other curves of the same task as context, as freeze-thaw methods do) is the obvious next step for B.

## 2026-11-20: Measure 2 harness; faker v0 and v1

**Harness** (`curvefaker/evaluate.py`, `scripts/evaluate_fakers.py`; `results/faker_eval.csv`,
`results/faker_gaps.csv`). Each task gets a descriptor vector. All descriptors are computed on one seed (what an
optimiser sees; it also makes sources with 1-5 seeds comparable):

- the atlas statistics: dead / diverged / turns fractions, median final value, final spread, t50, t90, early-rank
  correlations, top10_kept;
- the noise level (log10 robust sd of step-to-step changes);
- **landscape_rho**: rank correlation of boosted trees trained on 200 configs with the true final value of others,
  i.e. how predictable the final value is from the config.

Two measures compare fakers with the real *select* tasks (150 TaskSet, 4 FCNet, 35 bbomix; equal numbers
generated):

- **coverage**: share of real tasks whose nearest generated task, in standardised descriptor space, is closer
  than the median distance between a real task and its nearest real neighbour;
- **precision**: the reverse, the share of generated tasks that resemble some real task.
- **gap**: distance between real and generated distributions per descriptor (1-D Wasserstein, in units of the
  real sd).

**real-fit** is a reference: real tasks of the *fit* split, used as if they were generated. It shows what real
but *different* task families reach.

**Faker v0** (`curvefaker/faker_hgp.py`), fitted on *fit* only:

- Each generated task borrows a real fit task's "record": the mean and covariance of its curves' PCA-8 scores,
  dead fraction, diverged fraction, noise level and number of hps.
- Configs are uniform. A random GP kernel (sparse relevances, random additive/product mix) gives fields that set
  the 8 scores, a dead region and a diverging region.
- White noise is added.

**Faker v1** (one change from v0): the Gaussian over PCA scores is replaced by the real task's own curves (one
seed, including its dead and diverged ones). They are handed to configs in order of a GP field: rank of field =
rank of final value.

| faker | coverage | precision | TaskSet | bbomix | FCNet |
|---|---|---|---|---|---|
| ifBO | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| hgp-v0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| hgp-v1 | 0.12 | 0.10 | 0.13 | 0.06 | 0.00 |
| real-fit (reference) | 0.20 | 0.16 | 0.25 | 0.00 | 0.00 |

Gaps (Wasserstein / real sd), largest first for ifBO:

| faker | turns | rank_early30_live | rank_early10 | noise | top10_kept | diverged | final_spread | landscape_rho |
|---|---|---|---|---|---|---|---|---|
| ifBO | 5.22 | 3.15 | 2.96 | 1.37 | 1.16 | 0.72 | 0.75 | 0.62 |
| hgp-v0 | 2.41 | 3.17 | 2.68 | 0.68 | 0.88 | 0.47 | 0.27 | 0.66 |
| hgp-v1 | 0.38 | 1.20 | 0.62 | 0.29 | 0.70 | 0.43 | 0.11 | 0.31 |
| real-fit | 0.43 | 0.73 | 0.24 | 0.34 | 0.62 | 0.38 | 0.15 | 0.33 |

**Findings.**

1. **ifBO covers no real select task.** Its largest gaps: far too many curves that improve and then get worse
   ("turns"), and a far less stable order of configs between early and late training. Both are central to
   freeze-thaw.
2. **v0 shows why a Gaussian over shape scores fails.** Checked on six fit tasks against the record they came from:
   the Gaussian fills the space between real shapes with curves that do not exist. "Turns" go from 0.01-0.04 to
   0.18-0.53, and the early-late order breaks (0.88 → 0.48). Its tails also shift the bad-start level, so dead
   configs stop counting as dead after renormalisation (QuickTune 0.23 → 0.00). Per-task curve sets are strongly
   non-Gaussian.
3. **v1 gets most curve statistics right**, but it only re-uses fit tasks, so it cannot beat real-fit's coverage.
   No faker covers FCNet or bbomix, real-fit included: those tasks differ from every fit family. Its remaining gap
   in early ranking is mostly the mixture of task types: v1 draws JAHS / QuickTune records 29% of the time,
   real-fit about 4%, and those two have the least stable early rankings.
4. The coverage threshold is strict in 12 dimensions. Even real-fit covers only a fifth of the select tasks.

Next: make v1 generalise beyond the fit tasks, with random task-level transformations of the curve set (time
warps, noise level, dead fraction), one at a time, and check that coverage of FCNet / bbomix rises without
losing TaskSet.

## 2026-11-20: Measure 3 harness: existing HPO algorithms on real vs generated tasks

**Method** (`curvefaker/hpo.py`, `scripts/evaluate_hpo.py`; `results/hpo_regret.csv`, `results/hpo_summary.csv`).
A task is one seed's curves. Training a config for k steps reveals its first k values. The *incumbent* is the
config with the lowest value observed so far. Three algorithms with fixed settings, chosen before looking at any
result:

- **random**: random configs, each trained to the end;
- **hyperband**: η = 3, minimum budget about T/27;
- **freezethaw**: a simple heuristic. Start 10 configs for 1 step, then either start a new one (probability 0.2)
  or continue the started config with the lowest last value by one step.

**Rank regret** at a budget of 1, 2, 5, 10, 20 full trainings: the share of the task's configs whose final value
beats the incumbent's (0 = found the best, 0.5 = a typical config). An earlier version divided by
(median − best final). It blew up on tasks where most configs end at the best value, and was replaced.

Same real select tasks as measure 2; equal numbers of generated tasks; 3 algorithm seeds per task.

| | random @1 / @5 / @20 | hyperband @1 / @5 / @20 | freezethaw @1 / @5 / @20 | mae vs real | HB gain | FT gain |
|---|---|---|---|---|---|---|
| **real select** | 0.51 / 0.21 / 0.07 | 0.18 / 0.08 / 0.05 | 0.26 / 0.12 / 0.05 | – | 0.155 | 0.112 |
| ifBO | 0.47 / 0.26 / 0.18 | 0.32 / 0.22 / 0.16 | 0.28 / 0.20 / 0.16 | 0.088 | 0.065 | 0.078 |
| real-fit | 0.51 / 0.21 / 0.09 | 0.19 / 0.10 / 0.05 | 0.31 / 0.13 / 0.07 | 0.014 | 0.140 | 0.090 |
| hgp-v1 | 0.49 / 0.17 / 0.06 | 0.17 / 0.08 / 0.03 | 0.30 / 0.12 / 0.05 | 0.016 | 0.140 | 0.077 |
| hgp-v2 | 0.49 / 0.18 / 0.06 | 0.18 / 0.08 / 0.04 | 0.31 / 0.12 / 0.05 | 0.015 | 0.140 | 0.080 |

(*mae*: mean absolute difference from the real mean regrets, over all algorithms and budgets. *HB / FT gain*: mean
over budgets of regret(random) − regret(hyperband / freezethaw), i.e. how much the multi-fidelity method helps.)

**Findings.**

1. **On ifBO tasks the algorithms behave differently from real tasks.** Hyperband's advantage over random search
   is less than half of the real one (0.065 vs 0.155). Every algorithm stalls at regret 0.16-0.18 after 20
   trainings, vs 0.05-0.07 on real tasks. A likely reason, consistent with measure 2: ifBO's noise and its many
   "turns" make the best *observed* value a poor guide to the best *final* value. So partial training is less
   informative, and the optimiser is led to lucky configs.
2. **The bootstrap fakers reproduce real algorithm behaviour as closely as real fit tasks do** (mae 0.015 vs
   0.014). The freezethaw gain is slightly underestimated by all three (0.08-0.09 vs 0.11).
3. Measure 3 agrees with measure 2: ifBO is far from real data, while v1/v2 are about as close as data from other
   real task families.

## 2026-11-20: Faker coverage, repeated over 3 generation seeds

| faker | coverage | precision | TaskSet | bbomix | FCNet |
|---|---|---|---|---|---|
| ifBO | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 |
| hgp-v0 | 0.014 ± 0.016 | 0.005 | 0.00 | 0.06 | 0.00 |
| hgp-v1 | 0.097 ± 0.017 | 0.079 | 0.11 | 0.04 | 0.08 |
| hgp-v2 (+ level warp) | 0.092 ± 0.029 | 0.071 | 0.09 | 0.09 | 0.08 |
| hgp-v3 (+ time warp) | 0.108 ± 0.046 | 0.049 | 0.11 | 0.11 | 0.00 |
| real-fit | 0.178 ± 0.024 | 0.152 | 0.22 | 0.00 | 0.00 |

(± = sd over the 3 generation seeds.) The single-seed v2 / v3 numbers of the previous entry were within this
noise. **Neither warp changes coverage measurably.** The level warp shifts coverage from TaskSet to bbomix. The
time warp lowers precision (it produces tasks unlike any real one), so it is dropped. A diagnosis of the nearest
generated neighbour of each FCNet / bbomix task shows the recurring mismatch: the *shares* of curve classes. These
tasks have a few turning curves (2-7%) or diverged curves alongside a predictable, mostly good landscape. In fit
tasks those shares come as a package, so such combinations never occur in a bootstrap.

## 2026-11-20: Fakers v4, v1w, v5: what limits coverage

Three single changes, each measured with 3 generation seeds (coverage ± sd):

| faker | change | coverage | precision | TaskSet | bbomix | FCNet |
|---|---|---|---|---|---|---|
| hgp-v1 | (base) | 0.097 ± 0.017 | 0.079 | 0.11 | 0.04 | 0.08 |
| hgp-v4 | v2 + dead / turns / diverged shares each drawn from a different fit task | 0.065 ± 0.032 | 0.046 | 0.07 | 0.06 | 0.00 |
| hgp-v1w | v1 + records drawn in proportion to their source in the fit split | 0.111 ± 0.011 | 0.092 | 0.13 | 0.03 | 0.17 |
| **hgp-v5** | v1w + landscape predictability matched per record | **0.150 ± 0.019** | 0.12 | 0.19 | 0.00 | 0.00 |
| real-fit | (reference) | 0.178 ± 0.024 | 0.152 | 0.22 | 0.00 | 0.00 |

- **v4 (rejected).** Drawing the class shares independently does not help and may hurt. Curves taken from
  different tasks do not form a consistent task, even after normalisation.
- **v1w (diagnostic).** v1 drew its 566 records uniformly, so JAHS and QuickTune appeared in 29% of tasks instead
  of 4%. Weighting fixes the early-ranking gaps (rank_early10 0.59 → 0.20, rank_early30_live 1.17 → 0.50,
  same as real-fit). Coverage only rises a little: the mixture is not the main limit.
- **v5.** The remaining gap was the landscape. Generated tasks were too predictable (median landscape_rho 0.89 vs
  0.77 on real select tasks; d = 10: 0.75, d = 1: 0.99), because the final-value order was an exact function of a
  smooth GP field. v5 mixes in a nugget: field = √(1−η)·GP + √η·noise. η is chosen per record so that the
  generated task's expected landscape_rho equals that of the real fit task the record came from. The real values
  are measured on the fit split only (median 0.74, 10-90%: 0.28-0.95). The expected values come from a simulated
  table over (d, η), made monotone. landscape_rho gap 0.50 → 0.34, **coverage 0.111 → 0.150**, within about one sd
  of real-fit.

Measure 3 for the new fakers (rank regret; real select: HB gain 0.155, FT gain 0.112):

| faker | mae | HB gain | FT gain |
|---|---|---|---|
| ifBO | 0.088 | 0.065 | 0.078 |
| real-fit | 0.014 | 0.140 | 0.090 |
| hgp-v1 (earlier run) | 0.016 | 0.140 | 0.077 |
| hgp-v1w | 0.021 | 0.132 | 0.055 |
| hgp-v5 | 0.019 | 0.130 | 0.059 |

These three algorithms never look at the configs, so only the set of curves matters here, and v1w / v5 differ
from v1 only in which records are drawn. **Open:** why the freeze-thaw gain is lower for the TaskSet-weighted
mixtures (0.055-0.059) than for real-fit (0.090), which is also mostly TaskSet. Candidate causes: v1w re-uses 400
TaskSet records while real-fit draws from all 3775; the 51 → 50 resampling; or run-to-run noise. Not checked yet.

**Where this leaves the faker.** v5 is the best faker so far. On both measures it is about as close to the real
select tasks as real tasks of other families are, and far closer than ifBO. But it re-uses real curve sets and
creates new tasks only through new landscapes. The next step is a generative model of whole curve *sets* that
also produces new curve sets, judged by whether it keeps v5's coverage while adding precision and novelty.

## 2026-11-20: A learned generative model of curve sets (set-VAE): first attempt fails, and why

**Model** (`curvefaker/setvae.py`). A hierarchical VAE over sets, trained on 3944 fit tasks × 128 configs (one
seed, 50 points; 5% of tasks held back to monitor overfitting, none seen):

- a task latent τ (16-d) from a set encoder (mean of per-curve embeddings);
- a config latent c (4-d) per curve;
- a decoder (τ, c) → curve, with a Gaussian likelihood whose noise sd per time point is set by τ.

Generation: c(x) from four GP landscape fields with v5's nugget and number of hps. 200 epochs, about 4 min on one
GPU. Validation loss was still falling slowly (−1.73 nats per point).

| variant | coverage | turns gap | noise gap | dead gap | diverged gap | landscape gap |
|---|---|---|---|---|---|---|
| setvae: τ from the N(0, I) prior | 0.000 | 18.4 | 2.59 | 0.76 | 1.76 | 0.94 |
| setvae-v2: τ from a 20-component mixture fitted to the fit tasks' posterior τ | 0.002 | 3.28 | 1.54 | 0.63 | 0.83 | 0.28 |
| diagnostic: decoder mean + a fit record's real noise level | 0.011 | 0.73 | 0.62 | 0.63 | 0.94 | 0.83* |

(*the diagnostic drew d uniformly and used no nugget, so its landscape gap is not comparable.)

**Three failure modes, each confirmed by a direct check:**

1. **Prior hole.** Fit tasks' τ posteriors are narrow (sd 0.06) and occupy a small part of the prior. A prior draw
   lands where the noise head outputs sd 0.33 instead of about 0.03. Drawing τ from a mixture fitted to the
   posteriors fixes most of it.
2. **Noise inflation.** The Gaussian likelihood's sd (median 0.03) absorbs heavy-tailed residuals that the decoder
   cannot follow, mostly divergence jumps. It is about 10× the real step-to-step noise of TaskSet. Generating with
   it turns smooth curves into "turns". Replacing it by a real noise level brings the turns gap from 3.3 to 0.7.
3. **Smeared class structure.** Dead fractions, diverged fractions and the final-value distribution remain off
   (0.6-0.9 sd). The model blends the sharp clusters of real tasks (dead at the bad level, jumps to the cap).

**Conclusion.** A straightforward learned generator is far behind the bootstrap faker v5 (coverage 0.150). To be
competitive, a learned model needs a heavy-tailed or mixture likelihood and explicit discrete components (dead /
diverged / live) rather than one continuous latent. That fits the atlas finding that these are separate
phenomena. Until then, v5 is the faker to build on.

**Resolved: the lower freeze-thaw gain of v1w / v5 is mostly noise.** The real tasks behind v1w's records, loaded
raw, give an FT gain of 0.089 (= real-fit), against 0.055 for v1w. But on the *same* 80 TaskSet records, the raw,
resampled, stored (fit-time subsample) and v1-processed versions give 0.050-0.083, each ± 0.012. The two most
different versions (raw 0.052, stored 0.083) differ only in the order of the configs, i.e. in which random configs
the algorithms draw. The task-to-task spread of the gain is large: with about 190 tasks its standard error is about
0.01, so differences below about 0.03 mean nothing. `scripts/evaluate_hpo.py` now reports these standard errors.
