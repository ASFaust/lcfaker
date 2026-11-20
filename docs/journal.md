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

## 2026-11-20: Noise models (question C)

**Question.** What is the noise in a single learning curve, per source?

**Method** (`curvefaker/noise.py`, `scripts/noise_models.py`; `results/noise_models.csv`). Each curve (one seed,
normalised z, diverged curves left out) is modelled as smooth trend + noise in a Gaussian process. The trend is
Matern-5/2 in log-time plus Matern-5/2 in linear time, so both fast early drops and late schedule-driven drops
are smooth. The trend and the noise size are fitted per curve. Noise models:

- **white**: independent from step to step;
- **ar1**: correlated from step to step, Cov = σ²ρ^|s−t|;
- **phase**: white, but with sd σ·exp(β(u − ½)), u = log-time in [0, 1], so noise can grow or shrink during
  training;
- **ar1+phase**: both.

Up to 40 tasks × 30 configs per source (300 curves when T > 100, because the cost grows with T³), split by task
into halves A and B. The global shape parameters (ρ, β) are chosen on A by grid search (ρ ∈ {−0.5 … 0.9};
β ∈ {−2 … 2}, widened to {−4 … 2} for a rerun of the sources that hit the edge). They are frozen, the per-curve
parameters are refitted on B, and B's log likelihood is compared. **dLL** = nats per point over white noise,
± standard error over curves. A model counts as better only above 3 standard errors. (Fitting ρ and β by gradient
together with everything else was tried first; it ended in different local optima for equivalent models and was
replaced by the grid.)

Also, on B: **sigma_med**, the median noise sd (z units: share of the task's start-to-best range); **spike_frac**,
the share of noise values beyond 5 robust sd (Gaussian: 6·10⁻⁷); **shared**, the share of noise variance common to
all configs of a task at the same step (as if all were scored on one validation set).

| source | T | best | dLL phase (± se) | β | ρ (ar1) | sigma_med | spike_frac | shared |
|---|---|---|---|---|---|---|---|---|
| TaskSet | 51 | white | 0 | 0 | 0 | 0.003 | 0.026 | 0.000 |
| NAS-Bench-201 | 200 | phase | +0.081 ± 0.008 | −2 | 0 | 0.041 | 0.010 | 0.005 |
| FCNet | 100 | phase | +0.086 ± 0.011 | −2 | 0 | 0.009 | 0.035 | 0.000 |
| bbomix | 300 | phase | +0.058 ± 0.008 | −1 | 0 | 0.0006 | 0.035 | 0.001 |
| QuickTune | 50 | phase | +0.015 ± 0.004 | −1 | 0 | 0.010 | 0.033 | 0.003 |
| JAHS | 200 | phase | +0.108 ± 0.022 | −3 | 0 | 0.022 | 0.026 | 0.000 |
| **ifBO (check)** | 100 | white | 0 | 0 | 0 | 0.027 | 0.009 | 0.003 |

(NAS-Bench-201, FCNet and JAHS were rerun with β ∈ {−4 … 2} because −2 was the edge of the first grid. FCNet and NAS-Bench-201 stay at −2, now inside the grid. JAHS moves to −3: its noise sd falls about 20× from start to end. The rerun draws a different sample of curves, so its dLL values differ slightly from the first run, e.g. JAHS +0.155 → +0.108.)

**Findings.**

1. **The method recovers ifBO's noise model.** ifBO adds white Gaussian noise, and the comparison picks white
   (ρ = 0, β = 0).
2. **Noise shrinks during training on every real source except TaskSet.** β < 0 means noise falls in log-time;
   β = −2 means the sd falls by a factor of e² ≈ 7 from start to end. It is strongest with cosine schedules
   (JAHS, NAS-Bench-201), where the learning rate goes to zero at the end. TaskSet's noise shows no phase effect
   with this resolution. ifBO's noise is constant in time.
3. **No step-to-step correlation beyond the smooth trend** (ρ = 0 everywhere). Earlier measurements found
   correlated step-to-step changes (TaskSet +0.2). With a flexible trend, slow wandering cannot be told apart from
   the trend itself, so it is absorbed there. For a faker this distinction does not matter: it just needs smooth
   curves plus white, shrinking noise.
4. **Noise is heavy-tailed everywhere**: 1-3.5% of noise values are beyond 5 robust sd (Gaussian: practically
   none). These are the spikes seen in the atlas (bbomix, FCNet). ifBO has fewer (0.9%).
5. **No noise shared across configs** (shared < 0.5% of noise variance). Configs of a task do not share an
   evaluation-noise component.
6. **Noise size differs by two orders of magnitude between sources** (median 0.0006 bbomix to 0.04 NAS-Bench-201,
   in units of the task's range). ifBO (0.027) sits at the noisy end.

**For the faker:** smooth curve + heavy-tailed white noise whose size depends on the task and shrinks during
training, with no cross-config component. The bootstrap faker v5 gets all of this from the real curves it re-uses;
a generative faker must model it explicitly. (The set-VAE's single Gaussian sd per time point could represent the
shrinking but not the heavy tails, which is one reason its noise inflated.)

## 2026-11-20: State of the project after the first session (summary)

**What explains real curves** (all on allowed data; generators fitted on *fit*, compared on *select*):

- **Curve shapes (B).** About 4-8 linear components learned from real curves represent held-out curves better
  than classic 3-parameter saturating families. The gap is large for cosine-schedule sources, where much of the
  progress comes late. A VAE adds nothing over PCA at k ≥ 4. For ranking configs early, nothing beats "last value
  seen".
- **Config → performance (A).** Predictability from 200 configs ranges from about 0.2 (QuickTune) to 0.99
  (bbomix). One hp dominates on TaskSet, where interactions matter. FCNet, bbomix and NAS-Bench-201 are close to
  additive. A faker has to match *how predictable* a landscape is (faker v5).
- **Noise (C).** White and heavy-tailed, shrinking during training (except TaskSet), no cross-config component,
  size varying 100× between sources.
- **Task level.** Dead configs (0-38%) and abrupt divergence (up to 15%) are separate phenomena. Per-task curve
  sets are strongly non-Gaussian in shape space (faker v0).

**Fakers, head to head** (measure 2: coverage of real select tasks, ± sd over 3 generation seeds; measure 3: mean
abs. difference of HPO-algorithm regret from real):

| faker | idea | coverage | HPO mae |
|---|---|---|---|
| ifBO | random BNN → parametric basis curves + white noise | 0.000 | 0.088 |
| hgp-v0 | GP landscapes → Gaussian over PCA scores per task | 0.014 | – |
| setvae-v2 | learned hierarchical VAE over curve sets | 0.002 | – |
| hgp-v1w | real fit task's curve set, assigned to configs by a GP landscape | 0.111 ± 0.011 | 0.021 |
| hgp-v5 | v1w + landscape predictability matched per record | 0.150 ± 0.019 | 0.019 |
| real-fit | (reference: real tasks of other families) | 0.178 ± 0.024 | 0.014 |

**Main conclusions.**

1. ifBO is far from real data on everything that matters to multi-fidelity HPO. It has too many curves that get
   worse again, a weak link between early and final performance, constant and comparatively large noise, and
   almost no dead configs. On its tasks Hyperband's advantage over random search is less than half the real one.
2. The best faker so far (v5) re-uses real curve sets and gets new tasks from GP landscapes whose predictability is
   matched to real tasks. It is about as close to unseen task families as real tasks of other families are.
3. A first fully learned generator fails for identifiable reasons: prior hole, noise inflation by heavy tails, and
   blended classes.

**Next steps, in order.**

1. A generative faker with *explicit* components: live curves from a learned shape model with heavy-tailed,
   shrinking noise; dead and diverged configs as separate classes with their own regions. The goal is new curve
   sets at v5's coverage.
2. Add a novelty measure (distance of generated tasks to the nearest fit task), so that a faker that copies fit
   tasks is not rewarded.
3. Measure 3 with more algorithm seeds or more tasks (standard error of the gains is about 0.01), and with model-based
   methods that use the configs (BOHB-like), since the current three algorithms ignore X.
4. Use the FT-PFN / rogue-dagger checkpoints as additional (biased) evaluators.

## 2026-11-20: Novelty measure

`scripts/evaluate_fakers.py` now also reports **novelty**: the median distance of generated tasks to their nearest
of 600 random fit tasks (standardised descriptors, as for coverage), divided by the same median for the real
select tasks. About 0 means copies of fit tasks; about 1 means as new as an unseen real task family; well above 1
means unlike any real task.

| faker | coverage | precision | novelty |
|---|---|---|---|
| ifBO | 0.000 | 0.000 | 2.54 |
| setvae-v2 | 0.002 | 0.002 | 1.32 |
| real-fit | 0.178 | 0.152 | 0.64 |
| hgp-v1w | 0.111 | 0.092 | 0.78 |
| hgp-v5 | 0.150 | 0.120 | 0.76 |

v5 does not merely replay fit tasks: its new landscapes put its tasks a little further from the fit tasks than
real-fit's own. ifBO and the set-VAE are "novel" in the useless sense, far from every real task. A good faker should
combine high coverage with novelty near 1.

## 2026-11-20: Faker "comp": explicit components with learned live curves (fails, diagnosed)

**Design** (`curvefaker/faker_comp.py`), aimed at the three set-VAE failures:

- a set-VAE trained on *live* curves only (improving / turning), with a Student-t likelihood (ν = 3) against noise
  inflation;
- each generated task anchored to a fit record: its posterior task latent τ (+ jitter 0.3), number of hps, nugget,
  dead and diverged shares and dead level (against the prior hole and inconsistent class mixing);
- dead and diverged configs as explicit classes in their own GP regions (against blended classes).

Live model: 3890 fit tasks, validation −2.87 nats per point.

| faker | change | coverage | novelty | rank_early10 gap | turns gap | noise gap |
|---|---|---|---|---|---|---|
| comp | config latents: 4 independent GP fields, N(0, 1) marginals | 0.014 | 1.76 | 2.70 | 1.49 | 0.71 |
| comp-v2 | config latents: the record's own posterior latents (+ jitter), assigned by GP-field rank | 0.014 | 1.96 | 3.03 | 1.27 | 0.68 |

**Diagnosis** (checks on fit tasks / records):

1. *Config-level prior hole* (motivated comp-v2). Within a task, the posterior config latents are correlated
   (median |r| 0.42), and two of the four dimensions barely vary (sd 0.02-0.03 vs 1 under the prior). Rank
   correlation between the value at 10% and the final value: real 0.87, reconstruction 0.84, N(0, I) latents 0.77,
   dimensions shuffled 0.69. comp-v2 fixes this, but coverage does not move. So it was not the main limit.
2. *The decoder loses fine early-final consistency.* On six records, the decoded mean curves of the record's own
   latents give an early-final rank correlation of 0.71-0.86, vs 0.86-0.91 for the real curves (TaskSet). Early
   on, the differences between configs are small, and the decoder's reconstruction error (median 0.0025) is
   comparable to them. Real tasks vary little in this property, so a small loss is about 3 real sd.
3. *Noise still too large.* The learned noise scale is 3-8× the real step-to-step noise on TaskSet records, and on
   one record 0.6 at the first point. Even the Student-t scale absorbs decoder misfit.

**Conclusion.** On the property that matters most to freeze-thaw (how consistently configs keep their order), the
learned generators fall short even when everything else is anchored to real tasks. Re-using real curve sets (v5)
keeps it exactly. A learned generator would need far more precise reconstruction (more latent dimensions, a
residual model of the record's curves) or a different likelihood that separates decoder misfit from noise. Left
for later; v5 remains the faker of record.

## 2026-11-20: Correction: hgp-v1 to v5 are not curve models

The summary above calls v5 "the best faker so far", which overstates it. hgp-v1 to v5 contain **no curve model**,
neither analytical nor learned. Every curve of a generated task is a real curve of one fit task (one seed,
resampled to 50 points); only the assignment of curves to new, synthetic configs is generated (a GP field ranked
against the final values, with a nugget in v5). They are a resampling baseline ("wavetable"), and v5 is a result
about the **hyperparameter axis** (matching how predictable a landscape is mattered), not about curve generation.
Their coverage is bounded by the fit tasks they re-use (real-fit 0.178). On the curve-generation axis, every model
tried so far (v0, set-VAE, comp) is far from real data.

Consequences: v5 serves as a reference for "perfectly realistic curves + synthetic landscapes". The next step is
an analysis of what the data look like before more generators are built: `docs/analysis_plan.md`. It covers
metrics per axis (curve generation, hyperparameter influence, noise, task level, whole faker), prior knowledge
about training curves, analysis questions, and the insight → model table it should produce.

## 2026-11-20: Analysis, part 1: curve shapes, time axis, learning rate (S1-S3, S5, H2, H3)

Plan: `docs/analysis_plan.md`. Analysis on the *fit* split; *select* sources reported as replication.

### S1: which parametric families explain real curves? (`scripts/shape_families.py`, `curvefaker/families.py`)

Live curves (improving / turning), one seed, native length. Up to 3000 per source. Eight families, each fitted per
curve. Errors are divided by the curve's own noise sd (robust, from step-to-step changes). **ref**: error of
predicting each held-out point by the mean of its two neighbours, a nonparametric floor (white noise: about 1.22;
heavy tails raise it).

Interpolation error (every 5th point held out), median:

| family (params) | TaskSet fit | TaskSet sel | NB201 | QuickTune | JAHS | FCNet | bbomix |
|---|---|---|---|---|---|---|---|
| pow3 (3) | 6.47 | 3.25 | 2.75 | 2.82 | 2.29 | 2.19 | 10.0 |
| loglin: line in log-time (2) | 8.38 | 3.73 | 2.62 | 3.08 | 2.46 | 2.69 | 12.4 |
| exp3 (3) | 3.05 | 1.78 | 1.93 | 1.89 | 2.51 | 2.06 | 7.27 |
| weibull (4) | 1.78 | 1.40 | 1.65 | 1.49 | 1.95 | 1.77 | 5.53 |
| logsig: S-shape in log-time = ifBO's hill (4) | 1.73 | 1.36 | 1.67 | 1.40 | 1.97 | 1.73 | 5.57 |
| hinge1: 1 knee in log-time (4) | 2.33 | 1.50 | 1.52 | 1.86 | 1.93 | 1.90 | 5.59 |
| hinge2: 2 knees in log-time (6) | 1.45 | 1.24 | 1.47 | 1.44 | 1.85 | 1.73 | 3.59 |
| **logsig2: two S-stages in log-time (7)** | **1.16** | **1.15** | **1.45** | **1.29** | **1.82** | **1.63** | **2.73** |
| ref (neighbour mean) | 1.27 | 1.27 | 1.73 | 1.36 | 2.14 | 1.83 | 1.84 |

Extrapolation (last 30% from the first 70%), median: logsig2 is best or tied on all sources (TaskSet 1.46, QT 1.41,
FCNet 1.16, bbomix 2.31) except NB201, where hinge2 is best (1.55 vs 1.81); on JAHS the two tie (1.21 vs 1.15).
pow3 / loglin extrapolate worst (TaskSet 6.5 / 12.4).

Best family by BIC (share of curves): logsig2 40% (TaskSet fit), 30% (TaskSet sel), 35% (QT), 60% (bbomix);
hinge2 42% (NB201), 31% (JAHS); the classic pow3 / exp3 / loglin win on ≤ 11% of curves anywhere.

**Findings.** (1) The classic saturating families (power law, exponential, log-linear) are far from real curves:
2-8× the error of the best family. (2) A **two-stage S-shape in log-time** reaches the nonparametric floor on
TaskSet and QuickTune and beats it on NAS-Bench-201, JAHS and FCNet; a one-stage S-shape (ifBO's "hill") is clearly
worse. On cosine-schedule sources, a piecewise-linear shape in log-time with knees does as well. (3) bbomix is not
explained by any family (spikes; 300 points of detail).

### S2 / S3: are curve shapes multimodal, and in which clock? (`scripts/shape_clusters.py`, `scripts/shape_template.py`)

Shapes were separated from level and scale: progress curve p = (start − value) / (start − best), 0 → 1, on a
common grid of 40 points over the budget fraction 0.02-1 (log-time or linear time). 16.5k live curves with a gain
of at least 0.1.

- **Mixture clustering does not find discrete modes.** BIC keeps improving up to the largest k tried (12; with
  16k curves BIC rewards any extra flexibility). Bootstrap stability is only moderate (adjusted Rand index 0.45-0.67).
  A UMAP map (`figures/shape_umap.png`) shows **one connected manifold with two main branches**: fast-saturating
  shapes (TaskSet, QuickTune, FCNet, bbomix) and late-progress shapes (JAHS, NAS-Bench-201). Clusters are strongly
  source-specific. Within a task, curves spread over about 4 clusters (the main cluster holds 40-63%).
- **The clusters are largely one S-shape at different positions** (`figures/shape_clusters.png`). Fitting each
  progress curve with one shifted / stretched S-shape (location m, steepness k, offset, amplitude) explains 76% of
  curves within RMSE 0.05 and 93% within 0.10 (progress units), in either clock (log 0.931, linear 0.932 within
  0.10; the free steepness lets one clock imitate the other, so this does not decide the clock).
- **Where the multimodality is.** The template location m (`figures/shape_template_location.png`) has modes at both
  edges of the training window. Left: only the tail of the S is visible (curve converging from the first point).
  Right: only its start is visible (a late drop; most NAS-Bench-201, about 20% of JAHS). JAHS has both modes. TaskSet
  has a spike at about 4.5% of the budget: many curves do most of their progress between the initialisation point
  and the 2nd-3rd recorded point, a resolution limit of the data. Feature histograms
  (`figures/shape_features_hist.png`) agree. JAHS half-progress time has two humps (fast learners vs late learners),
  NAS-Bench-201's late share has two humps. TaskSet is a spike of very fast curves plus an even spread. QuickTune,
  FCNet and bbomix are unimodal and skewed. (The formal bimodality tests used, the bimodality coefficient and 1- vs
  2-component BIC, cannot tell bimodality from skew here and are not used as evidence.)
- **What the single template misses (7%)** (`figures/shape_template_residual.png`): two-stage curves (plateau, then
  a second rise; FCNet, TaskSet), curves that first get worse and then improve (FCNet), overfitting (rise then
  partial fall-back; bbomix, TaskSet), and unstable / spiky curves.

**So the hunch is partly right.** Shapes are not a set of discrete families. They are a continuum: mostly one
S-shape in log-time whose position and steepness vary. That continuum is multimodal in its *position*, because the
training budget cuts the S at different places (only tail / full / only start), and in a few sources (JAHS) learners
split into fast and late groups. Besides that there are a few distinct minority patterns.

### S5: are a task's curves one template moved per config? (`scripts/shape_invariant.py`)

Shape-invariant model (as SITAR in growth-curve research): per task, one smooth template f in log-time, and per
curve y_i(u) = α_i + a_i · f((u − β_i) · e^(γ_i)) with level α, amplitude a, shift β (speed), stretch γ. Fitted by
alternation (exact least squares for f, grid search for β, γ; a first gradient-descent version got stuck and gave
nested models worse fits, so it was replaced). Within-task variance explained on held-out points:

| source | level + amplitude (2) | + shift (3) | + stretch (4) | free two-stage S per curve (7) |
|---|---|---|---|---|
| TaskSet | 0.900 | 0.969 | 0.974 | 0.977 |
| QuickTune | 0.940 | 0.974 | 0.982 | 0.985 |
| NAS-Bench-201 | 0.875 | 0.899 | 0.906 | 0.924 |
| JAHS | 0.814 | 0.849 | 0.857 | 0.865 |
| FCNet | 0.848 | 0.859 | 0.851 | 0.862 |
| bbomix | 0.976 | 0.986 | 0.987 | 0.989 |

**Finding.** Within a task, curves are close to **one template, shifted in log-time and scaled** per config: 3-4
parameters per curve explain almost all of what 7 free parameters do. The shift (speed) is the most important
addition after level and amplitude. The remaining misfit is curve-specific detail; on TaskSet it is about 2× the
free fit's error, so a small but real per-curve residual is needed.

### H2: is the learning rate a time warp? (`scripts/lr_timewarp.py`, TaskSet adam1p / adam4p)

Pairs of curves within a task, the second with a clearly higher learning rate (median ratio e^1.2 ≈ 3.3). RMSE
remaining after alignment, relative to before: level shift only 0.50, log-time shift only 0.54-0.58, **both
0.14-0.18** (about 3% of the variance left). The higher-learning-rate curve is usually ahead in time (median shift
+0.35 to +0.5 in log-time, i.e. about 1.4-1.6× faster), but the shift does not scale with the size of the ratio
across pairs (rank correlation ≈ 0; presumably because past the best learning rate the relationship turns; not yet
checked). **So the learning rate acts as a partial time warp *and* a level change**, matching the shape-invariant
result.

### H3: does the schedule act as the clock? (`scripts/schedule_clock.py`, TaskSet nadamw / adam8p)

Schedules now come from the TaskSet source code (`curvefaker/schedules.py`): adam1p/4p constant; adam6p/8p linear and
exponential decay; nadamw warm-up, constant phase, cosine decay.

- **Per-curve fits** of the annealing scaling law (cumulative learning rate S1 plus annealing area S2, Tissue et al.
  2024) with the config's true schedule vs a shuffled schedule: no difference (the true one wins on 49-52%). This
  test is **inconclusive**: the fits were badly optimised (worse than pow3, which they nest) and the annealing term
  extrapolates freely.
- **Within-task test** (no curve fits): per task, the rank correlation between a config's share of progress made in
  the second half and properties of its schedule (299 nadamw / 298 adam8p tasks):

| property | nadamw: median ρ (share of tasks > 0) | adam8p |
|---|---|---|
| share of the cumulative learning rate spent in the 2nd half | +0.08 (88%) | +0.11 (90%) |
| … controlling for the learning rate | +0.09 (89%) | +0.08 (82%) |
| annealing area S2 | −0.08 (12%) | −0.11 (10%) |
| learning rate | −0.12 (34%) | −0.44 (1%) |

**Finding.** Consistent but weak support for the cumulative learning rate as the clock: configs that spend more of
their learning rate late make more progress late, in about 90% of tasks. The annealing "bonus" is not visible: more
annealing goes with *less* late progress, presumably because strong decay stops training. The learning rate itself
is the strong factor: higher rates finish earlier. TaskSet curves mostly converge early, so schedule effects are
small here. Sources with cosine schedules (JAHS, NAS-Bench-201) have one schedule for all configs and cannot test
this.

## 2026-11-20: Analysis, part 2: noise and seeds (N1-N3), landscape structure (H1, H5), what is task-level (T1)

### N1-N3: noise shape, spikes, seeds (`scripts/noise_shapes.py`; `results/noise_shapes.csv`)

Residuals of live curves (one seed) from their two-stage S-shape fit in log-time; up to 1600 curves per source.

| | TaskSet | NB201 | QuickTune | JAHS | FCNet | bbomix |
|---|---|---|---|---|---|---|
| excess kurtosis (Gaussian 0) | 0.5 | 1.9 | 0.6 | 3.5 | 3.0 | 8.0 |
| skew | 0.08 | 0.73 | 0.11 | 1.03 | 1.10 | 0.32 |
| spikes (> 5 robust sd) | 2.1% | 1.0% | 1.9% | 2.6% | 2.7% | 4.1% |
| share of spikes upward (worse) | 52% | 92% | 55% | 73% | 71% | 46% |
| lag-1 autocorrelation | 0.05 | −0.02 | 0.03 | −0.01 | −0.04 | 0.50 |
| noise sd, early / mid / late third (× whole curve) | 1.00 / 0.87 / 0.82 | 1.61 / 1.32 / 0.46 | 1.06 / 1.07 / 0.77 | 1.92 / 1.21 / 0.39 | 1.56 / 1.05 / 0.62 | 1.43 / 0.82 / 0.60 |
| rank corr. noise sd vs curve level | 0.32 | 0.44 | 0.10 | 0.47 | 0.40 | 0.44 |
| spike co-occurrence across configs (× base rate) | 0 | 1.2 | 0 | 0 | 0.5 | 0.3 |
| seed difference: level / log-time trend / white | 55 / 9 / 32% | 4 / 2 / 95% | – | 14 / 3 / 82% | 32 / 5 / 63% | 15 / 9 / 71% |
| seed share of final-value variance: all / best 10% | 6% / 37% | 0.4% / 27% | – | (unreliable) | 11% / 24% | 1% / 1% |

(JAHS "seeds" are de-duplicated re-runs, which exist only for a particular subset of configs; its seed numbers are
not used. Some TaskSet families, e.g. quadratic, are deterministic: identical seeds. Those pairs are skipped.)

**Findings.**

1. **Noise is heavy-tailed everywhere** (excess kurtosis 0.5-8). On cosine-schedule and tabular sources it is
   **skewed upward**: spikes make the loss worse, not better.
2. **Noise shrinks during training** (late third 0.4-0.8× the curve's level), most strongly with cosine schedules.
   This agrees with the GP result.
3. **Noise grows with the loss level**: worse curves are noisier.
4. **Spikes are per config**, not shared across configs at the same step.
5. **No serial correlation** of residuals, except in bbomix (0.5), where the S-shape fit misses structure.
6. **Seeds.** On TaskSet a different seed mostly shifts the curve's *level* (55% of the difference). On the other
   sources the seed difference is mostly white noise. Seed luck is a small share of the final-value variance
   overall (0.4-11%), but **a quarter to a third among the best 10% of configs**: near the optimum, seed luck is
   comparable to config differences.

### H1 / H5: landscape structure and bad regions (`scripts/landscape_structure.py`; `results/landscape_structure.csv`)

Per task, on a 10-nearest-neighbour graph of configs:

- **local optima ratio**: share of configs better than all their neighbours, relative to shuffled final values;
- **best basin**: share of configs from which greedy descent reaches the global best;
- **near best**: share of configs within 0.05 z of the best;
- for tasks with a learning rate: the median learning-rate rank (0 = lowest, 1 = highest in the task) of dead,
  diverged and top-10% configs; AUC of predicting diverged (dead) from a high (low) learning rate alone; and
  **gap**, the rank of the best config minus the 5th percentile rank of diverged configs.

| source | local optima ratio | best basin | near best | LR rank dead / diverged / top 10% | AUC diverged ← high LR | AUC dead ← low LR | gap best → divergence |
|---|---|---|---|---|---|---|---|
| TaskSet | 0.53 | 0.03 | 0.11 | 0.16 / 0.85 / 0.58 | 0.93 | 0.91 | +0.01 |
| JAHS | 0.54 | 0.09 | 0.17 | 0.66 / 0.92 / 0.62 | 0.85 | 0.39 | +0.04 |
| FCNet | 0.32 | 0.09 | 0.28 | 0.85 / 0.85 / 0.26 | 0.90 | 0.14 | −0.24 |
| bbomix | 0.28 | 0.13 | 0.46 | 0.61 / 0.87 / 0.51 | 0.79 | 0.46 | +0.41 |
| QuickTune | 0.77 | 0.16 | 0.03 | 0.35 / 0.56 / 0.57 | 0.54 | 0.65 | +0.52 |
| NB201 | 0.43 | 0.02 | 0.13 | (no LR hp) | | | |

**Findings.**

1. **Landscapes are multimodal**: smoother than random (local optima at 28-77% of the shuffled rate), but 3-7% of
   configs are local optima, and the global best's basin holds only 2-16% of configs. Caveat: with one seed, noise
   creates spurious optima, and in large one-hot spaces (JAHS, QuickTune) neighbourhoods are poorly defined. These
   are upper bounds on ruggedness.
2. **The learning rate organises the bad regions.** On TaskSet: dead at low LR (AUC 0.91), diverged at high LR
   (AUC 0.93), and the best config sits right at the divergence boundary (gap +0.01), the classic "largest stable
   learning rate". JAHS likewise (gap +0.04). FCNet differs: dead *and* diverged both at high LR (collapse), best at
   low LR. QuickTune's learning rate barely matters (fine-tuning: model choice dominates).
3. The near-best region ranges from 3% (QuickTune) to 46% (bbomix) of configs, a key difficulty parameter.

### T1: which shape properties are task-level? (`scripts/variance_levels.py`; `results/variance_levels.csv`)

Variance of shape features of live curves, split into between sources / between tasks within a source / within
tasks:

| feature | sources | tasks | configs (within task) |
|---|---|---|---|
| shape location m (speed) | 30% | 10% | 59% |
| steepness log k | 23% | 14% | 64% |
| half-progress time u50 | 26% | 17% | 57% |
| late share | 5% | 16% | 80% |
| template misfit | 3% | 17% | 80% |

**Finding.** Most shape variation is **between configs of the same task**, not between tasks: the config decides
the speed and steepness. Sources differ systematically (about a quarter of the variance), tasks within a source
modestly (10-17%). Together with S5 (one template per task, warped per config): the template is task-level, but
the warps carry most of the shape variation and are config-level.

### H4: does the config shape the curve beyond its final value? (`scripts/config_shape.py`; `results/config_shape.csv`)

Per task: speed (template location m), log steepness and half-progress time u50 of live curves, predicted by
boosted trees (5-fold cross-validation within the task) from the final value only, from the config only, and from
both. Median cross-validated R²:

| target | input | TaskSet (120) | JAHS (40) | FCNet (4) | bbomix (35) | QuickTune (23) |
|---|---|---|---|---|---|---|
| u50 | final only | 0.57 | 0.11 | 0.09 | −0.02 | 0.13 |
| | config only | 0.86 | 0.74 | 0.41 | 0.22 | −0.17 |
| | config + final | 0.88 | 0.80 | 0.44 | 0.30 | 0.21 |
| speed m | final only | 0.55 | −0.04 | −0.03 | 0.00 | 0.07 |
| | config + final | 0.82 | 0.58 | 0.28 | 0.13 | 0.04 |
| steepness | final only | 0.31 | 0.16 | 0.10 | 0.02 | −0.13 |
| | config + final | 0.67 | 0.52 | 0.30 | 0.24 | −0.19 |

**Finding.** The config determines the curve's speed and steepness **far beyond what its final value says**
(JAHS: u50 from the final value R² 0.11, from the config 0.74). A model of "final-value landscape + shape given
the level" misses most of the shape variation. Configs must drive the shape (warp) parameters directly. (QuickTune
has too few full-length configs per task for this test.)

## 2026-11-20: Faker swt: the model class from the analysis (shape-warp-template)

**Model** (`curvefaker/faker_swt.py`, `curvefaker/sim.py`). Fitted on *fit* tasks only; one record per fit task:

- the task's shape-invariant template in log-time;
- the fitted warps of up to 200 live curves (level, amplitude, log-time shift, stretch);
- noise: robust sd of the residuals, its profile over the budget, Student-t degrees of freedom from the kurtosis;
- class shares, dead level, divergence times.

Generation:
1. A record and its template (re-used, so tasks are new but templates are real).
2. GP landscape fields over uniform configs, which assign warps to configs.
3. Dead / diverged configs at the low / high end of an LR-like axis (first config dimension) mixed with a field.
4. Curves = template warped per config + Student-t noise with the record's time profile; dead curves flat;
   diverged curves jump to the cap at a divergence time from the record.

Unlike hgp-v1 to v5, the **curves are generated, not replayed**.

Two fixes during construction:
- A Gaussian copula over the four warp parameters produced warp combinations no real curve has (up to 18% of live
  final values below the task's best). Warps are now *joint vectors of real fitted curves*.
- Extrapolating the template beyond the range real curves covered is blocked (template held flat there).

**Per-axis evaluation** (`scripts/evaluate_axes.py`; real = the select tasks; one generation seed). B: shape-feature
gaps, a classifier two-sample test (AUC; 0.5 = indistinguishable), k-NN precision / recall of curves. C: noise
gaps. A: landscape properties per task.

| | real (median) | ifBO | real-fit | hgp-v5 | swt-v1 | swt-v2 |
|---|---|---|---|---|---|---|
| A: speed predictable from config (R², median) | 0.73 | 0.22 | 0.88 | 0.02 | 0.13 | 0.37 |
| A: local-optima ratio (median) | 0.46 | 0.53 | 0.55 | 0.71 | 0.72 | 0.77 |
| A: near-best share (median) | 0.14 | 0.01 | 0.09 | 0.15 | 0.05 | 0.11 |
| A: landscape_rho at 200 (median) | 0.77 | 0.73 | 0.78 | 0.78 | 0.70 | 0.72 |
| B: classifier AUC real vs generated curves | – | 0.91 | 0.82 | 0.87 | 0.89 | 0.87 |
| B: precision / recall of curves | – | 0.61 / 0.72 | 0.80 / 0.82 | 0.70 / 0.81 | 0.61 / 0.74 | 0.63 / 0.76 |
| B: gap half-progress time / late share / max rise | – | 0.59 / 0.94 / 0.89 | 0.16 / 0.12 / 0.10 | 0.08 / 0.11 / 0.10 | 0.16 / 0.06 / 0.18 | 0.08 / 0.07 / 0.10 |
| C: gap kurtosis / spikes / late-vs-early sd | – | 0.61 / 0.74 / 0.26 | 0.23 / 0.69 / 0.31 | 0.27 / 0.84 / 0.07 | 0.29 / 0.30 / 0.08 | 0.27 / 0.29 / 0.08 |
| E: coverage (± sd, 3 seeds) | – | 0.000 | 0.178 | 0.150 | 0.071 ± 0.034 | 0.088 ± 0.029 |
| E: novelty | – | 2.54 | 0.64 | 0.76 | 1.02 | 1.04 |
| E: HPO mae / HB gain / FT gain (real: 0.155 / 0.112, se ≈ 0.008) | – | 0.088 / 0.065 / 0.078 | 0.014 / 0.140 / 0.090 | 0.019 / 0.130 / 0.059 | 0.029 / 0.136 / 0.106 | 0.032 / 0.133 / 0.087 |

- **swt-v1** assigned speeds only within strata of 10 configs with similar final value. The config barely
  predicted speed (R² 0.13 vs 0.73 real).
- **swt-v2** (one change): each config gets a target (final-value rank, speed) from two fields, correlated like
  the task's real pairs, and takes the nearest real warp vector. Speed-from-config 0.13 → 0.37, near-best share
  and half-progress-time gaps roughly halved.

**Findings so far.**
1. swt generates genuinely new tasks (novelty ≈ 1, i.e. as new as an unseen real family; the replay fakers
   0.64-0.76) while keeping the HPO behaviour close to real: Hyperband gain within 2-3 standard errors, freeze-thaw
   gain within 1-3.
2. Its **noise is the most realistic of all fakers** (spike gap 0.29-0.30 vs 0.69-0.84 for the replay fakers, whose
   curves were resampled to 50 points, which smooths spikes away).
3. On curve shape (classifier AUC 0.87, precision 0.63) it is about level with hgp-v5, which replays real curves.
   That suggests the shape-invariant parametrisation captures most of what distinguishes real curves, though not
   everything; real-fit itself reaches only AUC 0.82.
4. Coverage is lower than v5's (0.09 vs 0.15) but more of its tasks are new. The remaining landscape gaps:
   ruggedness (local-optima ratio 0.77 vs 0.46, caused by the white-noise nugget) and speed-from-config (0.37 vs
   0.73).

## 2026-11-20: swt v3-v5: landscape roughness, noise level, duplicate warps

One change each, all evaluated on the three axes (one generation seed unless noted):

| | real | swt-v2 | v3: rough landscape part | v4: noise from differences | v5: all live curves fitted |
|---|---|---|---|---|---|
| A: speed from config (R²) | 0.73 | 0.37 | 0.49 | 0.49 | 0.39 |
| A: local-optima ratio | 0.46 | 0.75 | 0.68 | 0.65 | 0.66 |
| A: landscape_rho at 200 | 0.77 | 0.72 | 0.72 | 0.72 | 0.68 |
| A: near-best share | 0.14 | 0.11 | 0.11 | 0.11 | 0.07 |
| B: classifier AUC | – | 0.876 | 0.886 | 0.879 | 0.876 |
| B: gap share of rising steps | – | 0.25 | 0.29 | 0.13 | 0.03 |
| C: gap spikes / late-vs-early sd | – | 0.29 / 0.08 | 0.25 / 0.08 | 0.48 / 0.13 | 0.33 / 0.10 |
| E: coverage | – | 0.088 | 0.081 | 0.083 | 0.067 |
| E: novelty | – | 1.04 | 1.00 | 1.01 | 1.06 |
| E: HPO mae / HB gain / FT gain (real 0.155 / 0.112) | – | 0.032 / 0.133 / 0.087 | 0.042 / 0.121 / 0.070 | 0.041 / 0.123 / 0.069 | **0.024 / 0.145 / 0.089** |

- **v3**: the unpredictable part of the landscape is a rough but spatially coherent GP (length-scales × 0.15)
  instead of white noise, calibrated with its own (d, weight) table. Ruggedness gap 1.29 → 0.92, speed-from-config
  0.37 → 0.49. Limitation: with one hp the rough field stays learnable (expected landscape_rho ≈ 0.98 at any weight).
- **v4**: noise level and its time profile from robust step-to-step differences of the real curves instead of
  template-fit residuals; the residual-based sd was 1.42× larger (it includes template misfit). Rising-step gap
  0.29 → 0.13; but the late-vs-early noise ratio became too small (0.09 vs 0.15) and the spike gap grew.
- **HPO gains vs generation seed.** The FT gain had seemed to drift down from v1 to v4 (0.106 → 0.069). With two
  more generation seeds: v1 0.081 ± 0.004, v2 0.082 ± 0.006, v4 0.071 ± 0.014 (HB 0.119-0.129). So the "drift" was
  seed noise. All swt versions underestimated multi-fidelity gains by 20-35%; the single-seed comparisons of HPO
  gains between versions are not meaningful below about 0.03.
- **Why the gains were too low**: generated tasks kept the order of configs less consistently (rank_early30_live
  0.908 vs 0.949 real; top-10% kept 0.56 vs 0.72). A likely cause: warps were drawn from at most 200 fitted curves
  for 1000 configs, so each real warp vector was used about 5 times. Near-duplicate curves differ only in noise and
  cross at random.
- **v5**: records fitted on all live curves (median 545 warp vectors per task). HPO behaviour became the closest
  to real of all swt versions (mae 0.024; HB gain 0.145 vs 0.155). Rising steps match (gap 0.03). But the landscape
  got *less* predictable than calibrated (0.68) and the near-best share fell to 0.07. Many duplicates of the best
  warp vector had inflated the near-best share before; with distinct vectors, the 2-D nearest-vector matching
  trades final-value fidelity for speed fidelity, which adds unplanned noise to the final-value landscape.
- **v6** (testing the explanation above): the final-value coordinate weighted 4× in the matching. Nothing moved
  (landscape_rho 0.680, near-best 0.066, speed from config 0.38, coverage 0.069, HPO mae 0.025, gains 0.146 /
  0.086). So the explanation was wrong as stated.
- **Diagnostic of the matching step** (30 generated tasks, noise-free curves): realised final values have rank
  correlation only 0.73 (v4) / 0.78 (v6) with the best-fitting mix of the target fields. Two causes: (i) the warp
  fits put beta on a 41-point grid and gamma on a 9-point grid, so the "speed" coordinate is heavily tied and the
  matching is coarse; (ii) a finite set of real vectors cannot hit arbitrary (final, speed) targets. v4 used only
  17% distinct vectors per task, v6 33%. Changing the weights inside the match cannot fix either.
- **v7**: warps constructed instead of matched. Final value and beta are the target fields' quantiles of the
  task's real marginals (beta jittered within its grid cell), amplitude and gamma come from the nearest real vector,
  and alpha is solved so the noise-free curve ends exactly at the target final value. Every config now has its
  own vector, and realised finals follow the target better (0.84 vs 0.78). Per-axis result vs v6: landscape_rho
  0.68 → 0.69, near-best 0.066 → 0.082, optima ratio 0.65 → 0.63, precision 0.63 → 0.68, final-value gap
  0.25 → 0.20. Speed from config did **not** move (0.38 → 0.39), so the quantised beta was not what limited it.
  HB gain 0.104 / FT 0.083 (one seed, SE 0.01). Small gains overall; the landscape gap to real (0.77) is mostly
  elsewhere.
