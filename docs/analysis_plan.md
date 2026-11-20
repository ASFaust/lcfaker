# Analysis plan: what kind of learning-curve data are we modelling?

Written 2026-11-20, after the first session (see `docs/journal.md`). That session compared fakers with three
summary measures. The best one, hgp-v5, is **not a curve model**. It re-plays real curve sets from fit tasks and
only invents the mapping from configs to curves, so it is a study of the hyperparameter axis, not of curve
generation. Before building more generators, this plan asks what the data look like, which metrics fit which
modelling goal, and what the known theory and the data together say about how to parametrise a generative model.

The plan ends with the question it is meant to answer: **which insights (prior knowledge + our analysis) motivate
which kind of model?** (section 6, filled in as results come in).

---

## 1. Axes of the problem and the metrics that matter for each

A faker has to get several things right, and each needs its own metrics. One summary number (coverage) mixes them
and hides which part is wrong. Below, "real" always means held-out data (the *select* split, or held-out tasks /
configs / seeds within *fit* for diagnostics).

### Axis B: curve generation (what a single curve looks like, and a task's set of curves)

Per-curve fidelity (does the model represent and predict real curves?):
- **Rate-distortion**: held-out reconstruction error (RMSE in z) as a function of the number of parameters /
  latent dimensions. Which families reach the noise floor with how many parameters.
- **Held-out log likelihood per point** (for probabilistic curve models), per source, split into the mean-curve
  part and the noise part where the model allows it.
- **Extrapolation from prefixes** (10 / 20 / 50% of the budget): error of the predicted final value, *and
  calibration* of the predictive distribution (coverage of 50 / 80 / 95% intervals; CRPS). A generator that is right
  on average but over- or under-confident is a bad prior for freeze-thaw.

Distribution of curve features (do generated curves have the right properties, not just the right average?), each
compared per source, as distributions, with a 1-D Wasserstein distance and a two-sample test:
- speed: t10 / t50 / t90 (budget fraction at 10 / 50 / 90% of total improvement), in linear and log time;
- **knees**: number and position of slope changes in log-time (see 2.2), length of an initial plateau;
- late drop: share of the total improvement in the last 20% of the budget;
- monotonicity: share of steps that go up, size of the largest rise (overfitting, instability);
- spikes: rate and size of single-step excursions;
- level: start, final, best, final − best;
- class membership: improving / dead / diverged / turning, and the shape clusters of section 3.

Distribution of whole curves:
- **precision and recall** in a curve embedding (k-nearest-neighbour precision / recall at several k, as for image
  generators): are generated curves realistic (precision), and is every kind of real curve generated (recall)?
  Per source and per shape cluster. This measures mode coverage directly.
- **classifier two-sample test (C2ST)**: AUC of a classifier trained to tell real from generated curves. 0.5 =
  indistinguishable. Its most useful features show *what* gives the generator away.

Within a task (a task's set of curves, which is what an optimiser sees):
- distribution of pairwise distances between curves of one task (diversity);
- **crossing statistics**: share of curve pairs that cross after x% of the budget; rank stability
  (Kendall tau between the order at budget b and the final order, for several b); top-k retention.

### Axis A: hyperparameter influence (the landscape: config → curve)

- **Predictability vs sample size**: rank correlation of a fixed model (boosted trees and a GP) trained on 25 / 50 /
  100 / 200 / 400 configs. A curve, not a single point at n = 200.
- **Effective dimensionality**: fANOVA variance shares of first-order effects; number of hps that cover 80% of
  the variance; share of interactions.
- **Which hps matter, and how**: per hp, the main-effect shape (monotone? single optimum?). Learning rate in
  particular: position and width of its good region.
- **Smoothness**: GP length-scales per numeric hp; local roughness (rank correlation of final values between
  nearest-neighbour configs).
- **Modality of the landscape**: number and size of basins (local optima on a k-nearest-neighbour graph of
  configs), size of the near-best region (share of configs within ε of the best), shape of the distribution of
  final values (quantile function, number of modes).
- **Structure of the bad regions**: predictability of dead / diverged from the config (AUC), contiguity (share of
  their neighbours in the same class), and where they lie (e.g. dead at low learning rate, diverged at high).
- **Does the config shape the curve or only its level?** Predictability of shape features (t50, knee position,
  late drop, turning) from the config *given* the final value (partial correlation / conditional models).
- **Optimiser behaviour that depends on configs**: model-based methods (BO with a GP or random forest, BOHB-like).
  The current HPO measure uses only model-free methods, which ignore the configs and so cannot test this axis.

### Axis C: noise

- noise size vs time, vs loss level, vs learning rate (and schedule);
- shape of the noise distribution: tail index / kurtosis, spike rate and size, skew (spikes are mostly upward?);
- serial structure: autocorrelation of residuals at several trend flexibilities (the GP result showed a trend can
  absorb slow wandering, so state what is called "noise");
- seed effects: across-seed variance of the whole curve vs within-curve noise, and how much of the final-value
  variance is seed luck;
- shared components: across configs at the same step, and whether spikes co-occur across configs.
- Metrics for a noise model: held-out likelihood of residuals, plus the distributions above (generated vs real).

### Axis D: task level (how tasks differ from each other)

- **within-task vs between-task variance** of every curve feature (variance decomposition / intra-class
  correlation): which properties are task-level (shared by all configs of a task) and which config-level;
- dependence between task-level properties (copula / correlation): e.g. do tasks with many dead configs also have
  fast curves?
- number of task types: clustering of tasks on their descriptors, with stability checks;
- source effects: how much of the between-task variance is explained by the source (benchmark) vs within-source.

### Axis E: the whole faker (only after A-D)

- coverage / precision / recall of real tasks in descriptor space, at several k, not a single threshold;
- C2ST on whole tasks;
- novelty (distance to the nearest fit task) together with coverage;
- HPO behaviour: model-free *and* model-based methods; the existing FT-PFN / rogue-dagger checkpoints as additional
  (biased) evaluators; standard errors from enough tasks and seeds (the first session found the gain estimates have
  an SE of about 0.01 with 190 tasks).

---

## 2. What is already known about training curves

*From memory; citations to be checked when they are relied on.* Note that much of the "learning curve" literature
is about performance vs *training-set size* (e.g. Viering & Loog's survey, TPAMI 2022; the LCDB of Mohr et al.).
That is a different object from performance vs *training time*, which is what we model, although some shapes carry
over.

### 2.1 Parametric families used in HPO

- Domhan et al. (IJCAI 2015) fit an ensemble of 11 saturating families to epoch curves for early stopping
  (pow3, pow4, log power, exp3, exp4, Janoschek, Weibull, MMF, ilog2, vapor pressure, log-log-linear).
- Freeze-thaw BO (Swersky et al. 2014) uses an exponential-decay kernel: curves as sums of decaying exponentials.
- LC-BNN (Klein et al., ICLR 2017) and LC-PFN (Adriaensen et al., NeurIPS 2023) use subsets of these families.
- ifBO (Rakotoarison et al., ICML 2024) uses pow4, exp, log, hill with a "break point" after which the curve can
  change slope.
- All of these are **monotone saturating** shapes in their basic form. Our atlas found shapes they cannot make:
  late drops under cosine schedules and early plateaus.

### 2.2 Shape in log-time; plateaus and knees

- A saturating power law L(t) = L∞ + A·t^(−α) is a straight line in log(L − L∞) vs log t. Many curves look roughly
  **sigmoidal in log-time**: an initial plateau (before training "gets going"), a descent that is roughly linear in
  log t (the power-law phase), and saturation at a floor. The "knee" is a transition between these phases; there can
  be more than one.
- Stage-like learning: deep linear networks learn in sigmoidal steps with plateaus in between (Saxe et al. 2014).
  Plateaus and sudden drops also appear in practice (e.g. grokking, Power et al. 2022: late, sudden improvement of
  validation performance; rare, mostly on algorithmic tasks).

### 2.3 Learning-rate schedules determine the time axis

- With a constant learning rate, SGD-like training approaches a **noise floor** whose height grows with the
  learning rate. Lowering the rate lowers the floor, which is why the loss drops when the rate is decayed.
- Scaling-law work makes this quantitative. Tissue et al. (2024, "scaling law with learning-rate annealing") model
  the loss as a function of the *cumulative* learning rate ("forward area") plus a bonus for the annealed part,
  roughly L = L0 + A·S1^(−α) − C·S2. That is, the natural clock is not the step count but the integrated learning
  rate, and annealing adds a drop.
- Consequence for us: the late drops of JAHS / NAS-Bench-201 (cosine schedules) and the noise that shrinks during
  training (our noise result, strongest with cosine schedules) are both what this view predicts. **A hyperparameter
  (learning rate, schedule) can enter the curve as a warp of the time axis**, which is a parametrisation hint.

### 2.4 Non-monotone curves

- Overfitting: validation *loss* rises late while validation *error* often stays flat. So the metric type matters:
  our sources mix losses (TaskSet, bbomix, FCNet) and errors (NAS-Bench-201, JAHS, QuickTune).
- Epoch-wise double descent (Nakkiran et al. 2019): validation error falls, rises, falls again, mostly with label
  noise and large models.
- Instabilities: at too high learning rates, training enters the "edge of stability" regime (Cohen et al. 2021),
  shows loss spikes, or diverges. That suggests divergence and spikes are organised along the learning-rate axis
  of config space.

### 2.5 Seeds and noise

- Seed-to-seed variation of the final value is typically small compared to config-to-config variation (consistent
  with the earlier measurement: 0.1-5%), but not negligible near the optimum, where configs are close.

---

## 3. Analysis questions

Each question names its method. Results go to `docs/journal.md`; answers feed section 6.

### 3.1 Curve shape (axis B)

- **S1. Which parametric families explain real curves?** Fit a family set per curve (single seed and seed-mean):
  the Domhan families, ifBO's basis, and shapes in log-time: a sigmoid in log t; a piecewise-linear fit in log t
  with 1-3 knees; a power law to a floor plus an annealing term in cumulative-LR time (where the schedule is known:
  TaskSet adam8p / nadamw, FCNet). Compare by held-out points (interleaved and last-20% hold-out) and BIC. Report
  the share of curves for which each family is best, per source and per class, and which curves *no* family fits.
- **S2. Is the space of curve shapes multimodal?** (The hunch to test.) On curves normalised for level and scale
  (to separate shape from level), several views:
  (a) clustering of the normalised curve (k-means / Gaussian mixtures on PCA scores with BIC; hierarchical
  clustering with dynamic time warping as a distance), checking cluster stability by bootstrap (adjusted Rand
  index);
  (b) density of a few interpretable features (t50 in log-time, knee count, late-drop share): dip test for
  multimodality, mixture fits;
  (c) a 2-D embedding (UMAP) for inspection only, coloured by source, class and hyperparameters.
  Multimodal shapes argue for discrete latents or mixture components in the generator. A continuum argues for a
  smooth latent.
- **S3. Does the shape depend on the time axis?** The same clustering in linear time, log time and (where known)
  cumulative-LR time. If clusters merge in one of them, that clock is the natural one.
- **S4. Knees**: number and location of slope changes in log-time (piecewise-linear fits with BIC); does knee
  position depend on the learning rate / warm-up?
- **S5. Within-task vs between-task variation of shape**: variance decomposition of the shape features and of
  the cluster labels (how many clusters per task? do all configs of a task share one shape and differ only in level
  / speed?).
- **S6. Losses vs errors**: do the shape statistics differ systematically between loss-type and error-type metrics
  (overfitting visible in losses only)?

### 3.2 Noise (axis C)

- **N1. Noise shape**: residuals from a flexible smoother. Distribution (tails, skew, spikes), dependence on time,
  level, learning rate.
- **N2. Spikes**: rate, size, direction, recovery profile (instant vs decaying), co-occurrence across configs and
  seeds.
- **N3. Seeds**: whole-curve seed variation decomposed into level shift, speed change and white noise; share of
  final-value variance from seeds, especially among the best configs.

### 3.3 Hyperparameters (axis A)

- **H1. Landscape modality**: basins on k-NN graphs; size of the near-best region; distribution of final values
  (is it multimodal, e.g. good / bad / dead clusters?).
- **H2. Learning-rate role**: per task, the main effect of the learning rate on final value, speed (t50), knee
  position, spikes, dead / diverged share. Is the learning rate mostly a time-warp (faster / slower along the same
  curve), a level shift, or a shape change?
- **H3. Schedule hyperparameters** (TaskSet adam8p / nadamw, FCNet): effect on late drop, knee, noise decay. Test
  of the cumulative-learning-rate clock.
- **H4. Config → shape given level**: do configs with the same final value have different shapes depending on
  their hyperparameters? (Predict shape features from config and final value vs from final value alone.)
- **H5. Dead / diverged regions**: where in config space, how sharp the boundary, relation to the optimum (is the
  best region next to the divergence boundary, as "largest stable learning rate" heuristics suggest?).
- **H6. Effective dimensionality and interactions** as functions of the number of hps (fANOVA per optimizer
  family).

### 3.4 Joint and task-level (axis D)

- **T1. What is a task-level property?** Variance decomposition of all curve and landscape features into
  source / task / config / seed.
- **T2. Task types**: clustering of tasks on descriptors (with stability); dependence between task properties.
- **T3. Are curves and configs separable?** Compare (a) a model of curve | config with a separate landscape and
  (b) a joint latent model (a conditional VAE, curve | (task latent, config)), by held-out likelihood of curves of
  unseen configs within known tasks, and of unseen tasks. If (b) wins clearly, the generator should not separate
  parameters from curves.
- **T4. Sufficient statistics of a task**: which few task-level numbers (e.g. dead share, speed distribution,
  noise level, landscape predictability) are enough to reproduce the rest? Tested by predicting held-out
  descriptors from a small set.

---

## 4. Order of work

1. S1 + S2 + S3 (shape families, multimodality, time axis), because they decide the parametrisation of the curve
   model. Then S4-S6.
2. N1-N3 (noise), using the residuals from the S1 fits.
3. H2 + H3 (learning rate and schedule: time-warp or shape?), because they decide how configs enter the curve
   model. Then H1, H4-H6.
4. T1-T4 (what is task-level; are curves and configs separable).
5. Section 6: the modelling conclusions, then the metrics of section 1 implemented as an evaluation suite per axis.

## 5. Data notes for the analysis

- TaskSet curves have 51 points from step 0 (initialisation) to 10k steps. The other sources start after the first
  epoch, so "initial plateau" is only observable in TaskSet (and partially elsewhere).
- Schedules known per source: TaskSet adam1p / adam4p use a constant learning rate, adam8p linear / exponential
  decay (hps), nadamw warm-up + constant + decay (hps); FCNet `lr_schedule` (constant / cosine); NAS-Bench-201 and
  JAHS cosine; bbomix, QuickTune: per config (QuickTune `sched`).
- Metric type: losses (TaskSet, FCNet MSE, bbomix reconstruction) vs errors (NAS-Bench-201, JAHS, QuickTune).
- Held-out benchmarks stay untouched. All of this runs on *fit* and *select* data; nothing is tuned on *select*
  except where a comparison is explicitly a selection step.

## 6. Insights → modelling decisions (to be filled in by the analysis)

| finding | evidence | modelling consequence |
|---|---|---|
| Classic saturating families (power law, exponential) are 2-8× worse than the best; a two-stage S-shape in log-time reaches the noise floor on most sources (S1) | `results/shape_families.csv` | Parametrise curves in **log-time**; the basic unit is an S-shape (plateau, descent, floor), with room for a second stage / knee. Not ifBO's monotone saturating basis. |
| Shapes form a continuum, mostly one S-shape whose position (speed) and steepness vary; apparent modes come from the training window cutting the S (only tail / full / only start); JAHS has fast vs late learners; ~7% distinct minority patterns (S2) | `results/shape_template.csv`, figures | A **continuous** shape latent (position, steepness) is enough for most curves; the *distribution* of position needs to be multimodal (mixture) and must extend beyond the window; a small number of extra components for two-stage, worsen-then-improve, overfitting and unstable curves. |
| Within a task, curves ≈ one template shifted in log-time and scaled per config; 3-4 params/curve explain almost what 7 free do (S5) | `results/shape_invariant.csv` | **Hierarchical model: task-level template + per-config warps** (level, amplitude, log-time shift, stretch) + a small per-curve residual. The task latent is a curve shape; the config latent is a warp. |
| Learning rate = partial time warp + level change; higher LR → ahead in time (H2) | `results/lr_timewarp_*.csv` | Hyperparameters should enter (at least partly) **through the warp parameters**, not only through the final level. This argues against separating "landscape of final values" from "curve shape". |
| Cumulative learning rate as clock: consistent but weak in TaskSet; annealing bonus not visible (H3) | `results/schedule_clock_within.csv` | Where the schedule is known, use cumulative-LR time as an optional input to the warp; not a priority for TaskSet. |
| Noise: heavy-tailed everywhere (excess kurtosis 0.5-8), skewed upward (spikes = worse) on cosine / tabular sources; sd shrinks over training (late 0.4-0.8×) and grows with the loss level; no serial correlation; spikes per config, not shared (N1, N2, and the GP noise study) | `results/noise_shapes.csv`, `results/noise_models.csv` | Noise = **heavy-tailed, possibly skewed, white**, with sd = task level × f(time) × g(curve level); per-curve, no cross-config component. A Gaussian likelihood with one sd is wrong (it is what inflated the set-VAE's noise). |
| Seeds: on TaskSet a seed mostly shifts the level; elsewhere mostly white noise; seed luck is a small share of final-value variance overall but 24-37% among the best 10% (N3) | `results/noise_shapes.csv` | Seeds: per-curve random level offset + independent noise. Must be modelled: near the optimum it is as large as config differences. |
| Landscapes multimodal: smoother than random, but many local optima; best basin 2-16%; near-best region 3-46% (H1) | `results/landscape_structure.csv` | Landscape fields need **local structure (multiple basins)**, not one smooth bowl; the near-best share is a task-level difficulty parameter to sample. |
| Learning rate organises bad regions: dead at low LR, diverged at high LR, best right at the divergence boundary (TaskSet, JAHS); FCNet: collapse at high LR (H5) | `results/landscape_structure.csv` | Dead / diverged = **sharp regions along an LR-like axis** of config space, with the optimum next to the divergence boundary; not independent random regions (as in hgp-v0). |
| Most shape variation is between configs of a task (57-80%), sources ~25%, tasks within source 10-17% (T1) | `results/variance_levels.csv` | Task latent = template + noise + class shares; **config latent = warps**, which carry most of the shape variation. |
| Config predicts speed / steepness far beyond the final value (JAHS u50: 0.11 from final value vs 0.74 from config) (H4) | `results/config_shape.csv` | **Do not separate** "final-value landscape" from "curve shape": a joint config → (level, amplitude, shift, stretch, class) map, i.e. several correlated landscape fields driving the warps. |
| Not yet done: knees (S4), losses vs errors (S6), task types (T2), a fitted joint-vs-separated comparison (T3), sufficient task statistics (T4) | | |

### What this motivates (synthesis, 2026-11-20)

The analysis points to one model class, in which analytical structure and learning each have a clear place:

1. **Curves are warped templates in log-time.** Per task, one smooth template f in log-time, S-shaped with room
   for a second stage (two-stage S / knee); per config, a warp y(u) = α + a·f((u − β)·e^γ). This is a
   shape-invariant (SITAR-like) model. It explains almost as much within-task variance as 7 free parameters per curve,
   and it absorbs the apparent multimodality of shapes. Speed β is a continuous parameter; the training window then
   shows only the tail, the full S, or only the start. Its distribution must be wide and multimodal (fast vs late
   learners) and extend beyond the window.
2. **Configs drive the warps, jointly.** The config sets level, amplitude, speed, stretch and class (live / dead /
   diverged) together, through correlated landscape fields with several basins. A learning-rate-like direction moves
   speed and places the dead (low) and diverged (high) regions, with the optimum near the divergence boundary.
   Matching the landscape's predictability (faker v5) remains necessary.
3. **Task level** = template, noise parameters, class shares, near-best share, landscape predictability and
   dimensionality. These are sampled from a learned (or bootstrapped) distribution over real tasks, per source family.
4. **Noise** = heavy-tailed, optionally upward-skewed white noise whose size shrinks over training and grows with
   the loss level, plus a per-curve seed offset.
5. **Minority patterns** (two-stage plateaus, worsen-then-improve, overfitting, instability; about 7%) as explicit
   mixture components with small probabilities.

**Where learning fits** (the "VAE with the right parametrisation" question): the decoder should *be* this structure.
A task latent maps to a template (a low-dimensional family, e.g. two-stage S parameters or a few template
components). A config latent maps to the warps. The likelihood is heavy-tailed with time- and level-dependent noise.
Then a VAE or a flow only has to learn the *distributions* of task-level parameters and the joint config → warp maps,
both low-dimensional. That avoids the failures of the free-form generators (lost early-late order, inflated noise,
blurred classes): the structure keeps the order (one template per task, monotone warps), the likelihood separates
noise from misfit, and the classes are explicit.
