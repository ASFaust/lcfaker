# Avenues: brainstorm (2026-11-20)

Step 1 of the README plan. Input: the README's A / B / C lists, rogue-dagger's measurements (journal 2026-11-15/16,
`docs/future-ideas/prior-sim2real.md`). Output: a set of avenues, each with its premise, cost, and the measure that
would show it working, then a ranked plan. Nothing here has been run yet.

## 0. What the evidence so far says about the target

Three facts constrain every avenue below:

1. **Curve realism and downstream value came apart once already.** The noise fix moved 3 of 10 statistics to real
   values and made the policy worse on all three benchmarks. Possible reasons (none tested): large white noise
   regularises the scorer; the calibration target (TaskSet dev noise 0.017) is lower than on the test families
   (0.029-0.034); the remaining gaps interact with noise.
2. **The scorer's judgment already transfers** (pair accuracy real ≈ twin), except comparisons of *unseen configs on
   TaskSet*, where hyperparameters barely predict the final loss (config smoothness 0.01 real vs 0.15 prior). The
   biggest decision-relevant gap may be on the config → performance map (axis A), not on curve shape (axis B).
3. **The largest statistical gaps are task-level, not curve-level**: dead/flat configs (KS 0.94 on dev), final
   spread (0.83), the near-best cluster (PD1 0.87). Noise and shape gaps are smaller or similar.

Two readings follow, and they suggest different fakers:

- **Calibrate** (system identification): fit the faker so its distribution matches the real one. The noise fix was
  this. Risk: it narrows the distribution to the dev data, and the dev data (TaskSet dev: many quadratic, flow and
  AE tasks, half of all curves flat) isn't the test distribution.
- **Randomise** (domain randomisation, the usual winner in robotics sim-to-real): keep the faker wide, and add the
  missing phenomena as extra *mixture components* with sampled strengths, so that real tasks fall inside the
  faker's support. The noise result fits this reading: shrinking noise removed support; it didn't add any. M4
  (coverage) is the measure that matches this reading.

This choice is the first thing to settle experimentally (plan item 2), because it changes how every later avenue
is built (replace a component vs add one).

## 1. A new axis: what is randomised, and how wide

The README's A / B / C axes say *what* the faker generates. A fourth, cross-cutting axis says *how wide*:

- **D1 point-calibrated**: one setting per knob, fitted to dev data.
- **D2 hierarchically randomised**: per task, each knob is drawn from a hyperprior wide enough to contain both the
  ifBO value and the dev-data value (e.g. noise sd log-uniform over [0.25×, 1×] ifBO, smoothing over [0, 2]).
- **D3 mixture of fakers**: each task comes from one of several fakers (ifBO, a structured one, resampled real) with
  mixing weights. Mixture weights become the main tuning knob, and they are cheap to vary.

## 2. Notes on the README's axes

### A. Config → curve map

- The decision-relevant quantities are: **how much of the final-loss variance the config explains** (TaskSet: very
  little), **how concentrated that variance is in few dimensions** (fANOVA), and **cliffs** (dead regions). A
  random deep BNN gives one fixed regime of all three. A2/A3 make all three explicit knobs; A1 (ARD-GP) gives the
  first two but no cliffs.
- An **explicit config-independent term** is missing in ifBO: final loss = f(config) + per-config "luck" (seed,
  init) with a sampled variance ratio. On TaskSet that ratio is high. This is one scalar per task and is the
  cheapest way to reproduce config smoothness 0.01 vs 0.15. It may also be what the TaskSet unseen-config gap is.
- Dead configs should be a *region* (high LR), not random per config, since a method can learn to avoid a region.
  Real check before building: on dev data, are flat curves clustered in config space (e.g. a logistic regression of
  "flat" on the config; AUC near 0.5 means per-config luck, high means a region)?
- Categorical hyperparameters: keep out of scope until a numeric faker beats ifBO, as the README says; only the
  measurement (optimizer families on the same TaskSet task) is cheap enough to do early.

### B. Curve shape

- **B1** additions, in the order of measured gap: (a) flat/dead curves, (b) fewer overfitting and diverging shapes
  (these come from ifBO's basis shapes, not its noise), (c) autocorrelated noise *added to*, not replacing, white
  noise, (d) schedule-shaped endings (PD1 has 0% overfitting, consistent with decaying LR schedules ending at the
  best), (e) plateau-then-drop. Each is one ablation.
- **B2** (GP in log-time) is a natural home for (c) and (e) but loses ifBO's tie to its PFN; parking it.
- **B3** hybrid (shape model on `z`, avenue-A map to `z`) is the only way a learned model avoids the
  conditioning problem. Prerequisite: a decent unconditional curve model, which learning-faker had.
- **B4** resampling: cheapest real-looking curves, but the decision structure (how config relates to curve) comes
  from whatever map assigns curves to configs, so B4 is only as good as its A-part. Its value is as a *test*:
  if B4 + ifBO's map beats ifBO, shape matters; if not, it doesn't.
- **B5** fake by doing: see section 4.

### C. Smaller axes

- **Shared evaluation noise** (one validation set per task, so all configs' curves share a per-step offset) is
  probably harmless for rankings within a step but makes curves look more correlated; low priority.
- **Time axis**: the PD1 lesson from twins (the inversion was spec mismatch, not shape) says curve length and
  budget dominate shape. The faker must sample T and steps-per-epoch as widely as the benchmarks span, including
  subsampled (jump) curves.
- **Metric type**: bounded error rates that saturate at 0 vs unbounded losses; matters for normalisation, and for
  near-best clusters (many configs reach ~0 error on easy tasks).

## 3. Avenues

Cost: S = days, mostly reusing rogue-dagger code; M = 1-2 weeks; L = weeks plus compute. M5 always means 3 seeds of
`wide-focus03` on one GPU (memory: one GPU at a time for this project), evaluated on held-out *allowed* tasks.

| # | Avenue (A × B × C × D) | Premise | Cost | Shows it working |
|---|---|---|---|---|
| V0 | ifBO, and ifBO + noise fix | Reference pair with a known M5 ordering | done | Meta-check of M1-M4 |
| V1 | ifBO × ifBO × noise level and smoothing randomised (D2) | Breadth, not calibration, is what helps; tests the reading in section 0 directly | S | M5 ≥ ifBO while noise-fixed < ifBO; M4 coverage up |
| V2 | ifBO BNN + per-config luck term + dead region + near-best basin (A3) × ifBO curves × D2 | The task-level gaps (largest KS, TaskSet unseen-config gap) matter most | S-M | Config smoothness, flat fraction, near-best fraction in range; M2 unseen-config comparisons on TaskSet dev; M5 |
| V3 | ifBO map × B1 shape fixes, one at a time × D2 | Shape gaps (overfit/diverge excess, schedule ends) matter | M | M1 per statistic; M5 per addition |
| V4 | A2 sparse fANOVA landscape × ifBO curves | A random BNN gives the wrong importance structure | M | fANOVA importance distribution vs dev; M2 method ranking |
| V5 | ifBO map × B4 resampled dev curves | Isolates the value of real shapes | S-M | M5 vs ifBO; overfitting risk visible on held-out dev families |
| V6 | A-map → `z` × B3 learned shape model | Learned shapes without the conditioning problem | L | M3 accuracy toward 0.5; M5 |
| V7 | B5 trained small nets (as data, and as a faker) | Every real phenomenon for free | L | Is itself a held-out-able eval set; M5 when used for training |
| V8 | Mixture of the best of the above (D3) | No single faker covers everything | S once parts exist | M4 coverage; M5 |

## 4. B5 (fake by doing), worked out a little

It is the only source with true ground truth beyond the dev data, so it is worth a costing even if it comes later.

- **Throughput on one GPU**: train many configs of one small model in parallel with `torch.func.vmap` (or grouped
  batched matmuls): e.g. 256 configs of a 2-3 layer MLP on a 10k-row tabular set, 50 epochs, likely a few minutes
  per task. 1000 tasks × 256 configs ≈ a few GPU days. Small CNNs/RNNs are 10-50× more.
- **Leakage**: B5 must avoid the held-out families *by design*, not only by data. No LCBench OpenML datasets (the
  35 in the README), no PD1 dataset × model pairs (e.g. no wide-ResNet on CIFAR/SVHN, no simple CNN on
  (Fashion-)MNIST), no text RNN classification on IMDB. Using CIFAR-10 with a small MLP is a grey zone; deciding
  this up front belongs in the held-out list (plan item 1).
- **Uses**: (1) dev data with known configs and categorical effects; (2) a second, independent fit/select split
  next to TaskSet dev; (3) a faker on its own (V7) or a component of V8.

## 5. Ranked plan

1. **Infrastructure first (README steps 2-3)**: held-out list + guarded loaders; internal split of allowed data
   by family (proposal: TaskSet dev families {char, word, rnn, losg} for selection, the rest for fitting; then add
   NAS-Bench-201 / FCNet as a third, structurally different family); M1 and M4 on dev vs ifBO; the meta-check on
   V0. M2 needs HPO methods wired to tabular tasks; rogue-dagger's eval already does random search and the
   policies, so start with those plus ASHA.
2. **V1 (breadth vs calibration)**. Cheapest experiment that discriminates the two readings of the noise result,
   and its answer decides how every later avenue is built. Also a second data point for validating the measures.
3. **V2 (task-level structure)**, one component at a time: luck term, then dead region (after the clustering check
   in section 2A), then near-best basin.
4. **V5 (resampled real curves under ifBO's map)** as a cheap diagnostic of whether shape matters at all.
   If it doesn't, V3 and V6 drop in priority.
5. Then V3 or V4 depending on 3-4; B5 data generation can start in the background earlier if the leak rules are
   settled, since it only needs one GPU and no decisions.

## Open questions for the user

- Do you agree with treating breadth vs calibration (V1) as the first faker experiment, before the task-level
  structure (V2)?
- B5 leak rules: are small models on CIFAR-10 / MNIST acceptable if the model class differs from PD1's?
- Internal split: is splitting TaskSet dev by family enough, or should a non-TaskSet family (NAS-Bench-201,
  FCNet) be required for selection from the start?
