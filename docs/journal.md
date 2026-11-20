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
