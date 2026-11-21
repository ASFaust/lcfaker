# lcfaker: a data-driven prior of learning curves for freeze-thaw HPO

> **Draft release (bpl-v16).** The generator and the numbers below are final for this version; the paper is a
> draft. Known gaps are listed at the end.

`lcfaker` generates synthetic hyperparameter-optimization (HPO) tasks: a search space, up to 1000 configurations,
and a full validation learning curve for every configuration. It is meant as training data ("prior") for
learning-curve-based HPO methods such as FT-PFN / ifBO, and as a cheap testbed for multi-fidelity HPO algorithms.

The generator, **bpl** (broken power laws), is built from about 570 real HPO tasks (TaskSet, JAHS-Bench-201,
QuickTune, NAS-Bench-201). The paper is in [`paper/main.pdf`](paper/main.pdf).

![real and generated tasks](paper/figures/examples.png)

## Install and use

```bash
pip install -e .            # numpy + scipy only
```

```python
from lcfaker import generate

tasks = generate(n_tasks=100, seed=0)
t = tasks[0]
t.y              # [n_configs, T] float32: validation curves in raw units, lower is better
t.metric         # "loss" (unbounded) or "error" (bounded error rate)
t.X              # [n_configs, d] configurations in [0, 1]^d, d <= 10
t.info["log_lr"] # X[:, 0] as natural-log learning rate; TaskSet-like tasks: schedule hyperparameters in X[:, 1:]
t.info           # also: dead / diverged / rising configuration indices, schedule family, record id
t.normalised()   # curves on a common scale: 0 ~ the task's best, 1 ~ a typical start, 2 = diverged
```

`generate(n, seed, source_weights={"taskset": 1, "jahs": 1, "quicktune": 1, "nasbench201": 1})` rebalances the
record families. The default follows the number of real tasks per family (96% TaskSet); that is the setting
evaluated in the paper. `python examples/plot_tasks.py` plots a few tasks.

**Units.** Curves are in the raw units of the record's benchmark: losses (TaskSet; may be negative for density
models) or error rates (% for JAHS-Bench-201 / QuickTune records, a fraction for NAS-Bench-201). TaskSet-like tasks
include the value at step 0, before training, as their first point. Curves have T = 50-200 points.

## How a task is generated

Each curve is a broken power law on the scale log(loss − floor), in time measured as optimiser progress:

    y = L + d + A / (1 + (x / tau)^a),    x = (lr / lr_ref)^kappa * S1(t) / T_total

`S1` is the cumulative learning-rate multiplier of the configuration's schedule, so decaying schedules slow
progress late in training, and the learning rate acts as a partial time warp (kappa ≈ 0.15).

1. A **record** is drawn: the fitted description of one real task (floor, start level, the joint distribution of
   the four curve parameters and their relation to the learning rate, schedule hyperparameters, shares of dead,
   diverged and rising curves, noise).
2. **Configurations** are uniform in [0, 1]^d. Dimension 0 is the log learning rate; for TaskSet records the next
   dimensions are the schedule hyperparameters (linear / exponential decay; warm-up, constant phase, cosine floor).
3. Smooth random **Gaussian-process fields** give each configuration its curve parameters through a Gaussian
   copula with the record's real marginals and correlations, shifted by the real learning-rate relation.
4. **Rising curves** (overfitting / instability), **dead** configurations (low learning rates), **diverged**
   ones (high learning rates) and heavy-tailed noise whose size scales with the excess loss are added.

No real curve is replayed. What comes from real tasks are distributions (`lcfaker/assets/bpl_v16.npz`, 5 MB).

## How close is it to real data?

Measured on 189 real tasks from task families that were not used to build the generator (medians over tasks; HPO
rows: mean over three generation seeds). The ifBO prior is the prior FT-PFN was trained on.

| | **real (reference)** | ifBO prior | bpl | better |
|---|---|---|---|---|
| HPO: mean abs. difference of regrets from real | 0 | 0.086 | **0.019** | lower |
| HPO: Hyperband gain over random search | 0.155 | 0.071 | **0.124** | closer to real |
| HPO: freeze-thaw gain over random search | 0.112 | 0.083 | **0.089** | closer to real |
| curves: generated curves that look real (precision) | – | 0.61 | **0.71** | higher |
| curves: real curves the generator can produce (recall) | – | 0.72 | **0.79** | higher |
| noise: excess kurtosis of residuals (heavy tails) | 23 | 0.1 | **26** | closer to real |
| landscape: final value predictable from config (rank corr.) | 0.77 | **0.73** | 0.70 | closer to real |
| landscape: share of configs within 0.05 of the best | 0.14 | 0.01 | **0.09** | closer to real |
| landscape: curve speed predictable from config (R²) | 0.73 | 0.22 | **0.51** | closer to real |

**Known gaps:** curves level off too early (time to 90% of progress −1.5 vs −1.1 log budget fraction); late noise
is too small relative to early noise; rising curves are too rare (1.4% vs 3.9%), and overfitting and instability
share one rise term; landscapes are less predictable from the configuration than real ones; one seed per
configuration; NAS-Bench-201-like tasks clip diverged runs at 100% error.

## Provenance

The fitted records come from public benchmark data: TaskSet (Metz et al., 2020), JAHS-Bench-201 (Bansal et al.,
2022), QuickTune (Pineda Arango et al., 2024) and NAS-Bench-201 (Dong & Yang, 2020). LCBench, PD1 and TaskSet's
FixedTextRNNClassification tasks, the test benchmarks of FT-PFN, were not used in any way.

## Citation

See `paper/main.tex` (draft).
