# lcfaker: a data-driven prior of learning curves for freeze-thaw HPO

> **Draft release (bpl-v17).** The generator and the numbers below are final for this version; the paper is a
> draft. Known gaps are listed at the end.

`lcfaker` generates synthetic hyperparameter-optimization (HPO) tasks: a search space with configurations and a
validation learning curve for every configuration. It is meant as training data ("prior") for learning-curve-based
HPO methods such as FT-PFN / ifBO, and as a cheap testbed for multi-fidelity HPO algorithms.

The generator, **bpl** (broken power laws), is built from about 570 real HPO tasks (TaskSet, JAHS-Bench-201,
QuickTune, NAS-Bench-201). The paper is in [`paper/results.pdf`](paper/results.pdf); [`paper/main.pdf`](paper/main.pdf) is the longer version with the full development history.

![real and generated tasks](paper/figures/examples.png)

## Install

```bash
pip install -e .[torch]     # numpy, scipy, torch
```

## Use: `BplTasks` (bpl-v17, batched on the GPU)

Many tasks at once, with configuration maps that are functions: curves can be queried for any configurations,
at any curve length.

```python
import torch
from lcfaker import BplTasks

P = BplTasks(n_tasks=256, d_max=10, device="cuda", compile=True)
P.resample(idx=torch.arange(256), d=torch.randint(1, 11, (256,)))   # new tasks for rows idx with hp dims d
x = torch.rand(256, 64, 10, device="cuda")                           # configs in [0, 1]^d (dims >= d ignored)
T = torch.randint(4, 301, (256,))                                    # length of a full training, per task
y = P.curves(x, T, 300)                                              # [256, 64, 300]; point k at budget k / T
```

- **Tasks**: `resample(idx, d, record=None)` draws new tasks for rows `idx` with hp dims `d` (1..`d_max`); the record
  (fitted description of one real task) is drawn by the source weights (`BplTasks(..., source_weights={...})`;
  default as in the paper, 96% TaskSet) unless given.
- **Same config, same curve**: a task's landscape fields are random-Fourier-feature functions (Matern-5/2 per
  dimension, random relevances, additive / product mix, rough companion). The same `x` in the same task gives the
  same noise-free curve, whatever other configs are evaluated with it (`noise=False` returns it). Observation
  noise is drawn on each call. Class shares (dead, diverged, rising) are thresholds set on 512 reference configs at
  resample time.
- **Any T**: time is optimiser progress (cumulative learning rate of the config's schedule, in closed form), so any
  curve length works; point k is at budget fraction k / T. **Past T** a curve stays at its value at T (the run has
  finished, as in the benchmarks).
- **Dimensions**: dim 0 is the log learning rate; for TaskSet records dims 1.. are the schedule hyperparameters
  (`P.dims(row)` names them); the rest are generic. Shuffle columns yourself if your model is order-dependent.
- **Units**: `scale="raw"` (default; loss or error rate, `P.metric()`) or `scale="unit"`: (y - floor) / (start -
  floor), clipped to [0, 2], with 2 = diverged. TaskSet-like tasks also have a step-0 value (`P.start_values(x)`).
  `P.configs_info(x)` returns the noise-free per-config quantities (curve parameters, learning rate, classes).
- **Speed** (RTX 6000 Ada, `compile=True`): resampling 50 tasks ≈ 4.5 ms; 256 tasks × 64 configs × 300 points ≈
  2 ms. Without `compile` ≈ 7 ms for the curves. The first compiled call takes 10-60 s.

## Reference implementation: `generate()` (bpl-v16, numpy)

```python
from lcfaker import generate
tasks = generate(n_tasks=100, seed=0)   # whole tasks with a fixed sample of configs, numpy + scipy only
t = tasks[0]
t.y, t.X, t.metric, t.info              # curves [n, T] in raw units, configs [n, d], "loss" / "error", classes
t.normalised()                          # 0 ~ the task's best, 1 ~ a typical start, 2 = diverged
```

bpl-v17 is bpl-v16 with function-based configuration maps (above). [`consistency.md`](consistency.md) compares the
two on all statistics of the paper: HPO behaviour agrees within seed variation; v17's landscapes are somewhat
smoother (closer to real data).

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
3. Smooth random **fields** over the configuration space (random-Fourier-feature Gaussian processes) give each
   configuration its curve parameters through a Gaussian copula with the record's real marginals and correlations,
   shifted by the real learning-rate relation.
4. **Rising curves** (overfitting / instability), **dead** configurations (low learning rates), **diverged**
   ones (high learning rates) and heavy-tailed noise whose size scales with the excess loss are added.

No real curve is replayed. What comes from real tasks are distributions (`lcfaker/assets/bpl_records.npz`, 5 MB).

## How close is it to real data?

Measured on 189 real tasks from task families that were not used to build the generator (medians over tasks; HPO
rows: mean over three generation seeds). The ifBO prior is the prior FT-PFN was trained on.

| | **real (reference)** | ifBO prior | bpl-v17 | better |
|---|---|---|---|---|
| HPO: mean abs. difference of regrets from real | 0 | 0.086 | **0.018** | lower |
| HPO: Hyperband gain over random search | 0.155 | 0.071 | **0.130** | closer to real |
| HPO: freeze-thaw gain over random search | 0.112 | 0.083 | **0.092** | closer to real |
| curves: generated curves that look real (precision) | – | 0.61 | **0.72** | higher |
| curves: real curves the generator can produce (recall) | – | 0.72 | **0.80** | higher |
| noise: excess kurtosis of residuals (heavy tails) | 23 | 0.1 | **26** | closer to real |
| landscape: final value predictable from config (rank corr.) | 0.77 | **0.73** | 0.71 | closer to real |
| landscape: share of configs within 0.05 of the best | 0.14 | 0.01 | **0.11** | closer to real |
| landscape: curve speed predictable from config (R²) | 0.73 | 0.22 | **0.58** | closer to real |

**Known gaps:** curves level off too early (time to 90% of progress −1.5 vs −1.1 log budget fraction); late noise
is too small relative to early noise; rising curves are too rare (1.3% vs 3.9%), and overfitting and instability
share one rise term; landscapes are less predictable from the configuration than real ones; one seed per
configuration; NAS-Bench-201-like tasks clip diverged runs at 100% error.

## Provenance

The fitted records come from public benchmark data: TaskSet (Metz et al., 2020), JAHS-Bench-201 (Bansal et al.,
2022), QuickTune (Pineda Arango et al., 2024) and NAS-Bench-201 (Dong & Yang, 2020). LCBench, PD1 and TaskSet's
FixedTextRNNClassification tasks, the test benchmarks of FT-PFN, were not used in any way.

## Citation

See `paper/main.tex` (draft).
