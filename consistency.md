# Consistency: GPU prior vs generate()

Both are bpl-v16. Statistics of the paper on the select split (189 held-back real tasks, medians over
tasks; HPO rows: mean over three generation seeds). The reference is real data; the two right columns
should agree with each other.

| | statistic | **real (reference)** | generate() (bpl-v16) | BplTasks (bpl-v17) | better |
|---|---|---|---|---|---|
| Curves | time to 50% of a curve's progress (log budget fraction) | -2.708 | -2.809 | -2.909 | closer to real |
| Curves | progress made in the last 20% of the budget | 0.029 | 0.015 | 0.014 | closer to real |
| Curves | share of steps where the smoothed curve rises | 0.260 | 0.280 | 0.280 | closer to real |
| Curves | share of curves that rise after their minimum (overfitting / instability) | 0.039 | 0.014 | 0.013 | closer to real |
| Curves | final value (normalised) | 0.479 | 0.555 | 0.497 | closer to real |
| Curves | classifier telling real from generated curves (AUC; 0.5 = cannot) | – | 0.883 | 0.887 | lower |
| Curves | generated curves that look real (precision) | – | 0.712 | 0.717 | higher |
| Curves | real curves the faker can produce (recall) | – | 0.793 | 0.796 | higher |
| Noise | excess kurtosis of residuals (heavy tails) | 23.3 | 25.6 | 25.7 | closer to real |
| Noise | residual spread, last third / first third | 0.152 | 0.045 | 0.038 | closer to real |
| Landscape | final value predictable from config (rank corr., 200 configs) | 0.771 | 0.696 | 0.705 | closer to real |
| Landscape | share of configs within 0.05 of the best | 0.136 | 0.090 | 0.109 | closer to real |
| Landscape | ruggedness: local optima vs random (1 = random) | 0.461 | 0.604 | 0.558 | closer to real |
| Landscape | curve speed predictable from config (R²) | 0.730 | 0.511 | 0.581 | closer to real |
| HPO | Hyperband gain over random search | 0.155 | 0.124 | 0.130 | closer to real |
| HPO | freeze-thaw gain over random search | 0.112 | 0.089 | 0.092 | closer to real |
| HPO | mean abs. difference of regrets from real | 0.000 | 0.019 | 0.018 | lower |

Differences: the GPU prior's landscapes are somewhat smoother (fewer local optima, larger near-best
region, more predictable speed) and its median final value is lower. Likely causes: random Fourier
features approximate the rough landscape part with 256 features, and class shares are thresholds set on
512 reference configs instead of exact counts per task. HPO behaviour agrees within seed variation.
