| | statistic | **real (reference)** | ifbo | swt-v11 | bpl-v1 | bpl-v10 | bpl-v13 | bpl-v15 | better |
|---|---|---|---|---|---|---|---|---|---|
| Curves | time to 50% of a curve's progress (log budget fraction) | -2.708 | -1.806 | **-2.508** | -2.909 | -2.909 | -2.909 | -2.909 | closer to real |
| Curves | progress made in the last 20% of the budget | 0.029 | 0.092 | **0.027** | 0.014 | 0.016 | 0.014 | 0.014 | closer to real |
| Curves | share of steps where the smoothed curve rises | 0.260 | 0.340 | **0.266** | 0.320 | 0.240 | 0.300 | 0.286 | closer to real |
| Curves | share of curves that rise after their minimum (overfitting / instability) | 0.039 | 0.124 | **0.048** | 0.015 | 0.010 | 0.014 | 0.014 | closer to real |
| Curves | final value (normalised) | 0.479 | 0.683 | 0.645 | 0.706 | 0.612 | 0.655 | **0.603** | closer to real |
| Curves | classifier telling real from generated curves (AUC; 0.5 = cannot) | – | 0.911 | 0.890 | 0.882 | 0.887 | 0.888 | **0.877** | lower |
| Curves | generated curves that look real (precision) | – | 0.613 | 0.642 | **0.728** | 0.688 | 0.720 | 0.715 | higher |
| Curves | real curves the faker can produce (recall) | – | 0.721 | 0.760 | 0.802 | 0.811 | 0.801 | **0.816** | higher |
| Noise | excess kurtosis of residuals (heavy tails) | 23.3 | 0.140 | **22.3** | 25.7 | 25.7 | 25.7 | 25.7 | closer to real |
| Noise | residual spread, last third / first third | 0.152 | 0.810 | **0.111** | 0.037 | 0.042 | 0.037 | 0.041 | closer to real |
| Landscape | final value predictable from config (rank corr., 200 configs) | 0.771 | **0.734** | 0.707 | 0.675 | 0.719 | 0.677 | 0.667 | closer to real |
| Landscape | share of configs within 0.05 of the best | 0.136 | 0.013 | 0.082 | 0.077 | **0.093** | 0.071 | 0.083 | closer to real |
| Landscape | ruggedness: local optima vs random (1 = random) | 0.461 | **0.526** | 0.615 | 0.570 | 0.573 | 0.582 | 0.600 | closer to real |
| Landscape | curve speed predictable from config (R²) | 0.730 | 0.218 | 0.490 | 0.484 | 0.339 | 0.490 | **0.499** | closer to real |
| HPO | Hyperband gain over random search | 0.155 | 0.065 | 0.110 | 0.114 | 0.111 | **0.117** | 0.109 | closer to real |
| HPO | freeze-thaw gain over random search | 0.112 | 0.078 | 0.083 | **0.104** | 0.090 | 0.104 | 0.089 | closer to real |
| HPO | mean abs. difference of regrets from real | 0.000 | 0.088 | 0.031 | 0.018 | 0.021 | **0.017** | 0.020 | lower |
