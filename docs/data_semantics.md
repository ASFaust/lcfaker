# What the curves are (audit 2026-11-20, scripts/audit_metrics.py, results/audit_metrics.csv)

| source | split / metric | bounded? | values ≤ 0 | time axis | schedule information |
|---|---|---|---|---|---|
| TaskSet | validation loss (valid1); training loss also stored | no | 21% of tasks (quadratic 72%, maf 64%, nvp 64%, losg 38%, TwoD 75%: negative log-likelihoods / objectives) | 51 points, steps 0-10000 (first point = step 0, before training) | explicit: adam1p (lr), adam4p (+betas, eps), adam6p (+linear / exponential decay), adam8p (+l1, l2), nadamw (warm-up fraction, constant fraction, min-lr multiplier → warm-up / constant / cosine) |
| JAHS-Bench-201 | 100 − validation accuracy (%) | yes [0, 100] | 0 | 200 epochs | fixed cosine annealing over 200 epochs; lr, weight decay as hps |
| NAS-Bench-201 | validation error (fraction) | yes [0, 1] | 0 | 200 epochs | fixed cosine annealing over 200 epochs |
| QuickTune | 100 − top-1 (%) (fine-tuning) | yes | 11% of tasks (error 0 reached) | ≤ 50 epochs, varying length | `sched` hp (several schedulers), lr, warm-up implicit |
| FCNet | validation MSE | no | 0 | 100 epochs | `lr_schedule` hp (const / cosine), init_lr |
| bbomix | validation reconstruction loss | no | 0 | 300 epochs | none explicit |

All curves used by the models are **validation** metrics. Three sources are bounded error rates (log-scale
behaviour near 0 differs from losses); three are unbounded losses.

## Rising validation curves

Live curve rises after its minimum (min before 90% of the budget, rise ≥ max(10% of its drop, 3 noise sd)):

| source | share of live curves (median task) | tasks with > 10% such curves | training loss still improving at the end (TaskSet) |
|---|---|---|---|
| TaskSet | 0.036 (FixedImageConv 0.29, char 0.06, conv 0.05) | 20% | mostly no (0-10% of rising curves), except FixedImageConv 73%, FixedNVP 92% |
| bbomix | 0.107 | 57% | n/a |
| FCNet | 0.050 | 0% | n/a |
| QuickTune | 0.030 | 2% | n/a |
| JAHS-Bench-201, NAS-Bench-201 | ≈ 0 (cosine annealing to 0 over the full budget) | 0% | n/a |

So in TaskSet most rises are *instabilities* (training and validation worsen together; median rise larger than the
drop), not overfitting; classic overfitting (training improves, validation worsens) concentrates in some families
(image convnets on small data, density models) and in bbomix. Cosine-annealed sources show none within budget.
Both phenomena must be in the generator, with config-dependent rates.
