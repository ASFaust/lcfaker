# Journal

The narrative record: what was tried, why, what was learned, which decisions followed. Newest entries at the bottom.

## 2026-11-20: Setup: brainstorm, data, held-out guard, split

- Brainstorm written (`docs/avenues.md`). Its main point: the noise result can be read as "calibrating narrows
  the faker" rather than "realism doesn't matter". That suggests randomising widely (D2/D3) over point
  calibration, with coverage (M4) as the matching measure.
- User decisions: V1/V2 order left to Claude; all training *data* allowed (only benchmark results are held out);
  more data sources from the start.
- Data (`docs/data.md`): all 1149 allowed TaskSet tasks x 5 optimizer families (5745 HPO tasks), NAS-Bench-201,
  FCNet, 8 bbomix blackboxes (35 tasks), QuickTune (86 tasks). JAHS-Bench-201 is downloaded but not converted:
  it ships as pickles, and the permission check blocked unpickling them (see `docs/data.md`).
- `/shared/datasets` is read-only in this container, so the data lives in `curve-faker/data/` (git-ignored);
  `CURVEFAKER_DATA` can point elsewhere.
- Guard: `heldout.yaml` + `curvefaker/heldout.py`, checked by every loader and downloader (tests in
  `tests/test_heldout.py`, `tests/test_data.py`).
- Internal split (`splits.yaml`): select = TaskSet {char, conv, rnn, Associative, Copy} + FCNet + bbomix; fit =
  the rest. 1970 / 3775 TaskSet tasks.

## 2026-11-20: M1 / M4 on the broader data, and the first meta-check

`scripts/m1.py` (statistics as in rogue-dagger's `curve_stats.py`; one replica per config, as an HPO method sees
it; at most 1000 configs; 2 ifBO twins per real task with the same n, d, T). Logs: `runs/logs/m1_fit.log`,
`runs/logs/m1_select.log`; data `runs/m1/{fit,select}_first.json`.

The task-level gaps found on rogue-dagger's dev set appear on **every** source:

| source (split) | flat configs real / ifBO | final spread real / ifBO | divergence real / ifBO | overfit real / ifBO |
|---|---|---|---|---|
| TaskSet (fit, 300) | 0.34 / 0.00 | 0.99 / 0.46 | 0.03 / 0.17 | 0.13 / 0.56 |
| TaskSet (select, 296) | 0.28 / 0.00 | 0.94 / 0.45 | | |
| QuickTune (fit, 85) | 0.08 / 0.00 | 0.84 / 0.44 | 0.01 / 0.24 | 0.28 / 0.65 |
| NAS-Bench-201 (fit, 3) | 0.03 / 0.03 | 0.33 / 0.57 | 0.00 / 0.32 | 0.00 / 0.67 |
| FCNet (select, 4) | 0.30 / 0.02 | 0.98 / 0.54 | 0.02 / 0.26 | 0.15 / 0.62 |
| bbomix (select, 35) | 0.37 / 0.00 | 0.97 / 0.46 | 0.01 / 0.15 | 0.32 / 0.57 |

Noise differs by source: on TaskSet the increments of single-replica curves are mildly positively autocorrelated
(+0.22 fit; rogue-dagger's +0.57 used the 5-replica mean), while on NAS-Bench-201, FCNet, QuickTune and bbomix
they are negatively autocorrelated (-0.14 to -0.47), close to ifBO's white noise. The noise fix (+0.5 to +0.7) is
far off on all of them: it matched a TaskSet-dev artefact of replica averaging. Noise level: real 0.015-0.05 vs
ifBO 0.04-0.08.

Seed luck is small everywhere: across-seed variance of the final loss / across-config variance is 0.001-0.05
(median per source), and seed-to-seed rank correlation of configs 0.96-0.99. Rogue-dagger's "config smoothness
0.01 on TaskSet" doesn't reproduce here (0.15 fit, 0.09 select; probably the adam4p/8p-only, replica-mean
processing there); low smoothness is not seed noise.

**Meta-check** (the benchmarks say ifBO > noise-fixed prior, so a trustworthy measure must prefer ifBO):

| | KS sum, ifBO / noisefix | coverage recall, ifBO / noisefix |
|---|---|---|
| TaskSet fit | 6.07 / 4.83 (wrong) | 0.09 / 0.09 (tie) |
| TaskSet select | 5.18 / 4.79 (wrong) | **0.41 / 0.17** |
| QuickTune fit | 6.49 / 6.61 (right, small) | **0.31 / 0.19** |
| NAS-Bench-201 fit | 8.67 / 10.17 (right) | 1.00 / 1.00 (3 tasks) |
| bbomix select | 8.89 / 8.13 (wrong) | **0.11 / 0.00** |
| FCNet select | 8.25 / 8.00 (wrong) | 0.75 / 1.00 (4 tasks) |

The summed per-statistic KS distance prefers the noise fix on most sources, so it is misleading for exactly this
decision, as feared. **Coverage (k-NN recall of real tasks inside the fake distribution, M4) prefers ifBO
wherever it has enough tasks to say anything.** One data point, but it supports the breadth reading, and makes
coverage the primary cheap measure from here on. Caveat: coverage is noisy (k-NN radii from 600 fake tasks); see
the next entry, where versions of one faker moved it from 0.33 to 0.64.

## 2026-11-20: Dead configs are a region; the dead-region faker (V2a)

`scripts/flat_clustering.py`: per task, a random forest predicts "flat curve" from the config (5-fold CV AUC).
NAS-Bench-201 0.99, FCNet 0.93, bbomix 0.97, QuickTune 0.79 (median over tasks). Dead configs are a region of
config space, not bad luck per config.

So the first faker change is V2a (`curvefaker/fakers.py`, `dead`): with prob. 0.7 a task has a dead half-space
(on 1 dim with prob. 0.6, else 2) holding a fraction U(0.02, 0.5) of configs; configs in it stay at the initial
loss plus white noise with per-task sd log-uniform [1e-4, 1e-2]. Everything else is ifBO, draw for draw
(components use their own random stream; `tests/test_fakers.py`).

Choosing the dead-curve noise (on the fit split only): fixed sd 0.003 gave TaskSet coverage 0.64 but only 0.03
"flat" curves (the noise exceeds the flat threshold); sd 0.0005 gave flat 0.38 (real 0.34) but coverage 0.39; the
per-task range gives coverage 0.33 (TaskSet) / 0.42 (QuickTune) vs ifBO 0.09 / 0.22. The range was chosen for
breadth, not for the highest coverage number, since coverage varies this much between near-identical fakers.

M5 launched: `wf03-dead-s0/s1/s2` (wide-focus03 config, seeds 0-2 like the reference runs; GPU 6;
`runs/logs/train_dead.sh`). Early training pair accuracy is 0.87 vs about 0.71 for ifBO: dead-vs-live pairs are
easy. Whether that dilutes the useful signal is one of the things M5 will show.
