# Backlog

**2026-11-21: the representation was redone** (docs/status_2026-11-21.md). Current best generator: bpl-v13
(curvefaker/faker_bpl.py). Items below marked (swt) refer to the old generator.

## bpl next steps

- [ ] Curves converge too early (time to 90% −1.8 vs −1.1): better record fits for late progress (replay check),
      then generation.
- [ ] Noise profile: sharper first step and too little early-middle noise (late/early 0.04 vs 0.15).
- [ ] Rising curves 1.4% vs 3.9%; model instability (training also worsens) separately from overfitting.
- [ ] Landscape predictability 0.68 vs 0.77 and speed 0.49 vs 0.73 without losing HPO fidelity (v10 vs v13).
- [ ] Source-specific speed structure: JAHS speed comes from architecture / augmentation, not the learning rate.
- [ ] Rewrite the paper around bpl (process, alternatives tested, readable tables); update the release.

Open work after swt-v11, roughly by priority. Each item names the measurement that would show it worked.
Held-out benchmarks (LCBench, PD1, TaskSet FixedTextRNNClassification; `heldout.yaml`) stay untouched until the
final, once-only test evaluation (item 0).

## 0. Release and final test (one time, at the very end)

- [x] Freeze a faker version for the release: swt-v11 (2026-11-20). Standalone package in `release/`
      (reproduces faker_swt v11 exactly, `release/tests/test_equivalence.py`); published on the orphan branch
      `publication` together with the paper (`paper/`). Update the branch by copying `release/` and `paper/`.
- [x] License: MIT (2026-11-20). Paper authors: withheld for now.
- [ ] One-time evaluation on the held-out benchmarks of the FT-PFN paper, after the paper text and the faker are
      frozen. No change to the model after it. Report whatever comes out.

## 1. Landscape (axis A)

- [ ] **Landscape predictability below real** (0.69 vs 0.77) and **near-best share** (0.08 vs 0.14). Diagnostic
      so far: field 0.95 → after dead curves 0.83 → finished task 0.76 (without the rough part). Next: calibrate
      the rough-part weight on the whole generation pipeline (simulate generated tasks per (d, eta)) instead of on
      the bare field. Measure: A_median_rho200, A_median_near_best.
- [ ] **Too rugged** (local-optima ratio 0.62 vs 0.46). Check whether ruggedness comes from the rough field or
      from noise in the final value (noise-free trace already gave 0.49-0.58). Measure: A_median_optima_ratio.
- [ ] **Speed from config** 0.50 vs 0.73. Candidates: gamma and amplitude also as smooth functions of the config
      (now taken from the nearest real vector); a learned config → warp map (below). Measure:
      A_median_shape_from_config.
- [ ] **Learned config → warp map** in place of random GP fields: fit, per real fit task, a GP / trees from
      config to (final, beta, gamma, amplitude); sample new tasks from a distribution over such maps. The
      biggest structural step; the analysis (H4) says the config shapes the curve beyond its final value.

## 2. Multi-fidelity behaviour (axis E)

- [ ] **HPO gains too low** (HB 0.11 vs 0.155, FT 0.08 vs 0.112). Re-check after v11 (dead share fix).
- [ ] **Split difference**: TaskSet select tasks have more consistent early ranks than fit tasks (rank_early10
      0.896 vs 0.869). Quantify for all sources and report the bound in the paper; consider reweighting fit
      records.
- [ ] Repeat HPO evaluation over 3 generation seeds for the release version (single-seed differences < 0.03 are
      noise).

## 3. Curves (axis B) and noise (axis C)

- [ ] **First-value distribution**: generated first values reach higher than real (q90 1.03-1.29 in real units).
      Model the first value explicitly (template start + level).
- [ ] **Upward-skewed spikes** (analysis: noise skewed upward on some sources); now symmetric Student-t. Measure:
      C_gap_spikes.
- [ ] **Minority shapes** not generated: two-stage, worsen-then-improve, overfitting (late rise), unstable.
      Analysis S1/S3 gives their shares.
- [ ] **Late-vs-early noise ratio** too small (0.10 vs 0.15).
- [ ] **Seed replicas**: generate several seeds per config (seed luck = 24-37% of final-value variance among the
      top 10%).
- [ ] **Learned templates** (a generative model over templates) instead of bootstrapping the fitted ones; needed
      for novelty beyond the fit tasks.

## 4. Analysis items not done

- [ ] S4 knees; S6 losses vs error metrics; T2 task types; T3 fitted joint vs separate model; T4 sufficient task
      statistics (docs/analysis_plan.md).

## 5. Engineering

- [ ] The 2026-11-20 all-faker rerun of evaluate_axes printed log values in the C_median_late_vs_early_sd row
      (it started before a duplicated line was removed; gaps are correct). Re-run once for the paper tables.
- [ ] Generation speed: two n×n Cholesky factorisations per task (n = 1000) dominate; use inducing points or
      random features for larger tasks.
- [ ] Package API: `curvefaker.generate(n_tasks, seed)` returning (curves, configs, space) with docs.
