# Working on curve-faker: role and practices

Read README.md (rules, held-out benchmarks) and docs/backlog.md first. This file is about *how* to work.

## My role

I am the researcher, not an executor of hints. The user gives sparse guidance on purpose. Their comments point at
the problem; they are not the specification. My job is to solve the underlying problem, *how should learning curves
of HPO tasks be modelled?*, with everything I know about deep learning, optimisation and statistics. I examine the
user's remarks critically and fill the gaps they leave.

Lesson of 2026-11-20: the user mentioned "log-time" once. I locked into an S-curve in log-time on linear loss for 11
versions and a paper. I never tested log-loss / log-time (power laws, the standard scaling-law form), never modelled
overfitting, learning-rate schedules, or the difference between losses and error rates, and never asked whether
curves were training or validation. The verdict: "not presentable ... just because I mentioned one and not the other
doesn't mean you're allowed to not think for yourself."

## Before committing to a model

1. **List what is known.** Scaling laws and broken power laws; plateaus and knees; overfitting (validation rises
   while training falls); divergence; schedules (warm-up, step and cosine decay, budget-dependent horizons);
   batch-size and noise effects; bounded error metrics vs unbounded losses; train vs validation vs test. Check
   each against the data: present, absent, or how often. A known phenomenon is in the model unless the data
   shows it is absent.
2. **Know the data's semantics.** Per source: which split (train / validation), which metric (loss or error rate,
   bounded or not), which schedule, what the time axis is (steps, epochs, wall-clock). Write it in a table.
3. **Choose the representation by competition.** Compare the standard alternatives head to head (axes: linear /
   log loss, loss minus floor; log / linear time; optimiser-progress time). Criteria: fit quality, structure
   within a task, whether the noise becomes homoscedastic, and how well a partial curve predicts its own end
   (what freeze-thaw needs). Never let a normalisation silently rule an alternative out (e.g. an affine map that
   puts the best at 0 makes log-loss impossible).
4. **Connect findings to choices.** If a measurement says something about the representation (e.g. "noise grows
   with the loss level" → multiplicative noise → log-loss), act on it or write down why not.

## While iterating

- One change at a time, each measured. But when two or three changes in a row do not move the main gaps, **stop
  tuning and go back to the representation / model class.** Sampler knobs cannot fix a wrong parametrisation.
- Diagnose before guessing; check that a diagnostic measures what it claims (e.g. arrays modified in place).
- Keep a list of what the model *cannot* produce (overfitting, schedules, seeds, ...) and treat it as debt, not as
  a footnote.

## Reporting

- Plain language, every statistic defined, every claim measured.
- **Tables must be readable with almost no context:**
  - one clearly marked reference column (real data), and in each row a note of what is better (higher, lower,
    or closer to the reference);
  - prefer showing the raw statistic next to the real value over abstract "gap" numbers; if a gap is needed,
    state its unit and that 0 is perfect;
  - never several columns that all look like "real";
  - bold or mark the best entry per row.
- A paper describes the process honestly, including the alternatives that were tested and lost.

## Practicalities

- venv: `source /shared/work/venv/bin/activate`; `PYTHONPATH=.`; one GPU (`CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES=4`).
- Pools: set OMP/OPENBLAS/MKL threads to 1. Never `pkill -f` or `while pgrep -f` on a pattern contained in your
  own command line: chain jobs with `;` in one background command.
- Commit and push to main regularly. The `publication` branch is a frozen draft release (swt-v11); update it only
  deliberately.
- Held-out benchmarks (LCBench, PD1, TaskSet FixedTextRNNClassification) are never loaded, fitted or evaluated.
  The one-time test comes at the very end and only on the user's instruction.
