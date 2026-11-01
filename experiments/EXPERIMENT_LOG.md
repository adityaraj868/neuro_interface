# MUSE experiment log

This file records experiments actually run in this repository. Synthetic
unit-test data are software validation and are not EEG decoding experiments.

## 2026-11-01 — Subject 1 unilateral baseline

Dataset: PhysioNet EEGMMIDB, subject 1, runs 4/8/12 (imagined LEFT versus
RIGHT). The dataset audit found 64 channels at 160 Hz and the expected `T1`
LEFT / `T2` RIGHT mapping in every run.

Command:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_experiment.py \
  --task unilateral --subjects 1 --runs 4 8 12 \
  --output results/unilateral_lda_grouped
```

Configuration: 8–30 Hz band-pass, average reference, epochs from 1–4 seconds
after cue, 150 µV peak-to-peak rejection, four CSP components, shrinkage LDA,
and Leave-One-Recording-Group-Out validation.

Result: 22 retained epochs across 3 recording groups (`LEFT` 10,
`RIGHT` 12).

| Accuracy | Balanced accuracy | Macro F1 | Confusion matrix (`LEFT`, `RIGHT`) |
|---:|---:|---:|---|
| 59.09% | 59.17% | 59.01% | `[[6, 4], [5, 7]]` |

Per-recording accuracy was 100.00% for S001R04, 55.56% for S001R08, and
37.50% for S001R12. The result is a small within-subject feasibility result;
it is not evidence of subject-independent performance.

Artifacts: `results/unilateral_lda_grouped/metrics.json`, `config.json`,
`predictions.csv`, `confusion_matrix.png`, and `model.joblib`.

## 2026-11-01 — Subject 1 model comparison

Command:

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_models.py \
  --subjects 1 --runs 4 8 12 --output results/model_comparison.json
```

Under the same folds and preprocessing, CSP+LDA and CSP+linear-SVM produced
the same aggregate scores:

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| CSP + LDA | 59.09% | 59.17% | 59.01% |
| CSP + linear SVM | 59.09% | 59.17% | 59.01% |

## 2026-11-01 — Initial five-subject generalization

Subjects 1–5 and runs 4/8/12 were downloaded for a subject-held-out test.
With the default 150 µV rejection, four recordings produced no retained task
epochs, so this comparison was rerun with `--no-reject` as an explicit
preprocessing sensitivity. The loader now raises an error rather than
silently dropping such a recording.

Command:

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_models.py \
  --subjects 1 2 3 4 5 --runs 4 8 12 --grouping subject --no-reject \
  --output results/model_comparison_subject5_no_reject.json
```

Result: 225 epochs across 5 held-out subjects (`LEFT` 113, `RIGHT` 112).

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| CSP + LDA | 46.67% | 46.79% | 42.13% |
| CSP + linear SVM | 44.89% | 45.03% | 39.17% |

Both models were below the 50% two-class reference. This is an initial
five-subject sensitivity result, not a broad subject-independent benchmark.
The saved LDA artifacts are under
`results/unilateral_subject5_lda_no_reject/`.

## Next planned experiment

Run the supported bilateral imagery task on runs 6/10/14, reporting
`BOTH_HANDS` versus `BOTH_FEET` without relabeling those classes as UP/DOWN.

