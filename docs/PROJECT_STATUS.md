# MUSE project status

## Current phase

Milestone 1 is complete, including the first real-data LEFT/RIGHT result.
Milestone 2 has started with CSP+LDA/SVM comparison and an initial
five-subject held-out generalization experiment.

## Completed tasks

- Audited the initial scaffold, dependencies, virtualenv, empty data directory,
  disk space, and absence of Git metadata.
- Verified PhysioNet EEGMMIDB labels from the official documentation.
- Added explicit imagery run mappings and rejected executed runs from the
  imagery loader.
- Added bounded PhysioNet download and local-file discovery workflows.
- Added MNE filtering, average reference, event-based epoching, configurable
  amplitude rejection, and epoch/label alignment checks.
- Added CSP+shrinkage-LDA baseline and CSP+linear-SVM comparator.
- Added Leave-One-Recording-Group-Out evaluation with metrics and confusion
  matrix artifacts.
- Added model save/load support and synthetic unit tests.
- Downloaded the subject-1 LEFT/RIGHT subset and recorded its grouped result.
- Added explicit recording-held-out versus subject-held-out evaluation modes.
- Added a fail-fast check for recordings from which amplitude rejection removes
  every task epoch, plus an explicit no-rejection sensitivity option.
- Downloaded subjects 1--5 and recorded the first cross-subject comparison.
- Added README, data instructions, experiment configuration, and this status
  file.

## In progress

- Broaden the subject-held-out experiment beyond the initial five-subject
  subset and pre-specify artifact handling before treating it as a benchmark.
- Run the supported BOTH_HANDS versus BOTH_FEET experiment.

## Blocked or not yet supported

- UP/DOWN are not supported by the selected PhysioNet task labels. EEGMMIDB
  supplies imagined both-hands/both-feet runs and no tongue class; these are
  exposed under their actual labels for a separate experiment.
- Real-time inference is not implemented because no EEG stream or headset is
  configured.
- The subject-independent result is currently limited to five subjects and a
  no-amplitude-rejection sensitivity run; broader robustness results are not
  yet available.
- The default 150 uV peak-to-peak rejection removed all task epochs in some
  recordings in the five-subject subset. The loader now fails loudly for that
  condition instead of silently dropping a recording.
- The download attempt on 2026-10-22 was blocked by the shell environment's
  DNS/network policy, but the same requested files were successfully obtained
  on 2026-11-01 with approved network access.

## Exact commands run

```text
pwd && rg --files -g '!*.pyc' -g '!__pycache__/**'
git status --short --branch                  # no Git repository metadata present
python --version && python -m pip --version  # system Python 3.14.8; no system pip
./.venv/bin/python -m pip list              # MNE 1.12.1, NumPy 2.5.2, sklearn 1.9.0, etc.
df -h . /tmp                                 # approximately 210 GB free at audit time
PYTHONPATH=src .venv/bin/python -m compileall -q src scripts tests
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python scripts/download_eegbci.py --subjects 1 --runs 4 8 12
./.venv/bin/python scripts/inspect_dataset.py --subjects 1 --runs 4 8 12
PYTHONPATH=src .venv/bin/python scripts/run_experiment.py --task unilateral --subjects 1 --runs 4 8 12 --output results/unilateral_lda_grouped
PYTHONPATH=src .venv/bin/python scripts/compare_models.py --subjects 1 --runs 4 8 12 --output results/model_comparison.json
./.venv/bin/python scripts/download_eegbci.py --subjects 1 2 3 4 5 --runs 4 8 12
PYTHONPATH=src .venv/bin/python scripts/compare_models.py --subjects 1 2 3 4 5 --runs 4 8 12 --grouping subject --no-reject --output results/model_comparison_subject5_no_reject.json
PYTHONPATH=src .venv/bin/python scripts/run_experiment.py --task unilateral --subjects 1 2 3 4 5 --runs 4 8 12 --grouping subject --no-reject --output results/unilateral_subject5_lda_no_reject
```

## Tests

The current software test run passed: 10 tests, covering event mapping,
filename parsing, configuration validation, synthetic epoch shape and label
alignment, CSP pipeline fit/predict, grouped metrics, and model
save/load consistency.

## Experiments and artifacts

Measured artifacts are recorded under `results/` (ignored by Git). The first
baseline is in `results/unilateral_lda_grouped/`; the five-subject
subject-held-out LDA artifacts are in
`results/unilateral_subject5_lda_no_reject/`.

## Current best validated model

The within-subject, cross-recording CSP + shrinkage LDA baseline is the
strongest measured result so far: 59.09% accuracy, 59.17% balanced accuracy,
and 59.01% macro F1 on 22 retained epochs from subject 1. This is a small
feasibility result, not evidence of subject-independent performance. The
five-subject subject-held-out no-rejection result was below chance.

## Known issues

- `requirements.txt` reflects the existing environment's pinned versions;
  installation may require a Python version with wheels for those versions.
- Artifact handling currently uses a simple peak-to-peak EEG rejection
  threshold; a fuller artifact/quality pipeline is future work.

## Next concrete action

```bash
./.venv/bin/python scripts/download_eegbci.py --subjects 1 --runs 6 10 14
PYTHONPATH=src .venv/bin/python scripts/run_experiment.py \
  --task bilateral --subjects 1 --runs 6 10 14 \
  --output results/bilateral_lda_grouped
```

