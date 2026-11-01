# MUSE project status

## Current phase

Milestone 1 implementation is complete at the software level. The real-data
LEFT/RIGHT experiment is blocked here by shell network/DNS access, with the
download command ready for an environment that can reach PhysioNet.

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
- Added README, data instructions, experiment configuration, and this status
  file.

## In progress

- Download and execute the one-subject LEFT/RIGHT baseline when PhysioNet is
  reachable from the execution environment.
- Populate measured results and paths in README and the experiment log.

## Blocked or not yet supported

- UP/DOWN are not supported by the selected PhysioNet task labels. EEGMMIDB
  supplies imagined both-hands/both-feet runs and no tongue class; these are
  exposed under their actual labels for a separate experiment.
- Real-time inference is not implemented because no EEG stream or headset is
  configured.
- Subject-independent and broader session robustness results are not yet
  available.
- The download attempt on 2026-10-22 was blocked by the shell environment's
  DNS/network policy: `physionet.org` could not resolve while MNE requested
  `https://physionet.org/files/eegmmidb/1.0.0/S001/S001R04.edf`.

## Exact commands run

```text
pwd && rg --files -g '!*.pyc' -g '!__pycache__/**'
git status --short --branch                  # no Git repository metadata present
python --version && python -m pip --version  # system Python 3.14.8; no system pip
./.venv/bin/python -m pip list              # MNE 1.12.1, NumPy 2.5.2, sklearn 1.9.0, etc.
df -h . /tmp                                 # approximately 210 GB free at audit time
PYTHONPATH=src .venv/bin/python -m compileall -q src scripts tests
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python scripts/download_eegbci.py --subjects 1 --runs 4 8 12 # blocked: DNS
```

## Tests

The current software test run passed: 9 tests, covering event mapping,
filename parsing, configuration validation, synthetic epoch shape and label
alignment, CSP pipeline fit/predict, grouped metrics, and model
save/load consistency. No real EEG experiment has been run yet.

## Experiments and artifacts

No real experiment entry or measured artifact is recorded yet. The expected
first output directory is `results/unilateral_lda_grouped/`, containing
`metrics.json`, `config.json`, `predictions.csv`, `confusion_matrix.png`, and
`model.joblib` after the documented experiment command completes.

## Current best validated model

None on real EEG data yet. The software baseline is CSP + shrinkage LDA; its
performance must not be inferred from the synthetic tests.

## Known issues

- `requirements.txt` reflects the existing environment's pinned versions;
  installation may require a Python version with wheels for those versions.
- Artifact handling currently uses a simple peak-to-peak EEG rejection
  threshold; a fuller artifact/quality pipeline is future work.

## Next concrete action

```bash
./.venv/bin/python scripts/download_eegbci.py --subjects 1 --runs 4 8 12
```

