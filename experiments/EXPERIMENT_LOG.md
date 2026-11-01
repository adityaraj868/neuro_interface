# MUSE experiment log

This file records only experiments actually run in this repository. Synthetic
unit-test data are software validation and are not EEG decoding experiments.

## No real EEG experiment recorded yet

The implementation and tests are complete enough to run the first explicit
subject-1 PhysioNet subset. The next entry should be created after:

```bash
./.venv/bin/python scripts/download_eegbci.py --subjects 1 --runs 4 8 12
PYTHONPATH=src .venv/bin/python scripts/run_experiment.py \
  --task unilateral --subjects 1 --runs 4 8 12 \
  --output results/unilateral_lda_grouped
```

The first download attempt was made on 2026-10-22 but did not produce an
experiment: the execution environment could not resolve `physionet.org`
(`urllib3.exceptions.NameResolutionError`). No metrics, plot, or trained EEG
model were created.

