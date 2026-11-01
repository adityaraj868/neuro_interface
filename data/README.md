# MUSE data directory

Raw EEG is intentionally excluded from version control. The first supported
dataset is the PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB),
version 1.0.0.

From the repository root, download the small subject-1 LEFT/RIGHT imagery
subset with:

```bash
./.venv/bin/python scripts/download_eegbci.py --subjects 1 --runs 4 8 12
```

This downloads three EDF recordings (runs 4, 8, and 12) below
`data/physionet/`. The downloader checks available disk space first and never
fetches all 109 subjects unless explicitly asked. For the real bilateral
imagery task, use `--runs 6 10 14`; these are BOTH_HANDS versus BOTH_FEET,
not UP versus DOWN.

The public dataset is available from [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/).
Follow the dataset's usage terms and cite the dataset publication when using
the recordings outside this repository.

If a later milestone requires four experimentally defined imagery classes,
the [official BCI Competition IV page](https://www.bbci.de/competition/iv/)
describes Dataset 2a with left hand, right hand, feet, and tongue classes.
That is a separate source and must not be mixed with EEGMMIDB without an
explicit adapter and evaluation plan.

