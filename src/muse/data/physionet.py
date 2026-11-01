"""PhysioNet EEGMMIDB download, event mapping, and loading utilities."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from muse.event_mapping import (
    EEGBCI_BILATERAL_IMAGERY_RUNS as _BILATERAL_IMAGERY_RUNS,
    EEGBCI_UNILATERAL_IMAGERY_RUNS as _UNILATERAL_IMAGERY_RUNS,
    labels_for_run,
)
from muse.preprocessing.epochs import EpochBundle, PreprocessingConfig, epoch_raw

EEGBCI_UNILATERAL_IMAGERY_RUNS = _UNILATERAL_IMAGERY_RUNS
EEGBCI_BILATERAL_IMAGERY_RUNS = _BILATERAL_IMAGERY_RUNS
EEGBCI_EXECUTED_UNILATERAL_RUNS = (3, 7, 11)
EEGBCI_EXECUTED_BILATERAL_RUNS = (5, 9, 13)

_FILE_PATTERN = re.compile(r"S(?P<subject>\d{3})R(?P<run>\d{2})\.edf$", re.IGNORECASE)


class DatasetFilesMissingError(FileNotFoundError):
    """Raised when the requested EEGBCI EDF files are not locally available."""


@dataclass(frozen=True)
class EpochDataset:
    """Epoched EEG and trial metadata used by the decoding experiments."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    subjects: np.ndarray
    runs: np.ndarray
    sfreq: float
    preprocessing: PreprocessingConfig
    run_metadata: tuple[dict[str, object], ...]

    @property
    def n_epochs(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.X.shape[2])


def label_mapping_for_run(run: int) -> Mapping[str, str]:
    """Return the documented T1/T2 meaning for an imagery run.

    The PhysioNet annotation codes are intentionally interpreted in the
    context of the run.  T1/T2 do not have one global meaning across the
    dataset.
    """

    return dict(labels_for_run(run))


def task_for_run(run: int) -> str:
    """Return ``unilateral`` or ``bilateral`` for an imagery run."""

    run = int(run)
    if run in EEGBCI_UNILATERAL_IMAGERY_RUNS:
        return "unilateral"
    if run in EEGBCI_BILATERAL_IMAGERY_RUNS:
        return "bilateral"
    raise ValueError(f"Run {run} is not an imagery run.")


def labels_from_event_descriptions(run: int, descriptions: Iterable[str]) -> np.ndarray:
    """Map annotation descriptions to task labels with a clear error."""

    mapping = label_mapping_for_run(run)
    descriptions = list(descriptions)
    unknown = sorted(set(descriptions).difference(mapping))
    if unknown:
        raise ValueError(
            f"Unexpected event descriptions {unknown} for imagery run {run}; "
            f"expected only {sorted(mapping)}. Rest (T0) is excluded from task epochs."
        )
    return np.asarray([mapping[description] for description in descriptions], dtype=str)


def parse_subject_run(path: str | Path) -> tuple[int, int]:
    """Extract the subject and run number from an EEGBCI EDF filename."""

    match = _FILE_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(f"Not an EEGBCI EDF filename: {path}")
    return int(match.group("subject")), int(match.group("run"))


def _normalise_subjects(subjects: int | Sequence[int]) -> tuple[int, ...]:
    values = (subjects,) if isinstance(subjects, int) else tuple(int(x) for x in subjects)
    if not values or any(subject < 1 or subject > 109 for subject in values):
        raise ValueError("EEGBCI subjects must be integers in the range 1 through 109.")
    return values


def _normalise_runs(runs: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(x) for x in runs)
    if not values:
        raise ValueError("At least one EEGBCI run is required.")
    return values


def download_eegbci(
    data_dir: str | Path,
    subjects: int | Sequence[int],
    runs: Sequence[int],
    *,
    minimum_free_bytes: int = 100 * 1024 * 1024,
) -> list[Path]:
    """Download only the requested EEGBCI EDF files via MNE.

    The function checks free space before downloading.  It deliberately does
    not fetch the complete 109-subject archive unless the caller explicitly
    requests all subjects and runs.
    """

    subjects = _normalise_subjects(subjects)
    runs = _normalise_runs(runs)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(data_dir).free
    if free_bytes < minimum_free_bytes:
        raise OSError(
            f"Only {free_bytes / 1024**2:.1f} MiB is free at {data_dir}; "
            f"at least {minimum_free_bytes / 1024**2:.1f} MiB is required before download."
        )

    from mne.datasets import eegbci

    paths = eegbci.load_data(
        subjects=list(subjects),
        runs=list(runs),
        path=str(data_dir),
        update_path=False,
        verbose=False,
    )
    return [Path(path) for path in paths]


def find_eegbci_files(
    data_dir: str | Path,
    subjects: int | Sequence[int],
    runs: Sequence[int],
) -> list[Path]:
    """Find requested local EDF files or raise an actionable error."""

    subjects = _normalise_subjects(subjects)
    runs = _normalise_runs(runs)
    data_dir = Path(data_dir)
    found: list[Path] = []
    missing: list[str] = []
    for subject in subjects:
        for run in runs:
            expected = f"S{subject:03d}R{run:02d}.edf"
            matches = list(data_dir.rglob(expected))
            if matches:
                found.append(matches[0])
            else:
                missing.append(expected)
    if missing:
        raise DatasetFilesMissingError(
            f"Missing {len(missing)} EEGBCI file(s) below {data_dir}: {', '.join(missing)}. "
            "Run scripts/download_eegbci.py with the requested subjects and runs first."
        )
    return found


def _read_and_epoch(path: Path, config: PreprocessingConfig) -> EpochBundle:
    import mne
    from mne.datasets import eegbci

    subject, run = parse_subject_run(path)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    eegbci.standardize(raw)
    bundle = epoch_raw(raw, run=run, config=config)
    bundle.metadata.update({"subject": subject, "run": run, "path": str(path)})
    return bundle


def load_imagery_dataset(
    data_dir: str | Path,
    subjects: int | Sequence[int],
    runs: Sequence[int],
    *,
    config: PreprocessingConfig | None = None,
) -> EpochDataset:
    """Load, filter, epoch, and validate the requested imagery recordings."""

    subjects = _normalise_subjects(subjects)
    runs = _normalise_runs(runs)
    expected_tasks = {task_for_run(run) for run in runs}
    if len(expected_tasks) != 1:
        raise ValueError(
            "Do not combine unilateral and bilateral runs in one experiment; "
            "choose one task family so T1/T2 retain a consistent meaning."
        )
    config = config or PreprocessingConfig()
    paths = find_eegbci_files(data_dir, subjects, runs)
    bundles: list[EpochBundle] = []
    for path in paths:
        bundles.append(_read_and_epoch(path, config))

    X = np.concatenate([bundle.X for bundle in bundles], axis=0)
    y = np.concatenate([bundle.labels for bundle in bundles], axis=0)
    subject_values = np.concatenate(
        [np.full(len(bundle.labels), int(bundle.metadata["subject"]), dtype=int) for bundle in bundles]
    )
    run_values = np.concatenate(
        [np.full(len(bundle.labels), int(bundle.metadata["run"]), dtype=int) for bundle in bundles]
    )
    groups = np.asarray(
        [f"S{subject:03d}R{run:02d}" for subject, run in zip(subject_values, run_values)],
        dtype=str,
    )
    validate_epoch_dataset(X, y, groups)
    return EpochDataset(
        X=X,
        y=y,
        groups=groups,
        subjects=subject_values,
        runs=run_values,
        sfreq=float(bundles[0].sfreq),
        preprocessing=config,
        run_metadata=tuple(bundle.metadata for bundle in bundles),
    )


def validate_epoch_dataset(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> None:
    """Validate epoch dimensions and trial-level metadata alignment."""

    if X.ndim != 3:
        raise ValueError(f"EEG epochs must have shape (trials, channels, time), got {X.shape}.")
    if X.shape[0] == 0:
        raise ValueError("No valid EEG epochs remain after event selection/rejection.")
    if len(y) != X.shape[0] or len(groups) != X.shape[0]:
        raise ValueError(
            "EEG epoch, label, and group counts must match: "
            f"{X.shape[0]} epochs, {len(y)} labels, {len(groups)} groups."
        )
    if len(np.unique(y)) < 2:
        raise ValueError(f"At least two classes are required; observed {np.unique(y).tolist()}.")

