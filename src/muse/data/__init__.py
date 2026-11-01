"""Dataset acquisition and event-label handling."""

from .physionet import (
    EEGBCI_BILATERAL_IMAGERY_RUNS,
    EEGBCI_UNILATERAL_IMAGERY_RUNS,
    DatasetFilesMissingError,
    EpochDataset,
    download_eegbci,
    find_eegbci_files,
    label_mapping_for_run,
    load_imagery_dataset,
    parse_subject_run,
    task_for_run,
    validate_epoch_dataset,
)

__all__ = [
    "EEGBCI_BILATERAL_IMAGERY_RUNS",
    "EEGBCI_UNILATERAL_IMAGERY_RUNS",
    "DatasetFilesMissingError",
    "EpochDataset",
    "download_eegbci",
    "find_eegbci_files",
    "label_mapping_for_run",
    "load_imagery_dataset",
    "parse_subject_run",
    "task_for_run",
    "validate_epoch_dataset",
]

