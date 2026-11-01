"""Configurable, leakage-safe preprocessing for EEGBCI recordings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from muse.event_mapping import labels_for_run


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters that define the offline EEG preprocessing contract."""

    l_freq: float = 8.0
    h_freq: float = 30.0
    tmin: float = 1.0
    tmax: float = 4.0
    reject_uV: float | None = 150.0
    average_reference: bool = True

    def __post_init__(self) -> None:
        if self.l_freq <= 0 or self.h_freq <= self.l_freq:
            raise ValueError("Require 0 < l_freq < h_freq for band-pass filtering.")
        if self.tmax <= self.tmin:
            raise ValueError("Epoch tmax must be greater than tmin.")
        if self.reject_uV is not None and self.reject_uV <= 0:
            raise ValueError("reject_uV must be positive or None to disable amplitude rejection.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpochBundle:
    """Epoch data and labels extracted from one EDF run."""

    X: np.ndarray
    labels: np.ndarray
    sfreq: float
    metadata: dict[str, Any] = field(default_factory=dict)


def epoch_raw(raw: Any, run: int, config: PreprocessingConfig) -> EpochBundle:
    """Filter one MNE Raw object and epoch its task-onset annotations.

    Task epochs begin one second after the visual cue to reduce classification
    of the cue-evoked response.  T0 rest annotations are intentionally not
    included in this motor-imagery task.
    """

    import mne

    mapping = labels_for_run(run)
    working = raw.copy().load_data()
    working.filter(
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        method="fir",
        fir_design="firwin",
        skip_by_annotation="edge",
        verbose=False,
    )
    if config.average_reference:
        working.set_eeg_reference("average", projection=False, verbose=False)
    picks = mne.pick_types(working.info, eeg=True, meg=False, stim=False, eog=False, exclude="bads")
    if len(picks) < 2:
        raise ValueError("At least two usable EEG channels are required for CSP.")
    events, annotation_event_id = mne.events_from_annotations(working, verbose=False)
    missing = sorted(set(mapping).difference(annotation_event_id))
    if missing:
        raise ValueError(
            f"Run {run} is missing expected annotations {missing}; "
            f"available descriptions are {sorted(annotation_event_id)}."
        )
    selected_event_id = {label: annotation_event_id[description] for description, label in mapping.items()}
    selected_events = events[np.isin(events[:, 2], list(selected_event_id.values()))]
    code_to_label = {code: label for label, code in selected_event_id.items()}
    event_labels = np.asarray([code_to_label[int(code)] for code in selected_events[:, 2]], dtype=str)
    reject = None if config.reject_uV is None else {"eeg": config.reject_uV * 1e-6}
    epochs = mne.Epochs(
        working,
        selected_events,
        event_id=selected_event_id,
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=None,
        picks=picks,
        reject=reject,
        reject_by_annotation=True,
        preload=True,
        verbose=False,
    )
    # MNE can drop boundary/amplitude-rejected trials.  Use the retained event
    # codes rather than assuming every selected event survived.
    retained_codes = epochs.events[:, 2]
    retained_labels = np.asarray([code_to_label[int(code)] for code in retained_codes], dtype=str)
    X = epochs.get_data(copy=True)
    if X.shape[0] != len(retained_labels):
        raise RuntimeError("Internal epoch/label alignment error after MNE epoch rejection.")
    metadata = {
        "run": int(run),
        "event_mapping": dict(mapping),
        "n_selected_events": int(len(selected_events)),
        "n_retained_epochs": int(len(epochs)),
        "n_rejected_epochs": int(len(selected_events) - len(epochs)),
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "sfreq": float(epochs.info["sfreq"]),
    }
    return EpochBundle(X=X, labels=retained_labels, sfreq=float(epochs.info["sfreq"]), metadata=metadata)

