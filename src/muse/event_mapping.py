"""Pure PhysioNet EEGMMIDB event-label mapping."""

from __future__ import annotations

from typing import Mapping

EEGBCI_UNILATERAL_IMAGERY_RUNS = (4, 8, 12)
EEGBCI_BILATERAL_IMAGERY_RUNS = (6, 10, 14)


def labels_for_run(run: int) -> Mapping[str, str]:
    """Return the documented event meanings for an imagery run."""

    run = int(run)
    if run in EEGBCI_UNILATERAL_IMAGERY_RUNS:
        return {"T1": "LEFT", "T2": "RIGHT"}
    if run in EEGBCI_BILATERAL_IMAGERY_RUNS:
        return {"T1": "BOTH_HANDS", "T2": "BOTH_FEET"}
    raise ValueError(
        f"Run {run} is not a supported imagery run; expected one of "
        f"{EEGBCI_UNILATERAL_IMAGERY_RUNS + EEGBCI_BILATERAL_IMAGERY_RUNS}."
    )

