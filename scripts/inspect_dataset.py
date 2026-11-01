#!/usr/bin/env python3
"""Inspect EEGBCI annotations and metadata for selected local recordings."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import muse  # noqa: E402,F401
import mne  # noqa: E402

from muse.data import find_eegbci_files, label_mapping_for_run, parse_subject_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "physionet")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "dataset_audit.json")
    args = parser.parse_args()
    files = find_eegbci_files(args.data_dir, args.subjects, args.runs)
    records = []
    for path in files:
        subject, run = parse_subject_run(path)
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        descriptions = [str(description) for description in raw.annotations.description]
        record = {
            "path": str(path),
            "subject": subject,
            "run": run,
            "sfreq": float(raw.info["sfreq"]),
            "n_channels": int(raw.info["nchan"]),
            "duration_seconds": float(raw.times[-1]),
            "annotation_counts": dict(Counter(descriptions)),
            "task_event_mapping": dict(label_mapping_for_run(run)),
        }
        records.append(record)
        print(json.dumps(record, indent=2, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(f"Wrote audit to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

