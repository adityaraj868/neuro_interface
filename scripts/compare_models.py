#!/usr/bin/env python3
"""Compare CSP+LDA and CSP+linear-SVM under the same grouped folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from muse.data import load_imagery_dataset  # noqa: E402
from muse.evaluation import evaluate_grouped  # noqa: E402
from muse.models import make_csp_lda_pipeline, make_csp_svm_pipeline  # noqa: E402
from muse.preprocessing import PreprocessingConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "physionet")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "model_comparison.json")
    parser.add_argument("--csp-components", type=int, default=4)
    parser.add_argument(
        "--grouping",
        choices=["recording", "subject"],
        default="recording",
        help="Hold out each subject/run recording or each subject across all requested runs.",
    )
    parser.add_argument("--reject-uV", type=float, default=150.0)
    parser.add_argument(
        "--no-reject",
        action="store_const",
        const=None,
        dest="reject_uV",
        help="Disable amplitude rejection; use only as an explicitly labeled sensitivity run.",
    )
    args = parser.parse_args()
    preprocessing = PreprocessingConfig(reject_uV=args.reject_uV)
    dataset = load_imagery_dataset(
        args.data_dir,
        args.subjects,
        args.runs,
        config=preprocessing,
    )
    evaluation_groups = (
        dataset.groups
        if args.grouping == "recording"
        else np.asarray([f"S{subject:03d}" for subject in dataset.subjects], dtype=str)
    )
    results = {}
    for name, estimator in {
        "csp_lda": make_csp_lda_pipeline(args.csp_components),
        "csp_svm": make_csp_svm_pipeline(args.csp_components),
    }.items():
        results[name] = evaluate_grouped(estimator, dataset.X, dataset.y, evaluation_groups).as_dict()
    payload = {
        "dataset": "PhysioNet EEG Motor Movement/Imagery Database v1.0.0",
        "subjects": args.subjects,
        "runs": args.runs,
        "grouping": args.grouping,
        "preprocessing": preprocessing.as_dict(),
        "validation": (
            "Leave-One-Recording-Group-Out; each group is subject/run"
            if args.grouping == "recording"
            else "Leave-One-Subject-Out; all requested runs for one subject are held out together"
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {key: value[key] for key in ("accuracy", "balanced_accuracy", "macro_f1")} for name, value in results.items()}, indent=2))
    print(f"Wrote comparison to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

