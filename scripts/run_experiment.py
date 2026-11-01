#!/usr/bin/env python3
"""Run a grouped-held-out CSP experiment on local EEGBCI recordings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from muse.data import DatasetFilesMissingError, download_eegbci, load_imagery_dataset, task_for_run  # noqa: E402
from muse.evaluation import evaluate_grouped, save_evaluation_artifacts  # noqa: E402
from muse.models import make_csp_lda_pipeline, make_csp_svm_pipeline  # noqa: E402
from muse.preprocessing import PreprocessingConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "physionet")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--runs", type=int, nargs="+", default=None)
    parser.add_argument("--task", choices=["unilateral", "bilateral"], default="unilateral")
    parser.add_argument("--model", choices=["lda", "svm"], default="lda")
    parser.add_argument(
        "--grouping",
        choices=["recording", "subject"],
        default="recording",
        help="Hold out each subject/run recording or each subject across all requested runs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csp-components", type=int, default=4)
    parser.add_argument("--l-freq", type=float, default=8.0)
    parser.add_argument("--h-freq", type=float, default=30.0)
    parser.add_argument("--tmin", type=float, default=1.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--reject-uV", type=float, default=150.0)
    parser.add_argument(
        "--no-reject",
        action="store_const",
        const=None,
        dest="reject_uV",
        help="Disable amplitude rejection; use only as an explicitly labeled sensitivity run.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download exactly the requested files before running.",
    )
    args = parser.parse_args()
    runs = args.runs or ([4, 8, 12] if args.task == "unilateral" else [6, 10, 14])
    try:
        actual_tasks = {task_for_run(run) for run in runs}
    except ValueError as exc:
        parser.error(str(exc))
    if actual_tasks != {args.task}:
        parser.error(
            f"--task {args.task!r} does not match requested runs {runs}; "
            f"those runs represent {sorted(actual_tasks)}."
        )
    try:
        if args.download_missing:
            download_eegbci(args.data_dir, args.subjects, runs)
    except Exception as exc:
        parser.exit(1, f"Could not download EEGBCI data: {exc}\n")
    config = PreprocessingConfig(
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        tmin=args.tmin,
        tmax=args.tmax,
        reject_uV=args.reject_uV,
    )
    try:
        dataset = load_imagery_dataset(args.data_dir, args.subjects, runs, config=config)
    except DatasetFilesMissingError as exc:
        parser.exit(1, f"{exc}\n")
    estimator = (
        make_csp_lda_pipeline(args.csp_components)
        if args.model == "lda"
        else make_csp_svm_pipeline(args.csp_components)
    )
    evaluation_groups = (
        dataset.groups
        if args.grouping == "recording"
        else np.asarray([f"S{subject:03d}" for subject in dataset.subjects], dtype=str)
    )
    result = evaluate_grouped(estimator, dataset.X, dataset.y, evaluation_groups)
    output = args.output or ROOT / "results" / f"{args.task}_{args.model}_grouped"
    experiment_config = {
        "dataset": "PhysioNet EEG Motor Movement/Imagery Database v1.0.0",
        "data_dir": str(args.data_dir),
        "subjects": args.subjects,
        "runs": runs,
        "task": args.task,
        "classes": result.classes,
        "model": args.model,
        "csp_components": args.csp_components,
        "preprocessing": config.as_dict(),
        "sampling_rate_hz": dataset.sfreq,
        "n_epochs": dataset.n_epochs,
        "n_channels": dataset.n_channels,
        "n_times": dataset.n_times,
        "run_metadata": list(dataset.run_metadata),
        "grouping": args.grouping,
        "validation": (
            "Leave-One-Recording-Group-Out; each group is subject/run"
            if args.grouping == "recording"
            else "Leave-One-Subject-Out; all requested runs for one subject are held out together"
        ),
    }
    artifacts = save_evaluation_artifacts(
        result,
        output,
        config=experiment_config,
        estimator=estimator,
        X=dataset.X,
        y=dataset.y,
    )
    print(json.dumps({
        "classes": result.classes,
        "accuracy": result.accuracy,
        "balanced_accuracy": result.balanced_accuracy,
        "macro_f1": result.macro_f1,
        "chance_level": result.chance_level,
        "n_epochs": result.n_epochs,
        "n_groups": result.n_groups,
        "class_counts": result.class_counts,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

