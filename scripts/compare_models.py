#!/usr/bin/env python3
"""Compare CSP+LDA and CSP+linear-SVM under the same grouped folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    args = parser.parse_args()
    dataset = load_imagery_dataset(
        args.data_dir,
        args.subjects,
        args.runs,
        config=PreprocessingConfig(),
    )
    results = {}
    for name, estimator in {
        "csp_lda": make_csp_lda_pipeline(args.csp_components),
        "csp_svm": make_csp_svm_pipeline(args.csp_components),
    }.items():
        results[name] = evaluate_grouped(estimator, dataset.X, dataset.y, dataset.groups).as_dict()
    payload = {
        "dataset": "PhysioNet EEG Motor Movement/Imagery Database v1.0.0",
        "subjects": args.subjects,
        "runs": args.runs,
        "validation": "Leave-One-Recording-Group-Out; each group is subject/run",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {key: value[key] for key in ("accuracy", "balanced_accuracy", "macro_f1")} for name, value in results.items()}, indent=2))
    print(f"Wrote comparison to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

