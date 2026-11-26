#!/usr/bin/env python3
"""Benchmark CSP+LDA vs Riemannian-SVM vs EEGNet across within-subject and LOSO folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from muse.data import load_imagery_dataset  # noqa: E402
from muse.evaluation import evaluate_grouped, evaluate_loso  # noqa: E402
from muse.models import (  # noqa: E402
    make_csp_lda_pipeline,
    make_eegnet_classifier,
    make_riemannian_svm_pipeline,
)
from muse.preprocessing import PreprocessingConfig  # noqa: E402


def build_models(csp_components: int = 4, eegnet_epochs: int = 50) -> dict[str, Any]:
    return {
        "csp_lda": make_csp_lda_pipeline(n_components=csp_components),
        "riemannian_svm": make_riemannian_svm_pipeline(estimator="oas", C=1.0, probability=True),
        "eegnet": make_eegnet_classifier(epochs=eegnet_epochs, batch_size=16, lr=1e-3, random_state=42),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "physionet")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "model_benchmark.json")
    parser.add_argument("--csp-components", type=int, default=4)
    parser.add_argument("--eegnet-epochs", type=int, default=50)
    parser.add_argument("--reject-uV", type=float, default=150.0)
    parser.add_argument(
        "--no-reject",
        action="store_const",
        const=None,
        dest="reject_uV",
        help="Disable amplitude rejection for sensitivity evaluation.",
    )
    args = parser.parse_args()

    print(f"Loading dataset for subjects {args.subjects}, runs {args.runs}...")
    preprocessing = PreprocessingConfig(reject_uV=args.reject_uV)
    dataset = load_imagery_dataset(
        args.data_dir,
        args.subjects,
        args.runs,
        config=preprocessing,
    )
    print(f"Retained {dataset.n_epochs} epochs across {len(set(dataset.subjects))} subjects.")

    benchmark_results: dict[str, Any] = {
        "subjects": [int(s) for s in sorted(set(dataset.subjects))],
        "runs": args.runs,
        "n_epochs": int(dataset.n_epochs),
        "within_subject": {},
        "loso": {},
    }

    # 1. Within-subject evaluation (Leave-One-Run-Out per subject)
    unique_subs = sorted(set(dataset.subjects))
    for model_name, model in build_models(args.csp_components, args.eegnet_epochs).items():
        sub_baccs = []
        sub_f1s = []
        sub_results = {}

        for sub in unique_subs:
            sub_mask = dataset.subjects == sub
            sub_X = dataset.epochs[sub_mask]
            sub_y = dataset.labels[sub_mask]
            sub_groups = dataset.groups[sub_mask]

            # Need at least 2 distinct runs with at least 2 classes
            if len(set(sub_groups)) >= 2 and len(set(sub_y)) >= 2:
                eval_res = evaluate_grouped(model, sub_X, sub_y, sub_groups)
                sub_baccs.append(eval_res.balanced_accuracy)
                sub_f1s.append(eval_res.macro_f1)
                sub_results[f"S{sub:03d}"] = {
                    "balanced_accuracy": eval_res.balanced_accuracy,
                    "macro_f1": eval_res.macro_f1,
                    "n_epochs": eval_res.n_epochs,
                }

        benchmark_results["within_subject"][model_name] = {
            "mean_balanced_accuracy": float(np.mean(sub_baccs)) if sub_baccs else 0.0,
            "std_balanced_accuracy": float(np.std(sub_baccs)) if sub_baccs else 0.0,
            "mean_macro_f1": float(np.mean(sub_f1s)) if sub_f1s else 0.0,
            "std_macro_f1": float(np.std(sub_f1s)) if sub_f1s else 0.0,
            "per_subject": sub_results,
        }

    # 2. Leave-One-Subject-Out (LOSO) evaluation (if multiple subjects are present)
    if len(unique_subs) >= 2:
        for model_name, model in build_models(args.csp_components, args.eegnet_epochs).items():
            within_score = benchmark_results["within_subject"][model_name]["mean_balanced_accuracy"]
            loso_res = evaluate_loso(
                model,
                dataset.epochs,
                dataset.labels,
                dataset.subjects,
                within_subject_score=within_score,
            )
            benchmark_results["loso"][model_name] = loso_res.as_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\nBenchmark completed successfully! Results written to {args.output}")
    print("\nSummary (Within-Subject Balanced Accuracy):")
    for name, data in benchmark_results["within_subject"].items():
        print(f"  {name:15s}: {data['mean_balanced_accuracy']*100:.2f}% ± {data['std_balanced_accuracy']*100:.2f}%")

    if benchmark_results["loso"]:
        print("\nSummary (LOSO Generalization Balanced Accuracy):")
        for name, data in benchmark_results["loso"].items():
            gap = data.get("domain_shift_gap")
            gap_str = f"(Domain gap: {gap*100:+.2f}%)" if gap is not None else ""
            print(f"  {name:15s}: {data['mean_balanced_accuracy']*100:.2f}% {gap_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
