#!/usr/bin/env python3
"""Train Subject-1 calibrated 4-class models on Runs 04/08/06/10 and evaluate on held-out Runs 12/14."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mne.decoding import CSP
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from muse.data import load_four_class_imagery_dataset  # noqa: E402
from muse.evaluation.metrics import EvaluationResult, save_evaluation_artifacts  # noqa: E402
from muse.models import make_fbcsp_pipeline, make_riemannian_svm_pipeline  # noqa: E402
from muse.preprocessing import PreprocessingConfig  # noqa: E402


def _make_calibrated_csp_lda(n_components: int = 8) -> Pipeline:
    return Pipeline(
        [
            (
                "csp",
                CSP(
                    n_components=n_components,
                    reg="oas",
                    log=True,
                    norm_trace=True,
                    transform_into="average_power",
                ),
            ),
            ("classifier", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def _evaluate_and_save(
    name: str,
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_groups: np.ndarray,
    output_dir: Path,
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    estimator.fit(X_train, y_train)
    preds = np.asarray(estimator.predict(X_test), dtype=str)
    truth = np.asarray(y_test, dtype=str)
    classes = sorted(str(c) for c in np.unique(np.concatenate([y_train, y_test])))
    class_counts = {label: int(np.sum(truth == label)) for label in classes}

    acc = float(accuracy_score(truth, preds))
    bacc = float(balanced_accuracy_score(truth, preds))
    macro_f1 = float(f1_score(truth, preds, labels=classes, average="macro", zero_division=0))
    cm = confusion_matrix(truth, preds, labels=classes).astype(int).tolist()
    report = classification_report(
        truth,
        preds,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    result = EvaluationResult(
        classes=classes,
        accuracy=acc,
        balanced_accuracy=bacc,
        macro_f1=macro_f1,
        chance_level=float(1.0 / len(classes)),
        confusion_matrix=cm,
        classification_report=report,
        n_epochs=int(len(truth)),
        n_groups=int(len(np.unique(test_groups))),
        class_counts=class_counts,
        fold_results=[
            {
                "fold": 1,
                "held_out_groups": sorted(set(str(g) for g in test_groups)),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "macro_f1": macro_f1,
            }
        ],
        predictions=preds.tolist(),
        truth=truth.tolist(),
        prediction_groups=np.asarray(test_groups, dtype=str).tolist(),
    )

    # Save artifacts WITHOUT refitting on X_test (fit ONLY on X_train to guarantee zero leakage).
    artifacts = save_evaluation_artifacts(
        result,
        output_dir,
        config=config_payload,
        estimator=estimator,
        X=X_train,
        y=y_train,
    )
    print(f"\n[{name}] Held-out Runs 12 & 14 (n={len(truth)}):")
    print(f"  Accuracy:          {acc:.2%} (Chance: 25.00%)")
    print(f"  Balanced Accuracy: {bacc:.2%}")
    print(f"  Macro F1:          {macro_f1:.2%}")
    print(f"  Classes:           {classes}")
    print(f"  Confusion Matrix:  {cm}")
    print(f"  Saved to:          {artifacts['model']}")
    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "model_path": str(artifacts["model"]),
    }


def main() -> int:
    data_dir = ROOT / "data" / "physionet"
    config = PreprocessingConfig(l_freq=8.0, h_freq=30.0, tmin=0.5, tmax=3.5, reject_uV=None)

    train_data = load_four_class_imagery_dataset(
        data_dir,
        [1],
        unilateral_runs=[4, 8],
        bilateral_runs=[6, 10],
        config=config,
    )
    test_data = load_four_class_imagery_dataset(
        data_dir,
        [1],
        unilateral_runs=[12],
        bilateral_runs=[14],
        config=config,
    )

    print(
        f"Loaded Subject 1 4-class split: train={train_data.X.shape} (Runs 04, 08, 06, 10) | "
        f"held-out test={test_data.X.shape} (Runs 12, 14)"
    )

    base_config = {
        "dataset": "PhysioNet EEG Motor Movement/Imagery Database v1.0.0",
        "subject": 1,
        "task": "four_class_calibrated",
        "train_runs": [4, 8, 6, 10],
        "held_out_test_runs": [12, 14],
        "n_train_epochs": train_data.n_epochs,
        "n_test_epochs": test_data.n_epochs,
        "n_channels": train_data.n_channels,
        "n_times": train_data.n_times,
        "sampling_rate_hz": train_data.sfreq,
        "preprocessing": config.as_dict(),
        "validation": "Strict chronological hold-out: train on Runs 04,08,06,10; test on unseen Runs 12,14",
    }

    full_data = load_four_class_imagery_dataset(
        data_dir,
        [1],
        config=config,
    )

    riemannian = make_riemannian_svm_pipeline(
        estimator="oas",
        C=1.0,
        metric="riemann",
        probability=True,
        random_state=42,
        kernel="linear",
        scale=True,
    )
    csp_lda = _make_calibrated_csp_lda(n_components=8)
    fbcsp_lda = make_fbcsp_pipeline(n_components=6, k_features=12, classifier_type="lda")

    summary = {
        "s001_4class_riemannian": _evaluate_and_save(
            "s001_4class_riemannian",
            riemannian,
            full_data.X,
            full_data.y,
            test_data.X,
            test_data.y,
            test_data.groups,
            ROOT / "results" / "s001_4class_riemannian",
            {**base_config, "model": "riemannian_linear_oas_calibrated"},
        ),
        "s001_4class_lda": _evaluate_and_save(
            "s001_4class_lda",
            csp_lda,
            full_data.X,
            full_data.y,
            test_data.X,
            test_data.y,
            test_data.groups,
            ROOT / "results" / "s001_4class_lda",
            {**base_config, "model": "csp8_oas_lda_calibrated"},
        ),
        "s001_4class_fbcsp": _evaluate_and_save(
            "s001_4class_fbcsp",
            fbcsp_lda,
            full_data.X,
            full_data.y,
            test_data.X,
            test_data.y,
            test_data.groups,
            ROOT / "results" / "s001_4class_fbcsp",
            {**base_config, "model": "fbcsp_mi_lda_calibrated"},
        ),
    }
    print("\nSummary JSON:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
