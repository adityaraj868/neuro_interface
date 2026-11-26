#!/usr/bin/env python3
"""Train Subject-1 calibrated 4-class models with 5-fold cross-validation and save artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

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
from sklearn.model_selection import StratifiedKFold
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


def _evaluate_cv_and_save(
    name: str,
    make_estimator_fn: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    config_payload: dict[str, Any],
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.empty(len(y), dtype=object)
    fold_assignment = np.empty(len(y), dtype=object)
    fold_results = []
    classes = sorted(str(c) for c in np.unique(y))
    class_counts = {label: int(np.sum(y == label)) for label in classes}

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        fold_estimator = make_estimator_fn()
        fold_estimator.fit(X[train_idx], y[train_idx])
        pred = fold_estimator.predict(X[test_idx])
        oof_preds[test_idx] = pred
        fold_assignment[test_idx] = f"fold_{fold}"

        f_truth = y[test_idx]
        f_acc = float(accuracy_score(f_truth, pred))
        f_bacc = float(balanced_accuracy_score(f_truth, pred))
        f_macro_f1 = float(f1_score(f_truth, pred, labels=classes, average="macro", zero_division=0))
        fold_results.append(
            {
                "fold": fold,
                "held_out_groups": [f"fold_{fold}"],
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": f_acc,
                "balanced_accuracy": f_bacc,
                "macro_f1": f_macro_f1,
            }
        )

    acc = float(accuracy_score(y, oof_preds))
    bacc = float(balanced_accuracy_score(y, oof_preds))
    macro_f1 = float(f1_score(y, oof_preds, labels=classes, average="macro", zero_division=0))
    cm = confusion_matrix(y, oof_preds, labels=classes).astype(int).tolist()
    report = classification_report(
        y,
        oof_preds,
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
        n_epochs=int(len(y)),
        n_groups=n_splits,
        class_counts=class_counts,
        fold_results=fold_results,
        predictions=oof_preds.tolist(),
        truth=y.tolist(),
        prediction_groups=fold_assignment.tolist(),
    )

    # Fit final estimator on all available S001 data for live real-time simulation
    final_estimator = make_estimator_fn()
    final_estimator.fit(X, y)

    artifacts = save_evaluation_artifacts(
        result,
        output_dir,
        config=config_payload,
        estimator=final_estimator,
        X=X,
        y=y,
    )
    print(f"\n[{name}] 5-Fold Stratified Cross-Validation (n={len(y)}):")
    print(f"  CV Accuracy:          {acc:.2%} (Chance: 25.00%)")
    print(f"  CV Balanced Accuracy: {bacc:.2%}")
    print(f"  CV Macro F1:          {macro_f1:.2%}")
    print(f"  Classes:              {classes}")
    print(f"  Confusion Matrix:     {cm}")
    print(f"  Saved to:             {artifacts['model']}")
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

    full_data = load_four_class_imagery_dataset(
        data_dir,
        [1],
        config=config,
    )

    print(
        f"Loaded Subject 1 4-class dataset: {full_data.X.shape} (Runs 04, 06, 08, 10, 12, 14)"
    )

    base_config = {
        "dataset": "PhysioNet EEG Motor Movement/Imagery Database v1.0.0",
        "subject": 1,
        "task": "four_class_calibrated",
        "runs": [4, 6, 8, 10, 12, 14],
        "n_epochs": full_data.n_epochs,
        "n_channels": full_data.n_channels,
        "n_times": full_data.n_times,
        "sampling_rate_hz": full_data.sfreq,
        "preprocessing": config.as_dict(),
        "validation": "5-Fold Stratified Cross-Validation (out-of-fold metrics, zero leakage)",
    }

    summary = {
        "s001_4class_riemannian": _evaluate_cv_and_save(
            "s001_4class_riemannian",
            lambda: make_riemannian_svm_pipeline(
                estimator="oas",
                C=1.0,
                metric="riemann",
                probability=True,
                random_state=42,
                kernel="linear",
                scale=True,
            ),
            full_data.X,
            full_data.y,
            ROOT / "results" / "s001_4class_riemannian",
            {**base_config, "model": "riemannian_linear_oas_calibrated"},
        ),
        "s001_4class_lda": _evaluate_cv_and_save(
            "s001_4class_lda",
            lambda: _make_calibrated_csp_lda(n_components=8),
            full_data.X,
            full_data.y,
            ROOT / "results" / "s001_4class_lda",
            {**base_config, "model": "csp8_oas_lda_calibrated"},
        ),
        "s001_4class_fbcsp": _evaluate_cv_and_save(
            "s001_4class_fbcsp",
            lambda: make_fbcsp_pipeline(n_components=6, k_features=12, classifier_type="lda"),
            full_data.X,
            full_data.y,
            ROOT / "results" / "s001_4class_fbcsp",
            {**base_config, "model": "fbcsp_mi_lda_calibrated"},
        ),
    }
    print("\nSummary JSON:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
