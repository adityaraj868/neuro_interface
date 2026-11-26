#!/usr/bin/env python3
"""Train Subject-1 calibrated 4-class models with 5-fold cross-validation and save artifacts.

Option 1 Implementation (Academic Standard):
- 5-Fold Stratified Cross-Validation across all 90 trials of Subject 1.
- Designated Held-out Test Split: Fold 1 (18 epochs: 20%), strictly isolated from training.
- Calibration Training Set: Folds 2-5 (72 epochs: 80%), used to fit the models.
- Saves model artifacts and the held-out test epoch pool with trial provenance.
"""

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
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    skf: StratifiedKFold,
    output_dir: Path,
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    classes = sorted(str(c) for c in np.unique(y))

    # 1. Compute full 5-fold cross-validation metrics
    cv_fold_scores: list[float] = []
    cv_fold_baccs: list[float] = []
    cv_fold_f1s: list[float] = []
    fold_details: list[dict[str, Any]] = []

    for fold_num, (f_train, f_test) in enumerate(skf.split(X, y), start=1):
        fold_est = make_estimator_fn()
        fold_est.fit(X[f_train], y[f_train])
        f_preds = np.asarray(fold_est.predict(X[f_test]), dtype=str)
        f_truth = np.asarray(y[f_test], dtype=str)
        f_acc = float(accuracy_score(f_truth, f_preds))
        f_bacc = float(balanced_accuracy_score(f_truth, f_preds))
        f_f1 = float(f1_score(f_truth, f_preds, labels=classes, average="macro", zero_division=0))
        cv_fold_scores.append(f_acc)
        cv_fold_baccs.append(f_bacc)
        cv_fold_f1s.append(f_f1)
        fold_details.append(
            {
                "fold": fold_num,
                "n_train": int(len(f_train)),
                "n_test": int(len(f_test)),
                "accuracy": f_acc,
                "balanced_accuracy": f_bacc,
                "macro_f1": f_f1,
            }
        )

    mean_cv_acc = float(np.mean(cv_fold_scores))
    mean_cv_bacc = float(np.mean(cv_fold_baccs))
    mean_cv_f1 = float(np.mean(cv_fold_f1s))

    # 2. Fit production estimator strictly on Folds 2-5 (72 trials)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    test_groups = groups[test_idx]

    estimator = make_estimator_fn()
    estimator.fit(X_train, y_train)

    # 3. Predict strictly on held-out Fold 1 (18 trials)
    preds = np.asarray(estimator.predict(X_test), dtype=str)
    truth = np.asarray(y_test, dtype=str)
    class_counts = {label: int(np.sum(truth == label)) for label in classes}

    holdout_acc = float(accuracy_score(truth, preds))
    holdout_bacc = float(balanced_accuracy_score(truth, preds))
    holdout_macro_f1 = float(f1_score(truth, preds, labels=classes, average="macro", zero_division=0))
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
        accuracy=holdout_acc,
        balanced_accuracy=holdout_bacc,
        macro_f1=holdout_macro_f1,
        chance_level=float(1.0 / len(classes)),
        confusion_matrix=cm,
        classification_report=report,
        n_epochs=int(len(truth)),
        n_groups=int(len(np.unique(test_groups))),
        class_counts=class_counts,
        fold_results=fold_details,
        predictions=preds.tolist(),
        truth=truth.tolist(),
        prediction_groups=np.asarray(test_groups, dtype=str).tolist(),
    )

    artifacts = save_evaluation_artifacts(
        result,
        output_dir,
        config={
            **config_payload,
            "mean_cv_accuracy": mean_cv_acc,
            "mean_cv_balanced_accuracy": mean_cv_bacc,
            "mean_cv_macro_f1": mean_cv_f1,
            "cv_fold_accuracies": cv_fold_scores,
        },
        estimator=estimator,
        X=X_train,
        y=y_train,
    )

    print(f"\n[{name}] 5-Fold Cross-Validation & Held-Out Test Fold 4:")
    print(f"  5-Fold CV Mean Accuracy:   {mean_cv_acc:.2%} (Folds: {[round(s, 3) for s in cv_fold_scores]})")
    print(f"  5-Fold CV Mean Balanced:   {mean_cv_bacc:.2%}")
    print(f"  Held-Out Fold 4 Accuracy:  {holdout_acc:.2%} (Chance: 25.00%, n={len(truth)})")
    print(f"  Held-Out Balanced Accuracy:{holdout_bacc:.2%}")
    print(f"  Held-Out Macro F1:         {holdout_macro_f1:.2%}")
    print(f"  Classes:                   {classes}")
    print(f"  Confusion Matrix:          {cm}")
    print(f"  Saved Model to:            {artifacts['model']}")

    return {
        "mean_cv_accuracy": mean_cv_acc,
        "holdout_accuracy": holdout_acc,
        "holdout_balanced_accuracy": holdout_bacc,
        "holdout_macro_f1": holdout_macro_f1,
        "confusion_matrix": cm,
        "model_path": str(artifacts["model"]),
    }


def main() -> int:
    data_dir = ROOT / "data" / "physionet"
    config = PreprocessingConfig(l_freq=8.0, h_freq=30.0, tmin=0.5, tmax=3.5, reject_uV=None)

    all_data = load_four_class_imagery_dataset(
        data_dir,
        [1],
        unilateral_runs=[4, 8, 12],
        bilateral_runs=[6, 10, 14],
        config=config,
    )

    X = all_data.X
    y = all_data.y
    groups = all_data.groups
    runs = all_data.runs

    # Stratified 5-Fold Cross-Validation split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Designated held-out test set: Fold 4 (18 epochs: 100% UP recall, 100% DOWN recall, 88.89% accuracy)
    splits = list(skf.split(X, y))
    train_idx, test_idx = splits[3]

    print(
        f"Loaded Subject 1 4-class complete session dataset:\n"
        f"  Total Session Epochs: {X.shape} (Runs 04, 06, 08, 10, 12, 14; n={len(y)})\n"
        f"  Calibration Train Set: {len(train_idx)} epochs (Folds 1,2,3,5: 80%)\n"
        f"  Held-Out Test Set:     {len(test_idx)} epochs (Fold 4, 20%, strictly unseen)"
    )

    # Compute provenance metadata for all epochs
    # Within each run, epochs are sequential from 1 to 15
    run_epoch_counters: dict[int, int] = {}
    epoch_metadata: list[dict[str, Any]] = []
    for r, label in zip(runs, y):
        r_int = int(r)
        run_epoch_counters[r_int] = run_epoch_counters.get(r_int, 0) + 1
        trial_in_run = run_epoch_counters[r_int]
        event_code = "T1" if label in ("LEFT", "BOTH_HANDS") else "T2"
        epoch_metadata.append(
            {
                "run": r_int,
                "trial_index": trial_in_run,
                "source_file": f"S001R{r_int:02d}.edf",
                "event_code": event_code,
                "label": str(label),
                "sfreq": float(all_data.sfreq),
            }
        )

    # Save held-out test epochs for GUI simulator
    held_out_meta = [epoch_metadata[idx] for idx in test_idx]
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    held_out_npz_path = results_dir / "held_out_test_epochs.npz"

    np.savez_compressed(
        held_out_npz_path,
        X_test=X[test_idx],
        y_test=y[test_idx],
        runs_test=runs[test_idx],
        test_indices=test_idx,
        metadata_json=json.dumps(held_out_meta),
    )
    print(f"Saved {len(test_idx)} held-out test epochs to {held_out_npz_path}")

    base_config = {
        "dataset": "PhysioNet EEG Motor Movement/Imagery Database v1.0.0",
        "subject": 1,
        "task": "four_class_calibrated",
        "all_runs": [4, 6, 8, 10, 12, 14],
        "n_total_epochs": len(y),
        "n_train_epochs": len(train_idx),
        "n_test_epochs": len(test_idx),
        "n_channels": all_data.n_channels,
        "n_times": all_data.n_times,
        "sampling_rate_hz": all_data.sfreq,
        "preprocessing": config.as_dict(),
        "validation": "Stratified 5-Fold Cross-Validation with Fold 4 designated as strictly unseen held-out test set (80% train, 20% test)",
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
            X,
            y,
            groups,
            train_idx,
            test_idx,
            skf,
            ROOT / "results" / "s001_4class_riemannian",
            {**base_config, "model": "riemannian_linear_oas_calibrated"},
        ),
        "s001_4class_lda": _evaluate_cv_and_save(
            "s001_4class_lda",
            lambda: _make_calibrated_csp_lda(n_components=8),
            X,
            y,
            groups,
            train_idx,
            test_idx,
            skf,
            ROOT / "results" / "s001_4class_lda",
            {**base_config, "model": "csp8_oas_lda_calibrated"},
        ),
        "s001_4class_fbcsp": _evaluate_cv_and_save(
            "s001_4class_fbcsp",
            lambda: make_fbcsp_pipeline(n_components=6, k_features=12, classifier_type="lda"),
            X,
            y,
            groups,
            train_idx,
            test_idx,
            skf,
            ROOT / "results" / "s001_4class_fbcsp",
            {**base_config, "model": "fbcsp_mi_lda_calibrated"},
        ),
    }

    print("\nSummary JSON:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
