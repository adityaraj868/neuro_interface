"""Metrics for leakage-aware trial/group held-out evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut


@dataclass
class EvaluationResult:
    """Aggregate and fold-level results from grouped held-out evaluation."""

    classes: list[str]
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    chance_level: float
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]
    n_epochs: int
    n_groups: int
    class_counts: dict[str, int]
    fold_results: list[dict[str, Any]]
    predictions: list[str]
    truth: list[str]
    prediction_groups: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_grouped(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> EvaluationResult:
    """Evaluate with Leave-One-Group-Out, fitting CSP inside each fold."""

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (epochs, channels, time), got {X.shape}.")
    if len(y) != len(groups) or len(y) != X.shape[0]:
        raise ValueError("X, y, and groups must contain the same number of epochs.")
    classes = sorted(str(value) for value in np.unique(y))
    if len(classes) < 2:
        raise ValueError(f"Grouped evaluation requires at least two classes, got {classes}.")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(
            "Grouped evaluation requires at least two recording groups; "
            "provide at least two runs or use a separate exploratory split."
        )

    logo = LeaveOneGroupOut()
    truth: list[str] = []
    predictions: list[str] = []
    prediction_groups: list[str] = []
    fold_results: list[dict[str, Any]] = []
    for fold_index, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        train_classes = set(str(value) for value in np.unique(y[train_idx]))
        if set(classes) != train_classes:
            raise ValueError(
                f"Fold {fold_index} training data does not contain every class: "
                f"expected {classes}, got {sorted(train_classes)}."
            )
        fold_estimator = clone(estimator)
        fold_estimator.fit(X[train_idx], y[train_idx])
        fold_prediction = np.asarray(fold_estimator.predict(X[test_idx]), dtype=str)
        fold_truth = np.asarray(y[test_idx], dtype=str)
        truth.extend(fold_truth.tolist())
        predictions.extend(fold_prediction.tolist())
        prediction_groups.extend(np.asarray(groups[test_idx], dtype=str).tolist())
        fold_results.append(
            {
                "fold": fold_index,
                "held_out_groups": sorted(set(str(value) for value in groups[test_idx])),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(fold_truth, fold_prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(fold_truth, fold_prediction)),
                "macro_f1": float(
                    f1_score(fold_truth, fold_prediction, labels=classes, average="macro", zero_division=0)
                ),
            }
        )

    truth_array = np.asarray(truth, dtype=str)
    prediction_array = np.asarray(predictions, dtype=str)
    class_counts = {label: int(np.sum(np.asarray(y, dtype=str) == label)) for label in classes}
    return EvaluationResult(
        classes=classes,
        accuracy=float(accuracy_score(truth_array, prediction_array)),
        balanced_accuracy=float(balanced_accuracy_score(truth_array, prediction_array)),
        macro_f1=float(f1_score(truth_array, prediction_array, labels=classes, average="macro", zero_division=0)),
        chance_level=float(1.0 / len(classes)),
        confusion_matrix=confusion_matrix(truth_array, prediction_array, labels=classes).astype(int).tolist(),
        classification_report=classification_report(
            truth_array,
            prediction_array,
            labels=classes,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        n_epochs=int(len(y)),
        n_groups=int(len(unique_groups)),
        class_counts=class_counts,
        fold_results=fold_results,
        predictions=predictions,
        truth=truth,
        prediction_groups=prediction_groups,
    )


def save_evaluation_artifacts(
    result: EvaluationResult,
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    estimator: Any | None = None,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> dict[str, Path]:
    """Write metrics, predictions, and a confusion matrix to an experiment dir."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "config.json"
    predictions_path = output_dir / "predictions.csv"
    confusion_path = output_dir / "confusion_matrix.png"
    metrics_path.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True, default=str) + "\n")
    with predictions_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "true", "predicted"])
        writer.writerows(zip(result.prediction_groups, result.truth, result.predictions))

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(result.confusion_matrix),
        display_labels=result.classes,
    ).plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Grouped held-out confusion matrix")
    figure.tight_layout()
    figure.savefig(confusion_path, dpi=160)
    plt.close(figure)

    artifacts = {
        "metrics": metrics_path,
        "config": config_path,
        "predictions": predictions_path,
        "confusion_matrix": confusion_path,
    }
    if estimator is not None and X is not None and y is not None:
        import joblib

        estimator.fit(X, y)
        model_path = output_dir / "model.joblib"
        joblib.dump(estimator, model_path)
        artifacts["model"] = model_path
    return artifacts

