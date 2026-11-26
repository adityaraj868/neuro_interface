"""Leave-One-Subject-Out (LOSO) cross-validation and domain shift analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .metrics import EvaluationResult, evaluate_grouped


@dataclass
class LOSOResult:
    """Detailed Leave-One-Subject-Out generalization metrics."""

    overall: EvaluationResult
    subject_scores: dict[str, dict[str, float]]
    mean_balanced_accuracy: float
    std_balanced_accuracy: float
    domain_shift_gap: float | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["overall"] = self.overall.as_dict()
        return data


def evaluate_loso(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    within_subject_score: float | None = None,
) -> LOSOResult:
    """Evaluate a model using Leave-One-Subject-Out (LOSO) cross-validation.

    Parameters
    ----------
    estimator : scikit-learn compatible estimator or Pipeline
        Model to evaluate.
    X : np.ndarray
        Epoch data array of shape (n_epochs, n_channels, n_times).
    y : np.ndarray
        Labels array of shape (n_epochs,).
    subjects : np.ndarray
        Subject identifier array of shape (n_epochs,).
    within_subject_score : float, optional
        Precomputed within-subject baseline score (balanced accuracy) to compute
        the domain shift gap Delta = Score_within - Score_loso.

    Returns
    -------
    LOSOResult
        Aggregated LOSO metrics including per-subject scores and domain shift gap.
    """
    subjects_arr = np.asarray([f"sub-{s}" for s in subjects])
    overall = evaluate_grouped(estimator, X, y, subjects_arr)

    # Calculate per-subject balanced accuracy and accuracy
    subject_scores: dict[str, dict[str, float]] = {}
    bacc_list = []

    for fold in overall.fold_results:
        sub_name = fold["held_out_groups"][0] if fold.get("held_out_groups") else f"fold_{fold['fold']}"
        bacc = float(fold["balanced_accuracy"])
        acc = float(fold["accuracy"])
        subject_scores[sub_name] = {"balanced_accuracy": bacc, "accuracy": acc}
        bacc_list.append(bacc)

    mean_bacc = float(np.mean(bacc_list)) if bacc_list else 0.0
    std_bacc = float(np.std(bacc_list)) if bacc_list else 0.0

    domain_gap = None
    if within_subject_score is not None:
        domain_gap = float(within_subject_score - overall.balanced_accuracy)

    return LOSOResult(
        overall=overall,
        subject_scores=subject_scores,
        mean_balanced_accuracy=mean_bacc,
        std_balanced_accuracy=std_bacc,
        domain_shift_gap=domain_gap,
    )
