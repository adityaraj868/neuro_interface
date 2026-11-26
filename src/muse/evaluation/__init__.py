"""Trial-grouped and cross-subject evaluation and artifact generation."""

from .loso import LOSOResult, evaluate_loso
from .metrics import EvaluationResult, evaluate_grouped, save_evaluation_artifacts

__all__ = [
    "EvaluationResult",
    "LOSOResult",
    "evaluate_grouped",
    "evaluate_loso",
    "save_evaluation_artifacts",
]
