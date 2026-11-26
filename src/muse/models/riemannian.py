"""Riemannian Geometry pipelines for epoched motor-imagery EEG."""

from __future__ import annotations

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

SUPPORTED_ESTIMATORS = ("oas", "lwf", "corr", "cov")


def make_riemannian_svm_pipeline(
    estimator: str = "oas",
    C: float = 1.0,
    metric: str = "riemann",
    probability: bool = True,
    random_state: int = 42,
) -> Pipeline:
    """Create a Riemannian Tangent Space + Linear SVM classifier pipeline.

    Parameters
    ----------
    estimator : str, default='oas'
        Covariance estimator ('oas', 'lwf', 'corr', 'cov').
    C : float, default=1.0
        Regularization parameter for the linear SVM.
    metric : str, default='riemann'
        Riemannian metric for tangent space projection.
    probability : bool, default=True
        Whether to enable calibrated probability estimates (predict_proba).
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Pipeline
        scikit-learn Pipeline with Covariances -> TangentSpace -> Classifier.
    """
    if estimator not in SUPPORTED_ESTIMATORS:
        raise ValueError(
            f"Unsupported covariance estimator '{estimator}'. "
            f"Expected one of {SUPPORTED_ESTIMATORS}."
        )
    if C <= 0:
        raise ValueError("SVM C must be positive.")

    base_svc = SVC(kernel="linear", C=C, random_state=random_state)
    if probability:
        classifier = CalibratedClassifierCV(base_svc, ensemble=False)
    else:
        classifier = base_svc

    return Pipeline(
        [
            ("cov", Covariances(estimator=estimator)),
            ("tangentspace", TangentSpace(metric=metric)),
            ("classifier", classifier),
        ]
    )


def make_riemannian_ridge_pipeline(
    estimator: str = "oas",
    alpha: float = 1.0,
    metric: str = "riemann",
) -> Pipeline:
    """Create a Riemannian Tangent Space + Ridge Classifier pipeline."""
    if estimator not in SUPPORTED_ESTIMATORS:
        raise ValueError(
            f"Unsupported covariance estimator '{estimator}'. "
            f"Expected one of {SUPPORTED_ESTIMATORS}."
        )
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive.")

    return Pipeline(
        [
            ("cov", Covariances(estimator=estimator)),
            ("tangentspace", TangentSpace(metric=metric)),
            ("classifier", RidgeClassifier(alpha=alpha)),
        ]
    )
