"""Riemannian Geometry pipelines for epoched motor-imagery EEG."""

from __future__ import annotations

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SUPPORTED_ESTIMATORS = ("oas", "lwf", "corr", "cov")


def make_riemannian_svm_pipeline(
    estimator: str = "oas",
    C: float = 1.0,
    metric: str = "riemann",
    probability: bool = True,
    random_state: int = 42,
    kernel: str = "linear",
    scale: bool = False,
    class_weight: str | dict | None = None,
) -> Pipeline:
    """Create a Riemannian Tangent Space + SVM classifier pipeline.

    Parameters
    ----------
    estimator : str, default='oas'
        Covariance estimator ('oas', 'lwf', 'corr', 'cov').
    C : float, default=1.0
        Regularization parameter for the SVM.
    metric : str, default='riemann'
        Riemannian metric for tangent space projection.
    probability : bool, default=True
        Whether to enable calibrated probability estimates (predict_proba).
    random_state : int, default=42
        Seed for reproducibility.
    kernel : str, default='linear'
        SVM kernel ('linear', 'rbf', etc.).
    scale : bool, default=False
        Whether to standardize tangent space features before classification.
    class_weight : str, dict, or None, default=None
        Optional class weighting passed to SVC.

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

    base_svc = SVC(
        kernel=kernel,
        C=C,
        class_weight=class_weight,
        random_state=random_state,
    )
    if probability:
        classifier = CalibratedClassifierCV(base_svc, ensemble=False)
    else:
        classifier = base_svc

    steps: list[tuple[str, object]] = [
        ("cov", Covariances(estimator=estimator)),
        ("tangentspace", TangentSpace(metric=metric)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


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
