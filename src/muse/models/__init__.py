"""EEG decoding pipelines (Classical, Riemannian, Deep Learning)."""

from .classical import make_csp_lda_pipeline, make_csp_svm_pipeline
from .eegnet import EEGNet, EEGNetClassifier, make_eegnet_classifier
from .riemannian import make_riemannian_ridge_pipeline, make_riemannian_svm_pipeline

__all__ = [
    "make_csp_lda_pipeline",
    "make_csp_svm_pipeline",
    "make_riemannian_svm_pipeline",
    "make_riemannian_ridge_pipeline",
    "EEGNet",
    "EEGNetClassifier",
    "make_eegnet_classifier",
]
