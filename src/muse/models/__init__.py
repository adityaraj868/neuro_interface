"""EEG decoding pipelines (Classical, Riemannian, Deep Learning)."""

from .alignment import EuclideanAligner, euclidean_alignment
from .classical import make_csp_lda_pipeline, make_csp_svm_pipeline
from .eegnet import EEGNet, EEGNetClassifier, make_eegnet_classifier
from .fbcsp import FilterBankCSP, make_fbcsp_pipeline
from .riemannian import make_riemannian_ridge_pipeline, make_riemannian_svm_pipeline

__all__ = [
    "make_csp_lda_pipeline",
    "make_csp_svm_pipeline",
    "make_riemannian_svm_pipeline",
    "make_riemannian_ridge_pipeline",
    "FilterBankCSP",
    "make_fbcsp_pipeline",
    "EuclideanAligner",
    "euclidean_alignment",
    "EEGNet",
    "EEGNetClassifier",
    "make_eegnet_classifier",
]
