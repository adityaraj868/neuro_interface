"""Classical EEG decoding pipelines."""

from .classical import make_csp_lda_pipeline, make_csp_svm_pipeline

__all__ = ["make_csp_lda_pipeline", "make_csp_svm_pipeline"]

