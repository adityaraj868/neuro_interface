"""CSP-based classical models for epoched motor-imagery EEG."""

from __future__ import annotations

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def _csp(n_components: int) -> CSP:
    if n_components < 2:
        raise ValueError("CSP requires at least two spatial components.")
    return CSP(
        n_components=n_components,
        reg=None,
        log=True,
        norm_trace=False,
        transform_into="average_power",
    )


def make_csp_lda_pipeline(n_components: int = 4) -> Pipeline:
    """Create the MUSE baseline: CSP features followed by shrinkage LDA."""

    return Pipeline(
        [
            ("csp", _csp(n_components)),
            ("classifier", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def make_csp_svm_pipeline(n_components: int = 4, C: float = 1.0) -> Pipeline:
    """Create a linear-SVM comparator using the same fold-fitted CSP features."""

    if C <= 0:
        raise ValueError("SVM C must be positive.")
    return Pipeline(
        [
            ("csp", _csp(n_components)),
            ("classifier", SVC(kernel="linear", C=C)),
        ]
    )

