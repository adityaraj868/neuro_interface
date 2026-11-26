"""Filter Bank Common Spatial Pattern (FBCSP) feature extraction and pipelines."""

from __future__ import annotations

from typing import Sequence

from mne.decoding import CSP
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

DEFAULT_SUB_BANDS: tuple[tuple[float, float], ...] = (
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 26.0),
    (26.0, 30.0),
)


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Filter Bank Common Spatial Pattern (FBCSP) feature transformer.

    Splits multi-channel EEG into frequency sub-bands, applies spatial filtering
    (CSP) within each sub-band, and extracts log-variance power features.

    Parameters
    ----------
    sub_bands : sequence of (float, float), default=((8, 12), (12, 16), (16, 20), (20, 26), (26, 30))
        List of (low_freq, high_freq) bandpass cutoffs.
    n_components : int, default=4
        Number of CSP spatial filters per sub-band.
    sfreq : float, default=160.0
        Sampling frequency of the EEG signals.
    reg : str or float, default='oas'
        Covariance regularization method for CSP ('oas', 'ledoit_wolf', None, or float).
    """

    def __init__(
        self,
        sub_bands: Sequence[tuple[float, float]] = DEFAULT_SUB_BANDS,
        n_components: int = 4,
        sfreq: float = 160.0,
        reg: str | float | None = "oas",
    ) -> None:
        self.sub_bands = sub_bands
        self.n_components = n_components
        self.sfreq = sfreq
        self.reg = reg
        self.csps_: list[CSP] = []

    def _filter_band(self, X: np.ndarray, l_freq: float, h_freq: float) -> np.ndarray:
        nyq = 0.5 * self.sfreq
        low = max(0.1, l_freq) / nyq
        high = min(nyq - 0.5, h_freq) / nyq
        sos = butter(4, [low, high], btype="bandpass", output="sos")
        # X: (trials, channels, times) -> sosfiltfilt along axis=-1
        return sosfiltfilt(sos, X, axis=-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> FilterBankCSP:
        if X.ndim != 3:
            raise ValueError(f"Expected 3D EEG array (n_trials, n_channels, n_times), got {X.shape}")

        self.csps_ = []
        for l_freq, h_freq in self.sub_bands:
            X_band = self._filter_band(X, l_freq, h_freq)
            csp = CSP(
                n_components=self.n_components,
                reg=self.reg,
                log=True,
                norm_trace=True,
                transform_into="average_power",
            )
            csp.fit(X_band, y)
            self.csps_.append(csp)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.csps_:
            raise RuntimeError("FilterBankCSP must be fitted before transforming data.")

        band_features: list[np.ndarray] = []
        for (l_freq, h_freq), csp in zip(self.sub_bands, self.csps_):
            X_band = self._filter_band(X, l_freq, h_freq)
            feat = csp.transform(X_band)
            band_features.append(feat)

        # Concatenate features across all sub-bands: (n_trials, n_bands * n_components)
        return np.concatenate(band_features, axis=-1)


def _mutual_info_scoring(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return mutual_info_classif(X, y, random_state=42)


def make_fbcsp_pipeline(
    n_components: int = 4,
    k_features: int = 8,
    classifier_type: str = "lda",
    sub_bands: Sequence[tuple[float, float]] = DEFAULT_SUB_BANDS,
    sfreq: float = 160.0,
    reg: str | float | None = "oas",
    random_state: int = 42,
) -> Pipeline:
    """Build an end-to-end FBCSP pipeline with Mutual Information selection.

    Parameters
    ----------
    n_components : int, default=4
        Number of CSP spatial filters per sub-band.
    k_features : int, default=8
        Number of top features selected via Mutual Information.
    classifier_type : str, default='lda'
        'lda' for Shrinkage LDA or 'svm' for Linear SVM.
    sub_bands : sequence of (float, float)
        Frequency sub-bands for the filter bank.
    sfreq : float, default=160.0
        Sampling frequency.
    reg : str or float, default='oas'
        Covariance regularization for CSP.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    Pipeline
        scikit-learn Pipeline with FilterBankCSP -> SelectKBest -> Classifier.
    """
    total_features = len(sub_bands) * n_components
    k = min(k_features, total_features)

    fbcsp = FilterBankCSP(
        sub_bands=sub_bands,
        n_components=n_components,
        sfreq=sfreq,
        reg=reg,
    )

    selector = SelectKBest(score_func=_mutual_info_scoring, k=k)

    if classifier_type == "lda":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif classifier_type == "svm":
        clf = SVC(kernel="linear", C=1.0, probability=True, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier_type '{classifier_type}'; choose 'lda' or 'svm'.")

    return Pipeline([
        ("fbcsp", fbcsp),
        ("selector", selector),
        ("classifier", clf),
    ])
