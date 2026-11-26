"""Euclidean Alignment (EA) for cross-subject EEG transfer and normalization."""

from __future__ import annotations

import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.base import BaseEstimator, TransformerMixin


def compute_reference_matrix(X: np.ndarray) -> np.ndarray:
    """Compute the mean covariance matrix R across trials for a subject.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_trials, n_channels, n_times).

    Returns
    -------
    np.ndarray
        Reference covariance matrix of shape (n_channels, n_channels).
    """
    n_trials, n_channels, n_times = X.shape
    # Center each channel per trial
    centered = X - np.mean(X, axis=-1, keepdims=True)
    # Compute covariance per trial: (X @ X.T) / (n_times - 1)
    covs = np.matmul(centered, np.swapaxes(centered, -1, -2)) / (n_times - 1)
    # Arithmetic mean across trials
    r_mean = np.mean(covs, axis=0)
    # Regularize slightly for numerical stability
    r_reg = r_mean + 1e-6 * np.trace(r_mean) * np.eye(n_channels) / n_channels
    return r_reg


def inv_sqrt_matrix(R: np.ndarray) -> np.ndarray:
    """Compute R^(-1/2) via eigenvalue decomposition with eigenvalue clamping."""
    U, s, Vt = np.linalg.svd(R, hermitian=True)
    s_clamped = np.maximum(s, 1e-10)
    inv_sqrt_s = 1.0 / np.sqrt(s_clamped)
    return (U * inv_sqrt_s) @ Vt


def euclidean_alignment(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Align multi-subject EEG trials using Euclidean Alignment (EA).

    For each subject s:
        R_s = mean_{i in subject s} (Cov(X_i))
        X_tilde_i = R_s^(-1/2) @ X_i

    Parameters
    ----------
    X : np.ndarray
        EEG epochs of shape (n_trials, n_channels, n_times).
    subjects : np.ndarray
        Subject identifiers of shape (n_trials,).

    Returns
    -------
    np.ndarray
        Aligned EEG epochs of shape (n_trials, n_channels, n_times).
    """
    X_aligned = np.zeros_like(X)
    unique_subs = np.unique(subjects)

    for sub in unique_subs:
        mask = subjects == sub
        sub_X = X[mask]
        R_sub = compute_reference_matrix(sub_X)
        R_inv_sqrt = inv_sqrt_matrix(R_sub)
        # Apply transformation: X_tilde = R^(-1/2) @ X
        # R_inv_sqrt: (C, C), sub_X: (N, C, T) -> (N, C, T)
        X_aligned[mask] = np.matmul(R_inv_sqrt, sub_X)

    return X_aligned


class EuclideanAligner(BaseEstimator, TransformerMixin):
    """scikit-learn compatible transformer for subject-wise Euclidean Alignment."""

    def __init__(self) -> None:
        self.reference_matrices_: dict[str, np.ndarray] = {}
        self.global_ref_inv_sqrt_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, subjects: np.ndarray | None = None) -> EuclideanAligner:
        if X.ndim != 3:
            raise ValueError(f"Expected 3D EEG array (trials, channels, times), got {X.shape}")

        R_global = compute_reference_matrix(X)
        self.global_ref_inv_sqrt_ = inv_sqrt_matrix(R_global)

        if subjects is not None:
            for sub in np.unique(subjects):
                mask = subjects == sub
                R_sub = compute_reference_matrix(X[mask])
                self.reference_matrices_[str(sub)] = inv_sqrt_matrix(R_sub)

        return self

    def transform(self, X: np.ndarray, subjects: np.ndarray | None = None) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"Expected 3D EEG array (trials, channels, times), got {X.shape}")

        if subjects is not None:
            return euclidean_alignment(X, subjects)

        # Unsupervised trial-level or batch alignment using global reference
        if self.global_ref_inv_sqrt_ is not None:
            return np.matmul(self.global_ref_inv_sqrt_, X)

        R_batch = compute_reference_matrix(X)
        return np.matmul(inv_sqrt_matrix(R_batch), X)
