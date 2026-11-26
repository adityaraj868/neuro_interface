#!/usr/bin/env python3
"""Diagnostic benchmark script comparing Baseline vs Refactored MUSE EEG pipelines.

Benchmarks:
1. 2-Class (Left vs Right Hand) on S001
2. 4-Class Within-Subject (Left, Right, Both Hands, Both Feet) on S001 (5-fold Stratified CV)
3. 4-Class Cross-Subject (LOSO across S001, S002, S003)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mne.decoding import CSP
import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from muse.data import load_four_class_imagery_dataset, load_imagery_dataset
from muse.evaluation import evaluate_grouped, evaluate_loso
from muse.models import (
    EuclideanAligner,
    euclidean_alignment,
    make_csp_lda_pipeline,
    make_fbcsp_pipeline,
    make_riemannian_svm_pipeline,
)
from muse.preprocessing import PreprocessingConfig


def run_stratified_cv(estimator: Any, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict[str, Any]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    classes = sorted(list(np.unique(y)))
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in skf.split(X, y):
        est = clone(estimator)
        est.fit(X[train_idx], y[train_idx])
        preds = est.predict(X[test_idx])
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(preds)

    y_true_arr = np.asarray(y_true_all)
    y_pred_arr = np.asarray(y_pred_all)
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    bacc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes).astype(int).tolist()

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": f1,
        "classes": classes,
        "confusion_matrix": cm,
    }


def evaluate_benchmark_suite(
    data_dir: Path,
    config: PreprocessingConfig,
    use_ea: bool = False,
    tag: str = "pipeline",
) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUITE: [{tag.upper()}]")
    print(f"Config: tmin={config.tmin}s, tmax={config.tmax}s, channels={config.channels}, reject_uV={config.reject_uV}")
    print(f"{'='*60}")

    suite: dict[str, Any] = {"tag": tag, "config": config.as_dict()}

    # 1. 2-Class S001 (unilateral runs 4, 8, 12)
    print("\n--- [1/3] 2-Class (Left vs Right Hand) on S001 ---")
    ds_2c = load_imagery_dataset(data_dir, subjects=[1], runs=[4, 8, 12], config=config)
    print(f"Loaded {ds_2c.n_epochs} epochs, shape={ds_2c.X.shape}")

    res_2c: dict[str, Any] = {}
    csp_lda_2c = make_csp_lda_pipeline(n_components=min(4, ds_2c.n_channels))
    eval_csp = evaluate_grouped(csp_lda_2c, ds_2c.X, ds_2c.y, ds_2c.groups)
    res_2c["csp_lda"] = {
        "accuracy": eval_csp.accuracy,
        "balanced_accuracy": eval_csp.balanced_accuracy,
        "macro_f1": eval_csp.macro_f1,
    }
    print(f"  Standard CSP+LDA:    Acc={eval_csp.accuracy:.2%}, BAcc={eval_csp.balanced_accuracy:.2%}, F1={eval_csp.macro_f1:.2%}")

    riem_2c = make_riemannian_svm_pipeline(estimator="oas", C=1.0, probability=True)
    eval_riem = evaluate_grouped(riem_2c, ds_2c.X, ds_2c.y, ds_2c.groups)
    res_2c["riemannian"] = {
        "accuracy": eval_riem.accuracy,
        "balanced_accuracy": eval_riem.balanced_accuracy,
        "macro_f1": eval_riem.macro_f1,
    }
    print(f"  Riemannian SVM:      Acc={eval_riem.accuracy:.2%}, BAcc={eval_riem.balanced_accuracy:.2%}, F1={eval_riem.macro_f1:.2%}")

    fbcsp_2c = make_fbcsp_pipeline(n_components=min(4, ds_2c.n_channels), k_features=8, classifier_type="lda")
    eval_fbcsp = evaluate_grouped(fbcsp_2c, ds_2c.X, ds_2c.y, ds_2c.groups)
    res_2c["fbcsp_lda"] = {
        "accuracy": eval_fbcsp.accuracy,
        "balanced_accuracy": eval_fbcsp.balanced_accuracy,
        "macro_f1": eval_fbcsp.macro_f1,
    }
    print(f"  Filter Bank CSP+LDA: Acc={eval_fbcsp.accuracy:.2%}, BAcc={eval_fbcsp.balanced_accuracy:.2%}, F1={eval_fbcsp.macro_f1:.2%}")
    suite["2_class_s001"] = res_2c

    # 2. 4-Class Within-Subject S001 (5-Fold Stratified CV)
    print("\n--- [2/3] 4-Class Within-Subject on S001 (5-Fold CV) ---")
    ds_4c = load_four_class_imagery_dataset(data_dir, subjects=[1], config=config)
    print(f"Loaded {ds_4c.n_epochs} epochs, classes={sorted(list(np.unique(ds_4c.y)))}")

    res_4c: dict[str, Any] = {}
    csp_4c = Pipeline([
        ("csp", CSP(n_components=min(8, ds_4c.n_channels), reg="oas", log=True, norm_trace=True, transform_into="average_power")),
        ("classifier", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])
    cv_csp = run_stratified_cv(csp_4c, ds_4c.X, ds_4c.y, n_splits=5)
    res_4c["csp_lda"] = cv_csp
    print(f"  Standard CSP+LDA:    Acc={cv_csp['accuracy']:.2%}, BAcc={cv_csp['balanced_accuracy']:.2%}, F1={cv_csp['macro_f1']:.2%}")

    riem_4c = make_riemannian_svm_pipeline(estimator="oas", C=1.0, probability=True)
    cv_riem = run_stratified_cv(riem_4c, ds_4c.X, ds_4c.y, n_splits=5)
    res_4c["riemannian"] = cv_riem
    print(f"  Riemannian SVM:      Acc={cv_riem['accuracy']:.2%}, BAcc={cv_riem['balanced_accuracy']:.2%}, F1={cv_riem['macro_f1']:.2%}")

    fbcsp_4c = make_fbcsp_pipeline(n_components=min(6, ds_4c.n_channels), k_features=12, classifier_type="lda")
    cv_fbcsp = run_stratified_cv(fbcsp_4c, ds_4c.X, ds_4c.y, n_splits=5)
    res_4c["fbcsp_lda"] = cv_fbcsp
    print(f"  Filter Bank CSP+LDA: Acc={cv_fbcsp['accuracy']:.2%}, BAcc={cv_fbcsp['balanced_accuracy']:.2%}, F1={cv_fbcsp['macro_f1']:.2%}")
    suite["4_class_within_subject_s001"] = res_4c

    # 3. 4-Class Cross-Subject LOSO (S001, S002, S003)
    print("\n--- [3/3] 4-Class Cross-Subject LOSO (S001, S002, S003) ---")
    ds_multi = load_four_class_imagery_dataset(data_dir, subjects=[1, 2, 3], config=config)
    print(f"Loaded {ds_multi.n_epochs} epochs across subjects {sorted(list(set(ds_multi.subjects)))}")

    X_data = ds_multi.X
    if use_ea:
        print("  Applying Subject-wise Euclidean Alignment (EA)...")
        X_data = euclidean_alignment(ds_multi.X, ds_multi.subjects)

    res_loso: dict[str, Any] = {}
    riem_multi = make_riemannian_svm_pipeline(estimator="oas", C=1.0, probability=True)
    loso_riem = evaluate_loso(riem_multi, X_data, ds_multi.y, ds_multi.subjects)
    res_loso["riemannian"] = {
        "accuracy": loso_riem.overall.accuracy,
        "balanced_accuracy": loso_riem.overall.balanced_accuracy,
        "macro_f1": loso_riem.overall.macro_f1,
        "mean_balanced_accuracy": loso_riem.mean_balanced_accuracy,
        "std_balanced_accuracy": loso_riem.std_balanced_accuracy,
        "per_subject": loso_riem.subject_scores,
    }
    print(f"  Riemannian {'+ EA' if use_ea else 'Raw'}: Acc={loso_riem.overall.accuracy:.2%}, Mean BAcc={loso_riem.mean_balanced_accuracy:.2%}")

    fbcsp_multi = make_fbcsp_pipeline(n_components=min(6, ds_multi.n_channels), k_features=12, classifier_type="lda")
    loso_fbcsp = evaluate_loso(fbcsp_multi, X_data, ds_multi.y, ds_multi.subjects)
    res_loso["fbcsp_lda"] = {
        "accuracy": loso_fbcsp.overall.accuracy,
        "balanced_accuracy": loso_fbcsp.overall.balanced_accuracy,
        "macro_f1": loso_fbcsp.overall.macro_f1,
        "mean_balanced_accuracy": loso_fbcsp.mean_balanced_accuracy,
        "std_balanced_accuracy": loso_fbcsp.std_balanced_accuracy,
        "per_subject": loso_fbcsp.subject_scores,
    }
    print(f"  FBCSP+LDA  {'+ EA' if use_ea else 'Raw'}: Acc={loso_fbcsp.overall.accuracy:.2%}, Mean BAcc={loso_fbcsp.mean_balanced_accuracy:.2%}")

    suite["4_class_cross_subject_loso"] = res_loso
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "physionet")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "diagnostic_benchmark_comparison.json")
    args = parser.parse_args()

    # 1. Baseline: full 64 channels, tmin=1.0, tmax=4.0, reject=None for sensitivity
    baseline_cfg = PreprocessingConfig(tmin=1.0, tmax=4.0, channels="full", reject_uV=None)
    baseline_res = evaluate_benchmark_suite(args.data_dir, baseline_cfg, use_ea=False, tag="baseline")

    # 2. Refactored: sensorimotor strip (21 channels), tmin=0.5, tmax=3.5 (excludes VEP), reject=None
    refactored_cfg = PreprocessingConfig(tmin=0.5, tmax=3.5, channels="sensorimotor", reject_uV=None)
    refactored_res = evaluate_benchmark_suite(args.data_dir, refactored_cfg, use_ea=True, tag="refactored_with_ea")

    comparison = {
        "baseline": baseline_res,
        "refactored": refactored_res,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2))
    print(f"\nSaved benchmark comparison results to {args.output}")


if __name__ == "__main__":
    main()
