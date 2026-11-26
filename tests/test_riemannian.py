import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from muse.evaluation import evaluate_grouped
from muse.models import make_riemannian_ridge_pipeline, make_riemannian_svm_pipeline
from tests.test_models_evaluation import synthetic_epochs


class RiemannianModelTests(unittest.TestCase):
    def test_riemannian_svm_fits_and_predicts(self):
        X, y, groups = synthetic_epochs(seed=21)
        estimator = make_riemannian_svm_pipeline(estimator="oas", C=1.0, probability=True)
        estimator.fit(X, y)

        preds = estimator.predict(X[:6])
        self.assertEqual(preds.shape, (6,))

        # Verify .predict_proba contract for BCIInferenceEngine
        probs = estimator.predict_proba(X[:6])
        self.assertEqual(probs.shape, (6, 2))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

        # Verify grouped evaluation
        result = evaluate_grouped(
            make_riemannian_svm_pipeline(estimator="oas", C=1.0, probability=True),
            X,
            y,
            groups,
        )
        self.assertEqual(result.n_groups, 3)
        self.assertIn("balanced_accuracy", result.as_dict())
        self.assertEqual(sum(map(sum, result.confusion_matrix)), len(y))

    def test_riemannian_ridge_fits_and_predicts(self):
        X, y, _ = synthetic_epochs(seed=22)
        estimator = make_riemannian_ridge_pipeline(estimator="oas", alpha=1.0)
        estimator.fit(X, y)
        preds = estimator.predict(X[:5])
        self.assertEqual(preds.shape, (5,))

    def test_model_save_load_consistency(self):
        X, y, _ = synthetic_epochs(seed=23)
        estimator = make_riemannian_svm_pipeline(estimator="oas", C=1.0, probability=True).fit(X, y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "riemannian_model.joblib"
            joblib.dump(estimator, path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(estimator.predict(X[:6]), loaded.predict(X[:6]))
            np.testing.assert_allclose(
                estimator.predict_proba(X[:6]),
                loaded.predict_proba(X[:6]),
                rtol=1e-5,
            )

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            make_riemannian_svm_pipeline(estimator="invalid_estimator")
        with self.assertRaises(ValueError):
            make_riemannian_svm_pipeline(C=-1.0)
        with self.assertRaises(ValueError):
            make_riemannian_ridge_pipeline(alpha=0.0)
