import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from muse.evaluation import evaluate_grouped
from muse.models import make_csp_lda_pipeline


def synthetic_epochs(seed: int = 11):
    rng = np.random.default_rng(seed)
    n_per_group = 8
    X_parts = []
    y_parts = []
    groups = []
    for group_index in range(3):
        for label, sign in (("LEFT", 1.0), ("RIGHT", -1.0)):
            X_parts.append(rng.normal(size=(n_per_group, 8, 80)) * 0.05 + sign * np.eye(8, 1)[:, :, None])
            y_parts.extend([label] * n_per_group)
            groups.extend([f"run-{group_index}"] * n_per_group)
    return np.concatenate(X_parts), np.asarray(y_parts), np.asarray(groups)


class ModelEvaluationTests(unittest.TestCase):
    def test_pipeline_fits_and_predicts(self):
        X, y, groups = synthetic_epochs()
        estimator = make_csp_lda_pipeline(n_components=4)
        estimator.fit(X, y)
        self.assertEqual(estimator.predict(X[:5]).shape, (5,))
        result = evaluate_grouped(make_csp_lda_pipeline(n_components=4), X, y, groups)
        self.assertEqual(result.n_groups, 3)
        self.assertEqual(sum(map(sum, result.confusion_matrix)), len(y))
        self.assertIn("balanced_accuracy", result.as_dict())

    def test_model_save_load_consistency(self):
        X, y, _ = synthetic_epochs(seed=12)
        estimator = make_csp_lda_pipeline(n_components=4).fit(X, y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(estimator, path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(estimator.predict(X[:6]), loaded.predict(X[:6]))

