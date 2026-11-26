import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch

from muse.evaluation import evaluate_grouped
from muse.models import EEGNet, EEGNetClassifier, make_eegnet_classifier
from tests.test_models_evaluation import synthetic_epochs


class EEGNetModelTests(unittest.TestCase):
    def test_eegnet_module_forward(self):
        model = EEGNet(n_classes=2, n_channels=8, n_times=80, kernel_length=32)
        x = torch.randn(4, 1, 8, 80)
        out = model(x)
        self.assertEqual(out.shape, (4, 2))

    def test_eegnet_classifier_fit_and_predict(self):
        X, y, groups = synthetic_epochs(seed=31)
        clf = make_eegnet_classifier(epochs=5, batch_size=16, random_state=42)
        clf.fit(X, y)

        preds = clf.predict(X[:6])
        self.assertEqual(preds.shape, (6,))
        self.assertTrue(set(preds).issubset(set(clf.classes_)))

        # Verify .predict_proba contract
        probs = clf.predict_proba(X[:6])
        self.assertEqual(probs.shape, (6, 2))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

        # Verify grouped evaluation
        result = evaluate_grouped(
            make_eegnet_classifier(epochs=3, batch_size=16, random_state=42),
            X,
            y,
            groups,
        )
        self.assertEqual(result.n_groups, 3)
        self.assertIn("balanced_accuracy", result.as_dict())

    def test_eegnet_save_load_consistency(self):
        X, y, _ = synthetic_epochs(seed=32)
        clf = make_eegnet_classifier(epochs=5, batch_size=16, random_state=42).fit(X, y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eegnet.joblib"
            joblib.dump(clf, path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(clf.predict(X[:6]), loaded.predict(X[:6]))
            np.testing.assert_allclose(
                clf.predict_proba(X[:6]),
                loaded.predict_proba(X[:6]),
                rtol=1e-4,
            )
