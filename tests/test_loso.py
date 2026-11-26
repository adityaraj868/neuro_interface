import unittest
import numpy as np

from muse.evaluation import evaluate_loso
from muse.models import make_csp_lda_pipeline
from tests.test_models_evaluation import synthetic_epochs


class LOSOTests(unittest.TestCase):
    def test_evaluate_loso(self):
        X, y, _ = synthetic_epochs(seed=41)
        # 3 subjects across the 48 epochs
        subjects = np.array([1] * 16 + [2] * 16 + [3] * 16)

        model = make_csp_lda_pipeline(n_components=4)
        result = evaluate_loso(model, X, y, subjects, within_subject_score=0.85)

        self.assertIn("sub-1", result.subject_scores)
        self.assertIn("sub-2", result.subject_scores)
        self.assertIn("sub-3", result.subject_scores)
        self.assertIsNotNone(result.domain_shift_gap)
        self.assertEqual(result.overall.n_groups, 3)

        d = result.as_dict()
        self.assertIn("overall", d)
        self.assertIn("subject_scores", d)
