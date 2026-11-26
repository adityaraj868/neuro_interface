import unittest
import numpy as np

from muse.models.alignment import EuclideanAligner, compute_reference_matrix, euclidean_alignment, inv_sqrt_matrix


class AlignmentTests(unittest.TestCase):
    def test_reference_matrix_and_inv_sqrt(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(8, 4, 100))
        R = compute_reference_matrix(X)
        self.assertEqual(R.shape, (4, 4))
        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(R)
        self.assertTrue(np.all(eigvals > 0))

        R_inv_sqrt = inv_sqrt_matrix(R)
        # R^(-1/2) @ R @ R^(-1/2) should be approximately I
        identity_approx = R_inv_sqrt @ R @ R_inv_sqrt
        np.testing.assert_allclose(identity_approx, np.eye(4), atol=1e-5)

    def test_euclidean_alignment_function(self):
        rng = np.random.default_rng(42)
        # Subject 1 has larger scale, Subject 2 has smaller scale
        X1 = rng.normal(size=(6, 4, 100)) * 5.0
        X2 = rng.normal(size=(6, 4, 100)) * 0.2
        X = np.concatenate([X1, X2], axis=0)
        subjects = np.array([1] * 6 + [2] * 6)

        X_aligned = euclidean_alignment(X, subjects)
        self.assertEqual(X_aligned.shape, X.shape)

        # After alignment, mean covariance for both subjects should have trace ~ n_channels
        R1 = compute_reference_matrix(X_aligned[:6])
        R2 = compute_reference_matrix(X_aligned[6:])
        self.assertAlmostEqual(np.trace(R1), 4.0, delta=0.5)
        self.assertAlmostEqual(np.trace(R2), 4.0, delta=0.5)

    def test_aligner_transformer(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 4, 100))
        aligner = EuclideanAligner()
        aligner.fit(X)
        X_trans = aligner.transform(X)
        self.assertEqual(X_trans.shape, X.shape)


if __name__ == "__main__":
    unittest.main()
