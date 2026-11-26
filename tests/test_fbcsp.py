import unittest
import numpy as np

from muse.models.fbcsp import FilterBankCSP, make_fbcsp_pipeline


class FBCSPTests(unittest.TestCase):
    def test_fbcsp_fit_and_transform(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 8, 160)) * 1e-5
        y = np.array(["LEFT", "RIGHT"] * 5)

        fbcsp = FilterBankCSP(
            sub_bands=[(8.0, 12.0), (12.0, 16.0)],
            n_components=2,
            sfreq=160.0,
            reg="oas",
        )
        fbcsp.fit(X, y)
        feats = fbcsp.transform(X)
        self.assertEqual(feats.shape, (10, 4))

    def test_make_fbcsp_pipeline(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(12, 8, 160)) * 1e-5
        y = np.array(["LEFT", "RIGHT"] * 6)

        pipe = make_fbcsp_pipeline(
            n_components=2,
            k_features=3,
            classifier_type="lda",
            sub_bands=[(8.0, 12.0), (12.0, 16.0)],
            sfreq=160.0,
        )
        pipe.fit(X, y)
        preds = pipe.predict(X)
        self.assertEqual(len(preds), 12)
        self.assertTrue(set(preds).issubset({"LEFT", "RIGHT"}))


if __name__ == "__main__":
    unittest.main()
