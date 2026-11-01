import unittest

import numpy as np

import muse  # noqa: F401
import mne

from muse.data import validate_epoch_dataset
from muse.preprocessing import PreprocessingConfig, epoch_raw


class PreprocessingTests(unittest.TestCase):
    def test_config_validation(self):
        config = PreprocessingConfig()
        self.assertEqual(config.as_dict()["l_freq"], 8.0)
        with self.assertRaises(ValueError):
            PreprocessingConfig(l_freq=30, h_freq=8)
        with self.assertRaises(ValueError):
            PreprocessingConfig(tmin=2, tmax=1)

    def test_epoch_shape_and_label_alignment(self):
        sfreq = 100.0
        rng = np.random.default_rng(7)
        raw = mne.io.RawArray(rng.normal(size=(4, 1200)) * 1e-6, mne.create_info(4, sfreq, "eeg"), verbose=False)
        raw.set_annotations(mne.Annotations([2.0, 6.0, 10.0], [0.0, 0.0, 0.0], ["T1", "T2", "T1"]))
        bundle = epoch_raw(
            raw,
            run=4,
            config=PreprocessingConfig(tmin=0.5, tmax=1.5, reject_uV=None),
        )
        self.assertEqual(bundle.X.shape, (3, 4, 101))
        self.assertEqual(bundle.labels.tolist(), ["LEFT", "RIGHT", "LEFT"])
        validate_epoch_dataset(bundle.X, bundle.labels, np.array(["a", "b", "c"]))

    def test_epoch_label_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_epoch_dataset(np.zeros((2, 3, 4)), np.array(["LEFT"]), np.array(["a", "b"]))

