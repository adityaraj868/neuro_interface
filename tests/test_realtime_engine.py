import time
import unittest
from pathlib import Path

import numpy as np

from muse.models import make_eegnet_classifier, make_riemannian_svm_pipeline
from muse.realtime.buffer import SlidingRingBuffer
from muse.realtime.engine import BCIInferenceEngine
from muse.realtime.streamer import MockEEGStreamer


class _DummyModel:
    def __init__(self, classes, probs):
        self.classes_ = classes
        self._probs = np.asarray([probs], dtype=float)

    def predict_proba(self, X):
        assert X.shape == (1, 64, 481)
        return self._probs


class RealtimeEngineTests(unittest.TestCase):
    def test_single_axis_winner_take_all_gating(self):
        engine = BCIInferenceEngine(
            horizontal_model_path=_DummyModel(["LEFT", "RIGHT"], [0.20, 0.80]),
            vertical_model_path=_DummyModel(["BOTH_FEET", "BOTH_HANDS"], [0.35, 0.65]),
            single_axis=True,
            threshold=0.55,
        )
        window = np.random.default_rng(0).normal(size=(64, 481))
        res = engine.predict_window(window)
        self.assertEqual(res["h_command"], "RIGHT")
        self.assertGreater(res["vx"], 0.0)
        self.assertEqual(res["vy"], 0.0)
        self.assertEqual(res["v_command"], "NEUTRAL")

        engine_v = BCIInferenceEngine(
            horizontal_model_path=_DummyModel(["LEFT", "RIGHT"], [0.40, 0.60]),
            vertical_model_path=_DummyModel(["BOTH_FEET", "BOTH_HANDS"], [0.10, 0.90]),
            single_axis=True,
            threshold=0.55,
        )
        res_v = engine_v.predict_window(window)
        self.assertEqual(res_v["v_command"], "UP")
        self.assertGreater(res_v["vy"], 0.0)
        self.assertEqual(res_v["vx"], 0.0)
        self.assertEqual(res_v["h_command"], "NEUTRAL")

    def test_temporal_smoothing_and_reset(self):
        engine = BCIInferenceEngine(
            horizontal_model_path=_DummyModel(["LEFT", "RIGHT"], [0.10, 0.90]),
            vertical_model_path=_DummyModel(["BOTH_FEET", "BOTH_HANDS"], [0.50, 0.50]),
            single_axis=True,
            threshold=0.55,
            smoothing_alpha=0.5,
            history_len=3,
        )
        window = np.random.default_rng(1).normal(size=(64, 481))
        res1 = engine.predict_window(window)
        self.assertEqual(res1["h_command"], "RIGHT")
        self.assertIn("smoothed_probabilities", res1)

        engine.reset_state()
        self.assertEqual(len(engine._prob_history), 0)
        self.assertEqual(len(engine._cmd_history), 0)

    def test_riemannian_and_eegnet_481_compat(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(12, 64, 481)) * 1e-5
        y_h = np.array(["LEFT", "RIGHT"] * 6)
        y_v = np.array(["BOTH_FEET", "BOTH_HANDS"] * 6)

        r_h = make_riemannian_svm_pipeline(estimator="oas", C=1.0, probability=True).fit(X, y_h)
        r_v = make_riemannian_svm_pipeline(estimator="oas", C=1.0, probability=True).fit(X, y_v)
        r_engine = BCIInferenceEngine(r_h, r_v, single_axis=True)
        r_res = r_engine.predict_window(X[0])
        self.assertTrue(r_res["vx"] == 0.0 or r_res["vy"] == 0.0)

        e_h = make_eegnet_classifier(epochs=2, batch_size=6, random_state=42).fit(X, y_h)
        e_v = make_eegnet_classifier(epochs=2, batch_size=6, random_state=42).fit(X, y_v)
        e_engine = BCIInferenceEngine(e_h, e_v, single_axis=True)
        e_res = e_engine.predict_window(X[0])
        self.assertTrue(e_res["vx"] == 0.0 or e_res["vy"] == 0.0)

    def test_unified_four_class_blind_decoding(self):
        model_path = Path("results/s001_4class_riemannian/model.joblib")
        if not model_path.exists():
            self.skipTest(f"Model not found: {model_path}")

        engine = BCIInferenceEngine(
            horizontal_model_path=model_path,
            vertical_model_path=None,
            threshold=0.30,
            deadband=0.05,
            single_axis=True,
        )
        edf_path = Path("data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R14.edf")
        if not edf_path.exists():
            self.skipTest(f"EDF file not found: {edf_path}")

        stream = MockEEGStreamer(edf_path, chunk_size=16)
        buf = SlidingRingBuffer(n_channels=64, window_samples=481)
        for _ in range(31):
            buf.push(stream.get_chunk())

        res = engine.predict_window(buf.get_window(), axis_lock="AUTO")
        self.assertEqual(set(res["probabilities"].keys()), {"LEFT", "RIGHT", "BOTH_HANDS", "BOTH_FEET"})
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=4)
        self.assertIn(res["h_command"], {"LEFT", "RIGHT", "NEUTRAL"})
        self.assertIn(res["v_command"], {"UP", "DOWN", "NEUTRAL"})
        self.assertTrue(res["vx"] == 0.0 or res["vy"] == 0.0)

    def test_2d_inference_pipeline(self):
        h_model = Path("results/unilateral_lda_grouped/model.joblib")
        v_model = Path("results/bilateral_lda_grouped/model.joblib")
        edf_path = Path("data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R06.edf")

        if not (h_model.exists() and v_model.exists() and edf_path.exists()):
            self.skipTest("Required artifacts for 2D pipeline not available")

        stream = MockEEGStreamer(edf_path, chunk_size=16)
        buf = SlidingRingBuffer(n_channels=64, window_samples=481)
        engine = BCIInferenceEngine(
            horizontal_model_path=h_model,
            vertical_model_path=v_model,
            threshold=0.55,
        )

        for _ in range(31):
            buf.push(stream.get_chunk())
        self.assertTrue(buf.is_full())

        window = buf.get_window()
        self.assertEqual(window.shape, (64, 481))

        t0 = time.perf_counter()
        res = engine.predict_window(window)
        latency_ms = (time.perf_counter() - t0) * 1000

        self.assertIn(res["h_command"], ["LEFT", "RIGHT", "NEUTRAL"])
        self.assertIn(res["v_command"], ["UP", "DOWN", "NEUTRAL"])
        self.assertTrue(res["vx"] == 0.0 or res["vy"] == 0.0)
        self.assertLess(latency_ms, 100.0)


if __name__ == "__main__":
    unittest.main()
