import time
from pathlib import Path
import numpy as np
from muse.models import make_eegnet_classifier, make_riemannian_svm_pipeline
from muse.realtime.streamer import MockEEGStreamer
from muse.realtime.buffer import SlidingRingBuffer
from muse.realtime.engine import BCIInferenceEngine


class _DummyModel:
    def __init__(self, classes, probs):
        self.classes_ = classes
        self._probs = np.asarray([probs], dtype=float)

    def predict_proba(self, X):
        assert X.shape == (1, 64, 481)
        return self._probs


def test_single_axis_winner_take_all_gating():
    # Horizontal dominant: |0.80 - 0.20| = 0.60 > |0.65 - 0.35| = 0.30
    engine = BCIInferenceEngine(
        horizontal_model_path=_DummyModel(["LEFT", "RIGHT"], [0.20, 0.80]),
        vertical_model_path=_DummyModel(["BOTH_FEET", "BOTH_HANDS"], [0.35, 0.65]),
        single_axis=True,
    )
    window = np.random.default_rng(0).normal(size=(64, 481))
    res = engine.predict_window(window)
    assert res["h_command"] == "RIGHT"
    assert res["vx"] > 0.0
    assert res["vy"] == 0.0
    assert res["v_command"] == "NEUTRAL"

    # Vertical dominant: |0.90 - 0.10| = 0.80 > |0.60 - 0.40| = 0.20
    engine_v = BCIInferenceEngine(
        horizontal_model_path=_DummyModel(["LEFT", "RIGHT"], [0.40, 0.60]),
        vertical_model_path=_DummyModel(["BOTH_FEET", "BOTH_HANDS"], [0.10, 0.90]),
        single_axis=True,
    )
    res_v = engine_v.predict_window(window)
    assert res_v["v_command"] == "UP"
    assert res_v["vy"] > 0.0
    assert res_v["vx"] == 0.0
    assert res_v["h_command"] == "NEUTRAL"


def test_riemannian_and_eegnet_481_compat():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 64, 481)) * 1e-5
    y_h = np.array(["LEFT", "RIGHT"] * 6)
    y_v = np.array(["BOTH_FEET", "BOTH_HANDS"] * 6)

    r_h = make_riemannian_svm_pipeline(estimator="oas", C=1.0, probability=True).fit(X, y_h)
    r_v = make_riemannian_svm_pipeline(estimator="oas", C=1.0, probability=True).fit(X, y_v)
    r_engine = BCIInferenceEngine(r_h, r_v, single_axis=True)
    r_res = r_engine.predict_window(X[0])
    assert r_res["vx"] == 0.0 or r_res["vy"] == 0.0

    e_h = make_eegnet_classifier(epochs=2, batch_size=6, random_state=42).fit(X, y_h)
    e_v = make_eegnet_classifier(epochs=2, batch_size=6, random_state=42).fit(X, y_v)
    e_engine = BCIInferenceEngine(e_h, e_v, single_axis=True)
    e_res = e_engine.predict_window(X[0])
    assert e_res["vx"] == 0.0 or e_res["vy"] == 0.0


def test_unified_four_class_blind_decoding():
    for model_path in (
        Path("results/s001_4class_riemannian/model.joblib"),
        Path("results/four_class_riemannian_grouped/model.joblib"),
    ):
        assert model_path.exists(), f"Missing unified 4-class model: {model_path}"

        engine = BCIInferenceEngine(
            horizontal_model_path=model_path,
            vertical_model_path=None,
            threshold=0.55,
            deadband=0.15,
            single_axis=True,
        )
        edf_path = Path("data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R14.edf")
        stream = MockEEGStreamer(edf_path, chunk_size=16)
        buf = SlidingRingBuffer(n_channels=64, window_samples=481)
        for _ in range(31):
            buf.push(stream.get_chunk())

        res = engine.predict_window(buf.get_window(), axis_lock="AUTO")
        assert set(res["probabilities"].keys()) == {"LEFT", "RIGHT", "BOTH_HANDS", "BOTH_FEET"}
        assert abs(sum(res["probabilities"].values()) - 1.0) < 1e-5
        assert res["h_command"] in {"LEFT", "RIGHT", "NEUTRAL"}
        assert res["v_command"] in {"UP", "DOWN", "NEUTRAL"}
        assert res["vx"] == 0.0 or res["vy"] == 0.0


def test_2d_inference_pipeline():
    h_model = Path("results/unilateral_lda_grouped/model.joblib")
    v_model = Path("results/bilateral_lda_grouped/model.joblib")
    edf_path = Path("data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R06.edf")

    assert h_model.exists(), f"Missing horizontal model: {h_model}"
    assert v_model.exists(), f"Missing vertical model: {v_model}"
    assert edf_path.exists(), f"Missing bilateral EEG data: {edf_path}"

    stream = MockEEGStreamer(edf_path, chunk_size=16)
    buf = SlidingRingBuffer(n_channels=64, window_samples=481)
    engine = BCIInferenceEngine(
        horizontal_model_path=h_model,
        vertical_model_path=v_model,
        threshold=0.55,
    )

    for _ in range(31):
        buf.push(stream.get_chunk())
    assert buf.is_full(), "Ring buffer should be full at 481+ samples"

    window = buf.get_window()
    assert window.shape == (64, 481), f"Expected (64, 481), got {window.shape}"

    t0 = time.perf_counter()
    res = engine.predict_window(window)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\n[2D RESULT] H: {res['h_command']} (vx={res['vx']:+.2f}) | V: {res['v_command']} (vy={res['vy']:+.2f})")
    print(f"[TIMING]    Dual-model inference latency: {latency_ms:.2f} ms (Budget: <100 ms)")

    assert res["h_command"] in ["LEFT", "RIGHT", "NEUTRAL"]
    assert res["v_command"] in ["UP", "DOWN", "NEUTRAL"]
    assert res["vx"] == 0.0 or res["vy"] == 0.0, "Winner-Take-All gating must move along one axis at a time"
    assert latency_ms < 100.0, f"Dual inference too slow: {latency_ms:.2f} ms"
    print("[SUCCESS] Full 2D real-time inference chain verified.")


if __name__ == "__main__":
    test_2d_inference_pipeline()