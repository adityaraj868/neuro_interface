import time
from pathlib import Path
from muse.realtime.streamer import MockEEGStreamer
from muse.realtime.buffer import SlidingRingBuffer
from muse.realtime.engine import BCIInferenceEngine


def test_2d_inference_pipeline():
    h_model = Path("results/unilateral_lda_grouped/model.joblib")
    v_model = Path("results/bilateral_lda_grouped/model.joblib")
    edf_path = Path("data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R06.edf")

    assert h_model.exists(), f"Missing horizontal model: {h_model}"
    assert v_model.exists(), f"Missing vertical model: {v_model}"
    assert edf_path.exists(), f"Missing bilateral EEG data: {edf_path}"

    stream = MockEEGStreamer(edf_path, chunk_size=16)
    buf = SlidingRingBuffer(n_channels=64, window_samples=480)
    engine = BCIInferenceEngine(
        horizontal_model_path=h_model,
        vertical_model_path=v_model,
        threshold=0.55,
    )

    for _ in range(30):
        buf.push(stream.get_chunk())

    t0 = time.perf_counter()
    res = engine.predict_window(buf.get_window())
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\n[2D RESULT] H: {res['h_command']} (vx={res['vx']:+.2f}) | V: {res['v_command']} (vy={res['vy']:+.2f})")
    print(f"[TIMING]    Dual-model inference latency: {latency_ms:.2f} ms (Budget: <100 ms)")

    assert res["h_command"] in ["LEFT", "RIGHT", "NEUTRAL"]
    assert res["v_command"] in ["UP", "DOWN", "NEUTRAL"]
    assert latency_ms < 100.0, f"Dual inference too slow: {latency_ms:.2f} ms"
    print("[SUCCESS] Full 2D real-time inference chain verified.")


if __name__ == "__main__":
    test_2d_inference_pipeline()