import time
from pathlib import Path
from muse.realtime.streamer import MockEEGStreamer
from muse.realtime.buffer import SlidingRingBuffer
from muse.realtime.engine import BCIInferenceEngine


def test_inference_pipeline():
    model_path = Path("results/unilateral_lda_grouped/model.joblib")
    edf_path = Path("data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R04.edf")

    assert model_path.exists(), f"Missing model artifact: {model_path}"
    assert edf_path.exists(), f"Missing EEG data: {edf_path}"

    stream = MockEEGStreamer(edf_path, chunk_size=16)
    buf = SlidingRingBuffer(n_channels=64, window_samples=480)
    engine = BCIInferenceEngine(model_path=model_path, threshold=0.55)

    for _ in range(30):
        buf.push(stream.get_chunk())

    assert buf.is_full(), "Buffer should be full at 480 samples"

    t0 = time.perf_counter()
    result = engine.predict_window(buf.get_window())
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"[RESULT] Command: {result['command']} | Confidence: {result['confidence']:.2f}")
    print(f"[TIMING] Inference latency: {latency_ms:.2f} ms (Budget: <100 ms)")

    assert result["command"] in ["LEFT", "RIGHT", "NEUTRAL"]
    assert latency_ms < 100.0, f"Inference too slow: {latency_ms:.2f} ms"
    print("[SUCCESS] Full real-time inference chain verified.")


if __name__ == "__main__":
    test_inference_pipeline()