import time
from pathlib import Path
from muse.realtime.streamer import MockEEGStreamer
from muse.realtime.buffer import SlidingRingBuffer
from muse.realtime.engine import BCIInferenceEngine


def render_cursor_bar(position: float, width: int = 41) -> str:
    """Renders an ASCII cursor bar for position in [-1.0, +1.0]."""
    clamped = max(-1.0, min(1.0, position))
    center = width // 2
    cursor_idx = int(round(center + clamped * center))
    bar = ["-"] * width
    bar[center] = "|"
    bar[cursor_idx] = "O"
    return f"LEFT [{''.join(bar)}] RIGHT"


def run_demo(duration_sec: float = 20.0, alpha: float = 0.35):
    model_path = Path("results/unilateral_lda_grouped/model.joblib")
    edf_path = Path("data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R04.edf")

    stream = MockEEGStreamer(edf_path, chunk_size=16)       # 100 ms per chunk
    buf = SlidingRingBuffer(n_channels=64, window_samples=480)
    engine = BCIInferenceEngine(model_path=model_path, threshold=0.60)

    print("Warming up 3.0s ring buffer (30 chunks)...")
    while not buf.is_full():
        buf.push(stream.get_chunk())

    print("Live BCI stream active! Press Ctrl+C to stop.\n")
    smoothed_pos = 0.0
    step = 0
    start_time = time.time()

    try:
        while (time.time() - start_time) < duration_sec:
            buf.push(stream.get_chunk())
            step += 1

            # Run inference every 2 chunks (200 ms = 5 Hz decision rate)
            if step % 2 == 0:
                t0 = time.perf_counter()
                res = engine.predict_window(buf.get_window())
                latency_ms = (time.perf_counter() - t0) * 1000

                probs = res["probabilities"]
                raw_intent = probs.get("RIGHT", 0.0) - probs.get("LEFT", 0.0)
                smoothed_pos = alpha * raw_intent + (1.0 - alpha) * smoothed_pos

                bar = render_cursor_bar(smoothed_pos)
                elapsed = time.time() - start_time
                print(
                    f"\r[{elapsed:5.1f}s] {bar} | "
                    f"Cmd: {res['command']:<7} ({res['confidence']:.2f}) | "
                    f"{latency_ms:4.1f}ms",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass

    print("\n\n[DEMO COMPLETE] Real-time session finished.")


if __name__ == "__main__":
    run_demo()