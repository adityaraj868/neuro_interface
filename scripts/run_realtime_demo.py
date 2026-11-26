import argparse
import time
from pathlib import Path
from muse.realtime.streamer import MockEEGStreamer
from muse.realtime.buffer import SlidingRingBuffer
from muse.realtime.engine import BCIInferenceEngine


def render_2d_arena(x: float, y: float, width: int = 33, height: int = 13) -> str:
    """Renders a 2D ASCII cursor grid for coordinates in [-100, +100]."""
    cx, cy = width // 2, height // 2
    col = int(round(cx + (max(-100.0, min(100.0, x)) / 100.0) * cx))
    row = int(round(cy - (max(-100.0, min(100.0, y)) / 100.0) * cy))

    lines = ["        UP (BOTH_HANDS)", "  +" + "-" * width + "+"]
    for r in range(height):
        chars = [" "] * width
        if r == cy:
            chars = ["-"] * width
        chars[cx] = "+" if r == cy else "|"
        if r == row:
            chars[col] = "O"
        prefix = "L " if r == cy else "  "
        suffix = " R" if r == cy else ""
        lines.append(f"{prefix}|{''.join(chars)}|{suffix}")
    lines.append("  +" + "-" * width + "+")
    lines.append("       DOWN (BOTH_FEET)")
    return "\n".join(lines)


def run_2d_demo(edf_path: Path, duration_sec: float = 15.0, alpha: float = 0.30, step_size: float = 15.0):
    engine = BCIInferenceEngine(
        horizontal_model_path="results/unilateral_lda_grouped/model.joblib",
        vertical_model_path="results/bilateral_lda_grouped/model.joblib",
        threshold=0.55,
        deadband=0.15,
    )
    stream = MockEEGStreamer(edf_path, chunk_size=16)
    buf = SlidingRingBuffer(n_channels=64, window_samples=480)

    print(f"Loading stream: {edf_path.name} | Filling 3.0s ring buffer...")
    while not buf.is_full():
        buf.push(stream.get_chunk())

    x, y = 0.0, 0.0
    step = 0
    start_time = time.time()

    try:
        while (time.time() - start_time) < duration_sec:
            buf.push(stream.get_chunk())
            step += 1

            if step % 2 == 0:
                t0 = time.perf_counter()
                res = engine.predict_window(buf.get_window())
                latency_ms = (time.perf_counter() - t0) * 1000

                x = max(-100.0, min(100.0, x + alpha * res["vx"] * step_size))
                y = max(-100.0, min(100.0, y + alpha * res["vy"] * step_size))

                print("\033[H\033[J", end="")
                print(f"Elapsed: {time.time() - start_time:4.1f}s | Latency: {latency_ms:4.1f} ms")
                print(f"Pos: ({x:+6.1f}, {y:+6.1f}) | H: {res['h_command']:<7} | V: {res['v_command']:<7}\n")
                print(render_2d_arena(x, y))
    except KeyboardInterrupt:
        pass

    print("\n[2D DEMO COMPLETE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        choices=["horizontal", "vertical"],
        default="vertical",
        help="Test horizontal (S001R04) or vertical (S001R06) imagery run",
    )
    args = parser.parse_args()
    edf = (
        Path("data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R04.edf")
        if args.run == "horizontal"
        else Path("data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R06.edf")
    )
    run_2d_demo(edf)