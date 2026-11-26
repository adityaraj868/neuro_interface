import time
from pathlib import Path
from muse.realtime.streamer import MockEEGStreamer

def test_streamer():
    path = Path("data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R04.edf")
    assert path.exists(), f"Missing test file: {path}"

    stream = MockEEGStreamer(path, chunk_size=16)
    print(f"\n[STREAM INFO] sfreq: {stream.sfreq} Hz | Channels: {stream.n_channels}")

    print("Streaming 10 chunks of 16 samples (~1.0s real time)...")
    t0 = time.time()
    for i in range(10):
        chunk = stream.get_chunk()
        assert chunk.shape == (64, 16), f"Expected shape (64, 16), got {chunk.shape}"
        print(f"  Chunk {i + 1:02d}: shape={chunk.shape}")

    elapsed = time.time() - t0
    print(f"[TIMING] Total time elapsed: {elapsed:.2f}s (Target: ~1.00s)")
    assert 0.8 <= elapsed <= 1.5, f"Timing skewed: {elapsed:.2f}s"
    print("[SUCCESS] MockEEGStreamer verified.")

if __name__ == "__main__":
    test_streamer()