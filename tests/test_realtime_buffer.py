import numpy as np
from muse.realtime.buffer import SlidingRingBuffer

def test_ring_buffer():
    n_channels = 64
    window_samples = 480
    chunk_samples = 16

    buf = SlidingRingBuffer(n_channels=n_channels, window_samples=window_samples)
    assert not buf.is_full(), "Buffer should not be full initially"

    # Push 29 chunks (29 * 16 = 464 samples -> not yet 480)
    for _ in range(29):
        chunk = np.ones((n_channels, chunk_samples))
        buf.push(chunk)
    assert not buf.is_full(), "Buffer should be at 464/480 samples"

    # Push 30th chunk (30 * 16 = 480 samples -> now full)
    chunk = np.ones((n_channels, chunk_samples)) * 5.0
    buf.push(chunk)
    assert buf.is_full(), "Buffer should be full at 480 samples"

    window = buf.get_window()
    assert window.shape == (64, 480), f"Expected shape (64, 480), got {window.shape}"
    # Verify the most recent samples contain the latest chunk (5.0)
    assert np.allclose(window[:, -16:], 5.0), "Tail of buffer does not match newest chunk"
    print("[SUCCESS] SlidingRingBuffer verified: FIFO sliding & dimensions intact.")

if __name__ == "__main__":
    test_ring_buffer()