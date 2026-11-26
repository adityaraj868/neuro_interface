import numpy as np


class SlidingRingBuffer:
    """
    Maintains a rolling First-In-First-Out (FIFO) window of multichannel EEG data.
    """
    def __init__(self, n_channels: int = 64, window_samples: int = 481):
        """
        Args:
            n_channels: Number of EEG electrode channels (64 for PhysioNet).
            window_samples: Window length in samples (481 samples = [1.0s, 4.0s] at 160 Hz).
        """
        self.n_channels = n_channels
        self.window_samples = window_samples
        
        # Pre-allocate buffer in memory
        self.buffer = np.zeros((n_channels, window_samples), dtype=np.float64)
        self.samples_collected = 0

    def push(self, chunk: np.ndarray) -> None:
        """
        Appends a new chunk of data of shape (n_channels, n_new_samples)
        and slides out the oldest samples.
        """
        if chunk.ndim != 2:
            raise ValueError(f"Expected 2D chunk (channels, samples), got ndim={chunk.ndim}")
        
        n_ch, n_samples = chunk.shape
        if n_ch != self.n_channels:
            raise ValueError(f"Channel mismatch: expected {self.n_channels}, got {n_ch}")

        if n_samples >= self.window_samples:
            # If chunk is larger than buffer, take only the most recent samples
            self.buffer[:, :] = chunk[:, -self.window_samples:]
        else:
            # Shift buffer left by n_samples
            self.buffer[:, :-n_samples] = self.buffer[:, n_samples:]
            # Insert new chunk at the tail
            self.buffer[:, -n_samples:] = chunk

        self.samples_collected += n_samples

    def is_full(self) -> bool:
        """Returns True if the buffer contains at least window_samples of real data."""
        return self.samples_collected >= self.window_samples

    def get_window(self) -> np.ndarray:
        """
        Returns a copy of the current window ready for inference.
        Shape: (n_channels, window_samples)
        """
        return self.buffer.copy()