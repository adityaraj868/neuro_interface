import time
from pathlib import Path
import mne
import numpy as np


class MockEEGStreamer:
    """
    Simulates live EEG hardware by streaming an EDF recording in real time.
    """
    def __init__(self, edf_path: str | Path, chunk_size: int = 16):
        """
        Args:
            edf_path: Path to the .edf file.
            chunk_size: Number of time samples emitted per tick.
                        16 samples at 160 Hz = exactly 100 ms of data.
        """
        self.edf_path = Path(edf_path)
        if not self.edf_path.exists():
            raise FileNotFoundError(f"EDF file not found: {self.edf_path}")

        # Load raw recording without extra terminal spam and align EEG channel picks with training
        self.raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose="ERROR")
        mne.datasets.eegbci.standardize(self.raw)
        picks = mne.pick_types(
            self.raw.info, eeg=True, meg=False, stim=False, eog=False, exclude="bads"
        )
        self.sfreq = float(self.raw.info["sfreq"])
        self.ch_names = [self.raw.ch_names[idx] for idx in picks]

        # Shape: (n_eeg_channels, total_time_samples) in Volts
        self.data = self.raw.get_data(picks=picks)
        self.n_channels, self.total_samples = self.data.shape
        
        self.chunk_size = chunk_size
        self.delay_per_chunk = chunk_size / self.sfreq  # 16 / 160 = 0.1s
        self.cursor = 0

    def get_chunk(self) -> np.ndarray:
        """
        Fetches the next chunk of EEG data and pauses for the exact real-world duration.
        Returns:
            np.ndarray of shape (n_channels, chunk_size)
        """
        # If we reach the end of the recording, loop back to the beginning
        if self.cursor + self.chunk_size > self.total_samples:
            self.cursor = 0

        chunk = self.data[:, self.cursor : self.cursor + self.chunk_size]
        self.cursor += self.chunk_size

        # Simulate real-time hardware transmission delay
        time.sleep(self.delay_per_chunk)
        return chunk