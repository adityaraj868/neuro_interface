"""EEG filtering and trial epoching."""

from .epochs import SENSORIMOTOR_CHANNELS, EpochBundle, PreprocessingConfig, epoch_raw

__all__ = ["EpochBundle", "PreprocessingConfig", "SENSORIMOTOR_CHANNELS", "epoch_raw"]

