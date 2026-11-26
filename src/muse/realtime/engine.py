"""Real-time dual-model BCI inference engine with bandpass filtering and single-axis gating."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from scipy.signal import butter, filtfilt


class BCIInferenceEngine:
    """Runs 8-30 Hz bandpass filtering and orthogonal 2D motor imagery inference."""

    def __init__(
        self,
        horizontal_model_path: str | Path | Any,
        vertical_model_path: Optional[str | Path | Any] = None,
        threshold: float = 0.55,
        deadband: float = 0.15,
        sfreq: float = 160.0,
        l_freq: float = 8.0,
        h_freq: float = 30.0,
        single_axis: bool = True,
        average_reference: bool = True,
    ) -> None:
        self.horizontal_model = (
            joblib.load(horizontal_model_path)
            if isinstance(horizontal_model_path, (str, Path))
            else horizontal_model_path
        )
        self.h_classes = list(self.horizontal_model.classes_)

        if vertical_model_path is None:
            self.vertical_model = None
        elif isinstance(vertical_model_path, (str, Path)):
            self.vertical_model = joblib.load(vertical_model_path)
        else:
            self.vertical_model = vertical_model_path
        self.v_classes = (
            list(self.vertical_model.classes_) if self.vertical_model is not None else []
        )

        self.threshold = threshold
        self.deadband = deadband
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.single_axis = single_axis
        self.average_reference = average_reference

        nyq = 0.5 * sfreq
        self._b, self._a = butter(4, [l_freq / nyq, h_freq / nyq], btype="band")

    def _preprocess(self, window: np.ndarray) -> np.ndarray:
        """Apply MNE firwin 8-30 Hz bandpass filter (matching training) and CAR."""
        centered = window - np.mean(window, axis=-1, keepdims=True)
        if window.shape[-1] >= 120:
            import mne

            filtered = mne.filter.filter_data(
                centered.astype(np.float64, copy=False),
                sfreq=self.sfreq,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                method="fir",
                fir_design="firwin",
                verbose=False,
            )
        else:
            filtered = filtfilt(self._b, self._a, centered, axis=-1)
        if self.average_reference:
            filtered = filtered - np.mean(filtered, axis=0, keepdims=True)
        return filtered

    def predict_window(
        self,
        window: np.ndarray,
        axis_lock: str = "AUTO",
    ) -> Dict[str, Any]:
        """Predict orthogonal 2D cursor velocities from a (channels, samples) EEG window.

        Args:
            window: Raw EEG window of shape (n_channels, window_samples).
            axis_lock: "AUTO" (Winner-Take-All single axis), "HORIZONTAL", or "VERTICAL".
        """
        if window.ndim != 2:
            raise ValueError(f"Expected 2D array (n_channels, n_samples), got {window.shape}")
        filtered = self._preprocess(window)
        batch_window = np.expand_dims(filtered, axis=0)

        # 1. Decode probability distribution(s): unified 4-class or dual binary
        h_probs_arr = self.horizontal_model.predict_proba(batch_window)[0]
        h_probs = {str(cls): float(p) for cls, p in zip(self.h_classes, h_probs_arr)}
        vx = h_probs.get("RIGHT", 0.0) - h_probs.get("LEFT", 0.0)
        if abs(vx) < self.deadband:
            vx = 0.0

        is_unified_four_class = self.vertical_model is None and (
            "BOTH_HANDS" in h_probs or "BOTH_FEET" in h_probs
        )

        v_probs: Dict[str, float] = {}
        if is_unified_four_class:
            vy = h_probs.get("BOTH_HANDS", 0.0) - h_probs.get("BOTH_FEET", 0.0)
            if abs(vy) < self.deadband:
                vy = 0.0

            h_cmd = "RIGHT" if vx > 0.0 else ("LEFT" if vx < 0.0 else "NEUTRAL")
            v_cmd = "UP" if vy > 0.0 else ("DOWN" if vy < 0.0 else "NEUTRAL")
        else:
            h_top_idx = int(np.argmax(h_probs_arr))
            h_conf = float(h_probs_arr[h_top_idx])
            h_cmd = str(self.h_classes[h_top_idx]) if h_conf >= self.threshold else "NEUTRAL"

            vy = 0.0
            v_cmd = "NEUTRAL"

            if self.vertical_model is not None:
                v_probs_arr = self.vertical_model.predict_proba(batch_window)[0]
                v_probs = {str(cls): float(p) for cls, p in zip(self.v_classes, v_probs_arr)}
                vy = v_probs.get("BOTH_HANDS", 0.0) - v_probs.get("BOTH_FEET", 0.0)
                if abs(vy) < self.deadband:
                    vy = 0.0

                v_top_idx = int(np.argmax(v_probs_arr))
                v_conf = float(v_probs_arr[v_top_idx])
                raw_v = str(self.v_classes[v_top_idx])
                if v_conf >= self.threshold:
                    v_cmd = "UP" if raw_v == "BOTH_HANDS" else "DOWN"

        # 3. Single-axis orthogonal gating (human thinks of one movement axis at a time)
        if axis_lock == "HORIZONTAL":
            vy, v_cmd = 0.0, "LOCKED"
        elif axis_lock == "VERTICAL":
            vx, h_cmd = 0.0, "LOCKED"
        elif self.single_axis and (self.vertical_model is not None or is_unified_four_class):
            if abs(vx) >= abs(vy):
                vy, v_cmd = 0.0, "NEUTRAL"
            else:
                vx, h_cmd = 0.0, "NEUTRAL"

        return {
            "vx": float(vx),
            "vy": float(vy),
            "h_command": h_cmd,
            "v_command": v_cmd,
            "probabilities": {**h_probs, **v_probs},
        }