"""Real-time dual-model BCI inference engine with bandpass filtering and single-axis gating."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from scipy.signal import butter, filtfilt


class BCIInferenceEngine:
    """Runs 8-30 Hz bandpass filtering, temporal smoothing, and orthogonal 2D motor imagery inference."""

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
        smoothing_alpha: float = 0.35,
        history_len: int = 4,
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
        self.smoothing_alpha = smoothing_alpha
        self.history_len = max(1, history_len)

        self._prob_history: list[dict[str, float]] = []
        self._cmd_history: list[tuple[str, str]] = []
        self._smoothed_probs: dict[str, float] = {}

        nyq = 0.5 * sfreq
        self._b, self._a = butter(4, [l_freq / nyq, h_freq / nyq], btype="band")

    def reset_state(self) -> None:
        """Reset temporal smoothing and command history."""
        self._prob_history.clear()
        self._cmd_history.clear()
        self._smoothed_probs.clear()

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

        # 1. Decode raw probability distribution(s): unified 4-class or dual binary
        h_probs_arr = self.horizontal_model.predict_proba(batch_window)[0]
        h_probs = {str(cls): float(p) for cls, p in zip(self.h_classes, h_probs_arr)}

        is_unified_four_class = self.vertical_model is None and (
            "BOTH_HANDS" in h_probs or "BOTH_FEET" in h_probs
        )

        v_probs: Dict[str, float] = {}
        if not is_unified_four_class and self.vertical_model is not None:
            v_probs_arr = self.vertical_model.predict_proba(batch_window)[0]
            v_probs = {str(cls): float(p) for cls, p in zip(self.v_classes, v_probs_arr)}

        raw_all_probs = {**h_probs, **v_probs}

        # 2. Temporal Smoothing via Exponential Moving Average (EMA)
        if not self._smoothed_probs:
            self._smoothed_probs = dict(raw_all_probs)
        else:
            alpha = self.smoothing_alpha
            for k, val in raw_all_probs.items():
                self._smoothed_probs[k] = alpha * val + (1.0 - alpha) * self._smoothed_probs.get(k, val)

        # Re-normalize smoothed probabilities
        if is_unified_four_class:
            total_p = sum(self._smoothed_probs[k] for k in ["LEFT", "RIGHT", "BOTH_HANDS", "BOTH_FEET"] if k in self._smoothed_probs)
            if total_p > 0:
                smoothed = {k: self._smoothed_probs[k] / total_p for k in self._smoothed_probs}
            else:
                smoothed = dict(self._smoothed_probs)
        else:
            smoothed = dict(self._smoothed_probs)

        # 3. Decision Logic and Velocity Mapping
        if is_unified_four_class:
            four_classes = [c for c in ["LEFT", "RIGHT", "BOTH_HANDS", "BOTH_FEET"] if c in smoothed]
            top_class = max(four_classes, key=lambda c: smoothed[c])
            top_conf = smoothed[top_class]

            vx = smoothed.get("RIGHT", 0.0) - smoothed.get("LEFT", 0.0)
            vy = smoothed.get("BOTH_HANDS", 0.0) - smoothed.get("BOTH_FEET", 0.0)

            # Confidence thresholding
            if top_conf < self.threshold or (abs(vx) < self.deadband and abs(vy) < self.deadband):
                h_cmd, v_cmd = "NEUTRAL", "NEUTRAL"
                vx, vy = 0.0, 0.0
            else:
                if top_class == "RIGHT":
                    h_cmd, v_cmd = "RIGHT", "NEUTRAL"
                    vx = max(self.deadband, abs(vx))
                    vy = 0.0
                elif top_class == "LEFT":
                    h_cmd, v_cmd = "LEFT", "NEUTRAL"
                    vx = -max(self.deadband, abs(vx))
                    vy = 0.0
                elif top_class == "BOTH_HANDS":
                    h_cmd, v_cmd = "NEUTRAL", "UP"
                    vy = max(self.deadband, abs(vy))
                    vx = 0.0
                elif top_class == "BOTH_FEET":
                    h_cmd, v_cmd = "NEUTRAL", "DOWN"
                    vy = -max(self.deadband, abs(vy))
                    vx = 0.0
                else:
                    h_cmd, v_cmd = "NEUTRAL", "NEUTRAL"
                    vx, vy = 0.0, 0.0

        else:
            # Dual-binary setup
            h_right = smoothed.get("RIGHT", 0.0)
            h_left = smoothed.get("LEFT", 0.0)
            vx = h_right - h_left
            if abs(vx) < self.deadband:
                vx = 0.0

            h_top = "RIGHT" if h_right > h_left else "LEFT"
            h_conf = max(h_right, h_left)
            h_cmd = h_top if h_conf >= self.threshold and abs(vx) > 0.0 else "NEUTRAL"

            vy = 0.0
            v_cmd = "NEUTRAL"
            if self.vertical_model is not None:
                v_up = smoothed.get("BOTH_HANDS", 0.0)
                v_down = smoothed.get("BOTH_FEET", 0.0)
                vy = v_up - v_down
                if abs(vy) < self.deadband:
                    vy = 0.0

                v_top = "UP" if v_up > v_down else "DOWN"
                v_conf = max(v_up, v_down)
                v_cmd = v_top if v_conf >= self.threshold and abs(vy) > 0.0 else "NEUTRAL"

        # 4. Single-axis orthogonal gating
        if axis_lock == "HORIZONTAL":
            vy, v_cmd = 0.0, "LOCKED"
        elif axis_lock == "VERTICAL":
            vx, h_cmd = 0.0, "LOCKED"
        elif self.single_axis:
            if not is_unified_four_class and self.vertical_model is not None:
                if abs(vy) > abs(vx) + self.deadband:
                    vx, h_cmd = 0.0, "NEUTRAL"
                elif abs(vx) >= abs(vy):
                    vy, v_cmd = 0.0, "NEUTRAL"
                elif v_cmd == "UP" and abs(vx) >= self.deadband and h_cmd in ("LEFT", "RIGHT"):
                    vy, v_cmd = 0.0, "NEUTRAL"
                else:
                    vx, h_cmd = 0.0, "NEUTRAL"
            else:
                if abs(vx) >= abs(vy):
                    vy, v_cmd = 0.0, "NEUTRAL"
                else:
                    vx, h_cmd = 0.0, "NEUTRAL"

        # 5. Majority vote history buffer for jitter suppression
        self._cmd_history.append((h_cmd, v_cmd))
        if len(self._cmd_history) > self.history_len:
            self._cmd_history.pop(0)

        # Majority vote on active command
        h_votes = [c[0] for c in self._cmd_history]
        v_votes = [c[1] for c in self._cmd_history]
        from collections import Counter
        stable_h = Counter(h_votes).most_common(1)[0][0]
        stable_v = Counter(v_votes).most_common(1)[0][0]

        if stable_h == "NEUTRAL":
            vx = 0.0
            h_cmd = "NEUTRAL"
        if stable_v == "NEUTRAL":
            vy = 0.0
            v_cmd = "NEUTRAL"

        return {
            "vx": float(vx),
            "vy": float(vy),
            "h_command": h_cmd,
            "v_command": v_cmd,
            "probabilities": {**raw_all_probs},
            "smoothed_probabilities": {**smoothed},
        }