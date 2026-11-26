from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import numpy as np


class BCIInferenceEngine:
    """
    Consumes buffered EEG windows (n_channels, n_samples) and runs live
    2D motor-imagery classification using decoupled horizontal (LEFT/RIGHT)
    and vertical (BOTH_HANDS/BOTH_FEET) pipelines.
    """
    def __init__(
        self,
        horizontal_model_path: str | Path,
        vertical_model_path: Optional[str | Path] = None,
        threshold: float = 0.60,
        deadband: float = 0.15,
    ):
        self.horizontal_path = Path(horizontal_model_path)
        if not self.horizontal_path.exists():
            raise FileNotFoundError(f"Horizontal model not found: {self.horizontal_path}")

        self.horizontal_model = joblib.load(self.horizontal_path)
        self.h_classes = list(self.horizontal_model.classes_)

        self.vertical_model = None
        self.v_classes = []
        if vertical_model_path is not None:
            self.vertical_path = Path(vertical_model_path)
            if not self.vertical_path.exists():
                raise FileNotFoundError(f"Vertical model not found: {self.vertical_path}")
            self.vertical_model = joblib.load(self.vertical_path)
            self.v_classes = list(self.vertical_model.classes_)

        self.threshold = threshold
        self.deadband = deadband

    def predict_window(self, window: np.ndarray) -> Dict[str, Any]:
        if window.ndim != 2:
            raise ValueError(f"Expected 2D array (n_channels, n_samples), got {window.shape}")

        batch_window = np.expand_dims(window, axis=0)

        # 1. Horizontal axis (X): LEFT vs RIGHT
        h_probs_arr = self.horizontal_model.predict_proba(batch_window)[0]
        h_probs = {cls: float(p) for cls, p in zip(self.h_classes, h_probs_arr)}
        vx = h_probs.get("RIGHT", 0.0) - h_probs.get("LEFT", 0.0)
        if abs(vx) < self.deadband:
            vx = 0.0

        h_top_idx = int(np.argmax(h_probs_arr))
        h_cmd = self.h_classes[h_top_idx] if h_probs_arr[h_top_idx] >= self.threshold else "NEUTRAL"

        # 2. Vertical axis (Y): BOTH_HANDS (UP) vs BOTH_FEET (DOWN)
        v_probs: Dict[str, float] = {}
        vy = 0.0
        v_cmd = "NEUTRAL"
        if self.vertical_model is not None:
            v_probs_arr = self.vertical_model.predict_proba(batch_window)[0]
            v_probs = {cls: float(p) for cls, p in zip(self.v_classes, v_probs_arr)}
            vy = v_probs.get("BOTH_HANDS", 0.0) - v_probs.get("BOTH_FEET", 0.0)
            if abs(vy) < self.deadband:
                vy = 0.0

            v_top_idx = int(np.argmax(v_probs_arr))
            raw_v = self.v_classes[v_top_idx]
            if v_probs_arr[v_top_idx] >= self.threshold:
                v_cmd = "UP" if raw_v == "BOTH_HANDS" else "DOWN"

        return {
            "vx": float(vx),
            "vy": float(vy),
            "h_command": h_cmd,
            "v_command": v_cmd,
            "probabilities": {**h_probs, **v_probs},
        }