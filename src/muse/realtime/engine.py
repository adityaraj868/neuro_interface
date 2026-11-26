from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np


class BCIInferenceEngine:
    """
    Consumes buffered EEG windows and runs live classification against
    a serialized CSP + LDA pipeline (model.joblib).
    """
    def __init__(self, model_path: str | Path, threshold: float = 0.60):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        self.threshold = threshold
        self.classes_ = list(self.model.classes_)

    def predict_window(self, window: np.ndarray) -> Dict[str, Any]:
        if window.ndim != 2:
            raise ValueError(f"Expected 2D array (n_channels, n_samples), got {window.shape}")

        # Reshape (64, 480) -> (1, 64, 480) for pipeline inference
        batch_window = np.expand_dims(window, axis=0)

        probabilities = self.model.predict_proba(batch_window)[0]
        prob_dict = {cls: float(prob) for cls, prob in zip(self.classes_, probabilities)}

        top_idx = int(np.argmax(probabilities))
        top_class = self.classes_[top_idx]
        top_prob = float(probabilities[top_idx])

        command = top_class if top_prob >= self.threshold else "NEUTRAL"

        return {
            "command": command,
            "confidence": top_prob,
            "probabilities": prob_dict,
        }