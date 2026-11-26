# 2D BCI Cursor Control & Dual-Model Integration Guide

This document explains the end-to-end architecture, signal processing pipeline, model artifacts, and integration specifications for 2D cursor control in the **MUSE Brain-Computer Interface** system.

---

## 1. System Overview: Dual-Model 2D Cursor Architecture

To navigate a computer cursor on a 2D screen $(X, Y)$, the system uses two specialized binary decoders rather than a complex 4-class classifier. This design keeps decoding accuracy substantially higher and decouples horizontal and vertical motor intents:

```text
                     ┌────────────────────────────────────────────────────────┐
                     │                 Incoming EEG Signal                    │
                     │          (64 channels, 160 Hz, 3.0s window)            │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
                                                ▼
                     ┌────────────────────────────────────────────────────────┐
                     │                Preprocessing Pipeline                  │
                     │  • 8–30 Hz Bandpass Filter (Mu & Beta rhythms)         │
                     │  • Common Average Reference (CAR)                      │
                     │  • Epoch slicing: [tmin=1.0s, tmax=4.0s] (481 samples) │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       │                                                 │
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │     X-Axis Model (Horizontal)│                  │     Y-Axis Model (Vertical)  │
        │  Artifact:                  │                   │  Artifact:                  │
        │  results/unilateral_lda_    │                   │  results/bilateral_lda_     │
        │  grouped/model.joblib       │                   │  grouped/model.joblib       │
        ├─────────────────────────────┤                   ├─────────────────────────────┤
        │  Trained on: Runs 4, 8, 12  │                   │  Trained on: Runs 6, 10, 14 │
        │  Classes: [LEFT, RIGHT]     │                   │  Classes: [FEET, HANDS]     │
        └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                       │                                                 │
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │  Horizontal Velocity (vx):  │                   │  Vertical Velocity (vy):    │
        │  vx = P(RIGHT) - P(LEFT)    │                   │  vy = P(HANDS) - P(FEET)    │
        └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
                     ┌────────────────────────────────────────────────────────┐
                     │              2D Cursor Position Integrator             │
                     │       X(t) = X(t-1) + alpha * vx * step_size           │
                     │       Y(t) = Y(t-1) + alpha * vy * step_size           │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
                                                ▼
                                    Screen / Cursor Movement
```

---

## 2. Physiological Grounding & Class Mappings

The PhysioNet EEGMMIDB dataset records 4 distinct motor imagery actions across separate runs:

| Direction / Axis | Physical Imagery Cue | Run Numbers | Output Class Name | Neurophysiological Basis |
| :--- | :--- | :--- | :--- | :--- |
| **X- (Left)** | Imagined **Left Fist** | 4, 8, 12 | `LEFT` | Contralateral desynchronization over Right Motor Cortex ($C_4$) |
| **X+ (Right)** | Imagined **Right Fist** | 4, 8, 12 | `RIGHT` | Contralateral desynchronization over Left Motor Cortex ($C_3$) |
| **Y+ (Up)** | Imagined **Both Fists** | 6, 10, 14 | `BOTH_HANDS` | Bilateral desynchronization across primary motor strip ($C_3, C_4, C_z$) |
| **Y- (Down)** | Imagined **Both Feet** | 6, 10, 14 | `BOTH_FEET` | Medial central activation/desynchronization along midline ($C_z, P_z$) |

---

## 3. Detailed Component Breakdown

### 3.1 Preprocessing Pipeline (`src/muse/preprocessing/epochs.py`)
Each raw continuous EDF recording undergoes identical signal conditioning:
1. **Band-pass Filtering (`8.0 – 30.0 Hz`):** Isolates the Sensorimotor Rhythms (Mu rhythm: 8–12 Hz and Beta rhythm: 13–30 Hz) where Event-Related Desynchronization (ERD) occurs during imagined movement. Suppresses slow baseline DC drift (<0.5 Hz) and line noise harmonics (50/60 Hz).
2. **Common Average Reference (CAR):** Re-references each of the 64 electrode channels to the instantaneous spatial mean across all electrodes, removing shared volume-conduction noise.
3. **Temporal Epoching (`1.0s to 4.0s`):** Slices a 3.0-second task execution window starting 1.0 second after the visual cue. At 160 Hz, this yields $481$ time samples ($3 \times 160 + 1$).
4. **Peak-to-Peak Amplitude Rejection (`150 µV`):** Rejects corrupted epochs containing muscle twitches or high-amplitude eye blinks.

### 3.2 Common Spatial Patterns (`CSP`)
* **What it does:** Solves a generalized eigenvalue problem on trial covariance matrices to find spatial filters that maximize the variance for one class while simultaneously minimizing variance for the other.
* **Component Count:** 4 spatial filters (2 top components for Class 1, 2 top components for Class 2).
* **Feature Vector:** The log-variance of the spatially filtered signals, yielding a compact 4-dimensional feature vector per trial.

### 3.3 Classifier: Shrinkage Linear Discriminant Analysis (`LDA`)
* **Why Shrinkage LDA:** Uses Ledoit-Wolf analytic shrinkage covariance estimation. This prevents overfitting in high-dimensional settings and ensures numeric stability and well-calibrated class posterior probabilities (`predict_proba`).

---

## 4. Current Validated Model Artifacts

| Model Axis | Artifact Path | Training Data | Evaluation Metric | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Horizontal (X)** | `results/unilateral_lda_grouped/model.joblib` | S001 Runs 4, 8, 12 | Leave-One-Recording-Out | **59.09% Balanced Acc** (22 epochs) |
| **Vertical (Y)** | `results/bilateral_lda_grouped/model.joblib` | S001 Runs 6, 10, 14 | Leave-One-Recording-Out | **96.67% Balanced Acc** (30 epochs) |

Both models are saved as standard scikit-learn `Pipeline([('csp', CSP), ('lda', LinearDiscriminantAnalysis)])`.

---

## 5. Partner Integration Specification (`BCIInferenceEngine`)

To use these models in your partner's inference engine and cursor simulation, follow this integration pattern:

### 5.1 Loading the Models
```python
import joblib
from pathlib import Path

# Load horizontal (X-axis) and vertical (Y-axis) models
x_model_path = Path("results/unilateral_lda_grouped/model.joblib")
y_model_path = Path("results/bilateral_lda_grouped/model.joblib")

x_pipeline = joblib.load(x_model_path)
y_pipeline = joblib.load(y_model_path)

# Verify classes
print("X-axis classes:", x_pipeline.classes_)  # ['LEFT', 'RIGHT']
print("Y-axis classes:", y_pipeline.classes_)  # ['BOTH_FEET', 'BOTH_HANDS']
```

### 5.2 Processing an Epoch & Computing Velocity
```python
import numpy as np

def compute_2d_velocity(x_pipeline, y_pipeline, epoch_array):
    """
    epoch_array shape: (1, n_channels=64, n_times=481)
    Returns: (vx, vy) normalized between -1.0 and +1.0
    """
    # Horizontal prediction
    x_probs = x_pipeline.predict_proba(epoch_array)[0]
    # classes: [0: 'LEFT', 1: 'RIGHT']
    p_left, p_right = x_probs[0], x_probs[1]
    vx = p_right - p_left  # > 0 moves Right, < 0 moves Left

    # Vertical prediction
    y_probs = y_pipeline.predict_proba(epoch_array)[0]
    # classes: [0: 'BOTH_FEET', 1: 'BOTH_HANDS']
    p_feet, p_hands = y_probs[0], y_probs[1]
    vy = p_hands - p_feet  # > 0 moves Up, < 0 moves Down

    return vx, vy
```

### 5.3 Smoothing and Deadband Filtering (Recommended for Cursor Stability)
Raw model probabilities can fluctuate trial-to-trial. To achieve smooth cursor movement:
1. **Deadband Filter:** If $|v| < 0.15$, treat it as neutral ($v = 0$) so the cursor doesn't drift when intent is ambiguous.
2. **Exponential Smoothing (Leaky Integrator):**
   ```python
   def update_cursor_position(x, y, vx, vy, alpha=0.3, step_size=15.0, deadband=0.15):
       if abs(vx) < deadband:
           vx = 0.0
       if abs(vy) < deadband:
           vy = 0.0

       new_x = x + (alpha * vx * step_size)
       new_y = y + (alpha * vy * step_size)
       return new_x, new_y
   ```
