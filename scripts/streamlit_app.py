#!/usr/bin/env python3
"""Launch the MUSE offline experiment dashboard and Live 2D BCI Arena with Streamlit."""

from __future__ import annotations

import importlib
import inspect
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from muse.data import load_imagery_dataset
from muse.event_mapping import labels_for_run
from muse.models import make_eegnet_classifier, make_riemannian_svm_pipeline
from muse.preprocessing import PreprocessingConfig
from muse.realtime.buffer import SlidingRingBuffer
import muse.realtime.engine as _rt_engine
from muse.realtime.streamer import MockEEGStreamer

if "filter_data" not in inspect.getsource(_rt_engine.BCIInferenceEngine._preprocess):
    importlib.reload(_rt_engine)
BCIInferenceEngine = _rt_engine.BCIInferenceEngine

ARENA_MODEL_OPTIONS: dict[str, str] = {
    "S001 Calibrated 4-Class Riemannian (Blind 4-Way)": "s001_4class_riemannian",
    "S001 Calibrated 4-Class FBCSP + LDA (Blind 4-Way)": "s001_4class_fbcsp",
    "S001 Calibrated 4-Class CSP + LDA (Blind 4-Way)": "s001_4class_lda",
    "Unified 4-Class Riemannian (All Runs)": "four_class_riemannian",
    "Unified 4-Class CSP + LDA (All Runs)": "four_class_lda",
    "Dual Binary CSP + LDA": "csp_lda",
    "Dual Binary Riemannian (Tangent Space + SVM)": "riemannian",
    "Dual Binary EEGNet (CNN)": "eegnet",
}

ARENA_TARGETS: list[tuple[float, float]] = [
    (65.0, 65.0),
    (-65.0, 65.0),
    (-65.0, -65.0),
    (65.0, -65.0),
    (0.0, 75.0),
    (75.0, 0.0),
]

THOUGHT_SIM_RUNS: list[tuple[int, str]] = [
    (12, "S001R12.edf"),
    (14, "S001R14.edf"),
]


def _resolve_sensorimotor_indices(ch_names: list[str]) -> tuple[int, int, int]:
    """Return (C3, Cz, C4) indices using normalized names with fallback to (8, 10, 12)."""
    normalized = {name.rstrip(".").upper(): idx for idx, name in enumerate(ch_names)}
    return (
        normalized.get("C3", 8),
        normalized.get("CZ", 10),
        normalized.get("C4", 12),
    )


def extract_thought_epochs(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Extract raw (64, 481) [1.0s, 4.0s] task epochs and trial provenance from Subject 1."""
    import mne

    pools: dict[str, list[np.ndarray]] = {"LEFT": [], "RIGHT": [], "UP": [], "DOWN": []}
    meta_pools: dict[str, list[dict[str, Any]]] = {"LEFT": [], "RIGHT": [], "UP": [], "DOWN": []}
    s001_dir = data_dir / "MNE-eegbci-data/files/eegmmidb/1.0.0/S001"
    sfreq = 160.0
    ch_names: list[str] = []

    for run, filename in THOUGHT_SIM_RUNS:
        edf_path = s001_dir / filename
        if not edf_path.exists():
            raise FileNotFoundError(f"Missing Subject 1 EDF file for thought simulator: {edf_path}")

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw)
        sfreq = float(raw.info["sfreq"])
        picks = mne.pick_types(
            raw.info, eeg=True, meg=False, stim=False, eog=False, exclude="bads"
        )
        if not ch_names:
            ch_names = [raw.ch_names[idx].rstrip(".") for idx in picks]
        events, ann_id = mne.events_from_annotations(raw, verbose=False)
        mapping = labels_for_run(run)
        label_to_event = {label: desc for desc, label in mapping.items()}
        selected_id = {label: ann_id[desc] for desc, label in mapping.items()}
        selected_events = events[np.isin(events[:, 2], list(selected_id.values()))]
        epochs = mne.Epochs(
            raw,
            selected_events,
            event_id=selected_id,
            tmin=0.5,
            tmax=3.5,
            baseline=None,
            picks=picks,
            reject=None,
            reject_by_annotation=True,
            preload=True,
            verbose=False,
        )
        code_to_label = {code: label for label, code in selected_id.items()}
        epoch_data = epochs.get_data(copy=True)
        retained_codes = epochs.events[:, 2]
        retained_selections = epochs.selection

        for local_idx, (code, run_trial_idx) in enumerate(zip(retained_codes, retained_selections)):
            label = code_to_label[int(code)]
            key = {"BOTH_HANDS": "UP", "BOTH_FEET": "DOWN"}.get(label, label)
            pools[key].append(epoch_data[local_idx : local_idx + 1])
            meta_pools[key].append(
                {
                    "source_file": filename,
                    "run": int(run),
                    "trial_index": int(run_trial_idx) + 1,
                    "event_code": label_to_event[label],
                    "sfreq": sfreq,
                }
            )

    c3_idx, cz_idx, c4_idx = _resolve_sensorimotor_indices(ch_names)
    return {
        key: {
            "X": np.concatenate(chunks, axis=0),
            "meta": meta_pools[key],
            "sfreq": sfreq,
            "c3_idx": c3_idx,
            "cz_idx": cz_idx,
            "c4_idx": c4_idx,
        }
        for key, chunks in pools.items()
    }


def resolve_arena_models(
    model_key: str,
    results_dir: Path,
    data_dir: Path,
) -> tuple[Path, Path | None, str]:
    """Return horizontal/vertical .joblib paths (or single 4-class path + None)."""
    if model_key == "s001_4class_riemannian":
        model_path = results_dir / "s001_4class_riemannian/model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing Subject-1 calibrated 4-class Riemannian model: {model_path}. "
                "Run `PYTHONPATH=src .venv/bin/python scripts/train_s001_4class.py`."
            )
        return (
            model_path,
            None,
            "Loaded S001 calibrated 4-class Riemannian model (5-fold Stratified CV; calibrated on Runs 04/06/08/10/12/14)",
        )

    if model_key == "s001_4class_fbcsp":
        model_path = results_dir / "s001_4class_fbcsp/model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing Subject-1 calibrated 4-class FBCSP model: {model_path}. "
                "Run `PYTHONPATH=src .venv/bin/python scripts/train_s001_4class.py`."
            )
        return (
            model_path,
            None,
            "Loaded S001 calibrated 4-class Filter Bank CSP+LDA model (5-fold Stratified CV)",
        )

    if model_key == "s001_4class_lda":
        model_path = results_dir / "s001_4class_lda/model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing Subject-1 calibrated 4-class CSP+LDA model: {model_path}. "
                "Run `PYTHONPATH=src .venv/bin/python scripts/train_s001_4class.py`."
            )
        return (
            model_path,
            None,
            "Loaded S001 calibrated 4-class CSP+LDA model (5-fold Stratified CV)",
        )

    if model_key == "four_class_riemannian":
        model_path = results_dir / "four_class_riemannian_grouped/model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing unified 4-class Riemannian model: {model_path}. "
                "Run `PYTHONPATH=src .venv/bin/python scripts/run_experiment.py --task four_class --model riemannian`."
            )
        return model_path, None, "Loaded unified 4-class Riemannian joblib model (blind 4-way)"

    if model_key == "four_class_lda":
        model_path = results_dir / "four_class_lda_grouped/model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing unified 4-class CSP+LDA model: {model_path}. "
                "Run `PYTHONPATH=src .venv/bin/python scripts/run_experiment.py --task four_class --model lda`."
            )
        return model_path, None, "Loaded unified 4-class CSP+LDA joblib model (blind 4-way)"

    if model_key == "csp_lda":
        h_path = results_dir / "unilateral_lda_grouped/model.joblib"
        v_path = results_dir / "bilateral_lda_grouped/model.joblib"
        if not (h_path.exists() and v_path.exists()):
            raise FileNotFoundError(
                "Missing CSP+LDA artifacts (unilateral_lda_grouped / bilateral_lda_grouped)."
            )
        return h_path, v_path, "Loaded saved dual-binary CSP+LDA joblib pipelines"

    grouped_h = results_dir / f"unilateral_{model_key}_grouped/model.joblib"
    grouped_v = results_dir / f"bilateral_{model_key}_grouped/model.joblib"
    if grouped_h.exists() and grouped_v.exists():
        return grouped_h, grouped_v, f"Loaded saved {model_key} joblib models"

    cache_dir = results_dir / "realtime_models" / model_key
    cached_h = cache_dir / "horizontal.joblib"
    cached_v = cache_dir / "vertical.joblib"
    if cached_h.exists() and cached_v.exists():
        return cached_h, cached_v, f"Loaded cached {model_key} models"

    def _build_estimator() -> Any:
        if model_key == "riemannian":
            return make_riemannian_svm_pipeline(estimator="oas", C=1.0, probability=True)
        if model_key == "eegnet":
            return make_eegnet_classifier(epochs=30, batch_size=16, lr=1e-3, random_state=42)
        raise ValueError(f"Unsupported model key: {model_key}")

    config = PreprocessingConfig()
    h_data = load_imagery_dataset(data_dir, [1], [4, 8, 12], config=config)
    v_data = load_imagery_dataset(data_dir, [1], [6, 10, 14], config=config)

    h_model = _build_estimator().fit(h_data.X, h_data.y)
    v_model = _build_estimator().fit(v_data.X, v_data.y)

    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(h_model, cached_h)
    joblib.dump(v_model, cached_v)
    return cached_h, cached_v, f"Trained on S001 (481-sample epochs) & cached to {cache_dir.name}/"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _percentage(value: float) -> str:
    return f"{value:.1%}"


def discover_experiments(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []
    return sorted(
        path.parent
        for path in results_dir.glob("*/metrics.json")
        if path.is_file()
    )


def classification_report_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Convert sklearn's report dictionary into a display-friendly table."""
    rows = []
    for label, values in report.items():
        if isinstance(values, dict) and "precision" in values:
            rows.append(
                {
                    "class": label,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1-score"],
                    "support": int(values["support"]),
                }
            )
    return pd.DataFrame(rows)


def _render_arena_svg(
    x: float,
    y: float,
    tx: float,
    ty: float,
    trail: list[tuple[float, float]],
    hits: int,
) -> str:
    cx = 200 + (x / 100.0) * 170
    cy = 200 - (y / 100.0) * 170
    tcx = 200 + (tx / 100.0) * 170
    tcy = 200 - (ty / 100.0) * 170
    trail_pts = " ".join(
        f"{200 + (px / 100.0) * 170:.1f},{200 - (py / 100.0) * 170:.1f}"
        for px, py in trail[-20:]
    )
    return (
        '<div style="display:flex; justify-content:center; padding:8px 0; min-height:436px;">'
        '<svg width="420" height="420" viewBox="0 0 400 400" '
        'style="background:#0f172a; border-radius:14px; border:1px solid #334155; box-shadow:0 8px 24px rgba(0,0,0,0.45); contain: strict;">'
        '<line x1="200" y1="20" x2="200" y2="380" stroke="#334155" stroke-dasharray="4 4"/>'
        '<line x1="20" y1="200" x2="380" y2="200" stroke="#334155" stroke-dasharray="4 4"/>'
        '<rect x="20" y="20" width="360" height="360" fill="none" stroke="#475569" stroke-width="1.5"/>'
        '<text x="200" y="14" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="sans-serif">UP (BOTH HANDS)</text>'
        '<text x="200" y="395" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="sans-serif">DOWN (BOTH FEET)</text>'
        '<text x="12" y="204" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="sans-serif" transform="rotate(-90 12,204)">LEFT</text>'
        '<text x="390" y="204" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="sans-serif" transform="rotate(90 390,204)">RIGHT</text>'
        f'<circle cx="{tcx:.1f}" cy="{tcy:.1f}" r="26" fill="rgba(16, 185, 129, 0.22)" stroke="#10b981" stroke-width="2.5" style="transition: all 0.3s ease;"/>'
        f'<circle cx="{tcx:.1f}" cy="{tcy:.1f}" r="5" fill="#10b981" style="transition: all 0.3s ease;"/>'
        f'<polyline fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-opacity="0.65" points="{trail_pts}" style="transition: points 0.2s ease-out;"/>'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="11" fill="#38bdf8" stroke="#ffffff" stroke-width="2.5" style="transition: cx 0.22s cubic-bezier(0.25, 1, 0.5, 1), cy 0.22s cubic-bezier(0.25, 1, 0.5, 1);"/>'
        f'<text x="32" y="38" fill="#e2e8f0" font-size="12" font-family="monospace">Targets Hit: {hits}</text>'
        "</svg>"
        "</div>"
    )


@st.cache_resource(show_spinner=False)
def _cached_arena_models(selected_key: str, res_dir_str: str) -> tuple[Path, Path | None, str]:
    return resolve_arena_models(
        selected_key,
        Path(res_dir_str),
        ROOT / "data" / "physionet",
    )


@st.cache_data(show_spinner=False)
def _cached_thought_epochs(data_dir_str: str) -> dict[str, dict[str, Any]]:
    return extract_thought_epochs(Path(data_dir_str))


def _reset_arena_state() -> None:
    st.session_state["arena_x"] = 0.0
    st.session_state["arena_y"] = 0.0
    st.session_state["arena_tx"] = ARENA_TARGETS[0][0]
    st.session_state["arena_ty"] = ARENA_TARGETS[0][1]
    st.session_state["arena_trail"] = [(0.0, 0.0)]
    st.session_state["arena_hits"] = 0
    st.session_state["arena_last_trial"] = None
    st.session_state["session_trials"] = []


def _on_think_direction(
    pool_key: str,
    ground_truth_label: str,
    model_key: str,
    results_dir_str: str,
    alpha: float,
) -> None:
    try:
        h_model, v_model, _ = _cached_arena_models(model_key, results_dir_str)
    except Exception:
        return

    epoch_pools = _cached_thought_epochs(str(ROOT / "data" / "physionet"))
    if pool_key not in epoch_pools:
        return
    pool_bundle = epoch_pools[pool_key]
    pool_X = pool_bundle["X"]
    sample_idx = random.randrange(len(pool_X))
    epoch = pool_X[sample_idx]
    trial_meta = pool_bundle["meta"][sample_idx]

    engine = BCIInferenceEngine(
        horizontal_model_path=h_model,
        vertical_model_path=v_model,
        threshold=0.50,
        deadband=0.10,
        single_axis=True,
        smoothing_alpha=0.35,
        history_len=3,
    )
    filtered_epoch = engine._preprocess(epoch)
    t0 = time.perf_counter()
    res = engine.predict_window(epoch, axis_lock="AUTO")
    lat_ms = (time.perf_counter() - t0) * 1000.0

    c3_idx = pool_bundle["c3_idx"]
    cz_idx = pool_bundle["cz_idx"]
    c4_idx = pool_bundle["c4_idx"]
    filtered_uv = filtered_epoch * 1e6
    bandpower_uv2 = {
        "C3": float(np.mean(filtered_uv[c3_idx] ** 2)),
        "Cz": float(np.mean(filtered_uv[cz_idx] ** 2)),
        "C4": float(np.mean(filtered_uv[c4_idx] ** 2)),
    }
    rms_uv = {
        "C3": float(np.sqrt(bandpower_uv2["C3"])),
        "Cz": float(np.sqrt(bandpower_uv2["Cz"])),
        "C4": float(np.sqrt(bandpower_uv2["C4"])),
    }

    cur_x = st.session_state.get("arena_x", 0.0)
    cur_y = st.session_state.get("arena_y", 0.0)
    x = max(-95.0, min(95.0, cur_x + alpha * res["vx"] * 35.0))
    y = max(-95.0, min(95.0, cur_y + alpha * res["vy"] * 35.0))
    tx = st.session_state.get("arena_tx", ARENA_TARGETS[0][0])
    ty = st.session_state.get("arena_ty", ARENA_TARGETS[0][1])
    hits = st.session_state.get("arena_hits", 0)
    trail = list(st.session_state.get("arena_trail", [(0.0, 0.0)]))
    trail.append((x, y))

    if math.hypot(x - tx, y - ty) <= 22.0:
        hits += 1
        tx, ty = random.choice([t for t in ARENA_TARGETS if t != (tx, ty)])

    if res["h_command"] in ("LEFT", "RIGHT"):
        predicted_label = res["h_command"]
    elif res["v_command"] == "UP":
        predicted_label = "UP (BOTH_HANDS)"
    elif res["v_command"] == "DOWN":
        predicted_label = "DOWN (BOTH_FEET)"
    else:
        predicted_label = "NEUTRAL"

    matched = predicted_label == ground_truth_label
    conf_pct = (
        max(res["probabilities"].values()) * 100.0 if res["probabilities"] else 0.0
    )
    trials = list(st.session_state.get("session_trials", []))
    trials.append(
        {
            "Trial #": len(trials) + 1,
            "Source EDF": trial_meta["source_file"],
            "Ground Truth": ground_truth_label,
            "Blind Prediction": predicted_label,
            "Match": "✅" if matched else "❌",
            "Confidence %": f"{conf_pct:.1f}%",
            "Latency (ms)": round(lat_ms, 1),
        }
    )
    st.session_state["session_trials"] = trials
    st.session_state["arena_x"] = x
    st.session_state["arena_y"] = y
    st.session_state["arena_tx"] = tx
    st.session_state["arena_ty"] = ty
    st.session_state["arena_trail"] = trail
    st.session_state["arena_hits"] = hits
    st.session_state["arena_last_trial"] = {
        "source_file": trial_meta["source_file"],
        "trial_index": trial_meta["trial_index"],
        "event_code": trial_meta["event_code"],
        "ground_truth": ground_truth_label,
        "shape": tuple(epoch.shape),
        "sfreq": float(trial_meta["sfreq"]),
        "filtered_epoch": filtered_epoch,
        "c3_idx": c3_idx,
        "cz_idx": cz_idx,
        "c4_idx": c4_idx,
        "bandpower_uv2": bandpower_uv2,
        "rms_uv": rms_uv,
        "predicted": predicted_label,
        "matched": matched,
        "h_command": res["h_command"],
        "v_command": res["v_command"],
        "vx": res["vx"],
        "vy": res["vy"],
        "latency_ms": lat_ms,
        "probabilities": res["probabilities"],
        "model_label": ARENA_MODEL_OPTIONS.get(model_key, model_key),
    }


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised interactively
        raise SystemExit(
            "Streamlit is not installed. Install project requirements and run "
            "`streamlit run scripts/streamlit_app.py`."
        ) from exc

    st.set_page_config(page_title="MUSE EEG Dashboard", page_icon="🧠", layout="wide")
    st.title("MUSE EEG Experiment & Real-Time BCI Dashboard")
    st.caption("Offline inspection of saved motor-imagery benchmarks + Live 2D EEG cursor control")

    with st.sidebar:
        st.header("Experiment")
        results_text = st.text_input("Results directory", str(ROOT / "results"))
        results_dir = Path(results_text).expanduser()
        experiments = discover_experiments(results_dir)
        if not experiments:
            st.error("No saved experiment metrics were found.")
            st.code("PYTHONPATH=src .venv/bin/python scripts/run_experiment.py")
            st.stop()
        experiment_labels = [path.name for path in experiments]
        default_run_index = (
            experiment_labels.index("s001_4class_riemannian")
            if "s001_4class_riemannian" in experiment_labels
            else 0
        )
        selected_name = st.selectbox("Saved run", experiment_labels, index=default_run_index)
        selected_dir = experiments[experiment_labels.index(selected_name)]

    metrics = _read_json(selected_dir / "metrics.json")
    config = _read_json(selected_dir / "config.json")
    predictions = pd.read_csv(selected_dir / "predictions.csv")

    metric_columns = st.columns(5)
    metric_columns[0].metric("Accuracy", _percentage(metrics.get("accuracy", 0.0)), border=True)
    metric_columns[1].metric(
        "Balanced accuracy", _percentage(metrics.get("balanced_accuracy", 0.0)), border=True
    )
    metric_columns[2].metric("Macro F1", _percentage(metrics.get("macro_f1", 0.0)), border=True)
    metric_columns[3].metric("Chance level", _percentage(metrics.get("chance_level", 0.5)), border=True)
    metric_columns[4].metric(
        "Epochs / groups",
        f"{metrics.get('n_epochs', '-')} / {metrics.get('n_groups', '-')}",
        border=True,
    )

    overview_tab, predictions_tab, configuration_tab, audit_tab, live_tab = st.tabs(
        ["Overview", "Predictions", "Configuration", "Dataset audit", "Live 2D BCI Arena"]
    )

    with overview_tab:
        left, right = st.columns(2)
        with left:
            st.subheader("Per-class metrics")
            report_df = classification_report_frame(metrics.get("classification_report", {}))
            st.dataframe(report_df, hide_index=True, width="stretch")
        with right:
            st.subheader("Confusion matrix")
            cm = metrics.get("confusion_matrix", [])
            n_classes = len(cm)
            raw_labels = metrics.get("labels") or metrics.get("class_labels") or [
                k
                for k, v in metrics.get("classification_report", {}).items()
                if isinstance(v, dict) and k not in {"macro avg", "weighted avg"}
            ]
            labels = (
                raw_labels[:n_classes]
                if len(raw_labels) >= n_classes
                else [f"Class {i}" for i in range(n_classes)]
            )
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            st.dataframe(cm_df, width="stretch")

        st.subheader("Evaluation notes")
        st.write(
            {
                k: v
                for k, v in metrics.items()
                if k not in {"classification_report", "confusion_matrix"}
            }
        )

    with predictions_tab:
        st.subheader("Out-of-fold predictions")
        if "group" in predictions.columns:
            groups = sorted(predictions["group"].unique().tolist())
            selected_groups = st.multiselect("Groups", groups, default=groups)
            filtered = predictions[predictions["group"].isin(selected_groups)]
        else:
            filtered = predictions

        if "correct" in filtered.columns:
            selected_status = st.radio("Show", ["All", "Correct only", "Errors only"], horizontal=True)
            if selected_status == "Correct only":
                filtered = filtered[filtered["correct"]]
            elif selected_status == "Errors only":
                filtered = filtered[~filtered["correct"]]

        st.dataframe(filtered, hide_index=True, width="stretch")
        st.download_button(
            "Download filtered predictions",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_dir.name}_predictions.csv",
            mime="text/csv",
        )

    with configuration_tab:
        st.json(config)
        model_path = selected_dir / "model.joblib"
        if model_path.exists():
            st.success(f"Saved model available: {model_path.name}")
        else:
            st.warning("No saved model is present for this experiment.")

    with audit_tab:
        audit_path = results_dir / "dataset_audit.json"
        if not audit_path.exists():
            st.info("No dataset audit found. Run scripts/inspect_dataset.py first.")
        else:
            audit = _read_json(audit_path)
            audit_rows = []
            for record in audit:
                audit_rows.append(
                    {
                        "subject": record.get("subject"),
                        "run": record.get("run"),
                        "channels": record.get("n_channels"),
                        "sampling rate (Hz)": record.get("sfreq"),
                        "duration (s)": record.get("duration_seconds"),
                        "annotations": record.get("annotation_counts"),
                        "task mapping": record.get("task_event_mapping"),
                    }
                )
            st.dataframe(pd.DataFrame(audit_rows), hide_index=True, width="stretch")

    with live_tab:
        _render_live_bci_arena(results_dir)


@st.fragment
def _render_live_bci_arena(results_dir: Path) -> None:
    st.subheader("Live 2D Motor Imagery Cursor & Target-Reaching Game")

    top_col1, top_col2 = st.columns([1.3, 1.2])
    with top_col1:
        arena_mode = st.segmented_control(
            "Arena Mode",
            ["Interactive 4-Direction Simulator", "Continuous 160 Hz Stream Replay"],
            default="Interactive 4-Direction Simulator",
        )
        if arena_mode is None:
            arena_mode = "Interactive 4-Direction Simulator"
    with top_col2:
        model_label = st.selectbox(
            "Model Architecture",
            list(ARENA_MODEL_OPTIONS.keys()),
            help="Hot-swap between Subject-1 calibrated 4-class models, CSP+LDA, Riemannian, and EEGNet decoders.",
        )
        model_key = ARENA_MODEL_OPTIONS[model_label]

    alpha = 0.35
    duration_sec = 12
    stream_mode = "Held-Out Vertical Imagery (S001R14: Both Fists vs Both Feet)"
    start_btn = False

    if arena_mode == "Continuous 160 Hz Stream Replay":
        stream_col1, stream_col2, stream_col3, stream_col4 = st.columns([1.8, 1, 1, 1])
        with stream_col1:
            stream_mode = st.selectbox(
                "EEG Stream Run",
                [
                    "Held-Out Vertical Imagery (S001R14: Both Fists vs Both Feet)",
                    "Held-Out Horizontal Imagery (S001R12: Left Fist vs Right Fist)",
                    "Vertical Imagery (S001R06: Both Fists vs Both Feet)",
                    "Horizontal Imagery (S001R04: Left Fist vs Right Fist)",
                    "Vertical Imagery (S001R10: Both Fists vs Both Feet)",
                    "Horizontal Imagery (S001R08: Left Fist vs Right Fist)",
                ],
            )
        with stream_col2:
            alpha = st.slider("Smoothing (α)", min_value=0.10, max_value=0.80, value=0.35, step=0.05)
        with stream_col3:
            duration_sec = st.slider("Run Duration (s)", min_value=5, max_value=30, value=12, step=1)
        with stream_col4:
            st.write("")
            start_btn = st.button("▶ Start Live Session", type="primary", width="stretch")

    st.caption(
        f"Mode: **{arena_mode}** · 481-sample (3.0s @ 160 Hz) window · "
        f"Single-Axis Winner-Take-All `{model_label}` Inference (`axis_lock=\"AUTO\"`)"
    )

    if "arena_x" not in st.session_state:
        _reset_arena_state()
    if "session_trials" not in st.session_state:
        st.session_state["session_trials"] = []

    try:
        h_model, v_model, model_source = _cached_arena_models(model_key, str(results_dir))
        gating_desc = (
            "True Blind 4-Way (axis_lock='AUTO', Single-Axis WTA)"
            if v_model is None
            else "Dual-Binary Orthogonal WTA (axis_lock='AUTO')"
        )
        models_ready = True
    except Exception as exc:
        h_model, v_model, model_source = None, None, f"Error: {exc}"
        gating_desc = "Unavailable"
        models_ready = False

    arena_col, telemetry_col = st.columns([1.05, 1.15])
    with arena_col:
        # 1. Render Arena SVG (Fixed dimensions, smooth CSS transition)
        arena_slot = st.empty() if start_btn else None
        if not start_btn:
            st.markdown(
                _render_arena_svg(
                    st.session_state["arena_x"],
                    st.session_state["arena_y"],
                    st.session_state["arena_tx"],
                    st.session_state["arena_ty"],
                    st.session_state["arena_trail"],
                    st.session_state["arena_hits"],
                ),
                unsafe_allow_html=True,
            )

        # 2. Render Scorecard Metrics directly
        session_trials = st.session_state["session_trials"]
        scorecard_slot = st.empty() if start_btn else None
        if not start_btn:
            sc1, sc2, sc3 = st.columns(3)
            if session_trials:
                n_correct = sum(1 for row in session_trials if row["Match"] == "✅")
                n_total = len(session_trials)
                mean_lat = sum(float(row["Latency (ms)"]) for row in session_trials) / n_total
                sc1.metric(
                    "Blind Session Accuracy",
                    f"{n_correct / n_total:.0%} ({n_correct}/{n_total})",
                    border=True,
                )
                sc2.metric("Targets Hit", st.session_state["arena_hits"], border=True)
                sc3.metric("Mean Latency", f"{mean_lat:.1f} ms", border=True)
            else:
                sc1.metric("Blind Session Accuracy", "—", border=True)
                sc2.metric("Targets Hit", st.session_state["arena_hits"], border=True)
                sc3.metric("Mean Latency", "—", border=True)

        # 3. Render D-pad buttons (Using on_click callbacks for zero-lag state updates)
        if arena_mode == "Interactive 4-Direction Simulator":
            st.markdown("#### 4-Direction Thought Simulator (Unseen Held-Out Trials)")
            st.caption(
                "Sample a real held-out `(64, 481)` EEG epoch from `S001R12.edf` / `S001R14.edf` "
                "and run blind 4-way inference (`axis_lock=\"AUTO\"`)."
            )
            dpad_r1_c1, dpad_r1_c2, dpad_r1_c3 = st.columns(3)
            with dpad_r1_c2:
                st.button(
                    "⬆️ Think UP (Both Fists)",
                    key="btn_think_up",
                    on_click=_on_think_direction,
                    args=("UP", "UP (BOTH_HANDS)", model_key, str(results_dir), alpha),
                    width="stretch",
                )

            dpad_r2_c1, dpad_r2_c2, dpad_r2_c3 = st.columns(3)
            with dpad_r2_c1:
                st.button(
                    "⬅️ Think LEFT (Left Fist)",
                    key="btn_think_left",
                    on_click=_on_think_direction,
                    args=("LEFT", "LEFT", model_key, str(results_dir), alpha),
                    width="stretch",
                )
            with dpad_r2_c2:
                st.button("Reset Arena", key="btn_reset_arena", on_click=_reset_arena_state, width="stretch")
            with dpad_r2_c3:
                st.button(
                    "➡️ Think RIGHT (Right Fist)",
                    key="btn_think_right",
                    on_click=_on_think_direction,
                    args=("RIGHT", "RIGHT", model_key, str(results_dir), alpha),
                    width="stretch",
                )

            dpad_r3_c1, dpad_r3_c2, dpad_r3_c3 = st.columns(3)
            with dpad_r3_c2:
                st.button(
                    "⬇️ Think DOWN (Both Feet)",
                    key="btn_think_down",
                    on_click=_on_think_direction,
                    args=("DOWN", "DOWN (BOTH_FEET)", model_key, str(results_dir), alpha),
                    width="stretch",
                )

    with telemetry_col:
        st.info(
            f"**Active Decoder:** {model_label}  \n"
            f"**Source:** {model_source}  \n"
            f"**Gating:** {gating_desc}"
        )

        last_trial = st.session_state.get("arena_last_trial")
        telemetry_slot = st.empty() if start_btn else None
        if not start_btn:
            if last_trial is not None:
                status_badge = "✅ Match" if last_trial["matched"] else "❌ Misclassified"
                summary_msg = (
                    f"**{status_badge}** — **Ground Truth:** `{last_trial['ground_truth']}` | "
                    f"**Blind Model Prediction ({last_trial['model_label']}):** `{last_trial['predicted']}`"
                )
                if last_trial["matched"]:
                    st.success(summary_msg)
                else:
                    st.warning(summary_msg)

                m1, m2, m3 = st.columns(3)
                m1.metric("Latency", f"{last_trial['latency_ms']:.1f} ms")
                m2.metric("Horizontal", last_trial["h_command"], f"vx={last_trial['vx']:+.2f}")
                m3.metric("Vertical", last_trial["v_command"], f"vy={last_trial['vy']:+.2f}")
                st.metric("Targets Reached", st.session_state["arena_hits"])

                st.markdown("**Live Class Probabilities**")
                for label in ["LEFT", "RIGHT", "BOTH_HANDS", "BOTH_FEET"]:
                    val = last_trial["probabilities"].get(label, 0.0)
                    st.progress(
                        min(1.0, max(0.0, val)),
                        text=f"{label}: {val * 100:.1f}%",
                    )

                st.markdown("#### Sampled EEG Trial Inspector (Live Dataset Proof)")
                with st.container(border=True):
                    st.markdown(f"**Trial Provenance** — {status_badge}")
                    prov_col1, prov_col2 = st.columns(2)
                    prov_col1.markdown(
                        f"- **Source EDF:** `{last_trial['source_file']}`\n"
                        f"- **Trial Index:** `#{last_trial['trial_index']}`\n"
                        f"- **Event Marker:** `{last_trial['event_code']}`"
                    )
                    prov_col2.markdown(
                        f"- **Tensor Shape:** `{last_trial['shape']}`\n"
                        f"- **Sampling Rate:** `{last_trial['sfreq']:.0f} Hz`\n"
                        f"- **Inference Latency:** `{last_trial['latency_ms']:.1f} ms`"
                    )
                    st.markdown(
                        f"**Ground Truth:** `{last_trial['ground_truth']}` → "
                        f"**Blind Prediction:** `{last_trial['predicted']}`"
                    )

                    st.markdown("**Live Motor Cortex Signal Trace (8–30 Hz RMS & Waveform)**")
                    rms = last_trial["rms_uv"]
                    bp = last_trial["bandpower_uv2"]
                    bp_col1, bp_col2, bp_col3 = st.columns(3)
                    bp_col1.metric("C3 (Left Motor)", f"{rms['C3']:.2f} µV RMS", f"{bp['C3']:.1f} µV²")
                    bp_col2.metric("Cz (Midline)", f"{rms['Cz']:.2f} µV RMS", f"{bp['Cz']:.1f} µV²")
                    bp_col3.metric("C4 (Right Motor)", f"{rms['C4']:.2f} µV RMS", f"{bp['C4']:.1f} µV²")

                    filtered_uv = last_trial["filtered_epoch"] * 1e6
                    n_times = filtered_uv.shape[1]
                    time_axis = np.linspace(0.5, 3.5, n_times)
                    wave_df = pd.DataFrame(
                        {
                            "Time (s)": time_axis,
                            "C3 (Left Motor)": filtered_uv[last_trial["c3_idx"]],
                            "Cz (Midline)": filtered_uv[last_trial["cz_idx"]],
                            "C4 (Right Motor)": filtered_uv[last_trial["c4_idx"]],
                        }
                    )
                    st.caption(
                        f"`[0.5s – 3.5s]`, {n_times} samples @ {last_trial['sfreq']:.0f} Hz — "
                        f"`{last_trial['source_file']}` Trial #{last_trial['trial_index']}"
                    )
                    st.line_chart(
                        wave_df,
                        x="Time (s)",
                        y=["C3 (Left Motor)", "Cz (Midline)", "C4 (Right Motor)"],
                        x_label="Time (s) [0.5s – 3.5s]",
                        y_label="Amplitude (µV)",
                        height=240,
                    )

                    if session_trials:
                        st.markdown("**Recent Trials (Last 5)**")
                        recent_df = pd.DataFrame(session_trials[-5:][::-1])
                        st.dataframe(recent_df, hide_index=True, width="stretch")
            else:
                st.info("Tap any direction on the D-Pad to sample a real motor imagery epoch from unseen held-out runs (S001R12 / S001R14) and drive the cursor in real-time.")

    # Live Stream Replay Loop
    if start_btn:
        run_map = {
            "Held-Out Vertical Imagery (S001R14: Both Fists vs Both Feet)": ROOT
            / "data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R14.edf",
            "Held-Out Horizontal Imagery (S001R12: Left Fist vs Right Fist)": ROOT
            / "data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R12.edf",
            "Vertical Imagery (S001R06: Both Fists vs Both Feet)": ROOT
            / "data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R06.edf",
            "Horizontal Imagery (S001R04: Left Fist vs Right Fist)": ROOT
            / "data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R04.edf",
            "Vertical Imagery (S001R10: Both Fists vs Both Feet)": ROOT
            / "data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R10.edf",
            "Horizontal Imagery (S001R08: Left Fist vs Right Fist)": ROOT
            / "data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R08.edf",
        }
        edf_file = run_map[stream_mode]

        if not (models_ready and edf_file.exists()):
            st.error(
                "Missing model artifacts or EDF recording. Ensure models and Subject 1 EDF files are present."
            )
        else:
            engine = BCIInferenceEngine(
                horizontal_model_path=h_model,
                vertical_model_path=v_model,
                threshold=0.50,
                deadband=0.10,
                single_axis=True,
                smoothing_alpha=0.35,
                history_len=3,
            )
            streamer = MockEEGStreamer(edf_file, chunk_size=16)
            ring_buf = SlidingRingBuffer(n_channels=64, window_samples=481)

            while not ring_buf.is_full():
                ring_buf.push(streamer.get_chunk())

            x = st.session_state["arena_x"]
            y = st.session_state["arena_y"]
            tx = st.session_state["arena_tx"]
            ty = st.session_state["arena_ty"]
            hits = st.session_state["arena_hits"]
            trail = list(st.session_state["arena_trail"])
            step = 0
            t_start = time.time()

            while (time.time() - t_start) < duration_sec:
                ring_buf.push(streamer.get_chunk())
                step += 1

                if step % 3 == 0:
                    t0 = time.perf_counter()
                    res = engine.predict_window(ring_buf.get_window(), axis_lock="AUTO")
                    lat_ms = (time.perf_counter() - t0) * 1000.0

                    x = max(-95.0, min(95.0, x + alpha * res["vx"] * 18.0))
                    y = max(-95.0, min(95.0, y + alpha * res["vy"] * 18.0))
                    trail.append((x, y))

                    if math.hypot(x - tx, y - ty) <= 22.0:
                        hits += 1
                        tx, ty = random.choice([t for t in ARENA_TARGETS if t != (tx, ty)])

                    if arena_slot:
                        arena_slot.markdown(
                            _render_arena_svg(x, y, tx, ty, trail, hits),
                            unsafe_allow_html=True,
                        )
                    if telemetry_slot:
                        with telemetry_slot.container():
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Latency", f"{lat_ms:.1f} ms")
                            m2.metric("Horizontal", res["h_command"], f"vx={res['vx']:+.2f}")
                            m3.metric("Vertical", res["v_command"], f"vy={res['vy']:+.2f}")
                            st.metric("Targets Reached", hits)

                            st.markdown("**Live Class Probabilities**")
                            for label in ["LEFT", "RIGHT", "BOTH_HANDS", "BOTH_FEET"]:
                                val = res["probabilities"].get(label, 0.0)
                                st.progress(
                                    min(1.0, max(0.0, val)),
                                    text=f"{label}: {val * 100:.1f}%",
                                )

                time.sleep(0.02)

            st.session_state["arena_x"] = x
            st.session_state["arena_y"] = y
            st.session_state["arena_tx"] = tx
            st.session_state["arena_ty"] = ty
            st.session_state["arena_trail"] = trail
            st.session_state["arena_hits"] = hits


if __name__ == "__main__":
    main()