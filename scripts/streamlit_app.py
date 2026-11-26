#!/usr/bin/env python3
"""Launch the MUSE offline experiment dashboard with Streamlit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def discover_experiments(results_dir: Path) -> list[Path]:
    """Return result directories containing saved evaluation metrics."""

    if not results_dir.exists():
        return []
    return sorted(
        (path.parent for path in results_dir.glob("*/metrics.json")),
        key=lambda path: path.name.lower(),
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


def _percentage(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the CLI, not unit tests
        raise SystemExit(
            "Streamlit is not installed. Install project requirements, then run "
            "`streamlit run scripts/streamlit_app.py`."
        ) from exc

    st.set_page_config(page_title="MUSE EEG Dashboard", page_icon="🧠", layout="wide")
    st.title("MUSE EEG experiment dashboard")
    st.caption("Offline inspection of saved EEG motor-imagery experiments")

    with st.sidebar:
        st.header("Experiment")
        results_text = st.text_input("Results directory", str(ROOT / "results"))
        results_dir = Path(results_text).expanduser()
        experiments = discover_experiments(results_dir)
        if not experiments:
            st.error("No saved experiment metrics were found.")
            st.code("PYTHONPATH=src .venv/bin/python scripts/run_experiment.py --help")
            st.stop()
        experiment_labels = [path.name for path in experiments]
        selected_name = st.selectbox("Saved run", experiment_labels)
        selected_dir = experiments[experiment_labels.index(selected_name)]
        st.info("This dashboard does not connect to a live headset or issue computer commands.")

    metrics_path = selected_dir / "metrics.json"
    config_path = selected_dir / "config.json"
    metrics = _read_json(metrics_path)
    config = _read_json(config_path) if config_path.exists() else {}

    st.subheader(selected_dir.name)
    validation = config.get("validation", "Validation metadata unavailable")
    st.write(validation)

    metric_columns = st.columns(5)
    metric_columns[0].metric("Accuracy", _percentage(metrics["accuracy"]))
    metric_columns[1].metric("Balanced accuracy", _percentage(metrics["balanced_accuracy"]))
    metric_columns[2].metric("Macro F1", _percentage(metrics["macro_f1"]))
    metric_columns[3].metric("Chance level", _percentage(metrics["chance_level"]))
    metric_columns[4].metric("Epochs / groups", f"{metrics['n_epochs']} / {metrics['n_groups']}")

    overview_tab, predictions_tab, configuration_tab, audit_tab = st.tabs(
        ["Overview", "Predictions", "Configuration", "Dataset audit"]
    )

    with overview_tab:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Confusion matrix")
            classes = metrics.get("classes", [])
            confusion = pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes)
            st.dataframe(confusion, width="stretch")
            confusion_path = selected_dir / "confusion_matrix.png"
            if confusion_path.exists():
                st.image(str(confusion_path), caption="Saved grouped-held-out confusion matrix")
        with right:
            st.markdown("#### Class counts")
            class_counts = pd.DataFrame(
                [{"class": label, "count": count} for label, count in metrics.get("class_counts", {}).items()]
            )
            st.dataframe(class_counts, hide_index=True, width="stretch")
            st.markdown("#### Classification report")
            report = classification_report_frame(metrics.get("classification_report", {}))
            st.dataframe(report, hide_index=True, width="stretch")

        st.markdown("#### Held-out fold scores")
        folds = pd.DataFrame(metrics.get("fold_results", []))
        if folds.empty:
            st.info("No fold-level results were saved.")
        else:
            st.dataframe(folds, hide_index=True, width="stretch")
            chart = folds.set_index("fold")[["accuracy", "balanced_accuracy", "macro_f1"]]
            st.line_chart(chart)

    with predictions_tab:
        predictions_path = selected_dir / "predictions.csv"
        if not predictions_path.exists():
            st.info("No saved predictions were found for this experiment.")
        else:
            predictions = pd.read_csv(predictions_path)
            predictions["correct"] = predictions["true"] == predictions["predicted"]
            group_values = sorted(predictions["group"].dropna().unique().tolist())
            selected_groups = st.multiselect("Groups", group_values, default=group_values)
            selected_status = st.radio("Show", ["All", "Correct only", "Errors only"], horizontal=True)
            filtered = predictions[predictions["group"].isin(selected_groups)]
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


if __name__ == "__main__":
    main()
