from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="IoT Open-Set IDS", layout="wide")
st.title("Payload-aware IoT open-set IDS")
st.caption("Research prototype. Unknown means rejected by the model; it does not establish a novel exploit.")
directory = Path(st.text_input("Run directory", "runs/demo"))
metrics_path = directory / "metrics.json"
predictions_path = directory / "predictions.csv"
queue_path = directory / "review_queue.csv"
if not predictions_path.exists():
    st.info("Run the demo and training commands in README.md, then enter the output directory.")
    st.stop()

import json

metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
data = pd.read_csv(predictions_path)
queue = pd.read_csv(queue_path)
if metrics:
    st.warning(metrics.get("note", ""))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Known macro F1", f"{metrics['known_macro_f1']:.3f}")
    c2.metric("Unknown recall", f"{metrics['unknown_recall']:.3f}")
    c3.metric("Benign false-alert rate", f"{metrics['benign_false_alarm_rate']:.3f}")
    c4.metric("Rejection threshold", f"{metrics['threshold']:.3f}")
    st.subheader("Held-out evaluations")
    st.dataframe(pd.DataFrame([dict(category=c, samples=metrics[f"{c}_n"], unknown_recall=metrics[f"{c}_unknown_recall"])
                               for c in ("type_a", "type_b")]), hide_index=True)
    st.caption(f"Type A/B accuracy among {metrics['n_detected_heldout']} correctly rejected held-out flows: {metrics['hierarchical_accuracy_on_detected_unknown']}. Missed attacks are excluded from this figure.")
st.subheader("Prediction counts")
st.bar_chart(data.predicted.value_counts())
st.subheader("Flow inspection")
selection = st.multiselect("Predictions", sorted(data.predicted.unique()), default=list(data.predicted.unique()))
st.dataframe(data[data.predicted.isin(selection)], use_container_width=True, hide_index=True)
st.download_button("Download all predictions", data.to_csv(index=False), "predictions.csv", "text/csv")
st.subheader("Unknown-flow review")
st.caption("Ground truth labels shown in the evaluation table are for retrospective scoring only. Reviewers should judge the evidence independently.")
if queue.empty:
    st.info("No flows were rejected in this run.")
else:
    visible = queue.drop(columns=["true_label", "family", "experiment"], errors="ignore")
    st.dataframe(visible, use_container_width=True, hide_index=True)
    edited = st.data_editor(visible.assign(reviewed_label="", reviewed_family="", reviewer_note=""), hide_index=True, use_container_width=True)
    st.download_button("Export review decisions", edited.to_csv(index=False), "review_decisions.csv", "text/csv")
