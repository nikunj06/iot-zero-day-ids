"""Investigation dashboard for evaluation, local telemetry, and ablations."""
import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
VIEWS = {
    "Evaluation": RUNS / "demo",
    "Local lab": RUNS / "live",
    "Comparisons": RUNS / "comparison",
}
COLORS = {
    "unknown": "#ffbb72",
    "lab_alert": "#ffbb72",
    "benign": "#56d6c4",
    "DoS-SYN": "#8da9f4",
    "Recon-Scan": "#bf9df3",
}

st.set_page_config(
    page_title="Signal / IoT security",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --bg: #0a111d; --panel: #111e2e; --line: #26384d;
      --text: #eff5f8; --muted: #a3b2c0; --accent: #56d6c4;
    }
    .stApp, [data-testid="stSidebar"] { background: var(--bg); color: var(--text); }
    [data-testid="stSidebar"] {
      background: #0e1928; border-right: 1px solid var(--line);
    }
    .block-container { max-width: 1500px; padding: 2.2rem 3.4rem 5rem; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    h1, h2, h3 { letter-spacing: -.035em; }
    h1 { font-size: clamp(2.1rem, 3.1vw, 3.4rem) !important; font-weight: 720 !important; }
    h2 { font-size: 1.35rem !important; margin-top: 0 !important; }
    h3 { font-size: 1.05rem !important; }
    p, label { color: var(--muted); }
    [data-testid="stHorizontalBlock"] { gap: 1.1rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
      border-color: var(--line) !important; background: var(--panel);
      border-radius: 14px; padding: 0.35rem;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; }
    [data-baseweb="tab-list"] { gap: .5rem; border-bottom: 1px solid var(--line); }
    [data-baseweb="tab"] { padding: .8rem 1.25rem; font-weight: 600; }
    [data-baseweb="tab"][aria-selected="true"] { color: var(--accent); }
    [data-testid="stDownloadButton"] button, [data-testid="stButton"] button {
      border-radius: 9px; border-color: #355365; background: #152b38;
      color: #e5faf6; font-weight: 600;
    }
    [data-testid="stDownloadButton"] button:hover, [data-testid="stButton"] button:hover {
      border-color: var(--accent); color: var(--accent);
    }
    .brand { color: #f0f6f8; font-size: 1.4rem; font-weight: 760; letter-spacing: -.045em; }
    .brand-mark { color: var(--accent); margin-right: .5rem; }
    .brand-sub { color: #9fb1c0; font-size: .75rem; letter-spacing: .12em; margin: .15rem 0 2rem; }
    .eyebrow { color: var(--accent); font-size: .74rem; font-weight: 750; letter-spacing: .15em; text-transform: uppercase; }
    .lede { color: #b1c0ce; max-width: 760px; margin: -.5rem 0 1.6rem; }
    .section-note { color: #aab9c7; font-size: .9rem; margin: -.15rem 0 1.1rem; }
    .rule { border-top: 1px solid var(--line); margin: 1.3rem 0 1.5rem; }
    .kpi { min-height: 130px; padding: 1.1rem 1.2rem; border: 1px solid var(--line);
           border-radius: 13px; background: #121f30; }
    .kpi-label { color: #b9c8d4; font-size: .8rem; font-weight: 600; }
    .kpi-value { color: #f2f7f9; font-size: 2rem; font-weight: 720; letter-spacing: -.06em; line-height: 1.35; }
    .kpi-hint { color: #9eb2c1; font-size: .75rem; }
    .kpi.amber .kpi-value { color: #ffbb72; }
    .kpi.teal .kpi-value { color: #56d6c4; }
    .banner { padding: .82rem 1.05rem; border-radius: 10px;
              border: 1px solid #35516a; background: #142639; color: #c3d4df; font-size: .85rem; }
    .banner strong { color: #f2f7f9; }
    .banner.amber { background: #302716; border-color: #66502b; color: #ebd6b5; }
    .fact { padding: .7rem 0; border-bottom: 1px solid var(--line); font-size: .89rem; }
    .fact b { color: #eef5f8; font-weight: 600; }
    .fact span { float: right; color: #bccbd6; }
    .tag { display: inline-block; color: #6ce1cc; background: #163c3c;
           padding: .28rem .6rem; border-radius: 6px; font-size: .73rem; font-weight: 700; }
    .tag.alert { color: #ffc58a; background: #46311e; }
    @media (max-width: 900px) { .block-container { padding: 1.3rem 1rem 4rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def esc(value):
    if pd.isna(value):
        return "—"
    return html.escape(str(value))


def metric(label, value, hint, tone=""):
    st.markdown(
        f'<div class="kpi {tone}"><div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{esc(value)}</div><div class="kpi-hint">{esc(hint)}</div></div>',
        unsafe_allow_html=True,
    )


def banner(title, detail, amber=False):
    st.markdown(
        f'<div class="banner {"amber" if amber else ""}"><strong>{esc(title)}</strong>'
        f' &nbsp; {esc(detail)}</div>',
        unsafe_allow_html=True,
    )


def section(title, note=None):
    st.subheader(title)
    if note:
        st.markdown(f'<p class="section-note">{esc(note)}</p>', unsafe_allow_html=True)


def read_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_run(directory):
    path = directory / "predictions.csv"
    if not path.exists():
        return None, None, {}, {}
    try:
        data = pd.read_csv(path)
        review_path = directory / "review_queue.csv"
        reviews = pd.read_csv(review_path) if review_path.exists() else data.iloc[:0].copy()
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None, None, {}, {}
    return data, reviews, read_json(directory / "metrics.json"), read_json(directory / "run_info.json")


def palette(values):
    return alt.Scale(domain=values, range=[COLORS.get(v, "#7dabc4") for v in values])


def count_chart(data):
    counts = data.predicted.fillna("unspecified").value_counts().rename_axis("prediction").reset_index(name="flows")
    order = counts.prediction.tolist()
    chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=5, size=23).encode(
        y=alt.Y("prediction:N", sort=order, title=None, axis=alt.Axis(labelLimit=150)),
        x=alt.X("flows:Q", title="Flows", axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("prediction:N", scale=palette(order), legend=None),
        tooltip=["prediction:N", "flows:Q"],
    ).properties(height=max(180, len(order) * 43))
    st.altair_chart(chart.configure_view(stroke=None).configure_axis(
        labelColor="#b8c8d4", titleColor="#b8c8d4", gridColor="#28394b", domainColor="#405365"
    ), width="stretch")


def distance_chart(data, metrics):
    if not {"confidence", "prototype_distance", "predicted"}.issubset(data):
        st.caption("This run does not include model distance scores.")
        return
    plot = data.copy()
    plot["confidence"] = pd.to_numeric(plot.confidence, errors="coerce")
    plot["prototype_distance"] = pd.to_numeric(plot.prototype_distance, errors="coerce")
    plot = plot.dropna(subset=["confidence", "prototype_distance"])
    if plot.empty:
        st.caption("No finite distance scores to plot.")
        return
    labels = sorted(plot.predicted.dropna().unique())
    chart = alt.Chart(plot).mark_circle(size=66, opacity=.65).encode(
        x=alt.X("confidence:Q", title="Highest class score", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("prototype_distance:Q", title="Distance to predicted class"),
        color=alt.Color("predicted:N", scale=palette(labels), title="Prediction"),
        tooltip=["flow_id:N", "predicted:N", alt.Tooltip("confidence:Q", format=".3f"),
                 alt.Tooltip("prototype_distance:Q", format=".2f"), "rejection_reason:N"],
    )
    limit = metrics.get("prototype_threshold")
    if limit is not None:
        chart += alt.Chart(pd.DataFrame({"limit": [limit]})).mark_rule(
            color="#ffbb72", strokeDash=[5, 5], opacity=.9
        ).encode(y="limit:Q")
    st.altair_chart(chart.configure_view(stroke=None).configure_axis(
        labelColor="#b8c8d4", titleColor="#b8c8d4", gridColor="#28394b", domainColor="#405365"
    ).configure_legend(labelColor="#c4d0d9", titleColor="#c4d0d9"), width="stretch")


def timeline_chart(data):
    if "timestamp" not in data or data.empty:
        return
    plot = data.copy()
    plot["when"] = pd.to_datetime(pd.to_numeric(plot.timestamp, errors="coerce"), unit="s", utc=True)
    plot = plot.dropna(subset=["when"])
    if plot.empty:
        return
    plot["bucket"] = plot.when.dt.floor("5s")
    summary = plot.groupby(["bucket", "predicted"]).size().reset_index(name="sessions")
    labels = sorted(summary.predicted.unique())
    chart = alt.Chart(summary).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("bucket:T", title="Time (UTC)"),
        y=alt.Y("sessions:Q", title="Sessions / 5 s", stack=True, axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("predicted:N", scale=palette(labels), title="Result"),
        tooltip=[alt.Tooltip("bucket:T", title="Time"), "predicted:N", "sessions:Q"],
    ).properties(height=275)
    st.altair_chart(chart.configure_view(stroke=None).configure_axis(
        labelColor="#b8c8d4", titleColor="#b8c8d4", gridColor="#28394b", domainColor="#405365"
    ).configure_legend(labelColor="#c4d0d9", titleColor="#c4d0d9"), width="stretch")


def details(row, metrics, is_lab):
    prediction = str(row.get("predicted", "unknown"))
    alert = prediction in ("unknown", "lab_alert")
    st.markdown(
        f'<span class="tag {"alert" if alert else ""}">{esc(prediction)}</span> '
        f'<span style="color:#bdcbd5;font-size:.84rem">{esc(row.get("flow_id", ""))}</span>',
        unsafe_allow_html=True,
    )
    fields = [("Capture", row.get("capture_id")), ("Source", row.get("src")),
              ("Destination", row.get("dst"))]
    if is_lab:
        fields += [("Rule reason", row.get("rule_reason", "None")),
                   ("Scenario marker", row.get("lab_pattern", "—"))]
    else:
        fields += [("Class score", f"{row.get('confidence'):.3f}" if pd.notna(row.get("confidence")) else "—"),
                   ("Prototype distance", f"{row.get('prototype_distance'):.2f}" if pd.notna(row.get("prototype_distance")) else "—"),
                   ("Rejection reason", row.get("rejection_reason") or "—"),
                   ("Suggested novelty", row.get("novelty_type", "—"))]
        if metrics and "true_label" in row:
            fields += [("Held-out label", row.get("true_label"))]
    for name, value in fields:
        st.markdown(f'<div class="fact"><b>{esc(name)}</b><span>{esc(value)}</span></div>',
                    unsafe_allow_html=True)
    if is_lab:
        st.caption("Scenario marker is simulator context. The rule reason is the actual alert evidence.")
    else:
        st.caption("Class score alone does not determine hybrid acceptance. Type A/B is a suggestion only for rejected flows.")


def explorer(data, metrics, is_lab, prefix):
    section("Flow explorer", "Search and filter the scored records, then inspect an individual flow.")
    left, middle, right = st.columns([1.4, 1, 1])
    with left:
        query = st.text_input("Search ID, capture or address", key=f"{prefix}_search", placeholder="e.g. sim-benign-1")
    with middle:
        labels = sorted(data.predicted.dropna().astype(str).unique())
        selected_labels = st.multiselect("Prediction", labels, default=labels, key=f"{prefix}_labels")
    with right:
        if "rejection_reason" in data and data.rejection_reason.notna().any():
            reasons = sorted(data.rejection_reason.dropna().astype(str).unique())
            selected_reasons = st.multiselect("Rejection reason", reasons, key=f"{prefix}_reasons")
        elif "lab_pattern" in data:
            reasons = sorted(data.lab_pattern.dropna().astype(str).unique())
            selected_reasons = st.multiselect("Scenario", reasons, key=f"{prefix}_scenarios")
        else:
            selected_reasons = []
    filtered = data[data.predicted.astype(str).isin(selected_labels)].copy()
    if query:
        cols = [c for c in ("flow_id", "capture_id", "src", "dst") if c in filtered]
        matches = filtered[cols].fillna("").astype(str).apply(
            lambda col: col.str.contains(query, case=False, regex=False)
        ).any(axis=1)
        filtered = filtered[matches]
    if selected_reasons:
        column = "lab_pattern" if is_lab else "rejection_reason"
        filtered = filtered[filtered[column].isin(selected_reasons)]
    if not is_lab and "confidence" in filtered:
        threshold = st.slider("Minimum class score", 0.0, 1.0, 0.0, .05, key=f"{prefix}_confidence")
        filtered = filtered[pd.to_numeric(filtered.confidence, errors="coerce") >= threshold]
    st.caption(f"{len(filtered):,} of {len(data):,} flows shown")
    columns = ["flow_id", "predicted", "rule_reason" if is_lab else "rejection_reason",
               "lab_pattern" if is_lab else "confidence", "capture_id"]
    if not is_lab:
        columns += ["prototype_distance", "novelty_type"]
    visible = [c for c in columns if c in filtered]
    st.dataframe(filtered[visible], hide_index=True, width="stretch", height=360)
    st.download_button("Export filtered CSV", filtered.to_csv(index=False).encode(), "filtered_flows.csv",
                       "text/csv", disabled=filtered.empty, key=f"{prefix}_export")
    if filtered.empty:
        st.info("No flows match these filters.")
    else:
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        section("Inspect a flow")
        chosen = st.selectbox("Flow ID", filtered.flow_id.astype(str).tolist(), key=f"{prefix}_flow")
        details(filtered.loc[filtered.flow_id.astype(str) == chosen].iloc[0], metrics, is_lab)


def review(data, queue, is_lab, prefix):
    section("Review queue", "A focused list of rejected model flows or local rule alerts.")
    if queue is None or queue.empty:
        st.info("No alerts have been queued for this run.")
        return
    st.caption(f"{len(queue)} queued records")
    columns = ["flow_id", "predicted", "rule_reason" if is_lab else "rejection_reason",
               "lab_pattern" if is_lab else "novelty_type", "capture_id"]
    shown = queue[[c for c in columns if c in queue]].copy()
    st.dataframe(shown, hide_index=True, width="stretch", height=330)
    if is_lab:
        banner("Local rule alerts", "These are not CNN/GATv2 predictions or confirmed attacks.")
        return
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown("#### Record reviewer decisions")
    st.caption("Enter labels only after examining independent evidence. Export for the adaptation script.")
    reviews = queue.drop(columns=["true_label", "family", "experiment"], errors="ignore").copy()
    reviews["reviewed_label"] = ""
    reviews["reviewed_family"] = ""
    reviews["reviewer_note"] = ""
    edited = st.data_editor(
        reviews, hide_index=True, width="stretch", num_rows="fixed",
        disabled=[c for c in reviews if c not in ("reviewed_label", "reviewed_family", "reviewer_note")],
        column_order=["flow_id", "predicted", "rejection_reason", "reviewed_label",
                      "reviewed_family", "reviewer_note"],
        key=f"{prefix}_editor",
    )
    st.download_button("Export review decisions", edited.to_csv(index=False).encode(),
                       "review_decisions.csv", "text/csv", key=f"{prefix}_reviews")


def comparison(directory):
    path = directory / "summary.csv"
    if not path.exists():
        st.info("Run `python -m ids.compare --input data/demo.jsonl --out runs/comparison` to create this view.")
        return
    try:
        data = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        st.info("Waiting for a complete comparison file.")
        return
    banner("Simulation only", "The included comparison uses generated flows and does not validate real attacks.", True)
    st.write("")
    section("Model ablations", "Every row uses the same synthetic split and random seed.")
    metric_names = {
        "Unknown recall": "unknown_recall", "Type B recall": "type_b_recall",
        "Known macro F1": "known_macro_f1", "Benign false-alert rate": "benign_false_alarm_rate",
    }
    choice = st.selectbox("Compare metric", list(metric_names))
    field = metric_names[choice]
    selected = st.multiselect("Models", data.model.tolist(), default=data.model.tolist())
    plot = data[data.model.isin(selected)]
    if not plot.empty:
        chart = alt.Chart(plot).mark_bar(cornerRadiusEnd=5, size=30).encode(
            y=alt.Y("model:N", sort="-x", title=None),
            x=alt.X(f"{field}:Q", title=choice, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            color=alt.condition(alt.datum.model == "full_graph", alt.value("#56d6c4"), alt.value("#6587a5")),
            tooltip=["model:N", alt.Tooltip(f"{field}:Q", format=".3f")],
        ).properties(height=max(210, 52 * len(plot)))
        st.altair_chart(chart.configure_view(stroke=None).configure_axis(
            labelColor="#b8c8d4", titleColor="#b8c8d4", gridColor="#28394b", domainColor="#405365"
        ), width="stretch")
    st.dataframe(data, hide_index=True, width="stretch")
    if "seeds" in data:
        st.caption("Values are means across the listed seeds; dispersion and individual runs are saved beside this summary.")
    st.caption("A high unknown recall is not useful when known flows are also rejected; inspect F1 and false alerts together.")


st.sidebar.markdown('<div class="brand"><span class="brand-mark">◈</span>SIGNAL</div>'
                    '<div class="brand-sub">IoT SECURITY WORKBENCH</div>', unsafe_allow_html=True)
st.sidebar.markdown("### Workspace")
view = st.sidebar.radio("Choose a workspace", ["Evaluation", "Local lab", "Comparisons", "Custom run"],
                        label_visibility="collapsed")
directory = VIEWS.get(view)
if view == "Comparisons":
    directory = Path(st.sidebar.text_input("Comparison directory", "runs/comparison"))
    if not directory.is_absolute():
        directory = ROOT / directory
if view == "Custom run":
    directory = Path(st.sidebar.text_input("Run directory", "runs/fresh"))
    if not directory.is_absolute():
        directory = ROOT / directory
st.sidebar.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.sidebar.caption(f"READING  {directory}")
st.sidebar.caption("Research prototype · local files")

st.markdown(f'<div class="eyebrow">WORKSPACE / {esc(view.upper())}</div>', unsafe_allow_html=True)
titles = {
    "Evaluation": ("Model evaluation", "Inspect held-out results, failures, and the evidence behind each rejection."),
    "Local lab": ("Live lab", "Watch bounded loopback sessions and inspect the exact rules behind alerts."),
    "Comparisons": ("Model comparisons", "Test whether each architectural component adds value on the same split."),
    "Custom run": ("Run investigation", "Examine scored flows from a compatible model run."),
}
title, subtitle = titles[view]
st.title(title)
st.markdown(f'<p class="lede">{esc(subtitle)}</p>', unsafe_allow_html=True)

if view == "Comparisons":
    comparison(directory)
else:
    is_lab = view == "Local lab"

    @st.fragment(run_every="3s" if is_lab else None)
    def run_view():
        data, queue, metrics, info = load_run(directory)
        if data is None or data.empty:
            instruction = ("Run `python -m ids.live_lab --output data/live.jsonl --duration 45 --scenario mixed` "
                           "and `python -m ids.watch --input data/live.jsonl --out runs/live`."
                           if is_lab else "Train a model or score compatible flows to populate this run.")
            st.info(instruction)
            return
        model_run = not is_lab and bool(metrics)
        pilot = read_json(directory / "pilot_summary.json")
        source = info.get("source") or read_json(directory / "config.json").get("feature_source", "unrecorded")
        detector = info.get("mode") or ("CNN / GATv2" if model_run else "unrecorded")
        if is_lab:
            detector = info.get("mode", "local rules")
            last = pd.Timestamp((directory / "predictions.csv").stat().st_mtime, unit="s", tz="UTC")
            banner("LOCAL LAB · RULE DETECTOR",
                   f"Source: {source} · Last scored: {last.strftime('%d %b %H:%M:%S UTC')}. Refreshes every 3 s.")
            if detector != "local_lab_rules":
                st.warning("This directory does not contain current local lab rule output.")
        elif model_run:
            banner("SYNTHETIC EVALUATION" if source == "synthetic_flow_v1" else "LABELED EVALUATION",
                   f"Source: {source} · Decision: {metrics.get('decision_method', 'sigmoid')}. "
                   "Unknown means rejected by this model, not a confirmed new exploit.",
                   amber=source == "synthetic_flow_v1")
        elif pilot:
            banner("UNLABELED PILOT",
                   f"{pilot.get('observed_hours')} observed hours · {pilot.get('device_count')} devices · "
                   "Alerts require independent review.")
        else:
            banner("UNLABELED INFERENCE", f"Source: {source} · No ground-truth accuracy metrics are available.")
        st.write("")
        if is_lab:
            alerts = int(data.predicted.eq("lab_alert").sum())
            cards = [
                ("Completed sessions", f"{len(data):,}", "Latest scored window", ""),
                ("Rule alerts", f"{alerts:,}", "Inspect their exact reasons", "amber"),
                ("Ordinary sessions", f"{len(data) - alerts:,}", "No local rule matched", "teal"),
                ("Alert share", f"{alerts / len(data):.1%}", "Of completed sessions", ""),
            ]
        elif model_run:
            cards = [
                ("Known macro F1", f"{metrics['known_macro_f1']:.3f}", "Known classes only", ""),
                ("Unknown recall", f"{metrics['unknown_recall']:.3f}", "Held-out attack labels", "teal"),
                ("Type B recall", f"{metrics.get('type_b_unknown_recall', 0):.3f}", "Novel subtype of known family", "amber"),
                ("Benign false alerts", f"{metrics['benign_false_alarm_rate']:.1%}", "Benign test flows", ""),
            ]
        elif pilot:
            cards = [
                ("Scored flows", f"{len(data):,}", "Observation window", ""),
                ("Review alerts", f"{pilot['rejected_for_review']:,}", "Not confirmed attacks", "amber"),
                ("Alerts / device-day", f"{pilot['alerts_per_device_day']:.2f}", "Measured exposure", ""),
                ("Budget", "PASS" if pilot["within_predefined_budget"] else "OVER",
                 f"Max {pilot['predefined_max_alerts_per_device_day']:.2f} / day", "teal" if pilot["within_predefined_budget"] else "amber"),
            ]
        else:
            alerts = int(data.predicted.eq("unknown").sum())
            cards = [
                ("Scored flows", f"{len(data):,}", "Current file", ""),
                ("Rejected", f"{alerts:,}", "For independent review", "amber"),
                ("Accepted", f"{len(data) - alerts:,}", "Known class assigned", "teal"),
                ("Review share", f"{alerts / len(data):.1%}", "Of scored flows", ""),
            ]
        columns = st.columns(4)
        for col, (label, value, hint, tone) in zip(columns, cards):
            with col:
                metric(label, value, hint, tone)
        st.write("")
        tab_overview, tab_explorer, tab_review, tab_method = st.tabs(
            ["Overview", "Flow explorer", "Review queue", "How to read this"]
        )
        with tab_overview:
            if is_lab:
                section("Activity over time", "Sessions are grouped into five-second windows; this does not imply an attack.")
                timeline_chart(data)
            left, right = st.columns([1, 1.25])
            with left:
                section("Prediction mix")
                count_chart(data)
            with right:
                if model_run and "prototype_distance" in data:
                    section("Decision landscape", "Dashed line: calibrated prototype distance limit. Hover for individual flows.")
                    distance_chart(data, metrics)
                elif is_lab and "lab_pattern" in data:
                    section("Scenario × rule outcome")
                    st.caption("Scenario marker is generated by the simulator. It is not detector input.")
                    st.dataframe(pd.crosstab(data.lab_pattern.fillna("unmarked"), data.predicted),
                                 width="stretch")
                else:
                    section("Rejection reasons")
                    if "rejection_reason" in data:
                        st.dataframe(data.rejection_reason.fillna("accepted").value_counts().rename_axis(
                            "reason").reset_index(name="flows"), hide_index=True, width="stretch")
            if model_run:
                st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
                section("Held-out families", "Recall is the share of each held-out category rejected as unknown.")
                held = pd.DataFrame([
                    {"category": "Type A · unseen family", "flows": metrics.get("type_a_n"),
                     "unknown recall": metrics.get("type_a_unknown_recall")},
                    {"category": "Type B · unseen subtype", "flows": metrics.get("type_b_n"),
                     "unknown recall": metrics.get("type_b_unknown_recall")},
                ])
                st.dataframe(held, hide_index=True, width="stretch",
                             column_config={"unknown recall": st.column_config.ProgressColumn(
                                 "Unknown recall", min_value=0, max_value=1, format="%.3f")})
                st.caption(f"Unknown precision: {metrics.get('unknown_precision', '—')}; "
                           f"sigmoid-only unknown recall: {metrics.get('sigmoid_only_unknown_recall', '—')}. "
                           "Type A/B accuracy excludes missed held-out flows.")
                per_label = metrics.get("heldout_recall_by_label", {})
                if per_label:
                    st.markdown("#### Per held-out attack")
                    st.dataframe(pd.DataFrame([{"attack label": name, "unknown recall": value}
                                               for name, value in per_label.items()]),
                                 hide_index=True, width="stretch")
                if "benign_alerts_per_device_day" in metrics:
                    st.caption(f"{metrics['benign_false_alert_count']} benign test alerts over "
                               f"{metrics['benign_exposure_hours']} measured hours and "
                               f"{metrics['benign_device_count']} devices: "
                               f"{metrics['benign_alerts_per_device_day']:.2f} alerts/device-day.")
        with tab_explorer:
            explorer(data, metrics, is_lab, view.lower().replace(" ", "_"))
        with tab_review:
            review(data, queue, is_lab, view.lower().replace(" ", "_"))
        with tab_method:
            section("What these numbers mean")
            if is_lab:
                st.markdown("**Detector:** bounded rules on application messages. A session alerts on "
                            "20 or more messages or an unexpected `CONFIG` message. The watcher reads "
                            "`data/live.jsonl`; refreshing the dashboard only rereads output.")
                st.markdown("**Limits:** message counts are not packet counts. The local lab does not "
                            "run the CNN/GATv2 model or test a real zero-day exploit.")
            else:
                st.markdown("**Detector:** a CNN encodes payload bytes and a flow model uses numeric "
                            "traffic features; the GATv2 variant relates flows within each minibatch. "
                            "The hybrid decision rejects low class scores or flows far from a known-class prototype.")
                st.markdown("**Limits:** rejected traffic needs investigation. The bundled evaluation "
                            "uses generated flows. Prototype distance can catch more unknown flows while "
                            "raising false alerts; Type B detection is still weak.")
                if model_run:
                    st.caption(f"Class score threshold: {metrics.get('threshold', '—')} · "
                               f"Prototype distance limit: {metrics.get('prototype_threshold', '—')}. "
                               "The displayed novelty score is 1 − class score, not the full hybrid decision.")
            st.markdown("For experiment design, input provenance, and commands, see `README.md`.")

    run_view()
