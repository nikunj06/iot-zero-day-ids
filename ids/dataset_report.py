import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .data import FEATURES, feature_source


PAYLOAD_FIELDS = (
    "payload_hex",
    "payload_orig_hex",
    "payload_resp_hex",
)


def load_audit(path):
    if not path:
        return {}
    audit_path = Path(path)
    if not audit_path.exists():
        raise FileNotFoundError(f"audit file not found: {audit_path}")
    return json.loads(audit_path.read_text(encoding="utf-8"))


def analyze_dataset(input_path, audit=None):
    rows = []
    with open(input_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error
            rows.append(row)

    if not rows:
        raise ValueError("Dataset contains no records")

    source = feature_source(rows)
    capture_counts = Counter()
    label_counts = Counter()
    family_counts = Counter()
    capture_labels = defaultdict(Counter)
    missing_fields = Counter()
    invalid_features = Counter()
    payload_flow_counts = Counter()
    payload_bytes = Counter()

    required = ("flow_id", "capture_id", "label", "family")
    for index, row in enumerate(rows, 1):
        for field in required:
            if field not in row or row[field] in (None, ""):
                missing_fields[field] += 1

        capture = str(row.get("capture_id", "unknown"))
        label = str(row.get("label", "unknown"))
        family = str(row.get("family", "unknown"))

        capture_counts[capture] += 1
        label_counts[label] += 1
        family_counts[family] += 1
        capture_labels[capture][label] += 1

        for feature in FEATURES:
            value = row.get(feature)
            if value is None or value == "":
                missing_fields[feature] += 1
                continue
            try:
                number = float(value)
                if not np.isfinite(number):
                    invalid_features[feature] += 1
            except (TypeError, ValueError):
                invalid_features[feature] += 1

        for field in PAYLOAD_FIELDS:
            raw = row.get(field, "") or ""
            if raw:
                try:
                    decoded = bytes.fromhex(raw)
                except (TypeError, ValueError):
                    invalid_features[field] += 1
                    continue
                payload_flow_counts[field] += 1
                payload_bytes[field] += len(decoded)

    total = len(rows)
    benign = label_counts.get("benign", 0)
    attack = total - benign

    payload_summary = {}
    for field in PAYLOAD_FIELDS:
        count = payload_flow_counts[field]
        payload_summary[field] = {
            "flows_with_payload": count,
            "coverage_percent": round(100 * count / total, 2),
            "total_payload_bytes": payload_bytes[field],
        }

    report = {
        "dataset": {
            "input": str(input_path),
            "feature_source": source,
            "total_flows": total,
            "capture_count": len(capture_counts),
            "captures": dict(sorted(capture_counts.items())),
            "benign_flows": benign,
            "attack_flows": attack,
            "benign_percent": round(100 * benign / total, 2),
            "attack_percent": round(100 * attack / total, 2),
            "label_distribution": dict(sorted(label_counts.items())),
            "family_distribution": dict(sorted(family_counts.items())),
            "capture_label_distribution": {
                capture: dict(sorted(counts.items()))
                for capture, counts in sorted(capture_labels.items())
            },
        },
        "payload_coverage": payload_summary,
        "data_quality": {
            "missing_field_counts": dict(sorted(missing_fields.items())),
            "invalid_value_counts": dict(sorted(invalid_features.items())),
            "duplicate_flow_ids": total - len({
                row.get("flow_id") for row in rows
            }),
        },
        "pcap_import_audit": audit or {},
    }
    return report


def render_markdown(report):
    dataset = report["dataset"]
    lines = [
        "# IoT-23 Dataset Quality Report",
        "",
        "## Dataset overview",
        "",
        f"- **Input:** `{dataset['input']}`",
        f"- **Feature source:** `{dataset['feature_source']}`",
        f"- **Total flows:** {dataset['total_flows']:,}",
        f"- **Captures:** {dataset['capture_count']}",
        f"- **Benign flows:** {dataset['benign_flows']:,} "
        f"({dataset['benign_percent']}%)",
        f"- **Attack flows:** {dataset['attack_flows']:,} "
        f"({dataset['attack_percent']}%)",
        "",
        "## Label distribution",
        "",
        "| Label | Flows |",
        "|---|---:|",
    ]
    for label, count in dataset["label_distribution"].items():
        lines.append(f"| {label} | {count:,} |")

    lines += [
        "",
        "## Capture distribution",
        "",
        "| Capture | Total flows | Label breakdown |",
        "|---|---:|---|",
    ]
    for capture, count in dataset["captures"].items():
        labels = dataset["capture_label_distribution"][capture]
        breakdown = ", ".join(
            f"{label}: {number:,}" for label, number in labels.items()
        )
        lines.append(f"| {capture} | {count:,} | {breakdown} |")

    lines += [
        "",
        "## Payload availability",
        "",
        "| Payload field | Flows with payload | Coverage | Payload bytes |",
        "|---|---:|---:|---:|",
    ]
    for field, values in report["payload_coverage"].items():
        lines.append(
            f"| {field} | {values['flows_with_payload']:,} | "
            f"{values['coverage_percent']}% | "
            f"{values['total_payload_bytes']:,} |"
        )

    quality = report["data_quality"]
    lines += [
        "",
        "## Data quality checks",
        "",
        f"- **Duplicate flow IDs:** {quality['duplicate_flow_ids']:,}",
        f"- **Missing values:** {quality['missing_field_counts'] or 'None detected'}",
        f"- **Invalid values:** {quality['invalid_value_counts'] or 'None detected'}",
        "",
        "## PCAP importer audit",
        "",
    ]
    if report["pcap_import_audit"]:
        lines += ["| Audit field | Value |", "|---|---:|"]
        for key, value in report["pcap_import_audit"].items():
            lines.append(f"| {key} | {value:,} |" if isinstance(value, int)
                         else f"| {key} | {value} |")
    else:
        lines.append("No audit file supplied.")

    lines += [
        "",
        "## Interpretation notes",
        "",
        "- Payload coverage is measured per flow; packet audit counters are separate.",
        "- Unsupported packets are not necessarily malformed; they may use protocols or packet forms the importer does not currently extract.",
        "- Missing payload does not mean a flow is invalid. Statistical flow features may still be available.",
        "- This report describes the imported dataset and does not measure model performance.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a quality and coverage report for an imported flow JSONL dataset"
    )
    parser.add_argument("--input", required=True, help="Input flow-level JSONL file")
    parser.add_argument("--audit", help="Optional importer audit JSON file")
    parser.add_argument(
        "--out",
        default="runs/iot23_dataset_quality",
        help="Output directory for JSON and Markdown reports",
    )
    args = parser.parse_args()

    audit = load_audit(args.audit)
    report = analyze_dataset(args.input, audit)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    json_path = output / "dataset_quality_report.json"
    markdown_path = output / "dataset_quality_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"Report saved: {json_path}")
    print(f"Report saved: {markdown_path}")
    print()
    print(f"Flows: {report['dataset']['total_flows']:,}")
    print(f"Captures: {report['dataset']['capture_count']}")
    print(
        "Combined payload coverage: "
        f"{report['payload_coverage']['payload_hex']['coverage_percent']}%"
    )


if __name__ == "__main__":
    main()
