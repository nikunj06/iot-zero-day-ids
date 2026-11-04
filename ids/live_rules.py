"""Transparent local-lab indicators for application telemetry, separate from the ML model."""
import json
import os
import argparse
from pathlib import Path

import pandas as pd

from .data import feature_source


def score_live_rows(rows, out):
    if feature_source(rows) != "app_message_v1":
        raise ValueError("local-lab rules require app_message_v1 telemetry")
    output = []
    for r in rows:
        payload = bytes.fromhex(r.get("payload_hex", ""))
        reasons = []
        if int(r.get("packets", 0)) >= 20:
            reasons.append("20+ application messages in one local TCP session")
        if payload.startswith(b"CONFIG"):
            reasons.append("unexpected CONFIG command in local sensor message")
        output.append(dict(flow_id=r["flow_id"], capture_id=r["capture_id"], timestamp=r.get("timestamp"),
                           src=r.get("src"), dst=r.get("dst"), lab_pattern=r.get("lab_pattern", ""),
                           predicted="lab_alert" if reasons else "benign", rule_reason="; ".join(reasons),
                           detector="lab_rules", feature_source="app_message_v1"))
    table = pd.DataFrame(output)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, result in (("predictions.csv", table), ("review_queue.csv", table[table.predicted == "lab_alert"])):
        temp = out / f"{name}.tmp"
        result.to_csv(temp, index=False)
        os.replace(temp, out / name)
    (out / "run_info.json").write_text(json.dumps(dict(mode="local_lab_rules", source="app_message_v1",
                                                    note="Bounded local message patterns; rule alerts are not CNN-GATv2 predictions or zero-day detections."), indent=2))
    return len(table), int((table.predicted == "lab_alert").sum())


def main():
    parser = argparse.ArgumentParser(description="Score bounded local message scenarios with transparent rules")
    parser.add_argument("--input", default="data/live.jsonl")
    parser.add_argument("--out", default="runs/live")
    args = parser.parse_args()
    rows = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s.strip()]
    if not rows:
        parser.error("no completed flows")
    count, alerts = score_live_rows(rows, args.out)
    print(f"Scored {count} local telemetry flows; {alerts} lab-rule alerts")


if __name__ == "__main__":
    main()
