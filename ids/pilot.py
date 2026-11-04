"""Score an independently collected, unlabeled IoT-23-schema observation window."""
import argparse
import json
from pathlib import Path

from .data import feature_source
from .infer import score_rows


def summarize(predictions, hours, devices, budget):
    import pandas as pd
    table = pd.read_csv(predictions)
    count = int(table.predicted.eq("unknown").sum())
    rate = count * 24 / hours / devices
    return dict(flows=len(table), rejected_for_review=count, observed_hours=hours,
                device_count=devices, alerts_per_device_day=rate,
                predefined_max_alerts_per_device_day=budget,
                within_predefined_budget=rate <= budget,
                note="Unlabeled pilot: there are no recall or accuracy claims. Review alerts independently.")


def main():
    parser = argparse.ArgumentParser(description="Score an authorized, unlabeled IoT-23-schema pilot window")
    parser.add_argument("--input", required=True, help="fresh unlabeled JSONL from ids.iot23 --unlabeled")
    parser.add_argument("--run", required=True, help="model trained on iot23_zeek_v1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--hours", required=True, type=float, help="measured elapsed observation hours")
    parser.add_argument("--devices", required=True, type=int, help="number of monitored IoT devices")
    parser.add_argument("--max-alerts-per-device-day", required=True, type=float, help="budget chosen before scoring")
    args = parser.parse_args()
    if args.hours <= 0 or args.devices < 1 or args.max_alerts_per_device_day < 0:
        parser.error("hours >0, devices >=1 and alert budget >=0 required")
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    if not rows or feature_source(rows) != "iot23_zeek_v1" or any("label" in r or "family" in r for r in rows):
        parser.error("pilot requires fresh unlabeled iot23_zeek_v1 flows")
    count, rejected = score_rows(rows, args.run, args.out)
    summary = summarize(Path(args.out) / "predictions.csv", args.hours, args.devices, args.max_alerts_per_device_day)
    (Path(args.out) / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Scored {count} flows, {rejected} alerts; {summary['alerts_per_device_day']:.3f} alerts/device/day")


if __name__ == "__main__":
    main()
