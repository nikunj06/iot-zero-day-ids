"""Score newly completed local flow records repeatedly until interrupted."""
import argparse
import json
import time
from pathlib import Path

from .data import feature_source
from .infer import score_rows
from .live_rules import score_live_rows


def completed_rows(path):
    if not path.exists():
        return []
    content = path.read_bytes()
    # The writer may be appending; a line without its newline is not complete.
    lines = content.split(b"\n")[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/live.jsonl")
    parser.add_argument("--run", default="runs/demo", help="model run for compatible PCAP flows; local telemetry uses separate lab rules")
    parser.add_argument("--out", default="runs/live")
    parser.add_argument("--interval", type=float, default=3)
    args = parser.parse_args()
    if not 0.5 <= args.interval <= 60:
        parser.error("interval must be between 0.5 and 60 seconds")
    print("Watching completed local flows. Press Ctrl+C to stop.", flush=True)
    previous = None
    try:
        while True:
            rows = completed_rows(Path(args.input))
            identifiers = tuple(r["flow_id"] for r in rows)
            if rows and identifiers != previous:
                recent = rows[-256:]
                if feature_source(recent) == "app_message_v1":
                    count, rejected = score_live_rows(recent, args.out)
                    detector = "local lab rules"
                else:
                    count, rejected = score_rows(recent, args.run, args.out)
                    detector = "ML model"
                print(f"Scored latest {count} flows with {detector}; {rejected} alerts", flush=True)
                previous = identifiers
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Watcher stopped.")


if __name__ == "__main__":
    main()
