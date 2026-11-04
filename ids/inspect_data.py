"""Check provenance and split feasibility before fitting a model."""
import argparse
import json
from collections import Counter, defaultdict

from .data import feature_source, read_jsonl


def report(rows):
    labels = Counter(r["label"] for r in rows)
    captures = defaultdict(set)
    for row in rows:
        captures[row["label"]].add(row["capture_id"])
    by_capture = defaultdict(set)
    for row in rows:
        by_capture[row["capture_id"]].add(row["label"])
    mixed = sum(len(v) > 1 for v in by_capture.values())
    return dict(feature_source=feature_source(rows), flows=len(rows),
                capture_groups=len({r["capture_id"] for r in rows}),
                mixed_label_capture_groups=mixed,
                labels=dict(labels), capture_groups_per_label={k: len(v) for k, v in captures.items()},
                payload_available=sum(bool(r.get("payload_hex")) for r in rows),
                note=("Mixed-label captures require an explicit --splits manifest and independent recording groups."
                      if mixed else "Known labels need >=3 separate capture IDs; auxiliary unknowns need >=2."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    print(json.dumps(report(read_jsonl(args.input)), indent=2))


if __name__ == "__main__":
    main()
