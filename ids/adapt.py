"""Attach human-reviewed labels to fresh captures for a subsequent versioned run."""
import argparse
import csv
import json
from pathlib import Path

from .data import read_jsonl, save_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="original labeled JSONL")
    parser.add_argument("--new", required=True, help="fresh unlabeled JSONL scored with ids.infer")
    parser.add_argument("--reviews", required=True, help="dashboard export with reviewed_label and reviewed_family")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = read_jsonl(args.base)
    fresh = [json.loads(s) for s in Path(args.new).read_text().splitlines() if s.strip()]
    base_caps = {r["capture_id"] for r in base}
    if any(r["capture_id"] in base_caps for r in fresh):
        parser.error("new captures overlap original captures; evaluation would be contaminated")
    with open(args.reviews, newline="", encoding="utf-8") as handle:
        reviews = {r["flow_id"]: r for r in csv.DictReader(handle) if r.get("reviewed_label") and r.get("reviewed_family")}
    selected = []
    for row in fresh:
        if row["flow_id"] in reviews:
            review = reviews.pop(row["flow_id"])
            selected.append({**row, "label": review["reviewed_label"], "family": review["reviewed_family"]})
    if reviews:
        parser.error("reviews include flow IDs missing from the fresh observations")
    if not selected:
        parser.error("no rows have both reviewed_label and reviewed_family")
    save_jsonl(args.output, base + selected)
    print(f"Wrote {len(base)} original and {len(selected)} human-reviewed flows to {args.output}")


if __name__ == "__main__":
    main()
