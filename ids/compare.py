"""Run reproducible model ablations on the same source, split seed and epochs."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


EXPERIMENTS = (("stats_baseline", "stats", False), ("payload_only", "payload", False),
               ("combined_no_graph", "both", False), ("full_graph", "both", True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="runs/comparison")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seeds", default="42,43,44", help="comma-separated random seeds")
    parser.add_argument("--seed", type=int, help="single-seed compatibility shortcut")
    parser.add_argument("--splits", help="explicit capture assignments for mixed-label data")
    parser.add_argument("--benign-exposure-hours", type=float)
    parser.add_argument("--benign-device-count", type=int, default=1)
    parser.add_argument("--known", default="benign,DoS-SYN,Recon-Scan")
    parser.add_argument("--aux", default="Spoof-ARP")
    parser.add_argument("--heldout", default="DoS-HTTP,Mirai-Bot")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    try:
        seeds = [args.seed] if args.seed is not None else [int(s.strip()) for s in args.seeds.split(",")]
    except ValueError:
        parser.error("seeds must be comma-separated integers")
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("provide unique seeds")
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        for name, branch, graph in EXPERIMENTS:
            out = root / f"{name}_seed{seed}"
            cmd = [sys.executable, "-m", "ids.train", "--input", args.input, "--out", str(out),
                   "--epochs", str(args.epochs), "--seed", str(seed), "--known", args.known,
                   "--aux", args.aux, "--heldout", args.heldout, "--branch", branch,
                   "--graph" if graph else "--no-graph", "--cpu"]
            if args.splits:
                cmd += ["--splits", args.splits]
            if args.benign_exposure_hours is not None:
                cmd += ["--benign-exposure-hours", str(args.benign_exposure_hours),
                        "--benign-device-count", str(args.benign_device_count)]
            print(f"Training {name}, seed {seed}...", flush=True)
            subprocess.run(cmd, check=True)
            metrics = json.loads((out / "metrics.json").read_text())
            results.append(dict(model=name, seed=seed, branch=branch, graph=graph,
                                known_macro_f1=metrics["known_macro_f1"], unknown_recall=metrics["unknown_recall"],
                                type_a_recall=metrics["type_a_unknown_recall"], type_b_recall=metrics["type_b_unknown_recall"],
                                benign_false_alarm_rate=metrics["benign_false_alarm_rate"],
                                unknown_precision=metrics["unknown_precision"],
                                benign_alerts_per_device_day=metrics.get("benign_alerts_per_device_day"),
                                source=json.loads((out / "config.json").read_text())["feature_source"]))
    frame = pd.DataFrame(results)
    frame.to_csv(root / "per_seed.csv", index=False)
    measurements = ["known_macro_f1", "unknown_recall", "type_a_recall", "type_b_recall",
                    "benign_false_alarm_rate", "unknown_precision", "benign_alerts_per_device_day"]
    averages = frame.groupby(["model", "branch", "graph", "source"], as_index=False)[measurements].mean()
    spread = frame.groupby("model")[measurements].std().fillna(0).add_suffix("_std").rename_axis("model").reset_index()
    averages.merge(spread, on="model").assign(seeds=len(seeds)).to_csv(root / "summary.csv", index=False)
    (root / "experiment.json").write_text(json.dumps(dict(
        input=args.input, input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        splits=args.splits, seeds=seeds, epochs=args.epochs, known=args.known, aux=args.aux, heldout=args.heldout
    ), indent=2))
    print(f"Saved comparison to {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
