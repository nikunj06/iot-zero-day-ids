"""Score fresh unlabeled flows using a saved training run."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .model import PayloadGraphIDS
from .train import predict_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with flow_id/capture_id/features/payload_hex; no label needed")
    parser.add_argument("--run", required=True, help="directory containing model.pt/config.json/scaler.joblib")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.out)
    config = json.loads((run / "config.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    rows = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s.strip()]
    if not rows or any("flow_id" not in r or "capture_id" not in r for r in rows):
        parser.error("every observation needs flow_id and capture_id")
    if len(set(r["flow_id"] for r in rows)) != len(rows):
        parser.error("flow_id values must be unique")
    model = PayloadGraphIDS(len(config["features"]), len(config["classes"]), len(config["families"]), config["branch"], config["graph"])
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True))
    prob, family_prob, emb = predict_batches(model, rows, joblib.load(run / "scaler.joblib"), config["payload_len"], config["batch_size"], torch.device("cpu"))
    confidence = prob.max(axis=1)
    rejected = confidence < metrics["threshold"]
    clusters = np.full(len(rows), -1, dtype=int)
    if rejected.sum() >= 3:
        clusters[rejected] = DBSCAN(eps=2.5, min_samples=3).fit_predict(StandardScaler().fit_transform(emb[rejected]))
    table = pd.DataFrame([dict(flow_id=r["flow_id"], capture_id=r["capture_id"],
                               timestamp=r.get("timestamp", ""), src=r.get("src", ""), dst=r.get("dst", ""),
                               predicted=config["classes"][int(prob[i].argmax())] if not rejected[i] else "unknown",
                               confidence=float(confidence[i]), novelty_score=float(1-confidence[i]),
                               novelty_type="-" if not rejected[i] else ("type_b" if family_prob[i].max() >= metrics["family_threshold"] else "type_a"),
                               family_match=config["families"][int(family_prob[i].argmax())],
                               family_confidence=float(family_prob[i].max()), cluster=clusters[i])
                          for i, r in enumerate(rows)])
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "predictions.csv", index=False)
    table[rejected].to_csv(out / "review_queue.csv", index=False)
    print(f"Scored {len(rows)} flows; {rejected.sum()} rejected for review. Output: {out}")


if __name__ == "__main__":
    main()
