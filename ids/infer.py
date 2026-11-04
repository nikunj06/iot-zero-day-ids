"""Score fresh unlabeled flows using a saved training run."""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .data import feature_source
from .model import PayloadGraphIDS
from .train import predict_batches, prototype_distance


def score_rows(rows, run, out):
    run, out = Path(run), Path(out)
    config = json.loads((run / "config.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    if not rows or any("flow_id" not in r or "capture_id" not in r for r in rows):
        raise ValueError("every observation needs flow_id and capture_id")
    if len(set(r["flow_id"] for r in rows)) != len(rows):
        raise ValueError("flow_id values must be unique")
    source = feature_source(rows)
    trained_source = config.get("feature_source", "synthetic_flow_v1")
    if source != trained_source:
        raise ValueError(f"incompatible input: model trained on {trained_source}, input uses {source}; use the local lab rules for app telemetry or train a compatible model")
    model = PayloadGraphIDS(len(config["features"]), len(config["classes"]), len(config["families"]), config["branch"], config["graph"])
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True))
    prob, family_prob, emb = predict_batches(model, rows, joblib.load(run / "scaler.joblib"), config["payload_len"], config["batch_size"], torch.device("cpu"))
    confidence = prob.max(axis=1)
    if config.get("decision") == "hybrid":
        with np.load(run / "prototypes.npz") as reference:
            distance = prototype_distance(emb, prob.argmax(axis=1), reference["center"], reference["scale"], reference["centroids"])
        rejected = (confidence < metrics["threshold"]) | (distance > metrics["prototype_threshold"])
    else:
        distance = np.full(len(rows), np.nan)
        rejected = confidence < metrics["threshold"]
    clusters = np.full(len(rows), -1, dtype=int)
    if rejected.sum() >= 3:
        clusters[rejected] = DBSCAN(eps=2.5, min_samples=3).fit_predict(StandardScaler().fit_transform(emb[rejected]))
    table = pd.DataFrame([dict(flow_id=r["flow_id"], capture_id=r["capture_id"],
                               timestamp=r.get("timestamp", ""), src=r.get("src", ""), dst=r.get("dst", ""),
                               lab_pattern=r.get("lab_pattern", ""), detector="cnn_gatv2", feature_source=source,
                               predicted=config["classes"][int(prob[i].argmax())] if not rejected[i] else "unknown",
                               confidence=float(confidence[i]), novelty_score=float(1-confidence[i]),
                               prototype_distance=float(distance[i]),
                               rejection_reason=("low_class_score" if confidence[i] < metrics["threshold"] else
                                                 "far_from_known_prototype" if rejected[i] else ""),
                               novelty_type="-" if not rejected[i] else ("type_b" if family_prob[i].max() >= metrics["family_threshold"] else "type_a"),
                               family_match=config["families"][int(family_prob[i].argmax())],
                               family_confidence=float(family_prob[i].max()), cluster=clusters[i])
                          for i, r in enumerate(rows)])
    out.mkdir(parents=True, exist_ok=True)
    for name, result in (("predictions.csv", table), ("review_queue.csv", table[rejected])):
        temp = out / f"{name}.tmp"
        result.to_csv(temp, index=False)
        os.replace(temp, out / name)
    (out / "run_info.json").write_text(json.dumps(dict(mode="model_inference", source=source,
                                                    model_run=str(run), note="No accuracy can be computed without independent labels."), indent=2))
    return len(rows), int(rejected.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with flow_id/capture_id/features/payload_hex; no label needed")
    parser.add_argument("--run", required=True, help="directory containing model.pt/config.json/scaler.joblib")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s.strip()]
    try:
        count, rejected = score_rows(rows, args.run, args.out)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Scored {count} flows; {rejected} rejected for review. Output: {args.out}")


if __name__ == "__main__":
    main()
