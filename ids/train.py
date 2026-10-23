import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score, homogeneity_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from .data import FEATURES, matrices, read_jsonl, split_rows
from .model import PayloadGraphIDS


DEFAULT_KNOWN = ["benign", "DoS-SYN", "Recon-Scan"]
DEFAULT_AUX = ["Spoof-ARP"]
DEFAULT_HELDOUT = ["DoS-HTTP", "Mirai-Bot"]


def parse_labels(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def predict_batches(model, rows, scaler, payload_len, batch_size, device):
    model.eval()
    all_prob, all_family, all_embeddings = [], [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            s, p, m, _ = matrices(batch, payload_len, scaler)
            logits, family_logits, emb = model(torch.from_numpy(s).to(device), torch.from_numpy(p).to(device), torch.from_numpy(m).to(device))
            all_prob.append(torch.sigmoid(logits).cpu().numpy())
            all_family.append(torch.sigmoid(family_logits).cpu().numpy())
            all_embeddings.append(emb.cpu().numpy())
    return np.concatenate(all_prob), np.concatenate(all_family), np.concatenate(all_embeddings)


def select_threshold(prob, rows, known, aux):
    confidence = prob.max(axis=1)
    target = np.array([r["label"] in known for r in rows])
    # Balanced acceptance/rejection on validation known and auxiliary unknown.
    # A sigmoid value below 0.5 is not affirmative evidence for a known class.
    options = np.unique(np.r_[0.5, np.linspace(0.5, 0.99, 50), confidence[confidence >= 0.5]])
    scores = [(0.5 * ((confidence[target] >= t).mean() + (confidence[~target] < t).mean()), t) for t in options]
    return float(max(scores, key=lambda x: (x[0], -abs(x[1] - .5)))[1])


def classification_metrics(rows, predictions, known, categories, heldout):
    truth = np.array([r["label"] for r in rows])
    pred = np.asarray(predictions)
    known_mask = np.isin(truth, known)
    benign_mask = truth == "benign"
    result = {
        "n_test": len(rows),
        "known_macro_f1": float(f1_score(truth[known_mask], pred[known_mask], labels=known, average="macro", zero_division=0)),
        "benign_false_alarm_rate": float((pred[benign_mask] != "benign").mean()),
        "unknown_recall": float((pred[np.isin(truth, heldout)] == "unknown").mean()),
    }
    for category in ("type_a", "type_b"):
        mask = np.array([categories.get(label) == category for label in truth])
        result[f"{category}_unknown_recall"] = float((pred[mask] == "unknown").mean()) if mask.any() else None
        result[f"{category}_n"] = int(mask.sum())
    return result


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    known, aux, heldout = map(parse_labels, (args.known, args.aux, args.heldout))
    rows = read_jsonl(args.input)
    train, valid, test, categories = split_rows(rows, known, aux, heldout, args.seed)
    # Deterministic mixed batches: never group evaluation batches by label or capture.
    rng = random.Random(args.seed)
    for part in (train, valid, test):
        rng.shuffle(part)
    train_known = [r for r in train if r["label"] in known]
    raw = np.asarray([[float(r.get(k, 0)) for k in FEATURES] for r in train_known], dtype=np.float32)
    scaler = StandardScaler().fit(raw)
    classes = sorted(known)
    families_by_class = {r["label"]: r["family"] for r in train_known}
    families = sorted(set(families_by_class.values()) - {"benign"})
    if not families:
        raise ValueError("at least one known attack family is required for hierarchical novelty")
    model = PayloadGraphIDS(len(FEATURES), len(classes), len(families), args.branch, args.graph)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(train))
        losses = []
        for start in range(0, len(train), args.batch_size):
            batch = [train[int(i)] for i in order[start : start + args.batch_size]]
            s, p, m, _ = matrices(batch, args.payload_len, scaler)
            targets = torch.zeros((len(batch), len(classes)), device=device)
            family_targets = torch.zeros((len(batch), len(families)), device=device)
            for i, r in enumerate(batch):
                if r["label"] in known:
                    targets[i, classes.index(r["label"])] = 1
                    if r["family"] in families:
                        family_targets[i, families.index(r["family"])] = 1
            optimizer.zero_grad()
            logits, family_logits, _ = model(torch.from_numpy(s).to(device), torch.from_numpy(p).to(device), torch.from_numpy(m).to(device))
            loss = criterion(logits, targets) + 0.3 * criterion(family_logits, family_targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch {epoch + 1}/{args.epochs} loss={np.mean(losses):.4f}")

    val_prob, val_family, _ = predict_batches(model, valid, scaler, args.payload_len, args.batch_size, device)
    threshold = select_threshold(val_prob, valid, known, aux)
    family_threshold = select_threshold(val_family, valid, [k for k in known if k != "benign"], aux)
    prob, family_prob, embeddings = predict_batches(model, test, scaler, args.payload_len, args.batch_size, device)
    confidence = prob.max(axis=1)
    pred = [classes[int(i)] if confidence[n] >= threshold else "unknown" for n, i in enumerate(prob.argmax(axis=1))]
    novelty_type = ["-" if p != "unknown" else
                    ("type_b" if family_prob[i].max() >= family_threshold else "type_a")
                    for i, p in enumerate(pred)]
    metrics = classification_metrics(test, pred, classes, categories, heldout)
    rejected_true_unknown = [i for i, r in enumerate(test) if r["label"] in heldout and pred[i] == "unknown"]
    metrics.update(n_detected_heldout=len(rejected_true_unknown),
                   hierarchical_accuracy_on_detected_unknown=(float(np.mean([
                       novelty_type[i] == categories[test[i]["label"]] for i in rejected_true_unknown]))
                       if rejected_true_unknown else None),
                   threshold=threshold, family_threshold=family_threshold, n_train=len(train), n_validation=len(valid),
                   n_aux_train=sum(r["label"] in aux for r in train),
                   note="Synthetic input does not establish real-world attack detection." if all(r["capture_id"].startswith("sim-") for r in rows) else "Evaluate capture provenance and dataset leakage.")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output / "scaler.joblib")
    torch.save(model.state_dict(), output / "model.pt")
    config = dict(classes=classes, families=families, aux=aux, heldout=heldout, categories=categories, branch=args.branch,
                  graph=args.graph, payload_len=args.payload_len, batch_size=args.batch_size, seed=args.seed,
                  features=FEATURES)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    table = pd.DataFrame([dict(flow_id=r["flow_id"], capture_id=r["capture_id"], timestamp=r.get("timestamp", ""),
                               src=r.get("src", ""), dst=r.get("dst", ""), true_label=r["label"],
                               family=r["family"], predicted=p, confidence=float(c),
                               novelty_score=float(1-c), novelty_type=novelty_type[i],
                               family_match=families[int(family_prob[i].argmax())],
                               family_confidence=float(family_prob[i].max()),
                               experiment=categories.get(r["label"], "known"))
                          for i, (r, p, c) in enumerate(zip(test, pred, confidence))])
    table.to_csv(output / "predictions.csv", index=False)
    rejected = np.flatnonzero(np.asarray(pred) == "unknown")
    cluster_ids = np.full(len(test), -1, dtype=int)
    if len(rejected) >= 3:
        z = StandardScaler().fit_transform(embeddings[rejected])
        cluster_ids[rejected] = DBSCAN(eps=args.cluster_eps, min_samples=3).fit_predict(z)
    assigned = rejected[cluster_ids[rejected] >= 0]
    metrics["clustered_unknown_flows"] = int(len(assigned))
    metrics["cluster_homogeneity_on_labeled_test"] = (float(homogeneity_score(
        [test[i]["label"] for i in assigned], cluster_ids[assigned])) if len(assigned) >= 2 else None)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    table["cluster"] = cluster_ids
    table[table.predicted == "unknown"].to_csv(output / "review_queue.csv", index=False)
    table[["flow_id", "predicted", "true_label", "cluster"]].to_csv(output / "clusters.csv", index=False)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a payload + GATv2 open-set research prototype")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--known", default=",".join(DEFAULT_KNOWN))
    parser.add_argument("--aux", default=",".join(DEFAULT_AUX))
    parser.add_argument("--heldout", default=",".join(DEFAULT_HELDOUT))
    parser.add_argument("--branch", choices=("both", "stats", "payload"), default="both")
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--payload-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--cluster-eps", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 2 or args.payload_len < 8 or args.epochs < 1:
        parser.error("batch-size >=2, payload-len >=8, epochs >=1 required")
    run(args)


if __name__ == "__main__":
    main()
