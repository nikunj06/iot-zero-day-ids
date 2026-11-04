import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


FEATURES = ("duration", "packets", "bytes", "mean_size", "iat_mean", "tcp_syn", "dst_port")
SOURCES = {"synthetic_flow_v1", "pcap_packet_v1", "app_message_v1", "iot23_zeek_v1"}


def feature_source(rows):
    sources = {r.get("feature_source") for r in rows}
    if len(sources) != 1 or None in sources or not sources <= SOURCES:
        raise ValueError(f"all flows must declare one supported feature_source; found {sources}")
    return sources.pop()


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                for key in ("flow_id", "capture_id", "label", "family"):
                    if key not in row:
                        raise ValueError(f"line {line_no}: missing {key}")
                if len(bytes.fromhex(row.get("payload_hex", ""))) > 1_000_000:
                    raise ValueError(f"line {line_no}: excessive payload")
                rows.append(row)
    if not rows:
        raise ValueError("empty input")
    ids = [r["flow_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("flow_id must be unique")
    feature_source(rows)
    return rows


def matrices(rows, payload_len=128, scaler=None):
    stats = np.asarray([[float(r.get(k, 0)) for k in FEATURES] for r in rows], dtype=np.float32)
    if not np.isfinite(stats).all():
        raise ValueError("non-finite traffic feature")
    if scaler is None:
        scaler = StandardScaler().fit(stats)
    stats = np.clip(scaler.transform(stats), -10, 10).astype(np.float32)
    payload = np.zeros((len(rows), payload_len), dtype=np.float32)
    mask = np.zeros((len(rows), 1), dtype=np.float32)
    for i, row in enumerate(rows):
        raw = bytes.fromhex(row.get("payload_hex", ""))[:payload_len]
        if raw:
            payload[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8) / 255.0
            mask[i, 0] = 1
    return stats, payload, mask, scaler


def split_rows(rows, known, aux, heldout, seed=42):
    known, aux, heldout = set(known), set(aux), set(heldout)
    if not known or "benign" not in known or not aux or not heldout:
        raise ValueError("known must include benign; aux and heldout must be nonempty")
    if known & aux or known & heldout or aux & heldout:
        raise ValueError("known, aux, and heldout fine labels must be disjoint")
    present = {r["label"] for r in rows}
    if missing := (known | aux | heldout) - present:
        raise ValueError(f"missing labels: {sorted(missing)}")
    selected = [r for r in rows if r["label"] in known | aux | heldout]
    families = defaultdict(set)
    for row in selected:
        families[row["label"]].add(row["family"])
    if any(len(v) != 1 for v in families.values()):
        raise ValueError("each fine label must have exactly one family")
    known_families = {next(iter(families[k])) for k in known}
    categories = {k: ("type_b" if next(iter(families[k])) in known_families else "type_a") for k in heldout}
    # A capture cannot be assigned conflicting labels, or cross split boundaries.
    capture_labels = defaultdict(set)
    for r in selected:
        capture_labels[r["capture_id"]].add(r["label"])
    if any(len(v) != 1 for v in capture_labels.values()):
        raise ValueError("capture_id spans different labels; provide distinct homogeneous capture groups")
    train, valid, test = [], [], []
    for label in sorted(known | aux):
        subset = [r for r in selected if r["label"] == label]
        groups = [r["capture_id"] for r in subset]
        if len(set(groups)) < (3 if label in known else 2):
            raise ValueError(f"{label}: insufficient independent capture groups")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.4 if label in known else 0.5, random_state=seed)
        tr_idx, rest_idx = next(splitter.split(subset, groups=groups))
        train.extend(subset[i] for i in tr_idx)
        rest = [subset[i] for i in rest_idx]
        if label in aux:
            valid.extend(rest)
        else:
            rest_groups = [r["capture_id"] for r in rest]
            if len(set(rest_groups)) < 2:
                raise ValueError(f"{label}: need at least two held-aside capture groups")
            v_idx, t_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(rest, groups=rest_groups))
            valid.extend(rest[i] for i in v_idx)
            test.extend(rest[i] for i in t_idx)
    test.extend(r for r in selected if r["label"] in heldout)
    return train, valid, test, categories


def split_assigned(rows, known, aux, heldout, assignments):
    """Use independent capture/session assignments even when each capture has mixed labels."""
    known, aux, heldout = set(known), set(aux), set(heldout)
    if "benign" not in known or not aux or not heldout or known & aux or known & heldout or aux & heldout:
        raise ValueError("known (including benign), auxiliary and held-out labels must be nonempty and disjoint")
    if set(assignments) != {"train", "validation", "test"}:
        raise ValueError("split file requires train, validation and test capture ID lists")
    groups = [g for part in assignments.values() for g in part]
    if len(groups) != len(set(groups)):
        raise ValueError("a capture ID is assigned to multiple partitions")
    selected = [r for r in rows if r["label"] in known | aux | heldout]
    if {r["label"] for r in selected} != known | aux | heldout:
        raise ValueError("one or more requested labels are missing")
    observed = {r["capture_id"] for r in selected}
    if observed != set(groups):
        raise ValueError(f"split capture IDs differ from selected data: missing={sorted(observed-set(groups))}, extra={sorted(set(groups)-observed)}")
    family_sets = defaultdict(set)
    for r in selected:
        family_sets[r["label"]].add(r["family"])
    if any(len(v) != 1 for v in family_sets.values()):
        raise ValueError("a fine label maps to multiple families")
    known_families = {next(iter(family_sets[k])) for k in known}
    categories = {k: ("type_b" if next(iter(family_sets[k])) in known_families else "type_a") for k in heldout}
    parts = []
    for part in ("train", "validation", "test"):
        labels = known | (heldout if part == "test" else aux)
        captures = set(assignments[part])
        subset = [r for r in selected if r["capture_id"] in captures and r["label"] in labels]
        required = known | (heldout if part == "test" else aux)
        if not required <= {r["label"] for r in subset}:
            raise ValueError(f"{part} is missing required labels {sorted(required-{r['label'] for r in subset})}")
        parts.append(subset)
    return *parts, categories


def knn_edges(stats, k=4):
    """Directed edges based on normalized feature distance within the current batch."""
    import torch

    n = stats.size(0)
    if n == 1:
        return torch.tensor([[0], [0]], device=stats.device)
    d = torch.cdist(stats, stats)
    d.fill_diagonal_(float("inf"))
    near = d.topk(min(k, n - 1), largest=False).indices
    source = near.reshape(-1)
    target = torch.arange(n, device=stats.device).repeat_interleave(near.size(1))
    return torch.stack((source, target))


def save_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
