"""Import IoT-23 Zeek flow labels; optionally attach uniquely matched PCAP payload bytes."""
import argparse
import bisect
import json
import socket
from collections import Counter, defaultdict
from pathlib import Path

from .data import save_jsonl

SOURCE = "iot23_zeek_v1"


def zeek_rows(path):
    fields = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.rstrip("\r\n")
            if line.startswith("#fields"):
                fields = line.split()[1:]
                continue
            if not line or line.startswith("#"):
                continue
            if not fields:
                raise ValueError(f"{path}:{line_no}: missing Zeek #fields header")
            values = line.split("\t")
            # IoT-23 appends the last two label fields with spaces to some Zeek TSV logs.
            if len(values) == len(fields) - 2:
                values = values[:-1] + values[-1].rsplit(None, 2)
            if len(values) != len(fields):
                raise ValueError(f"{path}:{line_no}: {len(values)} values, expected {len(fields)}")
            yield dict(zip(fields, values))


def number(value):
    return float(value) if value not in (None, "-", "(empty)", "") else 0.0


def make_rows(log_path, capture_id, taxonomy=None, unlabeled=False):
    rows, skipped = [], Counter()
    required = {"ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto"}
    for item in zeek_rows(log_path):
        if not required <= item.keys():
            raise ValueError(f"{log_path}: missing required Zeek fields {sorted(required-item.keys())}")
        proto = item["proto"].upper()
        if proto not in ("TCP", "UDP"):
            skipped["unsupported_protocol"] += 1
            continue
        tag = item.get("label", "").strip()
        detail = item.get("detailed-label", "").strip()
        if not unlabeled:
            if tag.lower() == "benign":
                fine, family = "benign", "benign"
            else:
                key = detail if detail not in ("", "-") else tag
                mapped = (taxonomy or {}).get(key)
                if not mapped or not mapped.get("label") or not mapped.get("family"):
                    raise ValueError(f"unmapped IoT-23 label {tag!r}/{detail!r}; provide taxonomy entry {key!r}")
                fine, family = mapped["label"], mapped["family"]
        packets = int(number(item.get("orig_pkts"))) + int(number(item.get("resp_pkts")))
        ip_bytes = number(item.get("orig_ip_bytes")) + number(item.get("resp_ip_bytes"))
        duration = number(item.get("duration"))
        row = dict(
            flow_id=f"{capture_id}:{item['uid']}", capture_id=capture_id, zeek_uid=item["uid"],
            timestamp=float(item["ts"]), src=item["id.orig_h"], src_port=int(item["id.orig_p"]),
            dst=item["id.resp_h"], dst_port=int(item["id.resp_p"]), protocol=proto,
            duration=duration, packets=packets, bytes=ip_bytes,
            mean_size=ip_bytes / packets if packets else 0.0,
            iat_mean=duration / (packets - 1) if packets > 1 else 0.0,
            tcp_syn=item.get("history", "").count("S") + item.get("history", "").count("s") if proto == "TCP" else 0,
            payload_hex="", feature_source=SOURCE,
        )
        if not unlabeled:
            row.update(label=fine, family=family)
        rows.append(row)
    if not rows:
        raise ValueError(f"{log_path}: no supported TCP/UDP flows")
    if len({r["flow_id"] for r in rows}) != len(rows):
        raise ValueError(f"{log_path}: duplicate Zeek UID in capture {capture_id}")
    return rows, skipped


def packet_fields(frame):
    import dpkt
    try:
        ip = dpkt.ethernet.Ethernet(frame).data
        if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return None
        layer = ip.data
        if not isinstance(layer, (dpkt.tcp.TCP, dpkt.udp.UDP)):
            return None
        address_family = socket.AF_INET6 if isinstance(ip, dpkt.ip6.IP6) else socket.AF_INET
        return (socket.inet_ntop(address_family, ip.src), int(layer.sport),
                socket.inet_ntop(address_family, ip.dst), int(layer.dport),
                "TCP" if isinstance(layer, dpkt.tcp.TCP) else "UDP", bytes(layer.data))
    except (ValueError, dpkt.dpkt.UnpackError):
        return None


def add_payload(rows, paths, tolerance=0.05, length=128):
    """Match each packet to a unique Zeek 5-tuple and time interval, never by row order."""
    import dpkt
    indexed = defaultdict(list)
    for row in rows:
        key = (row["src"], row["src_port"], row["dst"], row["dst_port"], row["protocol"])
        indexed[key].append(row)
        reverse = (row["dst"], row["dst_port"], row["src"], row["src_port"], row["protocol"])
        if reverse != key:
            indexed[reverse].append(row)
    for key in indexed:
        indexed[key].sort(key=lambda r: r["timestamp"])
    starts = {key: [r["timestamp"] for r in group] for key, group in indexed.items()}
    longest = {key: max(r["duration"] for r in group) for key, group in indexed.items()}
    payload = defaultdict(bytearray)
    audit = Counter()
    for path in paths:
        with open(path, "rb") as handle:
            try:
                reader = dpkt.pcap.Reader(handle)
            except (ValueError, dpkt.dpkt.NeedData):
                handle.seek(0)
                reader = dpkt.pcapng.Reader(handle)
            for ts, frame in reader:
                packet = packet_fields(frame)
                if packet is None:
                    audit["unsupported_packets"] += 1
                    continue
                key, content = packet[:5], packet[5]
                group = indexed.get(key, [])
                stop = bisect.bisect_right(starts.get(key, []), ts + tolerance)
                first = bisect.bisect_left(starts.get(key, []), ts - longest.get(key, 0) - tolerance)
                # Overlapping windows or repeated 5-tuples are refused rather than guessed.
                candidates = [r for r in group[first:stop]
                              if ts >= r["timestamp"] - tolerance
                              and ts <= r["timestamp"] + r["duration"] + tolerance]
                if len(candidates) != 1:
                    audit["ambiguous_packets" if candidates else "unmatched_packets"] += 1
                    continue
                audit["matched_packets"] += 1
                if content:
                    uid = candidates[0]["flow_id"]
                    if len(payload[uid]) < length:
                        payload[uid].extend(content[:length - len(payload[uid])])
    for row in rows:
        row["payload_hex"] = payload[row["flow_id"]].hex()
    audit["flows_with_payload"] = sum(bool(r["payload_hex"]) for r in rows)
    return audit


def import_manifest(manifest, taxonomy=None, unlabeled=False, tolerance=0.05):
    rows, audit = [], Counter()
    seen = set()
    for item in manifest:
        capture_id = item.get("capture_id")
        if not capture_id or capture_id in seen or not item.get("zeek_log"):
            raise ValueError("each manifest entry needs a unique capture_id and zeek_log")
        seen.add(capture_id)
        subset, skipped = make_rows(item["zeek_log"], capture_id, taxonomy, unlabeled)
        audit.update(skipped)
        paths = item.get("pcap", [])
        if isinstance(paths, str):
            paths = [paths]
        if paths:
            audit.update(add_payload(subset, paths, tolerance))
        rows.extend(subset)
    if not rows:
        raise ValueError("empty manifest or no supported Zeek flows")
    audit["flows"] = len(rows)
    audit["captures"] = len(seen)
    audit["payload_available"] = sum(bool(r["payload_hex"]) for r in rows)
    return rows, dict(audit)


def label_inventory(manifest):
    counts = Counter()
    for item in manifest:
        for row in zeek_rows(item["zeek_log"]):
            counts[(row.get("label", ""), row.get("detailed-label", ""))] += 1
    return [{"label": label, "detailed_label": detail, "flows": n}
            for (label, detail), n in sorted(counts.items())]


def main():
    parser = argparse.ArgumentParser(description="Import labeled IoT-23 Zeek flows and optionally PCAP payload")
    parser.add_argument("--manifest", required=True, help="JSON array of capture_id, zeek_log, optional pcap path(s)")
    parser.add_argument("--taxonomy", help="JSON map of detailed-label to {label, family}; required for malicious flows")
    parser.add_argument("--unlabeled", action="store_true", help="fresh Zeek log for a pilot, with no ground-truth labels")
    parser.add_argument("--output", help="JSONL output path; required unless --labels-only")
    parser.add_argument("--labels-only", action="store_true", help="show raw label inventory before writing a taxonomy")
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.tolerance <= 1:
        parser.error("tolerance must be 0 to 1 second")
    manifest = json.loads(Path(args.manifest).read_text())
    if not isinstance(manifest, list) or not manifest:
        parser.error("manifest must be a nonempty JSON array")
    if args.labels_only:
        print(json.dumps(label_inventory(manifest), indent=2))
        return
    if not args.output:
        parser.error("--output is required unless --labels-only")
    taxonomy = json.loads(Path(args.taxonomy).read_text()) if args.taxonomy else {}
    try:
        rows, audit = import_manifest(manifest, taxonomy, args.unlabeled, args.tolerance)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    save_jsonl(args.output, rows)
    Path(args.output + ".audit.json").write_text(json.dumps(audit, indent=2))
    print(f"Wrote {len(rows)} flow-level {'unlabeled' if args.unlabeled else 'labeled'} records; audit: {args.output}.audit.json")


if __name__ == "__main__":
    main()
