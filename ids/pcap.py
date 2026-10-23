"""Simple offline PCAP-to-flow adapter for individually labeled homogeneous captures."""
import argparse
import json
import socket
from collections import defaultdict
from pathlib import Path

from .data import save_jsonl


def extract_capture(path, label, family, timeout=30):
    import dpkt

    flows = {}
    completed = []
    sequence = defaultdict(int)
    path = Path(path)
    with open(path, "rb") as handle:
        try:
            packets = dpkt.pcap.Reader(handle)
        except (ValueError, dpkt.dpkt.NeedData):
            handle.seek(0)
            packets = dpkt.pcapng.Reader(handle)
        for time, frame in packets:
            try:
                eth = dpkt.ethernet.Ethernet(frame)
                ip = eth.data
                if isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                    src, dst = socket.inet_ntop(socket.AF_INET6 if isinstance(ip, dpkt.ip6.IP6) else socket.AF_INET, ip.src), socket.inet_ntop(socket.AF_INET6 if isinstance(ip, dpkt.ip6.IP6) else socket.AF_INET, ip.dst)
                    layer = ip.data
                    if isinstance(layer, dpkt.tcp.TCP):
                        proto = "TCP"
                    elif isinstance(layer, dpkt.udp.UDP):
                        proto = "UDP"
                    else:
                        continue
                    sport, dport = int(layer.sport), int(layer.dport)
                    payload = bytes(layer.data)
                    is_syn = int(isinstance(layer, dpkt.tcp.TCP) and bool(layer.flags & dpkt.tcp.TH_SYN))
                elif isinstance(ip, dpkt.arp.ARP):
                    src, dst = socket.inet_ntop(socket.AF_INET, ip.spa), socket.inet_ntop(socket.AF_INET, ip.tpa)
                    proto, sport, dport, payload, is_syn = "ARP", 0, 0, bytes(ip.data), 0
                else:
                    continue
            except (ValueError, dpkt.dpkt.UnpackError):
                continue
            direction = (src, sport, dst, dport)
            reverse = (dst, dport, src, sport)
            key = (proto, min(direction, reverse), max(direction, reverse))
            time = float(time)
            if key in flows and time - flows[key]["last"] > timeout:
                completed.append(flows.pop(key))
            if key not in flows:
                sequence[key] += 1
                flows[key] = dict(flow_id=f"{path.stem}-{len(sequence)}-{sequence[key]}", capture_id=path.stem,
                                  timestamp=time, last=time, src=src, dst=dst, protocol=proto, dst_port=dport,
                                  packets=0, bytes=0, tcp_syn=0, gaps=[], payload=b"")
            flow = flows[key]
            if flow["packets"]:
                flow["gaps"].append(max(0, time - flow["last"]))
            flow["last"] = time
            flow["packets"] += 1
            flow["bytes"] += len(frame)
            flow["tcp_syn"] += is_syn
            if len(flow["payload"]) < 128:
                flow["payload"] += payload[:128 - len(flow["payload"])]
    output = []
    for flow in completed + list(flows.values()):
        output.append(dict(flow_id=flow["flow_id"], capture_id=flow["capture_id"],
                           timestamp=flow["timestamp"], src=flow["src"], dst=flow["dst"],
                           protocol=flow["protocol"], dst_port=flow["dst_port"],
                           duration=max(0, flow["last"] - flow["timestamp"]), packets=flow["packets"],
                           bytes=flow["bytes"], mean_size=flow["bytes"] / flow["packets"],
                           iat_mean=sum(flow["gaps"]) / len(flow["gaps"]) if flow["gaps"] else 0,
                           tcp_syn=flow["tcp_syn"], payload_hex=flow["payload"].hex(),
                           label=label, family=family))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON list with path, fine label and family per homogeneous capture")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    if not isinstance(manifest, list) or not manifest:
        parser.error("manifest must be a nonempty list")
    rows = []
    seen = set()
    for entry in manifest:
        path = Path(entry["path"])
        if path.stem in seen:
            parser.error("capture basenames must be unique")
        seen.add(path.stem)
        rows.extend(extract_capture(path, entry["label"], entry["family"]))
    save_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} flows from {len(manifest)} labeled captures")


if __name__ == "__main__":
    main()
