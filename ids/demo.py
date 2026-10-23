"""Deterministic synthetic data for pipeline demonstration, never benchmark evidence."""
import argparse
import random

from .data import save_jsonl


CLASSES = {
    "benign": ("benign", 3, 230, 0.5, 1883, b"sensor/temperature=24.2"),
    "DoS-SYN": ("DoS", 95, 6600, 0.15, 80, b""),
    "DoS-HTTP": ("DoS", 72, 9500, 0.4, 8080, b"GET /api/read HTTP/1.1\r\n"),
    "Recon-Scan": ("Recon", 8, 490, 0.09, 22, b""),
    "Spoof-ARP": ("Spoof", 18, 1300, 0.2, 0, b"ARP REPLY fake gateway"),
    "Mirai-Bot": ("Mirai", 42, 4200, 0.3, 23, b"login admin password"),
}


def generate(seed=7, groups=8, flows_per_group=20):
    rng = random.Random(seed)
    rows = []
    for label, (family, packets, volume, duration, port, data) in CLASSES.items():
        for g in range(groups):
            capture = f"sim-{label}-{g}"
            for i in range(flows_per_group):
                p = max(1, int(packets * rng.uniform(0.75, 1.25)))
                b = max(p, int(volume * rng.uniform(0.7, 1.3)))
                payload = data + bytes(rng.randrange(32, 127) for _ in range(8)) if data else b""
                rows.append(dict(flow_id=f"{capture}-{i}", capture_id=capture,
                                 timestamp=g * 100 + i, src=f"10.0.{g}.2", dst="10.0.0.1",
                                 protocol="TCP", dst_port=port, duration=duration * rng.uniform(.7, 1.3),
                                 packets=p, bytes=b, mean_size=b / p,
                                 iat_mean=duration / max(p - 1, 1),
                                 tcp_syn=1 if "SYN" in label or label == "Recon-Scan" else 0,
                                 payload_hex=payload.hex(), label=label, family=family))
    rng.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/demo.jsonl")
    args = parser.parse_args()
    rows = generate()
    save_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} SIMULATED flows to {args.output}")


if __name__ == "__main__":
    main()
