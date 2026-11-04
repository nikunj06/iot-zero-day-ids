"""Local-only IoT traffic demonstration. No exploit, scanning, or external target."""
import argparse
import json
import random
import socket
import socketserver
import threading
import time
import uuid
from pathlib import Path


HOST = "127.0.0.1"


class Collector(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port, output, capture_id):
        self.output = Path(output)
        self.capture_id = capture_id
        self.lock = threading.Lock()
        self.count = 0
        super().__init__((HOST, port), FlowHandler)

    def append(self, row):
        with self.lock:
            self.count += 1
            # One complete JSON line per completed TCP connection.
            with self.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()


class FlowHandler(socketserver.StreamRequestHandler):
    def handle(self):
        start = time.time()
        times, lengths = [], []
        payload = bytearray()
        self.request.settimeout(3)
        try:
            for line in self.rfile:
                if len(line) > 4096 or len(times) >= 64:
                    break
                times.append(time.time())
                lengths.append(len(line))
                if len(payload) < 128:
                    payload.extend(line[:128 - len(payload)])
        except (OSError, TimeoutError):
            pass
        if not times:
            return
        gaps = [b - a for a, b in zip(times, times[1:])]
        source_ip, source_port = self.client_address
        total = sum(lengths)
        lab_pattern = ("bounded_burst" if len(times) >= 20 else
                       "unusual_command" if bytes(payload).startswith(b"CONFIG") else "normal")
        row = dict(flow_id=f"live-{uuid.uuid4().hex[:12]}", capture_id=self.server.capture_id,
                   timestamp=start, src=source_ip, dst=HOST, protocol="TCP", dst_port=self.server.server_address[1],
                   duration=max(0.0, times[-1] - times[0]), packets=len(times), bytes=total,
                   mean_size=total / len(times), iat_mean=sum(gaps) / len(gaps) if gaps else 0.0,
                   tcp_syn=1, payload_hex=bytes(payload).hex(), lab_pattern=lab_pattern,
                   feature_source="app_message_v1")
        self.server.append(row)


def send_session(port, messages, gap=0.03):
    with socket.create_connection((HOST, port), timeout=3) as conn:
        for message in messages:
            conn.sendall(message + b"\n")
            if gap:
                time.sleep(gap)


def run_lab(output, duration=20, interval=0.5, port=18830, scenario="mixed", seed=51):
    if not 1024 <= port <= 65535 or not 1 <= duration <= 300 or not 0.05 <= interval <= 10:
        raise ValueError("port 1024-65535, duration 1-300 seconds, interval 0.05-10 seconds required")
    rng = random.Random(seed)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    capture_id = f"live-local-{int(time.time())}"
    scenario_counts = {"normal": 0, "burst": 0, "unusual_command": 0}
    with Collector(port, path, capture_id) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        deadline = time.monotonic() + duration
        n = 0
        print(f"Local lab on {HOST}:{port}; telemetry: {path}", flush=True)
        try:
            while time.monotonic() < deadline:
                n += 1
                kind = ("burst" if n % 7 == 0 else "unusual_command" if n % 5 == 0 else "normal") if scenario == "mixed" else scenario
                scenario_counts[kind] += 1
                if kind == "normal":
                    messages = [f"temp={rng.uniform(20, 28):.1f}".encode() for _ in range(3)]
                    gap = 0.03
                elif kind == "burst":
                    # Bounded volume to a local server: anomalous rate, not a DoS attack.
                    messages = [f"temp={rng.uniform(20, 28):.1f}".encode() for _ in range(40)]
                    gap = 0.001
                else:
                    messages = [b"CONFIG RESET?", b"CONFIG EXPORT?", b"CONFIG RESET?"]
                    gap = 0.03
                send_session(port, messages, gap)
                if n % 5 == 0:
                    print(f"Sent {n} sessions; collector completed {server.count} flows", flush=True)
                time.sleep(interval)
        finally:
            server.shutdown()
            thread.join(timeout=3)
    truth = path.with_suffix(".scenarios.json")
    truth.write_text(json.dumps({"capture_id": capture_id, "source": "local application telemetry",
                                 "note": "packets field is a message-count proxy, not a PCAP packet count; scenario counts are not ground-truth attack labels",
                                 "scenario_counts": scenario_counts, "completed_flows": server.count}, indent=2))
    print(f"Finished: {server.count} local flows. Scenario summary: {truth}", flush=True)
    return server.count


def main():
    parser = argparse.ArgumentParser(description="Bounded local TCP IoT simulation (127.0.0.1 only)")
    parser.add_argument("--output", default="data/live.jsonl")
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=18830)
    parser.add_argument("--scenario", choices=("normal", "burst", "unusual_command", "mixed"), default="mixed")
    args = parser.parse_args()
    try:
        run_lab(args.output, args.duration, args.interval, args.port, args.scenario)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
