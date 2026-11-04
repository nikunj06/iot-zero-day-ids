import json
import socket
import tempfile
import unittest
from pathlib import Path

import dpkt

from ids.data import split_assigned
from ids.iot23 import add_payload, import_manifest, make_rows
from ids.pilot import summarize


HEADER = ("#separator \\x09\n#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto"
          "\tduration\torig_pkts\torig_ip_bytes\tresp_pkts\tresp_ip_bytes\thistory"
          "\ttunnel_parents   label   detailed-label\n")


def log_line(ts, uid, tag, detail, sport=5001, duration="0.1"):
    return (f"{ts}\t{uid}\t10.0.0.1\t{sport}\t10.0.0.2\t80\ttcp\t{duration}"
            f"\t1\t60\t0\t0\tS\t-   {tag}   {detail}\n")


def packet(payload=b"hello"):
    tcp = dpkt.tcp.TCP(sport=5001, dport=80, seq=1, flags=dpkt.tcp.TH_SYN, data=payload)
    ip = dpkt.ip.IP(src=socket.inet_aton("10.0.0.1"), dst=socket.inet_aton("10.0.0.2"),
                    p=dpkt.ip.IP_PROTO_TCP, data=tcp)
    ip.len = 20 + len(tcp)
    return bytes(dpkt.ethernet.Ethernet(src=b"\x01"*6, dst=b"\x02"*6,
                                        type=dpkt.ethernet.ETH_TYPE_IP, data=ip))


class IoT23Test(unittest.TestCase):
    def test_mixed_labels_from_real_zeek_format(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "conn.log.labeled"
            log.write_text(HEADER + log_line(10, "a", "Benign", "-") +
                           log_line(20, "b", "Malicious", "C&C"))
            rows, audit = import_manifest([{"capture_id": "capture-a", "zeek_log": str(log)}],
                                          {"C&C": {"label": "C2-Variant", "family": "Botnet"}})
            self.assertEqual([r["label"] for r in rows], ["benign", "C2-Variant"])
            self.assertEqual({r["capture_id"] for r in rows}, {"capture-a"})
            self.assertEqual(audit["flows"], 2)
            self.assertEqual(rows[0]["feature_source"], "iot23_zeek_v1")
            with self.assertRaisesRegex(ValueError, "unmapped"):
                make_rows(log, "capture-a")
            fresh, _ = make_rows(log, "pilot-a", unlabeled=True)
            self.assertNotIn("label", fresh[0])

    def test_payload_join_is_time_and_tuple_based(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "conn.log"
            log.write_text(HEADER + log_line(20, "second", "Benign", "-") +
                           log_line(10, "first", "Benign", "-"))
            rows, _ = make_rows(log, "capture-a")
            capture = root / "test.pcap"
            with capture.open("wb") as handle:
                writer = dpkt.pcap.Writer(handle)
                writer.writepkt(packet(b"one"), ts=10.01)
                writer.writepkt(packet(b"two"), ts=20.01)
            audit = add_payload(rows, [capture])
            self.assertEqual([bytes.fromhex(r["payload_hex"]) for r in rows], [b"two", b"one"])
            self.assertEqual(audit["matched_packets"], 2)

    def test_explicit_capture_split_keeps_mixed_labels_separate(self):
        rows = []
        for capture in ("train", "validation", "test"):
            for label, family in (("benign", "benign"), ("known", "DoS"),
                                  ("aux", "Spoof"), ("heldout", "DoS")):
                rows.append(dict(flow_id=f"{capture}-{label}", capture_id=capture,
                                 label=label, family=family))
        split = {"train": ["train"], "validation": ["validation"], "test": ["test"]}
        train, valid, test, categories = split_assigned(
            rows, ["benign", "known"], ["aux"], ["heldout"], split
        )
        self.assertEqual(categories["heldout"], "type_b")
        self.assertEqual({r["label"] for r in train}, {"benign", "known", "aux"})
        self.assertEqual({r["label"] for r in test}, {"benign", "known", "heldout"})
        self.assertFalse({r["capture_id"] for r in train} & {r["capture_id"] for r in valid})
        split["test"] = ["validation"]
        with self.assertRaisesRegex(ValueError, "multiple"):
            split_assigned(rows, ["benign", "known"], ["aux"], ["heldout"], split)

    def test_pilot_rate_requires_explicit_exposure(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            pd.DataFrame({"predicted": ["unknown", "benign", "unknown"]}).to_csv(path, index=False)
            result = summarize(path, hours=12, devices=2, budget=1)
            self.assertEqual(result["alerts_per_device_day"], 2)
            self.assertFalse(result["within_predefined_budget"])


if __name__ == "__main__":
    unittest.main()
