import tempfile
import unittest
from pathlib import Path

import socket
import dpkt

from ids.pcap import extract_capture


class PcapTest(unittest.TestCase):
    def test_timeout_keeps_both_flows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.pcap"
            with open(path, "wb") as handle:
                writer = dpkt.pcap.Writer(handle)
                for time, payload in [(1, b"one"), (50, b"two")]:
                    tcp = dpkt.tcp.TCP(sport=1234, dport=80, data=payload)
                    ip = dpkt.ip.IP(src=socket.inet_aton("10.0.0.1"), dst=socket.inet_aton("10.0.0.2"), p=6, data=tcp)
                    ip.len = len(ip)
                    ethernet = dpkt.ethernet.Ethernet(src=b"\x00"*6, dst=b"\x11"*6, type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
                    writer.writepkt(bytes(ethernet), ts=time)
                writer.close()
            rows = extract_capture(path, "Test", "TestFamily", timeout=30)
            self.assertEqual(len(rows), 2)
            self.assertEqual({r["payload_hex"] for r in rows}, {b"one".hex(), b"two".hex()})
            self.assertEqual(len({r["flow_id"] for r in rows}), 2)


if __name__ == "__main__":
    unittest.main()
