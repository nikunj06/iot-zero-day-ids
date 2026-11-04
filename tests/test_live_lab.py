import json
import socket
import tempfile
import unittest
from pathlib import Path

from ids.live_lab import run_lab
from ids.watch import completed_rows


class LiveLabTest(unittest.TestCase):
    def test_bounded_local_sessions_emit_unlabeled_flows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.jsonl"
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            if port < 1024:
                self.skipTest("only an ephemeral privileged port was available")
            count = run_lab(path, duration=1, interval=0.1, port=port, scenario="mixed")
            rows = completed_rows(path)
            self.assertEqual(len(rows), count)
            self.assertGreater(count, 0)
            self.assertTrue(all(row["src"] == "127.0.0.1" and "label" not in row for row in rows))
            self.assertTrue(all(row["lab_pattern"] in {"normal", "bounded_burst", "unusual_command"} for row in rows))
            summary = json.loads(path.with_suffix(".scenarios.json").read_text())
            self.assertEqual(summary["completed_flows"], count)


if __name__ == "__main__":
    unittest.main()
