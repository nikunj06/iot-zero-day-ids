import unittest

from ids.data import split_rows
from ids.demo import generate


class DataSplitTest(unittest.TestCase):
    def test_disjoint_capture_groups_and_novelty_types(self):
        train, valid, test, kinds = split_rows(generate(),
            ["benign", "DoS-SYN", "Recon-Scan"], ["Spoof-ARP"], ["DoS-HTTP", "Mirai-Bot"])
        sets = [{r["capture_id"] for r in part} for part in (train, valid, test)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(kinds["DoS-HTTP"], "type_b")
        self.assertEqual(kinds["Mirai-Bot"], "type_a")
        self.assertTrue(all(r["label"] not in {"DoS-HTTP", "Mirai-Bot"} for r in train + valid))


if __name__ == "__main__":
    unittest.main()
