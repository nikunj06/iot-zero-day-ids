import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ids.data import feature_source
from ids.infer import score_rows
from ids.live_rules import score_live_rows
from ids.train import prototype_distance, select_distance_threshold


def observation(flow_id, payload, packets):
    return dict(flow_id=flow_id, capture_id=flow_id, feature_source="app_message_v1",
                payload_hex=payload.hex(), packets=packets, lab_pattern="normal")


class SourceAndRulesTest(unittest.TestCase):
    def test_mixed_sources_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one supported feature_source"):
            feature_source([observation("a", b"sensor", 1),
                            dict(observation("b", b"sensor", 1), feature_source="synthetic_flow_v1")])

    def test_live_rules_alert_and_provenance(self):
        rows = [observation("a", b"sensor", 2), observation("b", b"CONFIG:bad", 2),
                observation("c", b"sensor", 20)]
        with tempfile.TemporaryDirectory() as directory:
            count, alerts = score_live_rows(rows, directory)
            self.assertEqual((count, alerts), (3, 2))
            result = pd.read_csv(Path(directory) / "predictions.csv")
            self.assertEqual(result.predicted.tolist(), ["benign", "lab_alert", "lab_alert"])
            self.assertEqual(json.loads((Path(directory) / "run_info.json").read_text())["mode"], "local_lab_rules")

    def test_incompatible_model_source_fails_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            (run / "config.json").write_text(json.dumps({"feature_source": "synthetic_flow_v1"}))
            (run / "metrics.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "incompatible input"):
                score_rows([observation("a", b"sensor", 1)], run, Path(directory) / "out")

    def test_distance_threshold_selects_validation_separation(self):
        import numpy as np
        embeddings = np.array([[0.0], [1.0], [8.0]])
        distances = prototype_distance(embeddings, np.array([0, 0, 0]),
                                       np.array([0.0]), np.array([1.0]), np.array([[0.0]]))
        rows = [dict(label=x) for x in ["benign", "benign", "aux"]]
        threshold = select_distance_threshold(distances, np.array([.9, .9, .9]), rows, {"benign"}, .5)
        self.assertTrue(distances[1] <= threshold < distances[2])


if __name__ == "__main__":
    unittest.main()
