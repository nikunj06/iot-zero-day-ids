
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from ids.data import FEATURES, feature_source
from ids.infer import score_rows
from ids.live_rules import score_live_rows
from ids.model import PayloadGraphIDS, DirectionalPayloadGraphIDS
from ids.train import prototype_distance, select_distance_threshold


def observation(flow_id, payload, packets):
    return dict(
        flow_id=flow_id,
        capture_id=flow_id,
        feature_source="app_message_v1",
        payload_hex=payload.hex(),
        packets=packets,
        lab_pattern="normal"
    )


def model_observation(flow_id, source="synthetic_flow_v1"):
    return dict(
        flow_id=flow_id,
        capture_id=flow_id,
        feature_source=source,
        duration=1.0,
        packets=2,
        bytes=100,
        mean_size=50,
        iat_mean=0.5,
        tcp_syn=1,
        dst_port=80,
        payload_hex="01020304",
        payload_orig_hex="0102",
        payload_resp_hex="0304"
    )


def create_model_run(directory, model_type):
    run = Path(directory) / "run"
    run.mkdir()

    classes = ["benign", "attack"]
    families = ["DoS"]

    scaler = StandardScaler().fit(
        np.zeros((2, len(FEATURES)), dtype=np.float32)
    )

    if model_type == "directional":
        model = DirectionalPayloadGraphIDS(
            len(FEATURES), len(classes), len(families), graph=False
        )
        source = "iot23_zeek_v1"
    else:
        model = PayloadGraphIDS(
            len(FEATURES), len(classes), len(families),
            branch="both", graph=False
        )
        source = "synthetic_flow_v1"

    config = dict(
        classes=classes,
        families=families,
        aux=["aux"],
        heldout=["heldout"],
        categories={"heldout": "type_a"},
        model_type=model_type,
        branch="both",
        graph=False,
        payload_len=128,
        batch_size=2,
        seed=42,
        features=FEATURES,
        feature_source=source,
        decision="sigmoid"
    )

    metrics = {
        "threshold": 0.5,
        "family_threshold": 0.5
    }

    torch.save(model.state_dict(), run / "model.pt")
    joblib.dump(scaler, run / "scaler.joblib")

    (run / "config.json").write_text(json.dumps(config))
    (run / "metrics.json").write_text(json.dumps(metrics))

    return run


class SourceAndRulesTest(unittest.TestCase):

    def test_mixed_sources_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "one supported feature_source"
        ):
            feature_source([
                observation("a", b"sensor", 1),
                dict(
                    observation("b", b"sensor", 1),
                    feature_source="synthetic_flow_v1"
                )
            ])

    def test_live_rules_alert_and_provenance(self):
        rows = [
            observation("a", b"sensor", 2),
            observation("b", b"CONFIG:bad", 2),
            observation("c", b"sensor", 20)
        ]

        with tempfile.TemporaryDirectory() as directory:
            count, alerts = score_live_rows(rows, directory)

            self.assertEqual((count, alerts), (3, 2))

            result = pd.read_csv(
                Path(directory) / "predictions.csv"
            )

            self.assertEqual(
                result.predicted.tolist(),
                ["benign", "lab_alert", "lab_alert"]
            )

            run_info = json.loads(
                (Path(directory) / "run_info.json").read_text()
            )

            self.assertEqual(
                run_info["mode"],
                "local_lab_rules"
            )

    def test_incompatible_model_source_fails_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()

            (run / "config.json").write_text(
                json.dumps({
                    "feature_source": "synthetic_flow_v1"
                })
            )

            (run / "metrics.json").write_text("{}")

            with self.assertRaisesRegex(
                ValueError, "incompatible input"
            ):
                score_rows(
                    [observation("a", b"sensor", 1)],
                    run,
                    Path(directory) / "out"
                )

    def test_distance_threshold_selects_validation_separation(self):
        embeddings = np.array([
            [0.0],
            [1.0],
            [8.0]
        ])

        distances = prototype_distance(
            embeddings,
            np.array([0, 0, 0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([[0.0]])
        )

        rows = [
            dict(label=x)
            for x in ["benign", "benign", "aux"]
        ]

        threshold = select_distance_threshold(
            distances,
            np.array([.9, .9, .9]),
            rows,
            {"benign"},
            .5
        )

        self.assertTrue(
            distances[1] <= threshold < distances[2]
        )

    def test_baseline_checkpoint_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            run = create_model_run(directory, "baseline")

            rows = [
                model_observation("flow1"),
                model_observation("flow2")
            ]

            count, rejected = score_rows(
                rows,
                run,
                Path(directory) / "output"
            )

            self.assertEqual(count, 2)
            self.assertGreaterEqual(rejected, 0)

            output = Path(directory) / "output"

            predictions = pd.read_csv(
                output / "predictions.csv"
            )

            self.assertEqual(len(predictions), 2)
            self.assertTrue(
                (predictions["detector"] == "cnn_gatv2").all()
            )

            info = json.loads(
                (output / "run_info.json").read_text()
            )

            self.assertEqual(info["model_type"], "baseline")

    def test_directional_checkpoint_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            run = create_model_run(directory, "directional")

            rows = [
                model_observation("flow1", "iot23_zeek_v1"),
                model_observation("flow2", "iot23_zeek_v1")
            ]

            count, rejected = score_rows(
                rows,
                run,
                Path(directory) / "output"
            )

            self.assertEqual(count, 2)
            self.assertGreaterEqual(rejected, 0)

            output = Path(directory) / "output"

            predictions = pd.read_csv(
                output / "predictions.csv"
            )

            self.assertEqual(len(predictions), 2)

            self.assertTrue(
                (
                    predictions["detector"] ==
                    "directional_cnn_gatv2"
                ).all()
            )

            info = json.loads(
                (output / "run_info.json").read_text()
            )

            self.assertEqual(info["model_type"], "directional")

    def test_directional_inference_requires_both_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            run = create_model_run(directory, "directional")

            row = model_observation(
                "flow1", "iot23_zeek_v1"
            )

            del row["payload_resp_hex"]

            with self.assertRaisesRegex(
                ValueError, "missing directional payload fields"
            ):
                score_rows(
                    [row],
                    run,
                    Path(directory) / "output"
                )


if __name__ == "__main__":
    unittest.main()
