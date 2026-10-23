# Payload-aware graph open-set IDS for IoT traffic

An educational research prototype inspired by the **combination proposed in the project literature review**: a byte-level 1D-CNN, a GATv2 flow graph, one-vs-rest sigmoid heads with auxiliary-unknown rejection (DOC++ inspired), clustering of rejected flows, and reviewed retraining. It is **not** an implementation or reproduction of NERO or DOC++, and its synthetic demo does not establish real-world zero-day performance.

## Quick start

Use Python 3.10–3.12. PyTorch Geometric installation can vary with the installed PyTorch build; see its official installation guide if `pip install` cannot resolve compatible wheels.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
# CPU-only PyTorch (for laptops without a CUDA setup):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m ids.demo --output data/demo.jsonl
python -m ids.train --input data/demo.jsonl --out runs/demo --epochs 12
streamlit run dashboard.py
```

The demo writes generated toy flow records. Its attack labels and patterns are **simulated**, not packet captures or evidence of attack detection in deployment. Each run saves `model.pt`, `scaler.joblib`, `config.json`, `predictions.csv`, `metrics.json`, `clusters.csv`, and `review_queue.csv`. The dashboard opens a run directory and shows these results. The demo's Type B recall may be low; this is a measured limitation, not a hidden success.

## Real data contract

One JSON object per line, for example:

```json
{"flow_id":"captureA-1","capture_id":"captureA","timestamp":0.0,"src":"10.0.0.2","dst":"10.0.0.3","protocol":"TCP","dst_port":1883,"duration":0.25,"packets":4,"bytes":220,"mean_size":55,"iat_mean":0.08,"tcp_syn":1,"payload_hex":"74657374","label":"benign","family":"benign"}
```

`label` is the fine attack type; `family` is the broad taxonomy. Keep one clearly documented taxonomy for all experiments. `payload_hex` holds the first available application payload bytes, up to `--payload-len` after decoding; empty values are permitted. The payload-availability mask is fed to the model. Metadata-derived features use only the numeric fields in `ids/data.py`, not addresses, timestamps, labels, or capture IDs. Packet contents may be encrypted or absent; a byte branch cannot inspect encrypted application text.

For CICIoT2023, the publisher provides separate PCAPs and CSV features. Do **not** join PCAP flows to CSV labels by row order. Use a verified packet/flow-label mapping; alternatively, use homogeneous, documented capture segments with capture-level labels and state that limitation. `ids.pcap` imports individual labeled homogeneous PCAPs for a first experiment:

```bash
python -m ids.pcap --manifest data/captures.json --output data/flows.jsonl
```

Manifest format: `[{"path":"/path/to/capture.pcap","label":"AttackSubtype","family":"DoS"}, ...]`. Only capture files containing that single label are suitable for this route. The importer is not a Wi-Fi AP detector.

## Experimental protocol

`--known` specifies fine labels included in training; `--aux` specifies labels used as auxiliary unknowns during rejection training and calibration; `--heldout` contains final unknown labels **never read for threshold fitting or weight updates**. Benign flows are split across partitions. Capture groups are kept together within each label; ideally choose captures recorded on separate sessions/devices for evaluation. Run separate experiments:

- **Type B:** Hold out a subtype while other subtypes in its family are known.
- **Type A:** Hold out every subtype in an entire family.

The script verifies the family condition and label disjointness. It does not guarantee lack of capture artifacts or distribution leakage. The real dataset must be audited for these effects. The final report must include known macro F1, unknown recall, benign false-positive rate, and counts; the generated metrics provide those. Compare ablations with `--branch both`, `--branch stats`, and `--branch payload`, and `--graph` versus `--no-graph`.

A second sigmoid head learns broad known attack families. For a rejected flow, a sufficiently strong broad-family match suggests **Type B**; otherwise the output is **Type A**. Its threshold is also fitted on known and auxiliary-unknown validation groups. Report hierarchical accuracy **among correctly rejected unknown test flows**, along with its sample count. A wrong initial rejection means no usable Type A/B decision.

The threshold is selected using known validation data and **auxiliary unknown validation only**. The held-out evaluation labels do not influence selection. The known class with maximum sigmoid probability is used when above the threshold; otherwise the prediction is `unknown`. An `unknown` output is an alert, not a proof that a novel exploit occurred.

## Review and adaptation

Unknown test embeddings are grouped using DBSCAN, and a `review_queue.csv` is saved. For genuine adaptation, obtain **fresh captures outside the original evaluation dataset** and convert them to JSONL. Label and family fields are optional for inference:

```bash
python -m ids.infer --input data/fresh.jsonl --run runs/demo --out runs/fresh
streamlit run dashboard.py  # enter runs/fresh, review rejected flows, export review_decisions.csv
python -m ids.adapt --base data/demo.jsonl --new data/fresh.jsonl --reviews review_decisions.csv --output data/version2.jsonl
```

Train a newly versioned model on `version2.jsonl` using an updated `--known` label list and **a separate final held-out family**. The command refuses overlapping capture IDs, but cannot establish whether capture environments leak into one another. At least three independently labeled capture groups per known class are required by the group split. Never use the original held-out evaluation records to tune a model and then report performance on them again. The prototype does not automatically hot-swap a production model.

## Project files

- `ids/demo.py`: deterministic simulated flow generator
- `ids/pcap.py`: labeled PCAP import and flow aggregation
- `ids/infer.py`, `ids/adapt.py`: score fresh traffic and integrate human-reviewed labels
- `ids/data.py`: feature extraction, grouped split, graph construction
- `ids/model.py`: 1D-CNN + GATv2 + sigmoid classification heads
- `ids/train.py`: training, calibration, evaluation, clustering, saved outputs
- `dashboard.py`: inspection and human labeling interface

## Scope and limitations

This prototype makes graph edges by flow-feature similarity **within a minibatch** (no class labels or ground-truth edges). Thus predictions may depend on batch composition. Inference uses deterministic fixed-size batches, and the experiment must report that limitation. The generated demo is a smoke test, not a benchmark. The model should be assessed with original capture provenance, held-out sessions, repeat seeds, ablations, and meaningful baselines before any real-world claim.

## Source papers and dataset

- Cevallos Moreno et al., [NERO: Neural algorithmic reasoning for zero-day attack detection in the IoT](https://doi.org/10.1016/j.cose.2024.103898), *Computers & Security* 142, 103898 (2024).
- Soltani et al., [An Adaptable Deep Learning-Based Intrusion Detection System to Zero-Day Attacks](https://arxiv.org/abs/2108.09199), also published in *Journal of Information Security and Applications* 76, 103516 (2023).
- Canadian Institute for Cybersecurity, [CICIoT2023 dataset description and downloads](https://www.unb.ca/cic/datasets/iotdataset-2023.html).
