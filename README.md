# Payload-Aware IoT Open-Set Intrusion Detection System

An educational research prototype for IoT intrusion detection using 1D-CNN payload analysis, optional GATv2 graph learning, one-vs-rest classification, auxiliary-unknown training, prototype-distance rejection, and DBSCAN review grouping.

This project is inspired by the literature review; it is **not** a reproduction of NERO or DOC++. A flow rejected by the model is classified as unknown, **not proof of a zero-day exploit**.

## 1. Getting Started

### Requirements

- Python 3.10–3.12
- VS Code
- Windows, Linux, or macOS

### Installation

Open the project folder in VS Code and use a terminal inside that folder.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Generate and train the synthetic demo:

```powershell
.\.venv\Scripts\python.exe -m ids.demo --output data/demo.jsonl
.\.venv\Scripts\python.exe -m ids.train --input data/demo.jsonl --out runs/demo --epochs 12 --cpu
```

Launch the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

The repository includes a trained synthetic demo in `runs/demo`. After installing dependencies, you can launch the dashboard immediately and select **Saved evaluation**.

On macOS/Linux, create the environment with `python3 -m venv .venv` and use `./.venv/bin/python` for subsequent commands.

## 2. Project Workspaces

| Workspace | Input | Function |
|---|---|---|
| Saved evaluation | Generated evaluation data | Displays saved model predictions and metrics |
| Local live lab | Loopback TCP messages | Demonstrates application-level monitoring rules |
| Model comparison | Synthetic train/test split | Compares four model configurations |
| Custom run | Trained evaluation output | Loads a user-selected experiment |
| Dataset Overview | Dataset quality report | Displays dataset composition and import statistics |

### Local Live Lab

The live lab uses loopback TCP application messages. Its `packets` field refers to application messages, not captured IP packets.

The simulator's features are incompatible with the trained synthetic model. The watcher processes them separately using transparent application rules rather than silently feeding them into CNN/GATv2.

Run the following commands in three separate PowerShell terminals.

**Terminal 1 – Generate local traffic**

```powershell
.\.venv\Scripts\python.exe -m ids.live_lab --output data/live.jsonl --duration 45 --scenario mixed
```

**Terminal 2 – Start the watcher**

```powershell
.\.venv\Scripts\python.exe -m ids.watch --input data/live.jsonl --out runs/live
```

**Terminal 3 – Launch the dashboard**

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Select **Local live lab** in the dashboard. It refreshes every three seconds.

You can also score a completed file once:

```powershell
python -m ids.live_rules --input data/live.jsonl --out runs/live
```

The lab only connects to `127.0.0.1` and runs for at most 300 seconds. The `normal`, `burst`, and `unusual_command` options select bounded scenarios. `lab_pattern` is simulator context and is never used as a model feature.

Local alerts are based on 20 or more messages per session or an unexpected CONFIG message. They are **not model predictions or real attack validation**.

## 3. Real PCAP Evaluation

The PCAP importer supports labeled, homogeneous TCP/UDP captures. No real attack PCAP files are bundled with the repository.

A manifest is a JSON list:

```json
[
  {"path":"C:/captures/benign-1.pcap","label":"benign","family":"benign"},
  {"path":"C:/captures/syn-1.pcap","label":"DoS-SYN","family":"DoS"}
]
```

Use independent captures for each label. Each manifest entry is assigned a capture-level label, so mixed-label captures and unverified capture labels must not be used. Labels must not be associated with packets by CSV row order.

Generate flow records:

```powershell
.\.venv\Scripts\python.exe -m ids.pcap --manifest data/captures.json --output data/flows.jsonl
.\.venv\Scripts\python.exe -m ids.inspect_data --input data/flows.jsonl
```

Train a model using your dataset's taxonomy:

```powershell
.\.venv\Scripts\python.exe -m ids.train --input data/flows.jsonl --out runs/pcap --known "benign,DoS-SYN,Recon-Scan" --aux "Spoof-ARP" --heldout "DoS-HTTP,Mirai-Bot" --cpu
```

Replace the example labels with labels from your actual dataset. Each known label needs at least three independent capture IDs; each auxiliary unknown needs at least two.

The training pipeline supports capture-level partitioning to prevent a capture from appearing in multiple partitions.

Type B represents a held-out subtype in a known broad family. Type A represents a held-out broad family. Held-out samples are not used to fit model weights or thresholds.

After training, select **Custom run** in the dashboard and enter the corresponding output directory, such as `runs/pcap`.

The importer produces `feature_source=pcap_packet_v1`. The synthetic generator uses `synthetic_flow_v1`, while the local simulator uses `app_message_v1`. Incompatible sources are rejected during training or inference.

### Inference on Unlabeled Flows

To score unlabeled PCAP-extracted flows, use a model trained on compatible PCAP features:

```powershell
.\.venv\Scripts\python.exe -m ids.infer --input data/unlabeled_pcap_flows.jsonl --run runs/pcap --out runs/fresh
```

The input should contain `flow_id`, `capture_id`, `feature_source`, numeric features, and `payload_hex`. Remove label and family fields when scoring genuinely unlabeled data.

The PCAP importer is a limited TCP/UDP flow aggregator, not a Wi-Fi rogue access point detector. Encrypted payload bytes do not reveal cleartext application exploits. Capture-level labels and network-data provenance cannot be verified automatically.

## 4. IoT-23 Dataset Evaluation

The project supports flow-level evaluation using the IoT-23 dataset.

The importer processes labeled Zeek connection logs and can optionally associate PCAP packets with flows using connection tuples and timestamps. This enables extraction of original-direction and response-direction payload features.

### Selected Scenarios

| Capture ID | Scenario | Role |
|---|---|---|
| `mirai_34_1` | Mirai malware | Training |
| `hakai_8_1` | Hakai malware | Training |
| `philips_hue` | Philips Hue benign traffic | Validation |
| `torii_20_1` | Torii malware | Testing |

The detailed IoT-23 labels are mapped into the project's classes:

- `benign`
- `C&C`
- `DDoS`
- `PortScan`

### Dataset Preparation

Download the required scenarios from the official [Stratosphere Laboratory IoT-23 dataset](https://www.stratosphereips.org/datasets-iot23).

Place the extracted Zeek logs and optional PCAP files under `datasets/iot23/`.

The dataset configuration includes:

- `manifest.json` – capture locations and identifiers.
- `taxonomy.json` – detailed-label mapping.
- `splits_capture.json` – capture-level training, validation, and test assignments.

Raw PCAP files, Zeek logs, and generated flow records are not included in the repository. They must be obtained and prepared locally.

Generate the flow-level dataset:

```powershell
python -m ids.iot23 --manifest datasets/iot23/manifest.json --taxonomy datasets/iot23/taxonomy.json --output datasets/iot23/flows.jsonl
```

### Dataset Quality Report

Generate a dataset quality and coverage report:

```powershell
python -m ids.dataset_report --input datasets/iot23/flows.jsonl --audit datasets/iot23/flows.jsonl.audit.json --out runs/iot23_dataset_quality
```

This produces JSON and Markdown reports covering:

- Flow and capture counts.
- Label and family distributions.
- Payload availability and payload bytes.
- Missing and invalid values.
- Duplicate flow identifiers.
- PCAP importer audit statistics.

The dashboard's **Dataset Overview** workspace can display the generated report.

### Capture-Independent Evaluation

The supplied capture split is:

- **Training:** Mirai and Hakai.
- **Validation:** Philips Hue.
- **Testing:** Torii.

Run the stats-only experiment:

```powershell
python -m ids.train `
  --input datasets/iot23/flows.jsonl `
  --out runs/iot23_multi_capture `
  --known benign,DDoS `
  --aux PortScan `
  --heldout "C&C" `
  --splits datasets/iot23/splits_capture.json `
  --branch stats `
  --no-graph `
  --epochs 12 `
  --batch-size 64 `
  --decision hybrid
```

### Evaluation Results

The capture-independent test on the held-out Torii capture produced the following results:

| Metric | Result |
|---|---:|
| Held-out C&C recall | **87.50% (14/16)** |
| Unknown precision | **93.33% (14/15)** |
| Benign false-alert rate | **0.031% (1/3,193)** |
| Test flows | **3,209** |

The model identified 14 of the 16 held-out C&C flows as unknown. One benign flow was incorrectly flagged.

**Evaluation limitations:**

- The held-out test contains only 16 C&C flows, so recall is based on a small attack sample.
- DDoS traffic is not represented in the test partition.
- The validation capture contains benign traffic only. Therefore, thresholds were not calibrated against known attack-family or auxiliary unknown examples in validation.
- Results demonstrate performance on this particular held-out capture, not general zero-day detection across IoT networks.

Capture-independent results should be reported separately from random flow-level evaluation.

### Directional Payload Features

When matching PCAP files are available, the IoT-23 importer can extract original-direction and response-direction payloads.

Payload availability varies significantly across captures. Stats-only and payload-based experiments should therefore be compared rather than assuming payload features always improve detection.

## 5. Model Comparison and Interpretation

Run the model comparison experiment:

```powershell
.\.venv\Scripts\python.exe -m ids.compare --input data/demo.jsonl --out runs/comparison --epochs 12
```

This compares:

1. Stats-only model.
2. Payload-only model.
3. Combined model without graph.
4. Full GATv2 model.

The variants use the same seed and data split. Select **Model comparison** in the dashboard to inspect the results.

On the bundled synthetic data, the simpler stats-only model can outperform the full graph model. Synthetic structure and minibatch graph edges do not establish a graph benefit. Results should be interpreted according to the dataset and evaluation setup.

For real datasets, specify appropriate `--known`, `--aux`, and `--heldout` labels and use capture-independent partitions.

## 6. Important Limitations

- This is an educational prototype, not a production intrusion detection system.
- Unknown predictions indicate model rejection, not confirmed zero-day attacks.
- Dataset imbalance and limited payload availability can affect detection performance.
- Results from synthetic traffic are not evidence of real-world attack detection.
- The current PCAP pipeline focuses on TCP/UDP flows and does not implement Wi-Fi rogue access point discovery.
- Device risk scores are heuristic indicators based on observed flow alerts and source IP addresses, not verified physical-device identities.
- Model performance should be evaluated on independent captures and interpreted with the size and composition of the test set in mind.

## 7. Testing

Run the test suite from the project root:

```powershell
python -m pytest -v
```

The tests cover dataset handling, IoT-23 parsing, payload extraction, model inference, and source compatibility.

## 8. Project Structure

```text
iot-zero-day-ids/
│
├── dashboard.py
├── requirements.txt
├── README.md
│
├── ids/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── infer.py
│   ├── iot23.py
│   ├── pcap.py
│   ├── dataset_report.py
│   └── device_profile.py
│
├── data/
│   └── demo.jsonl
│
├── tests/
│
├── datasets/
│   └── iot23/
│
└── runs/
```

The `datasets/` and `runs/` directories may contain locally generated data and experiment outputs that are not distributed with the source repository.