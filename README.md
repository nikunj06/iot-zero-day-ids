# Payload-aware IoT open-set IDS

Educational research prototype: 1D-CNN over payload bytes, optional GATv2 over minibatch flow similarities, one-vs-rest sigmoid heads, auxiliary-unknown training, prototype-distance rejection, DBSCAN review grouping. This is inspired by the literature review; it is **not** a reproduction of NERO or DOC++. A rejected flow is unknown to the model, **not proof of a zero-day exploit**.

## Start in VS Code

Open the **zero_day_iot** folder itself, then open a PowerShell terminal there. Use Python 3.10–3.12.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ids.demo --output data/demo.jsonl
.\.venv\Scripts\python.exe -m ids.train --input data/demo.jsonl --out runs/demo --epochs 12 --cpu
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

If the executable is not found, run `pwd` and `Test-Path .\.venv\Scripts\python.exe`: you must be inside the project and finish creating the venv first. On macOS/Linux, use `python3 -m venv .venv`, then `./.venv/bin/python` in subsequent commands. The updated archive includes a trained **synthetic** demo in `runs/demo`, so you can launch the dashboard immediately after installing dependencies. Select **Saved evaluation**.

## Three distinct results

| View | Input | Detector | What changes |
| --- | --- | --- | --- |
| Saved evaluation | Generated `data/demo.jsonl` | CNN/GATv2 plus prototype threshold | Only when you retrain; held-out metrics do not update with local traffic |
| Local live lab | Loopback TCP messages in `data/live.jsonl` | Transparent application rules | New completed sessions and rule alerts when the lab runs |
| Model comparison | Same generated train/test split | Four trained variants | Changes when `ids.compare` runs |

The live collector's `packets` field means **application messages**, not captured IP packets. Its features are incompatible with the trained synthetic model. The watcher routes them to separate rules. It will not silently feed them to CNN/GATv2.

To see local activity, use three PowerShell terminals in the project:

```powershell
# Terminal 1: collector and bounded loopback scenarios
.\.venv\Scripts\python.exe -m ids.live_lab --output data/live.jsonl --duration 45 --scenario mixed
# Terminal 2: watcher, keep running; start before or during Terminal 1
.\.venv\Scripts\python.exe -m ids.watch --input data/live.jsonl --out runs/live
# Terminal 3: dashboard; select Local live lab (refreshes every 3 seconds)
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

You may score a finished file once with `python -m ids.live_rules --input data/live.jsonl --out runs/live`. The lab only connects to 127.0.0.1 and runs at most 300 seconds. `normal`, `burst`, and `unusual_command` select bounded scenarios. `lab_pattern` is simulator context, never a model feature. Local alerts are based on 20+ messages per session or an unexpected CONFIG message; they are **not model predictions or real attack validation**.

## Real PCAP evaluation

Bring **your own labeled, homogeneous PCAP captures**. No real attack PCAP is bundled. A manifest is a JSON list:

```json
[
  {"path":"C:/captures/benign-1.pcap","label":"benign","family":"benign"},
  {"path":"C:/captures/syn-1.pcap","label":"DoS-SYN","family":"DoS"}
]
```

Provide several independent captures per label and enough known/auxiliary/held-out labels. Each manifest entry is assigned a capture-level label; mixed-label captures or unverified capture labels **must not** be used. Do not join labels to packets by CSV row order.

```powershell
.\.venv\Scripts\python.exe -m ids.pcap --manifest data/captures.json --output data/flows.jsonl
.\.venv\Scripts\python.exe -m ids.inspect_data --input data/flows.jsonl
.\.venv\Scripts\python.exe -m ids.train --input data/flows.jsonl --out runs/pcap --known "benign,DoS-SYN,Recon-Scan" --aux "Spoof-ARP" --heldout "DoS-HTTP,Mirai-Bot" --cpu
```

Replace those example labels with your actual dataset taxonomy. Each known label needs at least three independent capture IDs; each auxiliary unknown needs at least two. Training prevents a capture from crossing its train/validation/test partitions. Type B is a held-out subtype in a known broad family; Type A holds out a whole broad family. The held-out samples are never used to fit model weights or thresholds. After training, select **Custom run** and enter `runs/pcap`.

The importer produces `feature_source=pcap_packet_v1`; the synthetic generator uses `synthetic_flow_v1`; the local simulator uses `app_message_v1`. Mixing sources in training or inference is rejected. To score **unlabeled PCAP-extracted flows**, use a model trained on compatible PCAP-extracted features:

```powershell
.\.venv\Scripts\python.exe -m ids.infer --input data/unlabeled_pcap_flows.jsonl --run runs/pcap --out runs/fresh
```

Remove label/family from fresh input if appropriate. Keep `flow_id`, `capture_id`, `feature_source`, numeric features, and `payload_hex`. The PCAP importer is a limited TCP/UDP flow aggregator, not Wi-Fi rogue AP detection. Encrypted payload bytes cannot reveal cleartext application exploits. Neither capture-level labels nor network data provenance can be verified automatically.

## Comparison and interpretation

```powershell
.\.venv\Scripts\python.exe -m ids.compare --input data/demo.jsonl --out runs/comparison --epochs 12
```

This trains stats-only, payload-only, combined without graph, and full GATv2 on the **same seed and split**. Select **Model comparison**. On the bundled synthetic data the simpler stats-only model can outperform the full graph. Report that finding honestly: synthetic structure and minibatch graph edges do not establish a graph benefit. For real data, pass explicit `--known`, `--aux`, and `--heldout` labels to `ids.compare` too.

The hybrid decision accepts a flow only if its highest class score meets a validation threshold **and** its embedding is close enough to that class prototype. This improves overall synthetic unknown recall in the bundled run, while Type B recall remains low and benign false alerts increase. `metrics.json` gives the actual numbers; `predictions.csv` shows `rejection_reason` and distance. Type A/B classification is reported only among successfully rejected held-out flows, with its count. `novelty_score=1-confidence` is a display aid, not a complete hybrid decision score.

## Review and adaptation

`review_queue.csv` holds rejected flows; `clusters.csv` groups some for human triage. For new, independently sourced, compatible records, export reviews in the **Custom run** view and adapt:

```powershell
.\.venv\Scripts\python.exe -m ids.adapt --base data/demo.jsonl --new data/fresh_demo.jsonl --reviews data/review_demo.csv --output data/version2.jsonl
```

The included fresh/review examples are synthetic. After adaptation, retrain to a **new output directory**, update `--known` and use a different untouched held-out set. Never tune on the original held-out evaluation then claim independent test performance. There is no automated production hot swap.

## Project map

- `ids/demo.py`, `ids/pcap.py`, `ids/live_lab.py`: synthetic, PCAP, and local telemetry inputs.
- `ids/inspect_data.py`, `ids/data.py`: source checks, feature matrices, grouped splits.
- `ids/model.py`, `ids/train.py`, `ids/infer.py`: CNN/GATv2, hybrid calibration, compatible scoring.
- `ids/live_rules.py`, `ids/watch.py`: local rules and live file watcher.
- `ids/compare.py`: reproducible ablations.
- `ids/adapt.py`, `dashboard.py`: reviewed retraining input and results.

The graph uses nearest flows **within each minibatch**, so outputs may change with batch composition. Stronger experiments require independent sessions/devices, provenance checks, multiple seeds, meaningful baselines, and clear false-alarm tradeoffs. This prototype should not be deployed as a production IDS.

## References

- Cevallos Moreno et al., [NERO: NEural algorithmic reasoning for zeRO-day attack detection in the IoT](https://doi.org/10.1016/j.cose.2024.103898), *Computers & Security* 142 (2024).
- Soltani et al., [An adaptable deep learning-based intrusion detection system to zero-day attacks](https://doi.org/10.1016/j.jisa.2023.103516), *Journal of Information Security and Applications* 76 (2023).
