# LogLite — Lightweight Real-Time Log Anomaly Detection

Self-supervised BERT-base anomaly detector for software logs. Zero labels required.

**Student:** Edia Lagerlov | **Course:** AI Expert | **Lecturer:** Dr. Yoram Segal

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Run Demo (offline, no API key needed)

```bash
python scripts/run_demo.py
```

Expected output:
```
[ANOMALY] Reconstruction Loss: X.XXXX (threshold=2.5000)
Pattern matched: hdfs_corrupt_block
Trigger lines: ...
Explanation: ...
```

## Results

*To be filled after M4/M7 — F1, latency, cost numbers go here.*

| Model | F1 (HDFS) | p95 Latency | Cost/1K logs |
|---|---|---|---|
| DeepLog | — | measured | ~$0 |
| LogBERT | — | measured | ~$0 |
| **LogLite (ours)** | **—** | **measured** | **—** |

## Project Structure

```
src/          Model, agent, data pipeline, training, streaming
tests/        Unit + integration tests
results/      Metrics, charts, streaming latency
docs/         Technical plan, prompt book
scripts/      run_demo.py, compare_baselines.py
data/demo/    20 pre-selected HDFS lines (committed)
config/       settings.py, .env.example
```

## Hypotheses

| | Claim | Status |
|---|---|---|
| H1 | F1 ≥ 0.90 on HDFS + BGL (zero labels) | pending M4 |
| H2a | p95 < 100ms per 20-line window | pending M4/M6 |
| H2b | LogLite p95 < LogBERT p95 (BGL, same code) | pending M4 |
| H3 | Detection cost < $0.001/1K logs (p50, Colab Pro+) | pending M4 |
| H4 | DAPT < 2h on T4 | engineering estimate |
| H5 | Agent mean score ≥ 3/5 (clarity/accuracy/actionability) | pending M5 |
