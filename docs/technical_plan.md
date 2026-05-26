# Technical Plan — LogLite
**Version:** 2.3 | **Date:** May 2026 | **Derived from:** preliminary_report.md

---

## 1. Repository Structure

```
loglite/
  src/
    models/
      loglite.py          ← BertForMaskedLM wrapper + compute_anomaly_score()
      tokenizer.py        ← BERT tokenizer wrapper
    agents/
      explanation_agent.py ← Anomaly explanation agent (Path B)
    streaming/
      simulator.py        ← Python queue-based streaming simulator
    training/
      train.py            ← MLM fine-tuning script
      evaluate.py         ← F1, latency, cost evaluation + threshold tuning
    data/
      load_dataset.py     ← Load HDFS/BGL from loghub
      preprocess.py       ← Regex normalization + windowing
  tests/
    unit/
      test_loglite.py     ← Model forward pass, anomaly score
      test_tokenizer.py   ← Tokenizer output shapes
      test_preprocess.py  ← Windowing logic
    integration/
      test_pipeline.py    ← End-to-end: raw log → anomaly score → label
  docs/
    technical_plan.md     ← This file
    prompt_book.md        ← Agent prompt entries (schema below)
  notebooks/
    experiments/
      hdfs_experiment.ipynb
      bgl_experiment.ipynb
  results/
    metrics/              ← JSON per model per dataset (schema in Section 5)
    baselines/
      hdfs_deeplog.json   ← Measured DeepLog latency on T4
      hdfs_logbert.json   ← Measured LogBERT latency on T4
      literature.json     ← Published F1/latency/cost: LogAnomaly, LogLLM, GPT-4
    charts/               ← PNG: F1 vs size, latency bar, cost bar
    streaming/
      latency.json        ← p50/p95/p99 from simulator
    agent_eval.json       ← 5 agent evaluation entries (schema in Section 6)
  scripts/
    compare_baselines.py  ← Builds final comparison table from results/
    run_demo.py           ← End-to-end demo: load → infer → explain
  data/
    demo/
      demo_block.txt      ← 20 pre-selected HDFS lines (committed to repo)
  config/
    settings.py           ← All constants (Section 9)
    .env.example          ← ANTHROPIC_API_KEY template
  requirements.txt
  .gitignore
  README.md
```

---

## 2. Model Architecture

### Transfer Learning Stages

LogLite applies transfer learning in three sequential stages, each requiring zero labels:

```
Stage 1 — General language pre-training (HuggingFace, already done)
  bert-base-uncased trained on Wikipedia + BookCorpus (~3.3B words)
  Learns: syntax, semantics, token co-occurrence patterns
           ↓  transfer
Stage 2 — Log domain adaptation (us, ~2–4h on Colab T4)
  MLM fine-tuning on HDFS + BGL unlabeled logs
  Learns: log-specific vocabulary, normal event sequences, system behavior
           ↓  transfer
Stage 3 — [Optional] Org-specific DAPT (~1–2h on Colab T4)
  MLM fine-tuning on a new organization's unlabeled logs
  Learns: org-specific formats, component names, normal operational patterns
```

### Model Stack

```
Input: raw log sequence (string)
  "081109 203615 148 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906"
         │
         ▼
Preprocessing (regex only):
  - Strip leading timestamp + numeric ID
  - Keep raw message text
  - Output: "INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906"
         │
         ▼
BERT WordPiece Tokenizer (bert-base-uncased):
  - max_length = 128
  - padding = "max_length"
  - truncation = True
  - Output: input_ids (128,), attention_mask (128,)
         │
         ▼
Token Embedding + Positional Encoding
         │
         ▼
Transformer Encoder × 12 layers:
  - num_heads = 12
  - d_model = 768
  - d_ff = 3072
  - dropout = 0.1
         │
         ▼
MLM Head (Linear 768 → vocab_size=30522):
  - Training: predict randomly masked tokens (self-supervised MLM)
  - Inference: measure cross-entropy loss on masked tokens
         │
         ▼
Reconstruction Loss Score:
  - mask 15% of tokens randomly at inference
  - mean CE loss over masked positions = anomaly score
  - high loss → model surprised → ANOMALY
  - low loss  → model confident → NORMAL
         │
         ▼
Output: anomaly_score (float), per_token_losses (list[float])
  ↑ per_token_losses used by explanation agent for trigger identification
  ↑ anomaly_score compared to val-tuned threshold → NORMAL / ANOMALY label
```

**Parameter count:** ~110M
**Base model:** `bert-base-uncased` — 12 layers, 12 heads, d=768
**Training objective:** Masked Language Modeling (MLM) — zero labels required
**Anomaly score:** Mean MLM cross-entropy loss over masked tokens. Higher = more anomalous.

---

## 3. Data Pipeline

### Active Datasets (this project)

| Dataset | Size | Window Type | Window Config | Label Use |
|---|---|---|---|---|
| HDFS v1 | 11.2M lines / 742K sessions | Session (block ID) | All lines per block ID | Eval only (never training) |
| BGL | 4.75M lines | Sliding | size=20, step=10 → ~400K train + ~75K test windows (~475K total) | Eval only |

> **BGL window counts:** The chronological split (first 4M lines → train, remaining ~750K → test) produces approximately **400K training windows** and **75K test windows** (~475K total). Step=1 would produce ~4.75M overlapping windows — too large for a single Colab session. Step=10 reduces this to ~475K while preserving anomaly pattern coverage.

> **Future work:** Thunderbird and Spirit are available on loghub but out of scope for the 40-hour budget.

### Train/Test Splits

Chronological splits following the canonical LogBERT/LogADEmpirical protocol — no future-leaking:

| Dataset | Train set | Test set |
|---|---|---|
| HDFS v1 | First 4,855 sessions (by block ID) | Remaining sessions |
| BGL | First 4,000,000 lines | Remaining ~750K lines |

Labels are attached **only at evaluation** — training uses raw text with no label column.

### Preprocessing

```python
import re
from collections import defaultdict

# Step 1: Load raw log file
# Step 2: Extract block ID (HDFS only) + strip timestamp/node ID

# HDFS line format: "081109 203615 148 INFO dfs.DataNode: ... blk_-1608999687919862906 ..."
BLOCK_ID_PATTERN = re.compile(r"(blk_-?\d+)")
STRIP_PATTERN    = re.compile(r"^\d{6}\s\d{6}\s\d+\s")

def parse_hdfs_line(raw_line: str) -> tuple[str | None, str]:
    """Returns (block_id, cleaned_message). block_id is None if not found."""
    m = BLOCK_ID_PATTERN.search(raw_line)
    block_id = m.group(1) if m else None
    message  = STRIP_PATTERN.sub("", raw_line).strip()
    return block_id, message

# Step 3: Chronological split (BEFORE windowing — no leakage)
#   HDFS: first 4,855 unique block IDs → train; remainder → test
#   BGL:  first 4,000,000 lines → train; remainder → test

# Step 4: Group into sequences
#   HDFS: group all lines sharing the same block ID → one sequence per session
sessions: dict[str, list[str]] = defaultdict(list)
for raw_line in hdfs_train_lines:
    block_id, message = parse_hdfs_line(raw_line)
    if block_id:
        sessions[block_id].append(message)
# Each session becomes one string: " ".join(sessions[block_id])

#   BGL: sliding window of 20 lines, step=10
def bgl_windows(lines: list[str], size: int = 20, step: int = 10):
    for i in range(0, len(lines) - size + 1, step):
        yield " ".join(lines[i: i + size])

# Step 5: Tokenize
# MODEL_REVISION is imported from config.settings — pins the HuggingFace commit hash
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", revision=MODEL_REVISION)
tokens = tokenizer(sequence, max_length=128, padding="max_length",
                   truncation=True, return_tensors="pt")

# Step 6: No label attachment for training — self-supervised
```

---

## 4. Training

### Phase 1 — Self-supervised MLM fine-tuning on HDFS + BGL

```python
import random, numpy as np, torch
from transformers import (BertForMaskedLM, BertTokenizer,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

RANDOM_SEED = 42
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

MODEL_NAME    = "bert-base-uncased"
EPOCHS        = 3
BATCH_SIZE    = 32
LEARNING_RATE = 2e-5
MLM_PROB      = 0.15

# Pin model revision for reproducibility — avoids silent weight changes on HuggingFace Hub.
# Obtain the commit hash with: huggingface-cli download bert-base-uncased --dry-run
# Example pinned revision (update when upgrading intentionally):
MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"  # bert-base-uncased, May 2026

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
model     = BertForMaskedLM.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB
)

args = TrainingArguments(
    output_dir="./results/loglite-mlm",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="best",          # save best checkpoint only (lowest val MLM loss)
    load_best_model_at_end=True,
    report_to="none",              # metrics saved to results/metrics/ as JSON
    seed=RANDOM_SEED,
)
```

**Metrics saved:** train/loss, eval/loss, epoch, learning_rate, train/runtime → `results/metrics/train_log.json`
**Hardware:** Google Colab T4
**Estimated time:** 2–4 hours per dataset
**Checkpoints:** saved to Google Drive every epoch (Colab session safety)
**Labels used during training:** None

### Phase 2 — [Optional] Domain-Adaptive Pre-Training (DAPT)

Same MLM training loop, applied to a new organization's unlabeled logs.
Estimated time: 1–2 hours on T4. No labels required.

```python
# Same Trainer + DataCollatorForLanguageModeling as Phase 1
# report_to="none" — metrics saved to results/metrics/ as JSON
```

> **H4 note:** The < 2 hour estimate is based on the BGL training time at similar data volume. It is an engineering estimate, not a formally measured result — stated as such in the report.

### Inference — Anomaly Scoring

Anomaly scoring uses **deterministic masking**: a fixed random seed is set before every masking call so the same log sequence always produces the same mask and the same score. This is required for reproducible threshold tuning — if the score changed between the tuning call and the evaluation call, the threshold would be meaningless.

```python
import torch
from config.settings import RANDOM_SEED

def mask_tokens(input_ids: torch.Tensor, tokenizer, mlm_prob: float = 0.15):
    """
    Deterministically masks mlm_prob of non-special tokens.
    Caller must set torch.manual_seed(RANDOM_SEED) before calling
    to ensure identical masks for identical inputs across runs.
    Returns (masked_input_ids, labels) where labels=-100 for non-masked positions.
    """
    labels = input_ids.clone()
    prob_matrix = torch.full(labels.shape, mlm_prob)
    special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id,
                      tokenizer.pad_token_id]
    for sp in special_tokens:
        prob_matrix[labels == sp] = 0.0
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100  # loss computed only on masked positions
    input_ids[masked_indices] = tokenizer.mask_token_id
    return input_ids, labels


def compute_anomaly_score(model, tokenizer, log_sequence: str,
                           mlm_prob: float = 0.15) -> float:
    """
    Deterministic anomaly score: mean CE loss over masked token positions.
    Seed is fixed before masking so the same input always yields the same score.
    Higher loss = model more surprised = more anomalous.
    Returns scalar float.
    """
    if not log_sequence or not log_sequence.strip():
        return 0.0   # empty input → treat as normal
    inputs = tokenizer(log_sequence, return_tensors="pt",
                       max_length=128, padding="max_length", truncation=True)
    torch.manual_seed(RANDOM_SEED)   # deterministic mask for reproducibility
    inputs["input_ids"], labels = mask_tokens(
        inputs["input_ids"].clone(), tokenizer, mlm_prob
    )
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    return outputs.loss.item()
```

> **Why deterministic masking?** `torch.bernoulli()` is stochastic. Without fixing the seed before each call, the same log sequence produces a different anomaly score on every run, making threshold tuning non-reproducible. Fixing the seed per call ensures identical behavior during tuning, evaluation, and demo.

---

## 5. Evaluation

### Threshold Tuning

`ANOMALY_THRESHOLD` is tuned per dataset on a small validation set (~20 operator-confirmed examples) to maximize binary F1 for the anomaly class (label=1). These examples are used **only** to find one threshold number — not for model training.

> **No data leakage:** The ~20 validation examples are drawn exclusively from the **training split** (held-out sessions not used in MLM training). They are never drawn from the test set. The test set is touched only once — for the final F1 evaluation reported in results.

> **How the ~20 examples are selected and labeled:** HDFS and BGL both ship with ground-truth labels (HDFS: per-session anomaly flags from the loghub annotation file; BGL: per-line alert tags). These labels exist in the dataset but are deliberately excluded from MLM training. For threshold tuning, ~20 sessions/windows are sampled at random from the training split — approximately 10 anomalous and 10 normal, to give the sweep enough signal on both classes. Their labels are read from the loghub annotation file at this point only. This is consistent with the zero-label training claim: the model never sees labels during training, but a small number of labeled examples are used post-hoc to select one decision threshold. This follows the LogADEmpirical evaluation protocol.

```python
# src/training/evaluate.py
import numpy as np
import random
from sklearn.metrics import f1_score

def tune_threshold(val_scores: list[float], val_labels: list[int]) -> float:
    """
    Sweeps the observed score range in 100 steps.
    Returns the threshold that maximizes binary F1 (anomaly class, label=1).

    val_scores: MLM reconstruction loss for each validation sequence
    val_labels: ground-truth labels (~20 operator-confirmed examples)
    """
    best_t, best_f1 = min(val_scores), 0.0
    for t in np.linspace(min(val_scores), max(val_scores), 100):
        preds = [1 if s >= t else 0 for s in val_scores]
        f1 = f1_score(val_labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def build_validation_set(
    train_sequences: list[str],
    label_file_path: str,
    dataset: str,
    model,
    tokenizer,
    n_per_class: int = 10,
    seed: int = 42,
) -> tuple[list[float], list[int]]:
    """
    Constructs the ~20-example validation set used for threshold tuning.

    Reads ground-truth labels from the loghub annotation file (HDFS: anomaly_label.csv;
    BGL: per-line alert tag in column 0). Labels are read here — never during training.
    Samples n_per_class anomalous and n_per_class normal sequences from train_sequences.
    Computes anomaly scores for each and returns (val_scores, val_labels).

    dataset: "hdfs" or "bgl"
    label_file_path: path to the loghub annotation file for the dataset
    train_sequences: list of (sequence_str, block_id_or_line_idx) tuples from the training split
    """
    import csv, pathlib

    # --- Load ground-truth labels from loghub annotation file ---
    labels: dict[str, int] = {}  # block_id/line_idx → 0 (normal) or 1 (anomaly)
    if dataset == "hdfs":
        # HDFS: anomaly_label.csv format — "BlockId,Label" where Label is "Normal"/"Anomaly"
        with open(label_file_path, newline="") as f:
            for row in csv.DictReader(f):
                labels[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0
    elif dataset == "bgl":
        # BGL: first column is "-" (normal) or alert tag (anomaly), per line
        with open(label_file_path) as f:
            for idx, line in enumerate(f):
                first_field = line.split()[0] if line.strip() else "-"
                labels[str(idx)] = 0 if first_field == "-" else 1
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Expected 'hdfs' or 'bgl'.")

    # --- Split train_sequences by label, sample n_per_class from each class ---
    # train_sequences is list of (seq_str, key) where key is block_id (HDFS) or str(line_idx) (BGL)
    anomalous = [(seq, key) for seq, key in train_sequences if labels.get(key, 0) == 1]
    normal    = [(seq, key) for seq, key in train_sequences if labels.get(key, 0) == 0]

    rng = random.Random(seed)
    sample_anomalous = rng.sample(anomalous, min(n_per_class, len(anomalous)))
    sample_normal    = rng.sample(normal,    min(n_per_class, len(normal)))
    sample = sample_anomalous + sample_normal

    # --- Compute anomaly scores ---
    val_scores = [compute_anomaly_score(model, tokenizer, seq) for seq, _ in sample]
    val_labels = [labels[key] for _, key in sample]

    return val_scores, val_labels
```

**Usage in training/evaluation pipeline:**
```python
# After MLM fine-tuning completes (M4):
val_scores, val_labels = build_validation_set(
    train_sequences=hdfs_train_seq,   # list of (seq_str, block_id) from training split
    label_file_path="data/HDFS/anomaly_label.csv",
    dataset="hdfs",
    model=model,
    tokenizer=tokenizer,
)
threshold = tune_threshold(val_scores, val_labels)
# threshold is saved to results/metrics/loglite_hdfs.json and used for test-set evaluation
```

### Metrics Recorded per Model per Dataset

All results saved as JSON to `results/metrics/` and `results/baselines/`:

```python
result = {
    "model":             "loglite",     # or "deeplog", "logbert", etc.
    "dataset":           "hdfs",        # or "bgl"
    "source":            "measured",    # or "literature"
    "f1":                0.0,           # binary F1, anomaly class (label=1)
    "precision":         0.0,
    "recall":            0.0,
    "threshold":         0.0,           # val-set-tuned threshold
    "latency_p50_ms":    0.0,           # measured on Colab T4
    "latency_p95_ms":    0.0,
    "latency_p99_ms":    0.0,
    "cost_per_1k_logs":  0.0,           # USD — formula below
    "model_params":      110_000_000,
}
```

Literature entries use `"source": "literature"` so `compare_baselines.py` can footnote them.

### Latency Measurement

Latency is measured per **window of 20 log lines** (HDFS session or BGL sliding window) — this is the natural inference unit. The PRD metric "< 100ms per batch" refers to this window size.

```python
import time, numpy as np

def measure_latency(model, tokenizer, sequences: list[str],
                    n_batches: int = 100) -> dict:
    """
    Runs inference on n_batches windows, records wall-clock time per window.
    Uses the same batch_size=1 (one window per call) for all models.
    """
    latencies = []
    for seq in sequences[:n_batches]:
        t0 = time.perf_counter()
        compute_anomaly_score(model, tokenizer, seq)
        latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }
```

Same `measure_latency()` function is used for DeepLog, LogBERT, and LogLite — ensuring a fair p95 comparison (H2).

> **H2b uses BGL 20-line windows only.** HDFS sessions are variable-length (some are 1 line, others 50+), making per-window latency comparisons across models unfair. H2b (LogLite p95 < LogBERT p95) is therefore measured exclusively on BGL non-overlapping 20-line windows — the same fixed-size unit for all three models. H2a (absolute p95 < 100ms) uses the simulator, which also uses 20-line windows.

### Cost Calculation

**H3 covers detection-only cost** — the GPU inference cost of running LogLite on log sequences. Agent explanation cost (Claude API calls) is a separate metric that applies only to flagged anomalies (~2.3% of HDFS logs) and is reported independently.

Cost is calculated as GPU-hour amortization using the **Colab Pro+ T4 rate** as a conservative upper bound. On the free tier the GPU cost is $0; using the Pro rate makes H3 a non-trivial, falsifiable claim rather than a trivially-true "$0 < $0.001" statement.

> **Agent cost (separate from H3):** Estimated at ~$0.003 **per anomaly API call** based on Anthropic's May 2026 pricing for claude-sonnet-4-6 (~$3/MTok input, ~$15/MTok output; a typical call uses ~200 tokens prompt + ~300 tokens output ≈ 500 tokens total ≈ $0.0015–$0.003). Using the conservative upper end ($0.003 per call): at 2.3% anomaly rate → 23 calls per 1K logs → **~$0.069 per 1K logs** when the agent is included. The comparison table reports both: "$0.003/anomaly (API call)" and "$0.069/1K logs (at 2.3% anomaly rate)" as separate rows so the distinction is unambiguous. H3 is about detection model cost only and is not affected by agent cost.

```python
COLAB_PRO_HOURLY_RATE_USD = 0.35   # Colab Pro+ T4 (conservative upper bound)

def compute_cost_per_1k(inference_time_sec: float, num_logs: int) -> float:
    """
    inference_time_sec: total wall-clock time for num_logs predictions
    num_logs: number of log sequences processed
    Returns: cost in USD per 1,000 sequences
    """
    hours = inference_time_sec / 3600
    return (hours * COLAB_PRO_HOURLY_RATE_USD / num_logs) * 1_000
```

### Baseline Comparison Protocol

**LogBERT** produces a continuous reconstruction loss score (not a classifier probability). To compare fairly, the same threshold-tuning protocol is applied to all models:

```python
def evaluate_model_on_test_set(scores: list[float], test_labels: list[int],
                                val_scores: list[float], val_labels: list[int],
                                model_name: str) -> dict:
    """
    Shared eval function for LogLite, LogBERT, and DeepLog.
    Tunes threshold on val set, evaluates on test set.
    """
    threshold = tune_threshold(val_scores, val_labels)
    preds = [1 if s >= threshold else 0 for s in scores]
    return {
        "model":     model_name,
        "source":    "measured",
        "f1":        f1_score(test_labels, preds, pos_label=1),
        "precision": precision_score(test_labels, preds, pos_label=1),
        "recall":    recall_score(test_labels, preds, pos_label=1),
        "threshold": threshold,
    }
```

All models evaluated with:
- Same HDFS/BGL test set (same session IDs and labels)
- Same `f1_score(pos_label=1)` call (binary F1, anomaly class)
- Val-set-tuned threshold per model
- Protocol follows LogADEmpirical (repo #3)

### Comparison Table Builder

```python
# scripts/compare_baselines.py
import json, pathlib, pandas as pd

def build_comparison_table(results_dir: str = "results") -> pd.DataFrame:
    rows = []
    for path in pathlib.Path(results_dir).rglob("*.json"):
        if path.name == "agent_eval.json":
            continue
        rows.append(json.loads(path.read_text()))
    df = pd.DataFrame(rows)[["model", "dataset", "source", "f1",
                              "latency_p95_ms", "cost_per_1k_logs", "model_params"]]
    return df.sort_values("f1", ascending=False)


def assert_beats_deeplog(df: pd.DataFrame) -> None:
    """H1 assertion: LogLite F1 > DeepLog F1 on HDFS and BGL."""
    for dataset in ["hdfs", "bgl"]:
        loglite_f1 = df[(df.model == "loglite") & (df.dataset == dataset)]["f1"].values[0]
        deeplog_f1 = df[(df.model == "deeplog") & (df.dataset == dataset)]["f1"].values[0]
        assert loglite_f1 > deeplog_f1, (
            f"H1 FAILED on {dataset}: LogLite={loglite_f1:.4f} <= DeepLog={deeplog_f1:.4f}"
        )
```

---

## 6. Explanation Agent (Path B)

The agent runs after anomaly detection. It identifies which lines caused the high reconstruction loss and calls the Claude API to produce a plain-English explanation.

### Trigger Identification

Trigger lines are identified by **per-token reconstruction loss** — not by attention weights. Tokens where the model predicted poorly (high CE loss) are the anomaly signal.

```python
# src/agents/explanation_agent.py
import torch, torch.nn.functional as F


def get_anomaly_context(log_id: str, all_lines: list[str],
                        window: int = 10) -> list[str]:
    """Returns ±window lines surrounding the flagged log_id.
    Raises ValueError if log_id is not found in any line."""
    idx = next((i for i, l in enumerate(all_lines) if log_id in l), None)
    if idx is None:
        raise ValueError(f"log_id '{log_id}' not found in provided log lines.")
    return all_lines[max(0, idx - window): idx + window + 1]


def compute_per_token_losses(model, tokenizer,
                              log_sequence: str) -> tuple[list[float], dict]:
    """
    Computes per-token CE loss using the SAME deterministic masking as
    compute_anomaly_score() — so trigger identification is consistent with
    the anomaly score that flagged this sequence.

    Steps:
      1. Tokenize with return_offsets_mapping=True (needed for line attribution)
      2. Apply identical deterministic mask (same seed → same masked positions)
      3. Forward pass → logits [1, 128, vocab]
      4. CE loss per position; non-masked positions have label=-100 (excluded)

    Returns:
      per_token_losses: list[float] of length 128 — loss at each position
                        (0.0 at non-masked positions, CE loss at masked positions)
      enc: tokenizer output dict including offset_mapping
    """
    enc = tokenizer(log_sequence, return_tensors="pt", max_length=128,
                    padding="max_length", truncation=True,
                    return_offsets_mapping=True)
    original_ids = enc["input_ids"].clone()
    torch.manual_seed(RANDOM_SEED)   # same seed as compute_anomaly_score
    masked_ids, labels = mask_tokens(original_ids.clone(), tokenizer)
    model_inputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
    model_inputs["input_ids"] = masked_ids
    with torch.no_grad():
        logits = model(**model_inputs).logits  # [1, 128, vocab]
    # Per-token CE loss; -100 labels are ignored by cross_entropy (returns 0)
    per_token = F.cross_entropy(
        logits[0], labels[0], reduction="none", ignore_index=-100
    ).tolist()   # [128] — non-zero only at masked positions
    return per_token, enc


def extract_line_reconstruction_loss(per_token_losses: list[float],
                                      tokenizer_encodings: dict,
                                      log_lines: list[str]) -> list[float]:
    """
    Aggregates per-token CE loss to per-line scores using offset_mapping.
    Only tokens at masked positions have non-zero loss (same mask as scorer).
    Lines with highest summed loss are the anomaly triggers.

    offset_mapping gives (start, end) byte positions in the joined string.
    Special/padding tokens have start==end and are skipped.
    Any token whose offset falls beyond the last line is assigned to the
    last line (fallback prevents StopIteration on truncated sequences).
    """
    offset_mapping = tokenizer_encodings["offset_mapping"][0]
    line_scores = [0.0] * len(log_lines)
    # cumulative char positions of line boundaries in the joined string
    line_ends = [sum(len(l) + 1 for l in log_lines[:i+1])
                 for i in range(len(log_lines))]
    last_line = len(log_lines) - 1
    for token_idx, (start, end) in enumerate(offset_mapping.tolist()):
        if start == end:
            continue   # padding or special token
        if token_idx >= len(per_token_losses):
            break
        loss = per_token_losses[token_idx]
        if loss == 0.0:
            continue   # non-masked position — no signal
        # find which line this token belongs to; default to last line if beyond boundary
        line_idx = next(
            (i for i, le in enumerate(line_ends) if start < le), last_line
        )
        line_scores[line_idx] += loss
    return line_scores


def highlight_trigger_events(token_losses: list[float],
                              log_lines: list[str]) -> list[str]:
    """Returns the top-3 lines by aggregated per-token reconstruction loss."""
    ranked = sorted(zip(token_losses, log_lines), reverse=True)
    return [line for _, line in ranked[:3]]


def lookup_known_patterns(event_sequence: list[str]) -> str | None:
    """
    Checks for known HDFS/BGL failure signatures by substring matching.
    Returns pattern name if matched, else None.
    """
    KNOWN_PATTERNS = {
        "hdfs_corrupt_block":        ["Received block", "is corrupt"],
        "hdfs_write_pipeline_error": ["Exception in receiveBlock"],
        "hdfs_datanode_lost":        ["has not received a heartbeat"],
        "bgl_ras_kernel_storm":      ["RAS KERNEL INFO", "RAS KERNEL INFO"],
        "bgl_abnormal_exit":         ["died with signal"],
    }
    joined = "\n".join(event_sequence)
    for name, keywords in KNOWN_PATTERNS.items():
        remaining, matched = joined, True
        for kw in keywords:
            idx = remaining.find(kw)
            if idx == -1:
                matched = False; break
            remaining = remaining[idx + len(kw):]
        if matched:
            return name
    return None


def generate_explanation(context: list[str], triggers: list[str],
                         pattern: str | None = None,
                         client: "anthropic.Anthropic | None" = None) -> str:
    """
    Calls Claude API to produce plain-English explanation + suggested action.

    client: an anthropic.Anthropic instance. Must be provided when called
            from the demo script or evaluation code (not at module scope here —
            caller is responsible for instantiation via _get_client()).
    """
    if client is None:
        raise ValueError(
            "client must be provided. Instantiate via _get_client() in the caller."
        )
    pattern_note = f"\nKnown pattern matched: {pattern}" if pattern else ""
    prompt = f"""You are a log analysis expert.

Flagged trigger lines (highest reconstruction loss — most anomalous tokens):
{triggers}

Surrounding context:
{context}{pattern_note}

Explain in 3-5 sentences what likely went wrong.
End with: "Suggested action: ..."
"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

### Agent Evaluation (M5 Done-Condition)

Five hand-picked anomaly examples from HDFS/BGL are evaluated and scored on three dimensions. All three dimensions must average ≥ 3/5 for H5 to pass.

```python
# M5 evaluation — run manually after explanation_agent.py is implemented

EVAL_EXAMPLES = [
    {"log_id": "blk_-1608999687919862906", "dataset": "hdfs",
     "expected_theme": "corrupt block / pipeline failure"},
    {"log_id": "blk_7503483334202473044",  "dataset": "hdfs",
     "expected_theme": "datanode lost / heartbeat timeout"},
    # + 3 BGL examples added during M5 (selected from known anomaly windows)
]

# Scoring rubric — each dimension rated 1–5:
#   clarity       — understandable to a non-expert operator?
#   accuracy      — correctly identifies the failure type?
#   actionability — suggested action makes sense?
#
# H5 passes when: mean(clarity + accuracy + actionability) >= 3.0 across all 5 examples

agent_eval_entry = {
    "log_id":          "...",
    "dataset":         "hdfs",
    "explanation":     "...",   # raw agent output
    "clarity":         0,       # 1–5
    "accuracy":        0,       # 1–5
    "actionability":   0,       # 1–5
    "pattern_matched": None,    # output of lookup_known_patterns()
}
# All 5 entries saved to results/agent_eval.json
```

```python
# src/training/evaluate.py (continued)
import json, pathlib, statistics

def check_h5_passes(agent_eval_path: str = "results/agent_eval.json") -> bool:
    """
    Reads agent_eval.json and asserts H5: mean(clarity + accuracy + actionability) >= 3.0
    across all evaluated examples.

    Scoring rubric (each dimension 1–5):
      1 = Poor / wrong / not actionable
      3 = Acceptable — understandable, roughly correct, plausible action
      5 = Excellent — clear, accurate, specific and useful action

    Returns True if H5 passes, raises AssertionError with details if it fails.
    """
    entries = json.loads(pathlib.Path(agent_eval_path).read_text())
    per_entry_means = []
    for e in entries:
        mean_score = statistics.mean([e["clarity"], e["accuracy"], e["actionability"]])
        per_entry_means.append(mean_score)
        print(f"  {e['log_id']}: clarity={e['clarity']} accuracy={e['accuracy']} "
              f"actionability={e['actionability']} → mean={mean_score:.2f}")

    overall_mean = statistics.mean(per_entry_means)
    print(f"\nH5 overall mean: {overall_mean:.2f} (threshold=3.0)")
    assert overall_mean >= 3.0, (
        f"H5 FAILED: mean={overall_mean:.2f} < 3.0 across {len(entries)} examples"
    )
    print("H5 PASSED")
    return True
```

> **H5 limitations:** Five examples is a small sample. Scoring is performed by the developer (no independent rater). Examples are hand-picked from known anomaly sessions — likely easier to explain than random anomalies. These limitations are acknowledged explicitly in the final report. H5 is treated as a proof-of-concept rather than a rigorous explainability benchmark.

> **Known patterns source:** The substrings in `KNOWN_PATTERNS` are derived from the HDFS and BGL loghub dataset documentation and Hadoop/BGL system event descriptions. During M5, the matched rate on the 5 evaluation examples will be measured and reported.

### Prompt Book Entry

File: `docs/prompt_book.md`

```
Prompt ID:      AGENT-EXPLAIN-001
Goal:           Explain why a log sequence was flagged as anomalous
Context:        Flagged log lines from HDFS/BGL; per-token reconstruction loss
                scores computed; top-3 trigger lines identified
Prompt:         "You are a log analysis expert. Given these flagged log lines
                 and their reconstruction loss scores, explain in plain English
                 what likely went wrong and suggest a next action for the operator."
Model:          claude-sonnet-4-6 (production); claude-haiku-4-5 during development
Expected Output: 3–5 sentence explanation + "Suggested action: ..."
Actual Output:  [filled during M5]
Evaluation:     Clarity, accuracy, actionability — each 1–5; mean ≥ 3 to pass H5
Decision:       [filled during M5]
```

---

## 7. Streaming Simulator

Implements a Python queue-based simulation of real-time log streaming — proving the latency claim without requiring Kafka or Flink infrastructure.

```python
# src/streaming/simulator.py
import queue, threading, time, numpy as np

WINDOW_SIZE       = 20     # lines per inference window (matches BGL sliding window)
LATENCY_TARGET_MS = 100    # H2 assertion threshold


def simulate_stream(log_file: str, model, tokenizer,
                    rate: int = 1000, n_lines: int = 10_000) -> dict:
    """
    Reads log_file line by line, feeds into queue at `rate` lines/sec.
    Consumer batches into windows of WINDOW_SIZE, runs inference, records latency.

    Returns p50/p95/p99 latency in ms.
    Asserts p95 < LATENCY_TARGET_MS (H2 evidence).
    """
    q: queue.Queue = queue.Queue()
    latencies: list[float] = []

    def producer():
        with open(log_file) as f:
            for i, line in enumerate(f):
                if i >= n_lines:
                    break
                q.put(line.strip())
                time.sleep(1 / rate)
        q.put(None)

    def consumer():
        window: list[str] = []
        while True:
            line = q.get()
            if line is None:
                break
            window.append(line)
            if len(window) == WINDOW_SIZE:
                t0 = time.perf_counter()
                compute_anomaly_score(model, tokenizer, " ".join(window))
                latencies.append((time.perf_counter() - t0) * 1000)
                window = []

    t_prod = threading.Thread(target=producer, daemon=True)
    t_cons = threading.Thread(target=consumer, daemon=True)
    t_prod.start(); t_cons.start()
    t_prod.join(); t_cons.join()

    results = {
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }
    assert results["latency_p95_ms"] < LATENCY_TARGET_MS, (
        f"H2 FAILED: p95={results['latency_p95_ms']:.1f}ms >= {LATENCY_TARGET_MS}ms"
    )
    return results
```

Results saved to `results/streaming/latency.json`. The assertion passing is the concrete evidence for H2a (absolute latency target).

> **Scope of the latency measurement:** The simulator measures pure inference latency (tokenization + forward pass per 20-line window). It does not include agent API call time (~500ms–2s per anomaly) — agent latency is not part of H2. This matches what is claimed: H2 is about detection latency, not end-to-end response time when anomalies are explained.

> **Window overlap:** The simulator uses non-overlapping windows (`window = []` after each inference) for the latency measurement (H2a). For the relative latency comparison (H2b — LogLite vs LogBERT), `measure_latency()` is used instead, which is also called with non-overlapping windows so both models are measured identically. The step=10 overlap is used for F1 evaluation only, not for latency measurement.

> **Baseline fairness (latency):** DeepLog and LogBERT latency is measured with `measure_latency()` using the same non-overlapping 20-line windows and the same `n_batches=100` parameter as LogLite — ensuring H2b is a fair comparison.

> **Baseline fairness (F1):** DeepLog and LogBERT F1 is evaluated with step=10 overlapping BGL windows — the same windowing as LogLite's F1 evaluation. Both latency and F1 comparisons use identical inputs per metric.

---

## 8. Technology Stack

| Layer | Tool | Version |
|---|---|---|
| Language | Python | 3.11 |
| ML Framework | PyTorch | 2.x |
| Transformers | HuggingFace Transformers | 4.x |
| Experiment tracking | JSON logs to results/metrics/ | — |
| Testing | pytest | 8.x |
| Linting | ruff | 0.4+ |
| Claude API | Anthropic SDK | 0.30+ |
| Environment | python-dotenv | 1.x |

---

## 9. Key Constants (config/settings.py)

```python
# Reproducibility
RANDOM_SEED = 42

# Model
MODEL_NAME     = "bert-base-uncased"                          # 12 layers, ~110M params
MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"  # pinned HuggingFace commit
MAX_LENGTH     = 128
NUM_LAYERS  = 12
NUM_HEADS   = 12
D_MODEL     = 768

# Training
EPOCHS        = 3
BATCH_SIZE    = 32
LEARNING_RATE = 2e-5
MLM_PROB      = 0.15   # fraction of tokens masked (training + inference scoring)

# Evaluation
ANOMALY_THRESHOLD        = None    # tuned per dataset — see Section 5
LATENCY_TARGET_MS        = 100     # p95 target (H2)
COST_TARGET_PER_1K       = 0.001   # USD (H3)
COLAB_PRO_HOURLY_RATE    = 0.35    # upper-bound cost assumption

# Datasets
HDFS_PATH = "data/HDFS/"
BGL_PATH  = "data/BGL/"

# Streaming simulator
WINDOW_SIZE = 20    # lines per inference window (BGL sliding window and simulator); HDFS uses variable-length session windows grouped by block ID
STREAM_RATE = 1000  # lines/sec

# Agent
AGENT_MODEL_PROD         = "claude-sonnet-4-6"   # used in final demo and results
AGENT_MODEL_DEV          = "claude-haiku-4-5"    # use during M5 development to control API costs
# Override: set AGENT_MODEL = AGENT_MODEL_DEV at top of explanation_agent.py while iterating
AGENT_MAX_TOKENS         = 300
AGENT_CONTEXT_WINDOW     = 10     # ±lines around flagged log_id
AGENT_TOP_K_TRIGGERS     = 3      # top-k lines by reconstruction loss
```

---

## 10. Demo Script

The live demo chains the full pipeline in one runnable script. No internet required when `USE_CACHED_RESPONSE = True`.

```python
# scripts/run_demo.py
import os
from pathlib import Path
import torch, torch.nn.functional as F
import anthropic
from src.models.loglite import load_model, compute_anomaly_score
from src.agents.explanation_agent import (
    get_anomaly_context, compute_per_token_losses,
    extract_line_reconstruction_loss, highlight_trigger_events,
    lookup_known_patterns, generate_explanation,
)
from config.settings import RANDOM_SEED, AGENT_TOP_K_TRIGGERS

# Validate API key at startup (only needed for live mode)
def _get_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key, "
            "or set USE_CACHED_RESPONSE=True to run offline."
        )
    return anthropic.Anthropic(api_key=api_key)

# Path relative to this script — works regardless of working directory
REPO_ROOT      = Path(__file__).resolve().parent.parent
DEMO_FILE      = REPO_ROOT / "data" / "demo" / "demo_block.txt"
DEMO_LOG_LINES = DEMO_FILE.read_text().splitlines()

USE_CACHED_RESPONSE = True   # set False for live Claude API call

CACHED_EXPLANATION = """Node 148 failed to receive block blk_-1608999687919862906
due to a pipeline exception, resulting in the block being marked corrupt. This is
consistent with a transient network interruption or disk write failure on the DataNode.
Suggested action: Re-replicate the block from a healthy DataNode and check disk health
on Node 148."""

DEMO_THRESHOLD = 2.5   # loaded from results/metrics/ in production


def run_demo():
    model, tokenizer = load_model()

    # 1. Compute anomaly score
    sequence = " ".join(DEMO_LOG_LINES)
    score = compute_anomaly_score(model, tokenizer, sequence)
    label = 1 if score >= DEMO_THRESHOLD else 0
    print(f"[{'ANOMALY' if label else 'NORMAL'}] "
          f"Reconstruction Loss: {score:.4f} (threshold={DEMO_THRESHOLD:.4f})")

    if label == 1:
        # 2. Per-token losses for trigger identification
        per_token_losses, enc = compute_per_token_losses(model, tokenizer, sequence)
        line_losses = extract_line_reconstruction_loss(
            per_token_losses, enc, DEMO_LOG_LINES
        )
        triggers = highlight_trigger_events(line_losses, DEMO_LOG_LINES)

        # 3. Pattern match
        pattern = lookup_known_patterns(DEMO_LOG_LINES)
        print(f"\nPattern matched: {pattern or 'none'}")
        print(f"Trigger lines:\n" + "\n".join(f"  {l}" for l in triggers))

        # 4. Explanation
        if USE_CACHED_RESPONSE:
            explanation = CACHED_EXPLANATION
        else:
            client = _get_client()
            explanation = generate_explanation(DEMO_LOG_LINES, triggers, pattern, client)
        print(f"\n{explanation}")


if __name__ == "__main__":
    run_demo()
```

`data/demo/demo_block.txt` — 20 pre-selected HDFS lines containing the `blk_-1608999687919862906` corruption sequence — is committed to the repo so the demo works on a clean clone with no data download.

---

## 11. Hypotheses and Measurement Map

| Hypothesis | Claim | How Measured | Where Saved |
|---|---|---|---|
| H1 (Accuracy) | Binary F1 (label=1) ≥ 0.90 on HDFS and BGL, zero labels | `evaluate_model_on_test_set()` — threshold tuned on held-out train examples, applied to test split | `results/metrics/loglite_hdfs.json` |
| H2a (Latency absolute) | LogLite p95 < 100ms per 20-line window on Colab T4 | `simulate_stream()` assertion — inference latency only, not agent | `results/streaming/latency.json` |
| H2b (Latency relative) | LogLite p95 < LogBERT p95 on Colab T4, same window | `measure_latency()` — identical code for LogLite and LogBERT | `results/metrics/`, `results/baselines/` |
| H3 (Efficiency) | Detection cost < $0.001/1K logs using Colab Pro+ rate (agent cost separate: ~$0.069/1K) | `compute_cost_per_1k()` — detection only | `results/metrics/loglite_hdfs.json` |
| H4 (Adaptability) | DAPT < 2h on T4 | Engineering estimate based on BGL training time — not a formal measurement | Noted in report as estimate |
| H5 (Explainability) | Mean(clarity + accuracy + actionability) ≥ 3/5 across 5 hand-picked anomaly examples | Manual 1–5 scoring by developer — acknowledged as limited, proof-of-concept only | `results/agent_eval.json` |

---

---

## 12. Limitations and Future Work

### Current Limitations

**Input representation:** LogLite is trained and evaluated on pre-parsed event sequences (E1–E29) extracted by the loghub team using the Drain log parser. In production, this parsing step must be performed before inference, adding ~0.4ms latency per 20-line window.

**Fixed template set:** The Drain parser used to generate event IDs was run offline on historical logs. If new log patterns appear in production (e.g. a new software version introduces new message formats), they will not match any known template and will be assigned an `[UNK]` token. The model's behavior on unknown patterns is undefined until the next retrain cycle.

**Small training set:** Only 4,855 HDFS sessions are used for fine-tuning. This is sufficient for event sequences (only 29 unique tokens) but would be inadequate for raw log text, which requires much larger datasets to generalize.

**Evaluation sample:** Due to Colab GPU time limits, the test set evaluation uses a stratified sample of 2,000 sessions (1,000 anomalous + 1,000 normal) rather than the full ~570,000 test sessions. Results may vary slightly on the full test set.

---

### Future Work

#### LLM-Assisted Template Drift Handling

The most significant production gap is handling **template drift** — new log patterns that appear after the Drain parser was trained. The proposed solution uses an LLM as a real-time classifier for unknown log lines:

```
Raw log line arrives at inference time
         ↓
Drain parser (locked template set)
         ↓ known pattern          ↓ unknown pattern
    Event ID (E5)              LLM classifier (Claude Haiku)
         ↓                              ↓
    BERT scoring               MATCH existing template → Event ID → BERT
                               NEW_PATTERN → tentative ID, flag for review
                               ANOMALY → alert immediately, skip BERT
```

**Production architecture:**
```
Raw line → Drain (known?)
              ↓ yes → Event ID → BERT scoring → anomaly score
              ↓ no  → Claude Haiku → MATCH / NEW_PATTERN / ANOMALY
                           ↓ ANOMALY → Claude Sonnet → plain-English explanation
```

**Why this works:**
- LLM understands log semantics — knows "Connection timeout after 45s" is similar to "Connection timeout after 30s" even if they have different numeric values
- Only called for unknown patterns (~1–5% of lines in a stable system) — cost is low
- Uses cheap model (Haiku ~$0.001/1K tokens) for classification; expensive model (Sonnet) only for confirmed anomalies requiring explanation
- No retraining needed when new patterns appear — the LLM handles them immediately

**Cost estimate:** At 1% unknown pattern rate and 1,000 lines/sec → 10 LLM calls/sec → ~$0.036/hr additional cost. Negligible compared to the GPU inference cost.

**Implementation scope:** Not implemented in this project due to time constraints. Estimated 2–3 days additional engineering work. The explanation agent (M5) already uses the same Claude API infrastructure, so the plumbing is in place.

#### User-Aware Anomaly Detection

For systems with millions of mixed-user logs (no session grouping), a per-user baseline approach would be more effective than a global threshold.

**The problem with global models:**
When logs from many users are mixed together, a single BERT model learns the average normal behavior across all users. A power user who routinely triggers E26 will always look anomalous to a model trained on average behavior — even though E26 is normal *for that user*.

**Proposed architecture:**

```
Raw log line: "2026-05-27 14:32 user_123 E5 E11 E9 E26"
         ↓
Prepend user token: "[USER_123] E5 E11 E9 E26"
         ↓
BERT MLM fine-tuning — learns per-user normal patterns
         ↓
Anomaly score = deviation from THIS user's learned baseline
         ↓
Alert only if score > user-specific threshold
```

**Implementation:**
```python
# Add all user IDs as special tokens before fine-tuning
user_tokens = [f"[USER_{uid}]" for uid in all_user_ids]
tokenizer.add_special_tokens({"additional_special_tokens": user_tokens})
model.resize_token_embeddings(len(tokenizer))

# Each training sequence prefixed with user token
sequence = f"[USER_{user_id}] " + " ".join(event_ids)
```

**Benefits:**
- No separate model per user — one BERT handles all users
- Per-user threshold tuning using that user's own history
- Naturally handles power users, admins, service accounts
- Detects anomalies relative to the user's own baseline — not the population mean

**Suitable datasets:** CERT Insider Threat Dataset (explicit user IDs + actions), LANL Unified Host and Network Dataset (user login events across enterprise network)

**Limitation:** Requires enough per-user history to learn a baseline (~100+ sessions per user). New users with sparse history fall back to the global threshold until enough data is collected.

#### Model Efficiency

For lower-latency or lower-cost deployments:
- **DistilBERT** (66M params, 2× faster, ~97% of BERT accuracy) — drop-in replacement
- **INT8 quantization** (`torch.quantization.quantize_dynamic`) — 4× smaller, 2× faster, minimal accuracy loss
- **4-layer BERT** trained from scratch on event sequences — sufficient for the 29-token vocabulary, much faster than 12-layer base

---

*Technical Plan v2.3 — LogLite Project, May 2026*
