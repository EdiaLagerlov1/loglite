# LogLite — TODO
**Student:** Edia Lagerlov | **Total budget:** 40h

---


## M1 — Environment & Data Setup (4h) ✓ DONE

- [x] Create directory structure (`src/`, `tests/`, `docs/`, `results/`, `data/`, `config/`, `scripts/`)
- [x] Create GitHub repo and push — https://github.com/EdiaLagerlov1/loglite
- [x] Set up Google Colab notebook with GPU (T4)
- [x] Install dependencies: `torch`, `transformers`, `datasets`, `pytest`, `ruff`, `scikit-learn`, `anthropic`, `python-dotenv`
- [x] Download HDFS v1 + BGL from loghub; verify both load without errors
- [x] Create `requirements.txt`, `.env.example`, `README.md` stub
- [x] Create `config/settings.py` with all constants (RANDOM_SEED=42, MODEL_NAME, MODEL_REVISION, WINDOW_SIZE, BGL label rule, H3 p50 note)

> Local check: `python scripts/check_setup.py` → **PASSED**

---

## M2 — Baseline Measurement (5h → budget risk: may need 10–15h)

> **Risk:** deep-loglizer and logbert repos are 2–4 years old. Allow extra time for dependency conflicts. If either repo fails to run after 2h of debugging, fall back to literature-only for that baseline.

- [ ] Set up deep-loglizer (DeepLog baseline) in Colab — resolve dependency conflicts
- [ ] Set up logbert repo in Colab — resolve dependency conflicts
- [x] Implement BERT WordPiece tokenizer pipeline: raw log → token IDs, max_length=128 (`src/data/preprocess.py`)
- [x] Implement HDFS session windowing (group by block ID) (`src/data/preprocess.py`)
- [x] Implement BGL sliding window (size=20, step=10) (`src/data/preprocess.py`)
- [x] **Tier 1 — Measure on Colab T4:**
  - [x] Run 50 warm-up windows before measurement starts
  - [x] DeepLog — fell back to literature (deep-loglizer unavailable); saved to `results/baselines/hdfs_deeplog.json`
  - [x] LogBERT p95 = **13.0ms** on T4 — saved to `results/baselines/hdfs_logbert.json`
- [x] **Tier 2 — Cite from literature:**
  - [x] DeepLog F1, LogAnomaly F1/latency, LogLLM F1/latency/cost, GPT-4 cost → `results/baselines/literature.json`
- [x] Verify `compare_baselines.py` produces table with Tier 1 and Tier 2 clearly labelled

**Done when:** latency measured for DeepLog + LogBERT on T4; literature.json populated. ✓ DONE

---

## M3 — LogLite Model Implementation (10h)

- [ ] Implement `src/models/loglite.py`
  - [ ] `BertForMaskedLM` wrapper loading `bert-base-uncased` at pinned `MODEL_REVISION`
  - [ ] `mask_tokens()` with deterministic seed: `torch.manual_seed(RANDOM_SEED)` before every call
  - [ ] `compute_anomaly_score()` — mean CE loss over masked positions; returns scalar float
  - [ ] `load_model()` helper for demo script
- [ ] Implement `src/training/evaluate.py`
  - [ ] `tune_threshold()` — sweeps score range in 100 steps, maximizes binary F1 (label=1)
  - [ ] `build_validation_set()` — reads loghub annotation file, samples ~10 anomalous + ~10 normal from training split, computes scores
  - [ ] `measure_latency()` — 100 non-overlapping windows, wall-clock per window; includes 50 warm-up calls before measurement
  - [ ] `compute_cost_per_1k()` — uses p50 latency (not p95) × Colab Pro+ $0.35/hr
  - [ ] `evaluate_model_on_test_set()` — shared eval for LogLite, LogBERT, DeepLog
  - [ ] `check_h5_passes()` — reads agent_eval.json, asserts mean ≥ 3.0
- [ ] Implement `src/data/preprocess.py`
  - [ ] `parse_hdfs_line()` — extract block ID + strip timestamp
  - [ ] `bgl_windows()` — sliding window size=20, step=10 (F1 eval); step=20 (latency measurement)
  - [ ] BGL window label rule: window is anomalous if any line has an alert tag
- [ ] Implement `src/data/load_dataset.py` — load HDFS/BGL, apply chronological split
- [ ] Write unit tests `tests/unit/test_loglite.py`:
  - [ ] Forward pass returns scalar loss
  - [ ] Anomaly score higher on corrupted sequence than normal
  - [ ] Parameter count ~110M
  - [ ] Same input + same seed → identical score (deterministic masking)
- [ ] Write `tests/unit/test_preprocess.py` — windowing logic, HDFS block ID extraction
- [ ] Verify model runs on dummy input on CPU (Mac)
- [ ] **Push to GitHub** — first commit: full scaffold + passing unit tests

**Done when:** `pytest tests/unit/ -v` passes all tests including determinism check, code is on GitHub.

---

## M4 — Training & Evaluation (8h active; plan for 12h if single GPU)

> **Execute sequentially if two GPU sessions unavailable. HDFS first, BGL second.**

### HDFS Training
- [ ] Fine-tune `bert-base-uncased` on HDFS training split (3–5 epochs, Colab T4)
  - [ ] Set all seeds: `random`, `numpy`, `torch`, `torch.cuda`
  - [ ] Use `DataCollatorForLanguageModeling` (mlm_prob=0.15)
  - [ ] Save checkpoints to Google Drive — **best-only** (not every epoch) to avoid Drive write overhead
  - [ ] Metrics saved automatically to `results/metrics/train_log.json` via `report_to="none"`
- [ ] Run 50 warm-up inference calls on HDFS before latency measurement
- [ ] Call `build_validation_set()` on HDFS training split → `(val_scores, val_labels)`
- [ ] Call `tune_threshold()` → save threshold to `results/metrics/loglite_hdfs.json`
- [ ] Evaluate LogLite on HDFS test set: binary F1 (label=1), Precision, Recall
- [ ] Measure latency with `measure_latency()` — non-overlapping windows, p50 used for cost
- [ ] Calculate `compute_cost_per_1k()` using p50 latency → must be < $0.001
- [ ] Save all metrics to `results/metrics/loglite_hdfs.json`

### BGL Training
- [ ] Fine-tune on BGL training split (first 4M lines, unlabeled)
  - [ ] Apply same label rule for windows: any anomalous line → anomalous window
  - [ ] Best-checkpoint-only save to Drive
- [ ] Call `build_validation_set()` on BGL training split
- [ ] Tune threshold, evaluate F1 on BGL test set
- [ ] Save to `results/metrics/loglite_bgl.json`

### Cross-model latency comparison (H2b — BGL only)
- [ ] Run `measure_latency()` for LogBERT on BGL 20-line windows (same non-overlapping code)
- [ ] Run `measure_latency()` for DeepLog on BGL 20-line windows
- [ ] Compare LogLite p95 vs LogBERT p95 — note if difference is within noise margin

### Optional — DAPT timing trial
- [ ] If time permits: run one DAPT pass on a small unlabeled sample, record wall-clock time
- [ ] If no time: note in report that H4 is an engineering estimate based on BGL training time

**Done when:** F1 ≥ 0.90 on HDFS, cost < $0.001/1K (at p50 latency), BGL F1 collected, all metrics in `results/metrics/`.

---

## M5 — Explanation Agent (5h)

- [ ] Implement `src/agents/explanation_agent.py`:
  - [ ] `get_anomaly_context(log_id, all_lines, window=10)` — raises ValueError if log_id not found
  - [ ] `compute_per_token_losses(model, tokenizer, log_sequence)` — same deterministic seed as scorer
  - [ ] `extract_line_reconstruction_loss(per_token_losses, encodings, log_lines)` — offset_mapping attribution
  - [ ] `highlight_trigger_events(token_losses, log_lines)` — top-3 lines by summed loss
  - [ ] `lookup_known_patterns(event_sequence)` — substring match against KNOWN_PATTERNS
  - [ ] `generate_explanation(context, triggers, pattern=None, client=None)` — raises ValueError if client is None; use `AGENT_MODEL_DEV` during development
- [ ] Test: feed 5 flagged HDFS log lines → verify plain-English explanation + "Suggested action: ..."
- [ ] **Score the 5 evaluation examples** (clarity, accuracy, actionability — each 1–5):
  - [ ] Fill `results/agent_eval.json` with all 5 entries
  - [ ] Run `check_h5_passes()` → assert mean ≥ 3.0
- [ ] Create `docs/prompt_book.md` and add entry `AGENT-EXPLAIN-001`

**Done when:** agent produces explanation on test input; `check_h5_passes()` assertion passes; `prompt_book.md` written.

---

## M6 — Streaming Simulator (2h)

- [ ] Implement `src/streaming/simulator.py`
  - [ ] Producer: reads log file line by line, feeds into `queue.Queue` at 1000 lines/sec
  - [ ] Consumer: batches into 20-line non-overlapping windows, calls `compute_anomaly_score()`, records wall-clock latency
  - [ ] Run 50 warm-up windows before recording latency
  - [ ] Assert p95 < 100ms (H2a)
- [ ] Process 10K lines, save p50/p95/p99 to `results/streaming/latency.json`

**Done when:** simulator processes 10K lines, assertion passes, latency file saved.

---

## M7 — Results, Charts & Report (4h)

- [ ] Select and commit `data/demo/demo_block.txt` — 20 HDFS lines from a known anomaly block (~30 min; requires reviewing HDFS annotation file to pick a clear corruption sequence)
- [ ] Run `compare_baselines.py` → generate final comparison table
  - [ ] Verify `assert_beats_deeplog()` uses only `source=="measured"` rows (not literature) for assertion
  - [ ] Confirm table shows both `$0.003/anomaly (API call)` and `$0.069/1K logs (2.3% anomaly rate)` as separate rows
- [ ] Create 3 charts with matplotlib, save to `results/charts/`:
  - [ ] F1 vs model size (scatter)
  - [ ] Latency comparison (bar — p95, all Tier 1 models)
  - [ ] Cost per 1K logs (bar — detection cost only; agent cost as separate annotated bar)
- [ ] Update `preliminary_report.md` with real numbers from `results/metrics/`
- [ ] Record 2–3 min demo video (screen capture of `run_demo.py`)

**Done when:** charts saved, report has real F1/latency/cost numbers, demo video recorded.

---

## M8 — Polish & Submission (2h)

- [ ] Run `ruff check src/` — fix all warnings
- [ ] Run `pytest tests/ -v --cov=src` — coverage ≥ 80%
- [ ] Implement `scripts/run_demo.py` (chains inference → agent → output)
  - [ ] `USE_CACHED_RESPONSE = True` by default (no API call needed)
  - [ ] `_get_client()` raises clear error if `ANTHROPIC_API_KEY` not set in live mode
- [ ] Verify demo runs offline on a clean clone: `python scripts/run_demo.py` prints anomaly label + explanation
- [ ] Final `README.md`: setup instructions, how to run demo, results summary table
- [ ] Tag GitHub release `v1.0.0`
- [ ] Submit

**Done when:** clean clone → `python scripts/run_demo.py` prints anomaly label + explanation with no API call.

---

## Open Decisions (must resolve before M4)

| Decision | Options | Recommended |
|---|---|---|
| H3 cost metric | p50 vs p95 latency for cost formula | Use p50 — p95 fails the $0.001 target by design |
| BGL window label rule | any line / majority / last line | Any line anomalous → window anomalous |
| H2b strategy | compare vs LogBERT / add margin / compare vs DeepLog | Add ≥10ms margin criterion, or compare LogLite vs DeepLog |
| H4 status | hypothesis / design assumption | Reclassify as design assumption in PRD |
| GPU strategy | two concurrent Colab sessions / sequential | Plan sequential; treat second session as a bonus |
| Baseline fallback | if deep-loglizer/logbert won't run | After 2h debugging: cite from literature, mark as Tier 2 |

---

*Generated from gantt.md + technical_plan.md v2.3 + R&D manager review, May 2026*
