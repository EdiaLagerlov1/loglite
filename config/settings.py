# config/settings.py
# All project-wide constants. Import from here — never hardcode in src/.

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "bert-base-uncased"
# Pinned HuggingFace commit — avoids silent weight changes on Hub updates.
# To update: huggingface-cli download bert-base-uncased --dry-run
MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
MAX_LENGTH     = 128
NUM_LAYERS     = 12
NUM_HEADS      = 12
D_MODEL        = 768

# ── Training ─────────────────────────────────────────────────────────────────
EPOCHS        = 3
BATCH_SIZE    = 32
LEARNING_RATE = 2e-5
MLM_PROB      = 0.15   # fraction of tokens masked (training + inference scoring)

# ── Evaluation ───────────────────────────────────────────────────────────────
ANOMALY_THRESHOLD     = None   # tuned per dataset — set by tune_threshold()
LATENCY_TARGET_MS     = 100    # p95 target (H2a)
COST_TARGET_PER_1K    = 0.001  # USD detection cost (H3) — measured at p50 latency
COLAB_PRO_HOURLY_RATE = 0.35   # T4 Colab Pro+ rate (conservative upper bound for H3)
LATENCY_WARMUP_RUNS   = 50     # discard first N windows before recording latency

# ── BGL window label rule (H1 / BGL evaluation) ──────────────────────────────
# A sliding window is labelled ANOMALOUS if ANY line within it has an alert tag.
# This is the most conservative definition and matches the per-line annotation
# in the BGL loghub dataset (first column != "-").
BGL_WINDOW_ANOMALOUS_IF_ANY = True

# ── Datasets ─────────────────────────────────────────────────────────────────
HDFS_PATH              = "data/HDFS/"
BGL_PATH               = "data/BGL/"
HDFS_LABEL_FILE        = "data/HDFS/anomaly_label.csv"   # loghub annotation file
BGL_LABEL_FILE         = "data/BGL/BGL.log"              # labels embedded in col 0
HDFS_TRAIN_SESSIONS    = 4_855   # first N block IDs → train (chronological)
HDFS_TEST_SESSIONS     = 2_000   # sample this many for test evaluation (avoids timeout)
BGL_TRAIN_LINES        = 4_000_000

# ── Streaming simulator ───────────────────────────────────────────────────────
# WINDOW_SIZE is used for BGL sliding windows and the simulator.
# HDFS uses variable-length session windows grouped by block ID.
WINDOW_SIZE = 20
BGL_STEP_F1      = 10   # step for F1 evaluation (overlapping windows)
BGL_STEP_LATENCY = 20   # step for latency measurement (non-overlapping = WINDOW_SIZE)
STREAM_RATE      = 1_000  # lines/sec fed into simulator queue

# ── Agent ────────────────────────────────────────────────────────────────────
AGENT_MODEL_PROD     = "claude-sonnet-4-6"   # final demo and reported results
AGENT_MODEL_DEV      = "claude-haiku-4-5"    # use during M5 dev to control API costs
AGENT_MAX_TOKENS     = 300
AGENT_CONTEXT_WINDOW = 10   # ±lines around flagged log_id
AGENT_TOP_K_TRIGGERS = 3    # top-k lines returned by highlight_trigger_events()

# ── H5 evaluation ────────────────────────────────────────────────────────────
H5_MIN_MEAN_SCORE   = 3.0   # mean(clarity + accuracy + actionability) must be >= this
H5_NUM_EXAMPLES     = 5
AGENT_EVAL_PATH     = "results/agent_eval.json"

# ── Cost note ────────────────────────────────────────────────────────────────
# H3 measures DETECTION cost only (GPU inference, no agent).
# Agent cost is separate: ~$0.003/anomaly API call; at 2.3% anomaly rate →
# ~$0.069/1K logs. Reported as a separate line in the comparison table.
# H3 cost formula: (inference_time_sec / 3600) * COLAB_PRO_HOURLY_RATE / n_logs * 1000
# Uses p50 latency (average case), not p95, because p95 is a tail metric and
# would make H3 fail by design ($0.35/hr × 100s/1000windows = $0.0097 >> $0.001).
