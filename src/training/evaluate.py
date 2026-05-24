# src/training/evaluate.py
import json
import pathlib
import random
import statistics
import time
from typing import Callable

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from config.settings import (
    COLAB_PRO_HOURLY_RATE,
    H5_MIN_MEAN_SCORE,
    LATENCY_WARMUP_RUNS,
    RANDOM_SEED,
)


# ── Threshold tuning ─────────────────────────────────────────────────────────

def tune_threshold(val_scores: list[float], val_labels: list[int]) -> float:
    """
    Sweeps the observed score range in 100 steps.
    Returns the threshold that maximizes binary F1 (anomaly class, label=1).
    """
    best_t, best_f1 = min(val_scores), 0.0
    for t in np.linspace(min(val_scores), max(val_scores), 100):
        preds = [1 if s >= t else 0 for s in val_scores]
        f1 = f1_score(val_labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def build_validation_set(
    train_sequences: list[tuple[str, str]],
    labels: dict[str, int],
    score_fn: Callable[[str], float],
    n_per_class: int = 10,
    seed: int = RANDOM_SEED,
) -> tuple[list[float], list[int]]:
    """
    Constructs the ~20-example validation set used for threshold tuning.

    train_sequences: list of (sequence_str, key) from the training split
    labels: {key: 0|1} loaded from loghub annotation file
    score_fn: compute_anomaly_score(model, tokenizer, seq) wrapped as a callable
    n_per_class: number of anomalous and normal examples to sample

    Returns (val_scores, val_labels).
    Labels are read here — never during MLM training.
    """
    anomalous = [(seq, key) for seq, key in train_sequences if labels.get(key, 0) == 1]
    normal    = [(seq, key) for seq, key in train_sequences if labels.get(key, 0) == 0]

    rng = random.Random(seed)
    sample_anomalous = rng.sample(anomalous, min(n_per_class, len(anomalous)))
    sample_normal    = rng.sample(normal,    min(n_per_class, len(normal)))
    sample = sample_anomalous + sample_normal

    val_scores = [score_fn(seq) for seq, _ in sample]
    val_labels = [labels[key] for _, key in sample]
    return val_scores, val_labels


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model_on_test_set(
    scores: list[float],
    test_labels: list[int],
    val_scores: list[float],
    val_labels: list[int],
    model_name: str,
    dataset: str,
) -> dict:
    """
    Tunes threshold on val set, evaluates on test set.
    Shared eval function for LogLite, LogBERT, and DeepLog.
    """
    threshold = tune_threshold(val_scores, val_labels)
    preds = [1 if s >= threshold else 0 for s in scores]
    return {
        "model":     model_name,
        "dataset":   dataset,
        "source":    "measured",
        "f1":        float(f1_score(test_labels, preds, pos_label=1, zero_division=0)),
        "precision": float(precision_score(test_labels, preds, pos_label=1, zero_division=0)),
        "recall":    float(recall_score(test_labels, preds, pos_label=1, zero_division=0)),
        "threshold": threshold,
    }


# ── Latency measurement ───────────────────────────────────────────────────────

def measure_latency(
    score_fn: Callable[[str], float],
    sequences: list[str],
    n_batches: int = 100,
) -> dict:
    """
    Measures wall-clock inference latency per 20-line window.
    Runs LATENCY_WARMUP_RUNS discarded calls first to avoid cold CUDA start.

    score_fn: compute_anomaly_score(model, tokenizer, seq) wrapped as a callable
    sequences: list of window strings (non-overlapping, 20 lines each)
    n_batches: number of windows to time

    Returns p50/p95/p99 latency in ms.
    """
    # Warm-up — discard results
    for seq in sequences[:LATENCY_WARMUP_RUNS]:
        score_fn(seq)

    latencies: list[float] = []
    for seq in sequences[LATENCY_WARMUP_RUNS: LATENCY_WARMUP_RUNS + n_batches]:
        t0 = time.perf_counter()
        score_fn(seq)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }


# ── Cost calculation ──────────────────────────────────────────────────────────

def compute_cost_per_1k(inference_time_sec: float, num_logs: int) -> float:
    """
    Calculates detection cost per 1,000 log sequences (H3).
    Uses total wall-clock time for num_logs predictions at Colab Pro+ rate.

    NOTE: call this with p50 total time (not p95) — p95 would fail H3 by design.
    """
    hours = inference_time_sec / 3600
    return (hours * COLAB_PRO_HOURLY_RATE / num_logs) * 1_000


# ── H5 assertion ─────────────────────────────────────────────────────────────

def check_h5_passes(agent_eval_path: str = "results/agent_eval.json") -> bool:
    """
    Reads agent_eval.json and asserts H5:
    mean(clarity + accuracy + actionability) >= H5_MIN_MEAN_SCORE across all examples.

    Scoring rubric (each dimension 1–5):
      1 = Poor   3 = Acceptable   5 = Excellent
    """
    entries = json.loads(pathlib.Path(agent_eval_path).read_text())
    per_entry_means = []
    for e in entries:
        mean_score = statistics.mean([e["clarity"], e["accuracy"], e["actionability"]])
        per_entry_means.append(mean_score)
        print(f"  {e['log_id']}: clarity={e['clarity']} accuracy={e['accuracy']} "
              f"actionability={e['actionability']} → mean={mean_score:.2f}")

    overall_mean = statistics.mean(per_entry_means)
    print(f"\nH5 overall mean: {overall_mean:.2f} (threshold={H5_MIN_MEAN_SCORE})")
    assert overall_mean >= H5_MIN_MEAN_SCORE, (
        f"H5 FAILED: mean={overall_mean:.2f} < {H5_MIN_MEAN_SCORE} "
        f"across {len(entries)} examples"
    )
    print("H5 PASSED")
    return True
