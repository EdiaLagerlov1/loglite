"""
scripts/measure_baselines.py

M2: Measure DeepLog and LogBERT latency on Colab T4.
Also writes the Tier 2 literature.json file.

Run on Colab after both baseline repos are installed.
Usage: python scripts/measure_baselines.py
"""
import json
import pathlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.load_dataset import load_hdfs
from src.data.preprocess import bgl_windows
from src.training.evaluate import measure_latency
from config.settings import HDFS_PATH, HDFS_LABEL_FILE, BGL_PATH, WINDOW_SIZE


def save_result(result: dict, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(result, indent=2))
    print(f"Saved: {path}")


def write_literature_json() -> None:
    """
    Tier 2 — published numbers cited from literature.
    Sources:
      DeepLog F1:     Du et al. 2017 (original paper)
      LogAnomaly F1:  Meng et al. 2019
      LogBERT F1:     Guo et al. 2021 (arXiv:2103.04475)
      LogLLM F1/lat:  Li et al. 2025 (arXiv:2411.08561)
      GPT-4 cost:     Patel 2026 (arXiv:2604.12218)
    """
    literature = [
        {
            "model": "deeplog", "dataset": "hdfs", "source": "literature",
            "f1": 0.933, "precision": None, "recall": None,
            "latency_p95_ms": None, "cost_per_1k_logs": 0.0,
            "model_params": 1_000_000,
            "citation": "Du et al. 2017",
        },
        {
            "model": "loganomaly", "dataset": "hdfs", "source": "literature",
            "f1": 0.958, "precision": None, "recall": None,
            "latency_p95_ms": 10.0, "cost_per_1k_logs": 0.0,
            "model_params": 1_000_000,
            "citation": "Meng et al. 2019",
        },
        {
            "model": "logbert", "dataset": "hdfs", "source": "literature",
            "f1": 0.82, "precision": None, "recall": None,
            "latency_p95_ms": None, "cost_per_1k_logs": 0.0,
            "model_params": 110_000_000,
            "citation": "Guo et al. 2021 arXiv:2103.04475",
        },
        {
            "model": "logllm", "dataset": "hdfs", "source": "literature",
            "f1": 0.99, "precision": None, "recall": None,
            "latency_p95_ms": 2000.0, "cost_per_1k_logs": 0.02,
            "model_params": 8_000_000_000,
            "citation": "Li et al. 2025 arXiv:2411.08561",
        },
        {
            "model": "gpt4", "dataset": "hdfs", "source": "literature",
            "f1": 0.94, "precision": None, "recall": None,
            "latency_p95_ms": 3000.0, "cost_per_1k_logs": 0.03,
            "model_params": 1_000_000_000_000,
            "citation": "Patel 2026 arXiv:2604.12218",
        },
    ]
    save_result(literature, "results/baselines/literature.json")


def measure_deeplog_latency(hdfs_test_sequences: list[str]) -> None:
    """
    Measure DeepLog inference latency on HDFS test set.
    Requires deep-loglizer installed: pip install deep-loglizer
    Falls back to literature value if import fails.
    """
    try:
        # deep-loglizer import — adjust based on installed package structure
        from loglizer.models import DeepLog  # noqa: F401
        print("DeepLog: loaded from deep-loglizer")
        # TODO: instantiate DeepLog model and wrap as score_fn
        # score_fn = lambda seq: deeplog_model.predict(seq)
        # result = measure_latency(score_fn, hdfs_test_sequences)
        # result.update({"model": "deeplog", "dataset": "hdfs", "source": "measured"})
        # save_result(result, "results/baselines/hdfs_deeplog.json")
        print("DeepLog latency measurement: implement after confirming deep-loglizer API")
    except ImportError:
        print("DeepLog: deep-loglizer not available — using literature value only")
        save_result({
            "model": "deeplog", "dataset": "hdfs", "source": "literature",
            "f1": 0.933, "latency_p95_ms": None, "cost_per_1k_logs": 0.0,
            "model_params": 1_000_000, "note": "latency not measured — repo unavailable",
        }, "results/baselines/hdfs_deeplog.json")


def measure_logbert_latency(hdfs_test_sequences: list[str]) -> None:
    """
    Measure LogBERT inference latency on HDFS test set.
    LogBERT uses the same bert-base-uncased architecture as LogLite —
    load it via HuggingFace and wrap with the same measure_latency() call.
    """
    try:
        from transformers import BertForMaskedLM, BertTokenizer
        import torch
        from config.settings import MODEL_NAME, MODEL_REVISION, RANDOM_SEED

        print("LogBERT: loading bert-base-uncased...")
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        model     = BertForMaskedLM.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        def logbert_score(seq: str) -> float:
            inputs = tokenizer(seq, return_tensors="pt", max_length=128,
                               padding="max_length", truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            return outputs.loss.item()

        print("LogBERT: measuring latency (50 warm-up + 100 timed windows)...")
        latency = measure_latency(logbert_score, hdfs_test_sequences)
        result = {
            "model": "logbert", "dataset": "hdfs", "source": "measured",
            "f1": 0.82,   # from literature — LogBERT paper
            "latency_p50_ms": latency["latency_p50_ms"],
            "latency_p95_ms": latency["latency_p95_ms"],
            "latency_p99_ms": latency["latency_p99_ms"],
            "cost_per_1k_logs": 0.0,
            "model_params": 110_000_000,
        }
        save_result(result, "results/baselines/hdfs_logbert.json")
        print(f"LogBERT p95: {latency['latency_p95_ms']:.1f}ms")

    except Exception as e:
        print(f"LogBERT measurement failed: {e}")


if __name__ == "__main__":
    print("=== M2: Baseline Measurement ===\n")

    # Load HDFS test sequences (non-overlapping 20-line windows for latency)
    print("Loading HDFS...")
    _, hdfs_test = load_hdfs(
        log_path=f"{HDFS_PATH}/HDFS.log",
        label_path=HDFS_LABEL_FILE,
    )
    # Use first 200 sequences (50 warm-up + 100 timed + buffer)
    hdfs_test_seqs = [seq for seq, _ in hdfs_test[:200]]

    # Tier 1: measure latency
    measure_logbert_latency(hdfs_test_seqs)
    measure_deeplog_latency(hdfs_test_seqs)

    # Tier 2: write literature numbers
    write_literature_json()

    print("\n=== M2 complete ===")
    print("Check results/baselines/ for output files.")
