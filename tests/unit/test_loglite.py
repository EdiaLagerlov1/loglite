# tests/unit/test_loglite.py
import torch
import pytest
from transformers import AutoTokenizer
from src.models.loglite import (
    load_model,
    mask_tokens,
    compute_anomaly_score,
    compute_per_token_losses,
)
from config.settings import RANDOM_SEED, MAX_LENGTH, MODEL_NAME, MODEL_REVISION


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model, tokenizer = load_model()
    return model, tokenizer


def test_forward_pass_returns_scalar_loss(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    score = compute_anomaly_score(model, tokenizer, "INFO dfs.DataNode: Receiving block")
    assert isinstance(score, float)
    assert score > 0.0


def test_anomaly_score_higher_on_corrupted_sequence(model_and_tokenizer):
    """
    A sequence of repeated common English words should have lower loss than
    a sequence of random number strings — the base BERT model is confident
    about common English regardless of log fine-tuning.
    """
    model, tokenizer = model_and_tokenizer
    normal = " ".join(["the cat sat on the mat"] * 10)   # very predictable English
    corrupted = " ".join([f"3847{i}xq9z" for i in range(30)])   # random tokens
    normal_score = compute_anomaly_score(model, tokenizer, normal)
    corrupted_score = compute_anomaly_score(model, tokenizer, corrupted)
    assert corrupted_score > normal_score, (
        f"Expected corrupted score ({corrupted_score:.4f}) > "
        f"normal score ({normal_score:.4f})"
    )


def test_deterministic_masking(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    seq = "INFO dfs.DataNode: Receiving block blk_-1608999687919862906"
    score1 = compute_anomaly_score(model, tokenizer, seq)
    score2 = compute_anomaly_score(model, tokenizer, seq)
    assert score1 == score2, (
        f"Scores differ between runs: {score1} vs {score2} — "
        "masking is not deterministic"
    )


def test_empty_input_returns_zero(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    assert compute_anomaly_score(model, tokenizer, "") == 0.0
    assert compute_anomaly_score(model, tokenizer, "   ") == 0.0


def test_parameter_count(model_and_tokenizer):
    model, _ = model_and_tokenizer
    n_params = sum(p.numel() for p in model.parameters())
    assert 100_000_000 < n_params < 130_000_000, (
        f"Expected ~110M parameters, got {n_params:,}"
    )


def test_per_token_losses_shape(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    seq = "INFO dfs.DataNode: Receiving block blk_-1608999687919862906"
    per_token, enc = compute_per_token_losses(model, tokenizer, seq)
    assert len(per_token) == MAX_LENGTH
    assert any(l > 0.0 for l in per_token), "All per-token losses are zero — masking failed"


def test_per_token_losses_consistent_with_anomaly_score(model_and_tokenizer):
    """Mean of non-zero per-token losses should be close to compute_anomaly_score."""
    model, tokenizer = model_and_tokenizer
    seq = "INFO dfs.DataNode: Receiving block blk_-1608999687919862906"
    score = compute_anomaly_score(model, tokenizer, seq)
    per_token, _ = compute_per_token_losses(model, tokenizer, seq)
    nonzero = [l for l in per_token if l > 0.0]
    mean_per_token = sum(nonzero) / len(nonzero) if nonzero else 0.0
    assert abs(score - mean_per_token) < 0.01, (
        f"Anomaly score ({score:.4f}) differs from mean per-token loss "
        f"({mean_per_token:.4f}) — seeds may differ"
    )


def test_mask_tokens_output_shapes():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_fast=True)
    # Use a long sequence to ensure enough real tokens for masking (15% of ~40 tokens = ~6 masks)
    seq = " ".join(["the quick brown fox jumps over the lazy dog"] * 5)
    input_ids = tokenizer(
        seq, return_tensors="pt",
        max_length=MAX_LENGTH, padding="max_length", truncation=True
    )["input_ids"]
    torch.manual_seed(RANDOM_SEED)
    masked_ids, labels = mask_tokens(input_ids.clone(), tokenizer)
    assert masked_ids.shape == input_ids.shape
    assert labels.shape == input_ids.shape
    assert (labels == -100).any(), "Expected some -100 labels (non-masked positions)"
    assert (labels != -100).any(), "Expected some non-(-100) labels (masked positions)"
