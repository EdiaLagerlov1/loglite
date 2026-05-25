# src/models/loglite.py
import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertTokenizer, AutoTokenizer

from config.settings import (
    MAX_LENGTH,
    MODEL_NAME,
    MODEL_REVISION,
    MLM_PROB,
    RANDOM_SEED,
)


def load_model(
    model_path: str | None = None,
) -> tuple[BertForMaskedLM, BertTokenizer]:
    """
    Loads tokenizer and model.
    If model_path is given, loads fine-tuned weights from that directory.
    Otherwise loads the base bert-base-uncased weights from HuggingFace.
    """
    source = model_path or MODEL_NAME
    revision = None if model_path else MODEL_REVISION

    # AutoTokenizer loads the fast tokenizer (PreTrainedTokenizerFast)
    # which supports return_offsets_mapping needed by compute_per_token_losses.
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, use_fast=True
    )
    model = BertForMaskedLM.from_pretrained(
        source, revision=revision
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def mask_tokens(
    input_ids: torch.Tensor,
    tokenizer: BertTokenizer,
    mlm_prob: float = MLM_PROB,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly masks mlm_prob of non-special tokens.

    Caller MUST call torch.manual_seed(RANDOM_SEED) before this function
    to ensure identical masks for identical inputs across all runs.
    This is required for reproducible threshold tuning and trigger identification.

    Returns:
      masked_input_ids: input_ids with [MASK] at selected positions
      labels: original token IDs at masked positions; -100 elsewhere
               (F.cross_entropy ignores -100 — loss computed only on masked tokens)
    """
    labels = input_ids.clone()
    prob_matrix = torch.full(labels.shape, mlm_prob)

    special_tokens = [
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    ]
    for sp in special_tokens:
        prob_matrix[labels == sp] = 0.0

    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100
    input_ids[masked_indices] = tokenizer.mask_token_id
    return input_ids, labels


def compute_anomaly_score(
    model: BertForMaskedLM,
    tokenizer: BertTokenizer,
    log_sequence: str,
    mlm_prob: float = MLM_PROB,
) -> float:
    """
    Deterministic anomaly score: mean CE loss over masked token positions.

    The same random seed is fixed before masking so identical inputs always
    produce identical scores — required for reproducible threshold tuning.

    Higher loss = model more surprised by the tokens = more anomalous.
    Returns 0.0 for empty input (treated as normal).
    """
    if not log_sequence or not log_sequence.strip():
        return 0.0

    inputs = tokenizer(
        log_sequence,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    torch.manual_seed(RANDOM_SEED)   # deterministic mask
    inputs["input_ids"], labels = mask_tokens(
        inputs["input_ids"].clone(), tokenizer, mlm_prob
    )
    if torch.cuda.is_available():
        labels = labels.cuda()

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    return outputs.loss.item()


def compute_per_token_losses(
    model: BertForMaskedLM,
    tokenizer: BertTokenizer,
    log_sequence: str,
) -> tuple[list[float], dict]:
    """
    Computes per-token CE loss using the SAME deterministic masking as
    compute_anomaly_score() — trigger identification is consistent with
    the score that flagged this sequence.

    Returns:
      per_token_losses: list[float] of length MAX_LENGTH
                        non-zero only at masked positions
      enc: tokenizer output dict including offset_mapping
           (used by extract_line_reconstruction_loss to map tokens → lines)
    """
    enc = tokenizer(
        log_sequence,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    original_ids = enc["input_ids"].clone()

    torch.manual_seed(RANDOM_SEED)   # same seed as compute_anomaly_score
    masked_ids, labels = mask_tokens(original_ids.clone(), tokenizer)

    model_inputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
    model_inputs["input_ids"] = masked_ids

    if torch.cuda.is_available():
        model_inputs = {k: v.cuda() for k, v in model_inputs.items()}
        labels = labels.cuda()

    with torch.no_grad():
        logits = model(**model_inputs).logits   # [1, MAX_LENGTH, vocab]

    per_token = F.cross_entropy(
        logits[0], labels[0], reduction="none", ignore_index=-100
    ).tolist()   # length MAX_LENGTH — non-zero only at masked positions

    return per_token, enc
