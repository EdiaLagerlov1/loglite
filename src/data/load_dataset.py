# src/data/load_dataset.py
import ast
import csv
from pathlib import Path
from config.settings import HDFS_TRAIN_SESSIONS, BGL_TRAIN_LINES
from src.data.preprocess import build_hdfs_sessions, hdfs_session_to_sequence


# ── HDFS (event sequences) ────────────────────────────────────────────────────

def load_hdfs_events(
    events_path: str,
    label_path: str,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Loads HDFS event sequences from preprocessed Event_traces.csv.
    Each session is represented as a space-separated event ID string
    (e.g. "E5 E22 E11 E9 E26") instead of raw log text.

    Applies chronological split: first HDFS_TRAIN_SESSIONS block IDs → train.

    Returns:
      train_sequences: list of (sequence_str, block_id)
      test_sequences:  list of (sequence_str, block_id)
    """
    import pandas as pd
    df = pd.read_csv(events_path)

    # Load labels for anomaly flag
    labels = load_hdfs_labels(label_path)

    # Preserve row order as chronological order
    block_ids = df['BlockId'].tolist()
    features  = df['Features'].tolist()

    sequences = []
    for bid, feat in zip(block_ids, features):
        # Features column: "[E5,E22,E11,...]" — parse to list then join with spaces
        events = ast.literal_eval(feat)
        seq = " ".join(events)
        sequences.append((seq, bid))

    train_sequences = sequences[:HDFS_TRAIN_SESSIONS]
    test_sequences  = sequences[HDFS_TRAIN_SESSIONS:]
    return train_sequences, test_sequences


# ── HDFS (raw log text) ───────────────────────────────────────────────────────

def load_hdfs(
    log_path: str,
    label_path: str,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Loads HDFS log file and annotation file.
    Applies chronological split: first HDFS_TRAIN_SESSIONS block IDs → train.

    Returns:
      train_sequences: list of (sequence_str, block_id) — no labels attached
      test_sequences:  list of (sequence_str, block_id) — labels loaded separately
    """
    with open(log_path) as f:
        raw_lines = f.readlines()

    sessions = build_hdfs_sessions(raw_lines)

    # Chronological order: block IDs in order of first appearance
    seen: list[str] = []
    seen_set: set[str] = set()
    for line in raw_lines:
        from src.data.preprocess import BLOCK_ID_PATTERN
        m = BLOCK_ID_PATTERN.search(line)
        if m:
            bid = m.group(1)
            if bid not in seen_set:
                seen.append(bid)
                seen_set.add(bid)

    train_ids = set(seen[:HDFS_TRAIN_SESSIONS])
    test_ids  = set(seen[HDFS_TRAIN_SESSIONS:])

    train_sequences = [
        (hdfs_session_to_sequence(sessions[bid]), bid)
        for bid in seen[:HDFS_TRAIN_SESSIONS]
        if bid in sessions
    ]
    test_sequences = [
        (hdfs_session_to_sequence(sessions[bid]), bid)
        for bid in seen[HDFS_TRAIN_SESSIONS:]
        if bid in sessions
    ]
    return train_sequences, test_sequences


def load_hdfs_labels(label_path: str) -> dict[str, int]:
    """
    Reads HDFS anomaly_label.csv.
    Returns {block_id: 0|1} where 1 = Anomaly.
    """
    labels: dict[str, int] = {}
    with open(label_path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0
    return labels


# ── BGL ──────────────────────────────────────────────────────────────────────

def load_bgl(log_path: str) -> tuple[list[str], list[str]]:
    """
    Loads BGL log file.
    Applies chronological split: first BGL_TRAIN_LINES → train.

    Returns:
      train_lines: raw lines (labels embedded in col 0, used only at eval)
      test_lines:  raw lines
    """
    with open(log_path) as f:
        lines = f.readlines()

    train_lines = lines[:BGL_TRAIN_LINES]
    test_lines  = lines[BGL_TRAIN_LINES:]
    return train_lines, test_lines
