# src/data/preprocess.py
import re
from collections import defaultdict
from typing import Generator

# ── HDFS ─────────────────────────────────────────────────────────────────────

BLOCK_ID_PATTERN = re.compile(r"(blk_-?\d+)")
STRIP_PATTERN    = re.compile(r"^\d{6}\s\d{6}\s\d+\s")


def parse_hdfs_line(raw_line: str) -> tuple[str | None, str]:
    """Returns (block_id, cleaned_message). block_id is None if not found."""
    m = BLOCK_ID_PATTERN.search(raw_line)
    block_id = m.group(1) if m else None
    message  = STRIP_PATTERN.sub("", raw_line).strip()
    return block_id, message


def build_hdfs_sessions(lines: list[str]) -> dict[str, list[str]]:
    """Groups cleaned log messages by block ID. Returns {block_id: [msg, ...]}."""
    sessions: dict[str, list[str]] = defaultdict(list)
    for raw_line in lines:
        block_id, message = parse_hdfs_line(raw_line)
        if block_id:
            sessions[block_id].append(message)
    return dict(sessions)


def hdfs_session_to_sequence(messages: list[str]) -> str:
    """Joins a session's messages into a single string for the tokenizer."""
    return " ".join(messages)


# ── BGL ──────────────────────────────────────────────────────────────────────

def parse_bgl_line(raw_line: str) -> tuple[int, str]:
    """
    BGL format: first field is '-' (normal) or an alert tag (anomaly).
    Returns (label, message) where label=0 (normal) or 1 (anomaly).
    """
    parts = raw_line.strip().split(None, 1)
    if not parts:
        return 0, ""
    label   = 0 if parts[0] == "-" else 1
    message = parts[1] if len(parts) > 1 else ""
    return label, message


def bgl_windows(
    lines: list[str],
    size: int = 20,
    step: int = 10,
) -> Generator[tuple[str, int], None, None]:
    """
    Sliding window over BGL lines.
    Yields (sequence_str, window_label) where:
      window_label = 1 if ANY line in the window is anomalous, else 0.
      (BGL_WINDOW_ANOMALOUS_IF_ANY rule from config/settings.py)

    Use step=10 for F1 evaluation (overlapping).
    Use step=20 (= size) for latency measurement (non-overlapping).
    """
    for i in range(0, len(lines) - size + 1, step):
        window_lines = lines[i: i + size]
        labels_and_messages = [parse_bgl_line(l) for l in window_lines]
        window_label   = 1 if any(lbl for lbl, _ in labels_and_messages) else 0
        sequence       = " ".join(msg for _, msg in labels_and_messages)
        yield sequence, window_label
