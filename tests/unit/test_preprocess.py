# tests/unit/test_preprocess.py
from src.data.preprocess import (
    parse_hdfs_line,
    build_hdfs_sessions,
    hdfs_session_to_sequence,
    parse_bgl_line,
    bgl_windows,
)


SAMPLE_HDFS_LINE = "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106"
SAMPLE_BGL_NORMAL = "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
SAMPLE_BGL_ANOMALY = "KERNSF 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL some error"


def test_parse_hdfs_line_extracts_block_id():
    block_id, message = parse_hdfs_line(SAMPLE_HDFS_LINE)
    assert block_id == "blk_-1608999687919862906"


def test_parse_hdfs_line_strips_timestamp():
    _, message = parse_hdfs_line(SAMPLE_HDFS_LINE)
    assert not message.startswith("081109")
    assert "DataXceiver" in message


def test_parse_hdfs_line_no_block_id():
    block_id, _ = parse_hdfs_line("081109 203518 143 INFO some message without block")
    assert block_id is None


def test_build_hdfs_sessions_groups_by_block_id():
    lines = [SAMPLE_HDFS_LINE, SAMPLE_HDFS_LINE]
    sessions = build_hdfs_sessions(lines)
    assert "blk_-1608999687919862906" in sessions
    assert len(sessions["blk_-1608999687919862906"]) == 2


def test_hdfs_session_to_sequence_joins_with_space():
    messages = ["msg one", "msg two", "msg three"]
    seq = hdfs_session_to_sequence(messages)
    assert seq == "msg one msg two msg three"


def test_parse_bgl_normal_line():
    label, message = parse_bgl_line(SAMPLE_BGL_NORMAL)
    assert label == 0
    assert "RAS KERNEL INFO" in message


def test_parse_bgl_anomaly_line():
    label, message = parse_bgl_line(SAMPLE_BGL_ANOMALY)
    assert label == 1
    assert "RAS KERNEL FATAL" in message


def test_bgl_windows_correct_count():
    lines = [SAMPLE_BGL_NORMAL] * 100
    windows = list(bgl_windows(lines, size=20, step=10))
    # (100 - 20) / 10 + 1 = 9
    assert len(windows) == 9


def test_bgl_windows_non_overlapping():
    lines = [SAMPLE_BGL_NORMAL] * 40
    windows = list(bgl_windows(lines, size=20, step=20))
    assert len(windows) == 2


def test_bgl_window_label_any_anomalous():
    """Window with one anomalous line should be labelled 1."""
    lines = [SAMPLE_BGL_NORMAL] * 19 + [SAMPLE_BGL_ANOMALY]
    windows = list(bgl_windows(lines, size=20, step=20))
    assert len(windows) == 1
    _, label = windows[0]
    assert label == 1


def test_bgl_window_label_all_normal():
    lines = [SAMPLE_BGL_NORMAL] * 20
    windows = list(bgl_windows(lines, size=20, step=20))
    _, label = windows[0]
    assert label == 0
