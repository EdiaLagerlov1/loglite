"""
scripts/compare_baselines.py

Builds final comparison table from results/.
Run after M4 and M7 when all metrics are collected.
Usage: python scripts/compare_baselines.py
"""
import json
import pathlib
import pandas as pd


def build_comparison_table(results_dir: str = "results") -> pd.DataFrame:
    rows = []
    for path in pathlib.Path(results_dir).rglob("*.json"):
        if path.name in ("agent_eval.json",):
            continue
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                rows.extend(data)   # literature.json is a list of dicts
            else:
                rows.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    df = pd.DataFrame(rows)
    cols = ["model", "dataset", "source", "f1", "latency_p95_ms",
            "cost_per_1k_logs", "model_params"]
    present = [c for c in cols if c in df.columns]
    return df[present].sort_values("f1", ascending=False)


def assert_beats_deeplog(df: pd.DataFrame) -> None:
    """H1 assertion: LogLite measured F1 > DeepLog measured F1 on HDFS and BGL."""
    measured = df[df["source"] == "measured"]   # only compare measured vs measured
    for dataset in ["hdfs", "bgl"]:
        loglite = measured[(measured.model == "loglite") & (measured.dataset == dataset)]
        deeplog = measured[(measured.model == "deeplog") & (measured.dataset == dataset)]
        if loglite.empty or deeplog.empty:
            print(f"  H1 [{dataset}]: skipped — measured rows missing for loglite or deeplog")
            continue
        lf1 = loglite["f1"].values[0]
        df1 = deeplog["f1"].values[0]
        assert lf1 > df1, (
            f"H1 FAILED on {dataset}: LogLite={lf1:.4f} <= DeepLog={df1:.4f}"
        )
        print(f"  H1 [{dataset}]: PASSED — LogLite={lf1:.4f} > DeepLog={df1:.4f}")


if __name__ == "__main__":
    df = build_comparison_table()
    if df.empty:
        print("No results found. Run training (M4) first.")
    else:
        print(df.to_string(index=False))
        print()
        assert_beats_deeplog(df)
