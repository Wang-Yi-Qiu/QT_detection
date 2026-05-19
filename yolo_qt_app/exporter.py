import csv
import json
from collections import Counter
from pathlib import Path


def export_csv(path: str | Path, records: list[dict]):
    if not records:
        raise ValueError("暂无检测结果可导出。")
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def export_json(
    path: str | Path,
    records: list[dict],
    model_path: str,
    total_counts: Counter,
):
    if not records:
        raise ValueError("暂无检测结果可导出。")
    payload = {
        "model": model_path,
        "total": sum(total_counts.values()),
        "counts": dict(total_counts),
        "records": records,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
