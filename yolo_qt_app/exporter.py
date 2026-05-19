import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def export_csv(path: str | Path, records: list[dict], metadata: dict | None = None):
    if not records:
        raise ValueError("暂无检测结果可导出。")
    rows = _with_metadata(records, metadata)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_tsv(path: str | Path, records: list[dict], metadata: dict | None = None):
    if not records:
        raise ValueError("暂无检测结果可导出。")
    rows = _with_metadata(records, metadata)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def export_json(
    path: str | Path,
    records: list[dict],
    model_path: str,
    total_counts: Counter,
    metadata: dict | None = None,
):
    if not records:
        raise ValueError("暂无检测结果可导出。")
    payload = {
        "version": "1.1",
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_path,
        "total": sum(total_counts.values()),
        "counts": dict(total_counts),
        "metadata": metadata or {},
        "records": records,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def export_txt_summary(
    path: str | Path,
    records: list[dict],
    total_counts: Counter,
    metadata: dict | None = None,
):
    if not records:
        raise ValueError("暂无检测结果可导出。")
    lines = [
        "YOLO 检测摘要",
        f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"总目标数: {sum(total_counts.values())}",
        "",
        "类别统计:",
    ]
    for class_name, count in sorted(total_counts.items()):
        lines.append(f"- {class_name}: {count}")
    lines.extend(["", "会话元数据:"])
    for key, value in sorted((metadata or {}).items()):
        lines.append(f"- {key}: {value}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def detect_export_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    mapping = {
        ".csv": "csv",
        ".json": "json",
        ".tsv": "tsv",
        ".txt": "txt",
    }
    if suffix not in mapping:
        raise ValueError(f"不支持的导出类型: {suffix}")
    return mapping[suffix]


def _with_metadata(records: list[dict], metadata: dict | None) -> list[dict]:
    if not metadata:
        return records
    rows = []
    for record in records:
        row = dict(record)
        for key, value in metadata.items():
            row[f"session_{key}"] = value
        rows.append(row)
    return rows
