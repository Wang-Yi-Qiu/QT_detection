import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from .config import HISTORY_DIR, HISTORY_INDEX_FILE, HISTORY_MAX_LINES


def append_history(
    export_path: str | Path,
    export_type: str,
    total_counts: Counter,
    metadata: dict,
):
    HISTORY_DIR.mkdir(exist_ok=True)
    item = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "path": str(export_path),
        "type": export_type,
        "total": sum(total_counts.values()),
        "counts": dict(total_counts),
        "metadata": metadata,
    }
    with open(HISTORY_INDEX_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    if HISTORY_INDEX_FILE.stat().st_size > 512 * 1024:
        trim_history_file(HISTORY_MAX_LINES)


def load_history_items(limit: int = 100) -> list[dict]:
    if not HISTORY_INDEX_FILE.exists():
        return []
    lines = HISTORY_INDEX_FILE.read_text(encoding="utf-8").splitlines()
    items = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(items) >= limit:
            break
    return items


def trim_history_file(max_lines: int):
    lines = HISTORY_INDEX_FILE.read_text(encoding="utf-8").splitlines()
    if len(lines) <= max_lines:
        return
    HISTORY_INDEX_FILE.write_text("\n".join(lines[-max_lines:]) + "\n", encoding="utf-8")
