import json
from pathlib import Path


def save_task_config(path: str | Path, payload: dict):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_task_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
