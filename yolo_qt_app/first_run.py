from pathlib import Path

from .config import FIRST_RUN_MARKER_FILE


def is_first_run(marker: str | Path = FIRST_RUN_MARKER_FILE) -> bool:
    return not Path(marker).exists()


def mark_first_run_complete(marker: str | Path = FIRST_RUN_MARKER_FILE):
    path = Path(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
