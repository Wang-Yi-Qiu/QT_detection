from pathlib import Path

from .config import MODEL_EXTENSIONS, MODEL_SEARCH_ROOTS


def find_models() -> list[Path]:
    seen = set()
    models = []
    for root in MODEL_SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS:
                resolved = path.resolve()
                if resolved not in seen:
                    models.append(path)
                    seen.add(resolved)
    return sorted(models, key=lambda item: str(item).lower())
