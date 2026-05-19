from pathlib import Path

from .config import MODEL_DIR_SUFFIXES, MODEL_FILE_EXTENSIONS, MODEL_SEARCH_ROOTS


def is_supported_model_path(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() in MODEL_FILE_EXTENSIONS
    if path.is_dir():
        name = path.name.lower()
        return any(name.endswith(suffix) for suffix in MODEL_DIR_SUFFIXES)
    return False


def find_models() -> list[Path]:
    seen = set()
    models = []
    for root in MODEL_SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if is_supported_model_path(path):
                resolved = path.resolve()
                if resolved not in seen:
                    models.append(path)
                    seen.add(resolved)
    return sorted(models, key=lambda item: str(item).lower())
