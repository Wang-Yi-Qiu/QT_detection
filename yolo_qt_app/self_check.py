import importlib
from pathlib import Path

from .config import MODEL_SEARCH_ROOTS
from .detector import available_devices
from .model_finder import find_models


REQUIRED_MODULES = ["cv2", "PyQt6", "ultralytics"]


def _check_model_roots() -> list[dict]:
    roots = []
    for root in MODEL_SEARCH_ROOTS:
        path = Path(root)
        roots.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "readable": path.exists() and path.is_dir(),
            }
        )
    return roots


def _check_model_paths(models: list[Path]) -> list[dict]:
    status = []
    for path in models[:5]:
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "readable": path.exists() and path.is_file(),
            }
        )
    return status


def run_startup_self_check() -> dict:
    dep_status: dict[str, bool] = {}
    for name in REQUIRED_MODULES:
        dep_status[name] = importlib.util.find_spec(name) is not None

    devices = available_devices()
    gpu_devices = [name for name in devices if name in {"cuda", "mps"}]
    models = find_models()
    model_paths = _check_model_paths(models)

    return {
        "dependencies": dep_status,
        "devices": devices,
        "gpu_devices": gpu_devices,
        "models_found": len(models),
        "model_examples": [str(path) for path in models[:5]],
        "model_roots": _check_model_roots(),
        "model_paths": model_paths,
        "cwd": str(Path.cwd()),
    }
