import importlib
from pathlib import Path

from .detector import available_devices
from .model_finder import find_models


REQUIRED_MODULES = ["cv2", "PyQt6", "ultralytics"]


def run_startup_self_check() -> dict:
    dep_status: dict[str, bool] = {}
    for name in REQUIRED_MODULES:
        dep_status[name] = importlib.util.find_spec(name) is not None

    models = find_models()
    return {
        "dependencies": dep_status,
        "devices": available_devices(),
        "models_found": len(models),
        "model_examples": [str(path) for path in models[:5]],
        "cwd": str(Path.cwd()),
    }
