from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = APP_DIR.parent
EXPORT_DIR = APP_DIR / "exports"
LOG_DIR = APP_DIR / "logs"
CONFIG_DIR = APP_DIR / "configs"

MODEL_EXTENSIONS = {".pt", ".onnx", ".engine"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.m4v)"

TASK_CONFIG_FILE = CONFIG_DIR / "last_task_config.json"
FIRST_RUN_MARKER_FILE = CONFIG_DIR / ".startup_self_check_done"

MODEL_SEARCH_ROOTS = [APP_DIR, WORKSPACE_DIR]
