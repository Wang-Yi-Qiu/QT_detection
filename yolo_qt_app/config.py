from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = APP_DIR.parent
EXPORT_DIR = APP_DIR / "exports"
LOG_DIR = APP_DIR / "logs"
HISTORY_DIR = APP_DIR / "history"
CONFIG_DIR = APP_DIR / "saved_configs"
APP_LOG_FILE = LOG_DIR / "app.log"
HISTORY_INDEX_FILE = HISTORY_DIR / "history.jsonl"
TASK_CONFIG_FILE = CONFIG_DIR / "last_task_config.json"
FIRST_RUN_MARKER_FILE = CONFIG_DIR / ".startup_self_check_done"

MODEL_FILE_EXTENSIONS = {
    ".pt",
    ".torchscript",
    ".onnx",
    ".engine",
    ".mlpackage",
    ".pb",
    ".tflite",
    ".mnn",
    ".rknn",
}
MODEL_DIR_SUFFIXES = {
    "_openvino_model",
    "_saved_model",
    "_web_model",
    "_paddle_model",
    "_ncnn_model",
    "_imx_model",
    "_rknn_model",
    "_executorch_model",
    "_axelera_model",
    "_deepx_model",
}
MODEL_FILE_FILTER = (
    "YOLO Models (*.pt *.torchscript *.onnx *.engine *.mlpackage "
    "*.pb *.tflite *.mnn *.rknn)"
)
IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.m4v)"
BATCH_IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_EXTENSIONS = MODEL_FILE_EXTENSIONS
IMAGE_EXTENSIONS = IMAGE_SUFFIXES
EXPORT_FILTER = (
    "CSV (*.csv);;JSON (*.json);;TSV (*.tsv);;TXT Summary (*.txt)"
)

MODEL_SEARCH_ROOTS = [APP_DIR, WORKSPACE_DIR]

DEFAULT_WINDOW_WIDTH = 1440
DEFAULT_WINDOW_HEIGHT = 860
THREAD_SHUTDOWN_TIMEOUT_MS = 1500
BATCH_PROCESSING_DELAY_MS = 1
HISTORY_MAX_LINES = 2000
