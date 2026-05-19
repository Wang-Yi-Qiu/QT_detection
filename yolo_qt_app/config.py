from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = APP_DIR.parent
EXPORT_DIR = APP_DIR / "exports"

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

MODEL_SEARCH_ROOTS = [APP_DIR, WORKSPACE_DIR]
