from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = APP_DIR.parent
EXPORT_DIR = APP_DIR / "exports"

MODEL_EXTENSIONS = {".pt", ".onnx", ".engine"}
IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.m4v)"

MODEL_SEARCH_ROOTS = [APP_DIR, WORKSPACE_DIR]
