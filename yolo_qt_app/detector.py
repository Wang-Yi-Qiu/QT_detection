from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

try:
    import torch
except Exception:  # pragma: no cover - defensive import for runtime envs
    torch = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - defensive import for runtime envs
    YOLO = None


@dataclass
class DetectionOptions:
    confidence: float
    image_size: int
    device: str


class YoloDetector:
    def __init__(self):
        self.model = None
        self.model_path: Path | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, model_path: str | Path):
        if YOLO is None:
            raise RuntimeError("未检测到 ultralytics，请先安装依赖。")
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))

    def predict(self, frame, options: DetectionOptions):
        if self.model is None:
            raise RuntimeError("请先加载模型。")
        return self.model.predict(
            source=frame,
            conf=options.confidence,
            imgsz=options.image_size,
            device=options.device,
            verbose=False,
        )[0]


def available_devices() -> list[str]:
    devices = ["auto", "cpu"]
    if torch is None:
        return devices
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compare_models(model_paths: list[str], frame, options: DetectionOptions) -> list[dict]:
    summary = []
    for path in model_paths:
        detector = YoloDetector()
        detector.load(path)
        start = perf_counter()
        result = detector.predict(frame, options)
        elapsed_ms = (perf_counter() - start) * 1000
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        summary.append(
            {
                "model": str(path),
                "inference_ms": round(elapsed_ms, 2),
                "detections": int(count),
            }
        )
    return summary


def records_from_result(result, source: str, source_type: str, frame_index: int) -> list[dict]:
    records = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return records

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for box in boxes:
        cls_id = int(box.cls[0].detach().cpu().item())
        conf = float(box.conf[0].detach().cpu().item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].detach().cpu().tolist()]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        records.append(
            {
                "time": now,
                "source": source,
                "source_type": source_type,
                "frame": frame_index,
                "class_id": cls_id,
                "class_name": result.names.get(cls_id, str(cls_id)),
                "confidence": round(conf, 6),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "width": round(width, 2),
                "height": round(height, 2),
                "area": round(width * height, 2),
            }
        )
    return records
