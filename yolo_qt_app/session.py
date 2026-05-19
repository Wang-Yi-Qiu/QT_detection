from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class SessionMetadata:
    model_path: str = ""
    source_type: str = ""
    source: str = ""
    device: str = ""
    confidence: float = 0.5
    image_size: int = 640
    frame_skip: int = 0
    max_fps: int = 30
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    ended_at: str = ""

    def close(self):
        self.ended_at = datetime.now().isoformat(timespec="seconds")

    def to_payload(self) -> dict:
        payload = asdict(self)
        if self.started_at and self.ended_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.ended_at)
            payload["duration_seconds"] = max(0.0, round((end - start).total_seconds(), 3))
        else:
            payload["duration_seconds"] = None
        return payload
