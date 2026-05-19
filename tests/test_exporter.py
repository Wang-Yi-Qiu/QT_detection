import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from yolo_qt_app.exporter import (
    detect_export_type,
    export_csv,
    export_json,
    export_tsv,
    export_txt_summary,
)


class ExporterTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "time": "2026-01-01 00:00:00",
                "source": "demo.jpg",
                "source_type": "image",
                "frame": 1,
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.99,
                "x1": 1.0,
                "y1": 2.0,
                "x2": 3.0,
                "y2": 4.0,
                "width": 2.0,
                "height": 2.0,
                "area": 4.0,
            }
        ]
        self.counts = Counter({"person": 1})
        self.metadata = {"session_id": "abc123", "device": "cpu"}

    def test_detect_export_type(self):
        self.assertEqual(detect_export_type("a.csv"), "csv")
        self.assertEqual(detect_export_type("a.json"), "json")
        self.assertEqual(detect_export_type("a.tsv"), "tsv")
        self.assertEqual(detect_export_type("a.txt"), "txt")

    def test_export_json(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.json"
            export_json(path, self.records, "m.pt", self.counts, self.metadata)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model"], "m.pt")
            self.assertEqual(payload["counts"], {"person": 1})
            self.assertEqual(payload["metadata"]["session_id"], "abc123")

    def test_export_csv_tsv_txt(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "out.csv"
            tsv_path = Path(td) / "out.tsv"
            txt_path = Path(td) / "out.txt"
            export_csv(csv_path, self.records, self.metadata)
            export_tsv(tsv_path, self.records, self.metadata)
            export_txt_summary(txt_path, self.records, self.counts, self.metadata)
            self.assertIn("session_id", csv_path.read_text(encoding="utf-8-sig"))
            self.assertIn("session_id", tsv_path.read_text(encoding="utf-8"))
            self.assertIn("会话元数据", txt_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
