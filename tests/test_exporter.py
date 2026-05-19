import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from yolo_qt_app.exporter import export_csv, export_json


class ExporterTests(unittest.TestCase):
    def test_export_csv_and_json(self):
        records = [
            {
                "time": "2026-01-01 00:00:00",
                "source": "demo",
                "source_type": "image",
                "frame": 1,
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.95,
                "x1": 1.0,
                "y1": 2.0,
                "x2": 10.0,
                "y2": 12.0,
                "width": 9.0,
                "height": 10.0,
                "area": 90.0,
            }
        ]
        counts = Counter({"person": 1})

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "out.csv"
            json_path = Path(tmpdir) / "out.json"
            export_csv(csv_path, records)
            export_json(
                json_path,
                records,
                model_path="model.pt",
                total_counts=counts,
                session={"device": "cpu"},
                history={"by_class": {"person": 1}},
            )

            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model"], "model.pt")
            self.assertEqual(payload["total"], 1)
            self.assertIn("session", payload)
            self.assertIn("history", payload)

    def test_export_raises_without_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.json"
            with self.assertRaises(ValueError):
                export_json(path, [], "", Counter())


if __name__ == "__main__":
    unittest.main()
