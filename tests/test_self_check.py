import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yolo_qt_app.first_run import is_first_run, mark_first_run_complete
from yolo_qt_app.self_check import run_startup_self_check


class FirstRunTests(unittest.TestCase):
    def test_first_run_marker_flow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            marker = Path(temp_dir) / "marker"
            self.assertTrue(is_first_run(marker))
            mark_first_run_complete(marker)
            self.assertFalse(is_first_run(marker))


class SelfCheckTests(unittest.TestCase):
    def test_run_startup_self_check_contains_required_fields(self):
        with patch("yolo_qt_app.self_check.find_models", return_value=[Path("/tmp/a.pt")]), patch(
            "yolo_qt_app.self_check.available_devices", return_value=["auto", "cpu", "cuda"]
        ):
            report = run_startup_self_check()
        self.assertIn("dependencies", report)
        self.assertIn("devices", report)
        self.assertIn("gpu_devices", report)
        self.assertIn("model_roots", report)
        self.assertIn("model_paths", report)
        self.assertEqual(report["models_found"], 1)
        self.assertIn("cuda", report["gpu_devices"])


if __name__ == "__main__":
    unittest.main()
