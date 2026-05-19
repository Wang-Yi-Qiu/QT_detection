import unittest
from pathlib import Path

from yolo_qt_app import config


class ConfigTests(unittest.TestCase):
    def test_core_paths_are_paths(self):
        self.assertIsInstance(config.APP_DIR, Path)
        self.assertIsInstance(config.EXPORT_DIR, Path)
        self.assertIsInstance(config.LOG_DIR, Path)
        self.assertIsInstance(config.CONFIG_DIR, Path)

    def test_filters_and_extensions(self):
        self.assertIn(".pt", config.MODEL_EXTENSIONS)
        self.assertIn(".jpg", config.IMAGE_EXTENSIONS)
        self.assertIn("Images", config.IMAGE_FILTER)
        self.assertIn("Videos", config.VIDEO_FILTER)


if __name__ == "__main__":
    unittest.main()
