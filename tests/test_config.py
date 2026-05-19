import unittest

from yolo_qt_app import config


class ConfigTests(unittest.TestCase):
    def test_basic_paths(self):
        self.assertTrue(config.APP_DIR.exists())
        self.assertTrue(str(config.EXPORT_DIR).endswith("exports"))
        self.assertTrue(str(config.LOG_DIR).endswith("logs"))

    def test_model_extensions_contains_common_types(self):
        self.assertIn(".pt", config.MODEL_FILE_EXTENSIONS)
        self.assertIn(".onnx", config.MODEL_FILE_EXTENSIONS)


if __name__ == "__main__":
    unittest.main()
