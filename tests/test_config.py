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

    def test_startup_and_task_config_paths(self):
        self.assertTrue(str(config.TASK_CONFIG_FILE).endswith("last_task_config.json"))
        self.assertTrue(str(config.FIRST_RUN_MARKER_FILE).endswith(".startup_self_check_done"))


if __name__ == "__main__":
    unittest.main()
