import unittest
from unittest.mock import patch

try:
    from yolo_qt_app.detector import available_devices, resolve_device
except ModuleNotFoundError:  # pragma: no cover
    available_devices = None
    resolve_device = None


@unittest.skipIf(available_devices is None or resolve_device is None, "torch/ultralytics not installed")
class DetectorUtilityTests(unittest.TestCase):
    @patch("yolo_qt_app.detector.torch.cuda.is_available", return_value=False)
    @patch("yolo_qt_app.detector.torch.backends.mps.is_available", return_value=False)
    def test_available_devices_cpu_only(self, _mps, _cuda):
        self.assertEqual(available_devices(), ["auto", "cpu"])

    @patch("yolo_qt_app.detector.torch.cuda.is_available", return_value=True)
    @patch("yolo_qt_app.detector.torch.backends.mps.is_available", return_value=False)
    def test_resolve_device_auto_prefers_cuda(self, _mps, _cuda):
        self.assertEqual(resolve_device("auto"), "cuda")

    @patch("yolo_qt_app.detector.torch.cuda.is_available", return_value=False)
    @patch("yolo_qt_app.detector.torch.backends.mps.is_available", return_value=True)
    def test_resolve_device_auto_fallback_mps(self, _mps, _cuda):
        self.assertEqual(resolve_device("auto"), "mps")


if __name__ == "__main__":
    unittest.main()
