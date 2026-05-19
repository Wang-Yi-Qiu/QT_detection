import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import yolo_qt_app.detector as detector
    from yolo_qt_app.detector import (
        DetectionOptions,
        available_devices,
        compare_models,
        records_from_result,
        resolve_device,
    )
except ModuleNotFoundError:  # pragma: no cover
    detector = None
    available_devices = None
    resolve_device = None
    compare_models = None
    records_from_result = None
    DetectionOptions = None


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return self.value


class _FakeArray:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class _FakeBox:
    def __init__(self):
        self.cls = [_FakeScalar(0)]
        self.conf = [_FakeScalar(0.88)]
        self.xyxy = [_FakeArray([1.0, 2.0, 10.0, 20.0])]


@unittest.skipIf(records_from_result is None, "torch/ultralytics not installed")
class DetectorRecordTests(unittest.TestCase):
    def test_records_from_result(self):
        result = SimpleNamespace(boxes=[_FakeBox()], names={0: "person"})
        records = records_from_result(result, source="a.jpg", source_type="image", frame_index=1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["class_name"], "person")


@unittest.skipIf(compare_models is None, "torch/ultralytics not installed")
class DetectorCompareTests(unittest.TestCase):
    def test_compare_models_calls_predict(self):
        fake_result = SimpleNamespace(boxes=[1, 2], names={})
        with patch.object(detector.YoloDetector, "load") as mock_load, patch.object(
            detector.YoloDetector,
            "predict",
            return_value=fake_result,
        ) as mock_predict:
            summary = compare_models(
                ["a.pt", "b.pt"],
                frame=object(),
                options=DetectionOptions(confidence=0.5, image_size=640, device="cpu"),
            )
        self.assertEqual(len(summary), 2)
        self.assertEqual(mock_load.call_count, 2)
        self.assertEqual(mock_predict.call_count, 2)


@unittest.skipIf(getattr(detector, "torch", None) is None, "torch not installed")
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
