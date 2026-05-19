import unittest
from types import SimpleNamespace
from unittest.mock import patch

import yolo_qt_app.detector as detector
from yolo_qt_app.detector import DetectionOptions, available_devices, records_from_result, resolve_device


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


class DetectorTests(unittest.TestCase):
    def test_records_from_result(self):
        result = SimpleNamespace(boxes=[_FakeBox()], names={0: "person"})
        records = records_from_result(result, source="a.jpg", source_type="image", frame_index=1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["class_name"], "person")
        self.assertEqual(records[0]["frame"], 1)

    def test_available_devices_without_torch(self):
        with patch.object(detector, "torch", None):
            devices = available_devices()
        self.assertIn("auto", devices)
        self.assertIn("cpu", devices)

    def test_resolve_device_auto_without_accelerator(self):
        with patch.object(detector, "torch", None):
            self.assertEqual(resolve_device("auto"), "cpu")

    def test_compare_models_calls_predict(self):
        fake_result = SimpleNamespace(boxes=[1, 2], names={})
        with patch.object(detector.YoloDetector, "load") as mock_load, patch.object(
            detector.YoloDetector,
            "predict",
            return_value=fake_result,
        ) as mock_predict:
            summary = detector.compare_models(
                ["a.pt", "b.pt"],
                frame=object(),
                options=DetectionOptions(confidence=0.5, image_size=640, device="cpu"),
            )
        self.assertEqual(len(summary), 2)
        self.assertEqual(mock_load.call_count, 2)
        self.assertEqual(mock_predict.call_count, 2)


if __name__ == "__main__":
    unittest.main()
