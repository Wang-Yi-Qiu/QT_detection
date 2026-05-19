from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt

from .config import APP_DIR, EXPORT_DIR, IMAGE_FILTER, MODEL_FILE_FILTER, VIDEO_FILTER
from .detector import (
    DetectionOptions,
    YoloDetector,
    available_devices,
    records_from_result,
    resolve_device,
)
from .exporter import export_csv, export_json
from .image_utils import frame_to_pixmap
from .model_finder import find_models
from .styles import APP_STYLE
from .widgets import VideoLabel, section_label, stat_value


class YoloMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = YoloDetector()
        self.capture = None
        self.source_path: Path | None = None
        self.source_type = ""
        self.frame_index = 0
        self.last_tick = QtCore.QElapsedTimer()
        self.records: list[dict] = []
        self.current_counts = Counter()
        self.total_counts = Counter()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.process_next_frame)

        self.build_ui()
        self.refresh_model_list()
        self.update_controls()

    def build_ui(self):
        self.setWindowTitle("YOLO 实时检测与统计导出")
        self.resize(1360, 820)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        header = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("YOLO 检测工作台")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel("图片、视频、摄像头实时检测，自动统计目标数量并导出结果")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        root.addLayout(header)

        content = QtWidgets.QHBoxLayout()
        content.setSpacing(14)
        root.addLayout(content, 1)

        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setSpacing(12)
        content.addLayout(left_panel, 4)

        video_grid = QtWidgets.QHBoxLayout()
        video_grid.setSpacing(12)
        self.original_view = VideoLabel("原始画面")
        self.detected_view = VideoLabel("检测结果")
        video_grid.addWidget(self.original_view)
        video_grid.addWidget(self.detected_view)
        left_panel.addLayout(video_grid, 1)

        self.log_box = QtWidgets.QTextBrowser()
        self.log_box.setMinimumHeight(150)
        self.log_box.setObjectName("LogBox")
        left_panel.addWidget(self.log_box)

        right_panel = QtWidgets.QWidget()
        right_panel.setObjectName("SidePanel")
        right_panel.setFixedWidth(370)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(12)
        content.addWidget(right_panel)

        self.build_model_panel(right_layout)
        self.build_input_panel(right_layout)
        self.build_stats_panel(right_layout)
        self.build_export_panel(right_layout)

        self.setStyleSheet(APP_STYLE)
        self.log("请先选择或加载 YOLO 模型。")

    def build_model_panel(self, parent_layout):
        parent_layout.addWidget(section_label("模型"))
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setMinimumHeight(36)
        self.model_combo.currentIndexChanged.connect(self.model_selection_changed)

        model_buttons = QtWidgets.QHBoxLayout()
        self.load_model_button = QtWidgets.QPushButton("加载模型")
        self.browse_model_button = QtWidgets.QPushButton("选择模型")
        self.refresh_model_button = QtWidgets.QPushButton("刷新")
        self.load_model_button.clicked.connect(self.load_model)
        self.browse_model_button.clicked.connect(self.browse_model)
        self.refresh_model_button.clicked.connect(self.refresh_model_list)
        model_buttons.addWidget(self.load_model_button)
        model_buttons.addWidget(self.browse_model_button)
        model_buttons.addWidget(self.refresh_model_button)

        parent_layout.addWidget(self.model_combo)
        parent_layout.addLayout(model_buttons)

        threshold_row = QtWidgets.QHBoxLayout()
        self.conf_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(50)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.50)
        self.conf_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.conf_spin.setFixedWidth(68)
        self.conf_slider.valueChanged.connect(lambda value: self.conf_spin.setValue(value / 100))
        self.conf_spin.valueChanged.connect(lambda value: self.conf_slider.setValue(round(value * 100)))
        threshold_row.addWidget(QtWidgets.QLabel("置信度"))
        threshold_row.addWidget(self.conf_slider, 1)
        threshold_row.addWidget(self.conf_spin)
        parent_layout.addLayout(threshold_row)

        options_row = QtWidgets.QHBoxLayout()
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(available_devices())
        self.imgsz_spin = QtWidgets.QSpinBox()
        self.imgsz_spin.setRange(160, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        options_row.addWidget(QtWidgets.QLabel("设备"))
        options_row.addWidget(self.device_combo, 1)
        options_row.addWidget(QtWidgets.QLabel("尺寸"))
        options_row.addWidget(self.imgsz_spin)
        parent_layout.addLayout(options_row)

    def build_input_panel(self, parent_layout):
        parent_layout.addWidget(section_label("输入"))
        input_grid = QtWidgets.QGridLayout()
        self.image_button = QtWidgets.QPushButton("图片检测")
        self.video_button = QtWidgets.QPushButton("视频检测")
        self.camera_button = QtWidgets.QPushButton("摄像头实时")
        self.stop_button = QtWidgets.QPushButton("停止")
        self.image_button.clicked.connect(self.open_image)
        self.video_button.clicked.connect(self.open_video)
        self.camera_button.clicked.connect(self.open_camera)
        self.stop_button.clicked.connect(self.stop_detection)
        input_grid.addWidget(self.image_button, 0, 0)
        input_grid.addWidget(self.video_button, 0, 1)
        input_grid.addWidget(self.camera_button, 1, 0)
        input_grid.addWidget(self.stop_button, 1, 1)
        parent_layout.addLayout(input_grid)

    def build_stats_panel(self, parent_layout):
        parent_layout.addWidget(section_label("统计"))
        stat_grid = QtWidgets.QGridLayout()
        self.status_value = stat_value("待加载")
        self.frame_value = stat_value("0")
        self.current_value = stat_value("0")
        self.total_value = stat_value("0")
        self.fps_value = stat_value("0.0")
        labels = ["状态", "帧数", "当前目标", "累计目标", "FPS"]
        values = [
            self.status_value,
            self.frame_value,
            self.current_value,
            self.total_value,
            self.fps_value,
        ]
        for row, (label, value) in enumerate(zip(labels, values)):
            stat_grid.addWidget(QtWidgets.QLabel(label), row, 0)
            stat_grid.addWidget(value, row, 1)
        parent_layout.addLayout(stat_grid)

        self.count_table = QtWidgets.QTableWidget(0, 3)
        self.count_table.setHorizontalHeaderLabels(["类别", "当前", "累计"])
        self.count_table.verticalHeader().setVisible(False)
        self.count_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.count_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.count_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        parent_layout.addWidget(self.count_table, 1)

    def build_export_panel(self, parent_layout):
        export_row = QtWidgets.QHBoxLayout()
        self.export_csv_button = QtWidgets.QPushButton("导出 CSV")
        self.export_json_button = QtWidgets.QPushButton("导出 JSON")
        self.clear_button = QtWidgets.QPushButton("清空统计")
        self.export_csv_button.clicked.connect(lambda: self.export_records("csv"))
        self.export_json_button.clicked.connect(lambda: self.export_records("json"))
        self.clear_button.clicked.connect(self.clear_records)
        export_row.addWidget(self.export_csv_button)
        export_row.addWidget(self.export_json_button)
        parent_layout.addLayout(export_row)
        parent_layout.addWidget(self.clear_button)

    def refresh_model_list(self):
        current_path = self.model_combo.currentData()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        models = find_models()
        if models:
            for path in models:
                self.model_combo.addItem(path.name, str(path))
        else:
            self.model_combo.addItem("未发现模型，请手动选择", "")

        if current_path:
            index = self.model_combo.findData(current_path)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)
        self.model_selection_changed()

    def model_selection_changed(self):
        self.load_model_button.setEnabled(bool(self.model_combo.currentData()))

    def browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 YOLO 模型",
            str(APP_DIR),
            MODEL_FILE_FILTER,
        )
        if not path:
            return
        index = self.model_combo.findData(path)
        if index < 0:
            self.model_combo.addItem(Path(path).name, path)
            index = self.model_combo.count() - 1
        self.model_combo.setCurrentIndex(index)
        self.load_model()

    def load_model(self):
        model_path = self.model_combo.currentData()
        if not model_path:
            self.log("没有可加载的模型，请点击“选择模型”。")
            return

        try:
            self.status_value.setText("加载中")
            QtWidgets.QApplication.processEvents()
            self.detector.load(model_path)
            self.log(f"模型加载成功：{self.detector.model_path}")
            self.status_value.setText("模型已加载")
        except Exception as exc:
            self.status_value.setText("加载失败")
            self.log(f"模型加载失败：{exc}")
            QtWidgets.QMessageBox.critical(self, "模型加载失败", str(exc))
        self.update_controls()

    def update_controls(self):
        has_model = self.detector.is_loaded
        self.image_button.setEnabled(has_model)
        self.video_button.setEnabled(has_model)
        self.camera_button.setEnabled(has_model)
        self.stop_button.setEnabled(self.timer.isActive())
        has_records = bool(self.records)
        self.export_csv_button.setEnabled(has_records)
        self.export_json_button.setEnabled(has_records)
        self.clear_button.setEnabled(has_records)

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片", str(APP_DIR), IMAGE_FILTER
        )
        if not path:
            return
        self.stop_detection(reset_views=False)
        self.source_type = "image"
        self.source_path = Path(path)
        frame = cv2.imread(path)
        if frame is None:
            self.log(f"图片读取失败：{path}")
            return
        self.show_frame(self.original_view, frame)
        self.detect_frame(frame, source=str(self.source_path), frame_index=1)
        self.frame_index = 1
        self.frame_value.setText("1")
        self.status_value.setText("图片完成")
        self.update_controls()

    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频", str(APP_DIR), VIDEO_FILTER
        )
        if not path:
            return
        self.stop_detection(reset_views=False)
        self.source_type = "video"
        self.source_path = Path(path)
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            self.log(f"视频打开失败：{path}")
            self.capture = None
            return
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 25
        self.frame_index = 0
        self.last_tick.restart()
        self.timer.start(max(1, int(1000 / fps)))
        self.status_value.setText("视频检测中")
        self.log(f"开始视频检测：{path}")
        self.update_controls()

    def open_camera(self):
        self.stop_detection(reset_views=False)
        self.source_type = "camera"
        self.source_path = None
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.log("摄像头打开失败。请检查系统摄像头权限。")
            self.capture = None
            return
        self.frame_index = 0
        self.last_tick.restart()
        self.timer.start(30)
        self.status_value.setText("实时检测中")
        self.log("开始摄像头实时检测。")
        self.update_controls()

    def process_next_frame(self):
        if self.capture is None:
            self.stop_detection()
            return

        ok, frame = self.capture.read()
        if not ok:
            self.log("视频检测完成。")
            self.stop_detection(reset_views=False)
            self.status_value.setText("检测完成")
            return

        self.frame_index += 1
        self.frame_value.setText(str(self.frame_index))
        elapsed = max(1, self.last_tick.restart())
        self.fps_value.setText(f"{1000 / elapsed:.1f}")
        self.show_frame(self.original_view, frame)
        source = str(self.source_path) if self.source_path else "camera"
        self.detect_frame(frame, source=source, frame_index=self.frame_index)

    def detect_frame(self, frame, source: str, frame_index: int):
        try:
            result = self.detector.predict(frame, self.detection_options())
        except Exception as exc:
            self.log(f"检测失败：{exc}")
            self.stop_detection(reset_views=False)
            QtWidgets.QMessageBox.critical(self, "检测失败", str(exc))
            return

        self.show_frame(self.detected_view, result.plot())
        frame_records = records_from_result(
            result,
            source=source,
            source_type=self.source_type,
            frame_index=frame_index,
        )
        self.add_records(frame_records)

    def detection_options(self) -> DetectionOptions:
        return DetectionOptions(
            confidence=self.conf_spin.value(),
            image_size=self.imgsz_spin.value(),
            device=resolve_device(self.device_combo.currentText()),
        )

    def add_records(self, frame_records: list[dict]):
        self.current_counts = Counter(record["class_name"] for record in frame_records)
        self.total_counts.update(self.current_counts)
        self.records.extend(frame_records)
        self.current_value.setText(str(sum(self.current_counts.values())))
        self.total_value.setText(str(sum(self.total_counts.values())))
        self.update_count_table()
        self.update_controls()

    def update_count_table(self):
        classes = sorted(set(self.current_counts) | set(self.total_counts))
        self.count_table.setRowCount(len(classes))
        for row, class_name in enumerate(classes):
            values = [
                class_name,
                str(self.current_counts.get(class_name, 0)),
                str(self.total_counts.get(class_name, 0)),
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.count_table.setItem(row, col, item)

    def show_frame(self, label: VideoLabel, frame):
        pixmap = frame_to_pixmap(frame)
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def stop_detection(self, reset_views=True):
        if self.timer.isActive():
            self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if reset_views:
            self.original_view.clear_frame()
            self.detected_view.clear_frame()
        if self.status_value.text() in {"视频检测中", "实时检测中"}:
            self.status_value.setText("已停止")
            self.log("检测已停止。")
        self.update_controls()

    def clear_records(self):
        self.records.clear()
        self.current_counts.clear()
        self.total_counts.clear()
        self.frame_index = 0
        self.frame_value.setText("0")
        self.current_value.setText("0")
        self.total_value.setText("0")
        self.fps_value.setText("0.0")
        self.count_table.setRowCount(0)
        self.log("统计结果已清空。")
        self.update_controls()

    def export_records(self, file_type: str):
        if not self.records:
            self.log("暂无检测结果可导出。")
            return

        EXPORT_DIR.mkdir(exist_ok=True)
        default_name = EXPORT_DIR / f"yolo_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_type}"
        filter_text = "CSV (*.csv)" if file_type == "csv" else "JSON (*.json)"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出检测结果", str(default_name), filter_text
        )
        if not path:
            return

        try:
            if file_type == "csv":
                export_csv(path, self.records)
            else:
                model_path = str(self.detector.model_path) if self.detector.model_path else ""
                export_json(path, self.records, model_path, self.total_counts)
            self.log(f"检测结果已导出：{path}")
        except Exception as exc:
            self.log(f"导出失败：{exc}")
            QtWidgets.QMessageBox.critical(self, "导出失败", str(exc))

    def closeEvent(self, event):
        self.stop_detection(reset_views=False)
        super().closeEvent(event)

    def log(self, message: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{stamp}] {message}")
