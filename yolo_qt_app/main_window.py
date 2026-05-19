import json
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt

from .app_logging import get_app_logger
from .config import (
    APP_DIR,
    BATCH_IMAGE_FILTER,
    CONFIG_DIR,
    EXPORT_DIR,
    EXPORT_FILTER,
    IMAGE_FILTER,
    MODEL_FILE_FILTER,
    VIDEO_FILTER,
)
from .detector import (
    DetectionOptions,
    YoloDetector,
    available_devices,
    records_from_result,
    resolve_device,
)
from .exporter import (
    detect_export_type,
    export_csv,
    export_json,
    export_tsv,
    export_txt_summary,
)
from .history import append_history, load_history_items
from .image_utils import frame_to_pixmap
from .model_finder import find_models
from .styles import APP_STYLE
from .widgets import VideoLabel, section_label, stat_value


class InferenceWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object, str, int, float)
    failed = QtCore.pyqtSignal(str, str, int)

    def __init__(self, detector: YoloDetector, frame, options: DetectionOptions, source: str, frame_index: int):
        super().__init__()
        self.detector = detector
        self.frame = frame
        self.options = options
        self.source = source
        self.frame_index = frame_index

    @QtCore.pyqtSlot()
    def run(self):
        start = time.perf_counter()
        try:
            result = self.detector.predict(self.frame, self.options)
        except Exception as exc:
            self.failed.emit(str(exc), self.source, self.frame_index)
            return
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.finished.emit(result, self.frame, self.source, self.frame_index, elapsed_ms)


class YoloMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = get_app_logger()
        self.detector = YoloDetector()
        self.capture = None
        self.source_path: Path | None = None
        self.source_type = ""
        self.frame_index = 0
        self.last_tick = QtCore.QElapsedTimer()
        self.records: list[dict] = []
        self.current_counts = Counter()
        self.total_counts = Counter()
        self.session_id = self.new_session_id()
        self.session_started_at = datetime.now()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.process_next_frame)
        self.infer_thread: QtCore.QThread | None = None
        self.infer_worker: InferenceWorker | None = None
        self.infer_busy = False
        self.pending_frame_data = None

        self.batch_queue: list[Path] = []
        self.batch_mode = False
        self.last_infer_ms = 0.0
        self.last_processed_time = 0.0

        self.build_ui()
        self.refresh_model_list()
        self.update_controls()
        self.run_first_self_check()

    def new_session_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def build_ui(self):
        self.setWindowTitle("YOLO 实时检测与统计导出")
        self.resize(1440, 860)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        header = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("YOLO 检测工作台")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel("图片、视频、摄像头实时检测，支持批量、对比、导出与历史回放")
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
        self.log_box.setMinimumHeight(170)
        self.log_box.setObjectName("LogBox")
        left_panel.addWidget(self.log_box)

        right_panel = QtWidgets.QWidget()
        right_panel.setObjectName("SidePanel")
        right_panel.setFixedWidth(410)
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

        perf_row = QtWidgets.QHBoxLayout()
        self.frame_policy_combo = QtWidgets.QComboBox()
        self.frame_policy_combo.addItems(["实时优先(忙时丢帧)", "全帧优先(排队处理)"])
        self.max_fps_spin = QtWidgets.QSpinBox()
        self.max_fps_spin.setRange(1, 120)
        self.max_fps_spin.setValue(30)
        perf_row.addWidget(QtWidgets.QLabel("策略"))
        perf_row.addWidget(self.frame_policy_combo, 1)
        perf_row.addWidget(QtWidgets.QLabel("限帧"))
        perf_row.addWidget(self.max_fps_spin)
        parent_layout.addLayout(perf_row)

    def build_input_panel(self, parent_layout):
        parent_layout.addWidget(section_label("输入"))
        input_grid = QtWidgets.QGridLayout()
        self.image_button = QtWidgets.QPushButton("图片检测")
        self.video_button = QtWidgets.QPushButton("视频检测")
        self.camera_button = QtWidgets.QPushButton("摄像头实时")
        self.batch_button = QtWidgets.QPushButton("批量图片/目录")
        self.compare_button = QtWidgets.QPushButton("多模型对比")
        self.stop_button = QtWidgets.QPushButton("停止")
        self.image_button.clicked.connect(self.open_image)
        self.video_button.clicked.connect(self.open_video)
        self.camera_button.clicked.connect(self.open_camera)
        self.batch_button.clicked.connect(self.open_batch)
        self.compare_button.clicked.connect(self.compare_models)
        self.stop_button.clicked.connect(self.stop_detection)
        input_grid.addWidget(self.image_button, 0, 0)
        input_grid.addWidget(self.video_button, 0, 1)
        input_grid.addWidget(self.camera_button, 1, 0)
        input_grid.addWidget(self.batch_button, 1, 1)
        input_grid.addWidget(self.compare_button, 2, 0)
        input_grid.addWidget(self.stop_button, 2, 1)
        parent_layout.addLayout(input_grid)

        cfg_row = QtWidgets.QHBoxLayout()
        self.save_cfg_button = QtWidgets.QPushButton("保存配置")
        self.load_cfg_button = QtWidgets.QPushButton("加载配置")
        self.save_cfg_button.clicked.connect(self.save_task_config)
        self.load_cfg_button.clicked.connect(self.load_task_config)
        cfg_row.addWidget(self.save_cfg_button)
        cfg_row.addWidget(self.load_cfg_button)
        parent_layout.addLayout(cfg_row)

        history_row = QtWidgets.QHBoxLayout()
        self.replay_button = QtWidgets.QPushButton("历史回放")
        self.replay_button.clicked.connect(self.show_history_summary)
        history_row.addWidget(self.replay_button)
        parent_layout.addLayout(history_row)

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
        self.export_button = QtWidgets.QPushButton("导出结果")
        self.clear_button = QtWidgets.QPushButton("清空统计")
        self.export_button.clicked.connect(self.export_records)
        self.clear_button.clicked.connect(self.clear_records)
        export_row.addWidget(self.export_button)
        export_row.addWidget(self.clear_button)
        parent_layout.addLayout(export_row)

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
            self.log("没有可加载的模型，请点击“选择模型”。", "warning")
            return

        try:
            self.status_value.setText("加载中")
            QtWidgets.QApplication.processEvents()
            self.detector.load(model_path)
            self.log(f"模型加载成功：{self.detector.model_path}")
            self.status_value.setText("模型已加载")
        except FileNotFoundError as exc:
            self.status_value.setText("加载失败")
            self.log(f"模型文件不存在：{exc}", "error")
            QtWidgets.QMessageBox.critical(self, "模型加载失败", f"模型文件不存在：{exc}")
        except RuntimeError as exc:
            self.status_value.setText("加载失败")
            self.log(f"模型运行环境错误：{exc}", "error")
            QtWidgets.QMessageBox.critical(self, "模型加载失败", f"运行环境错误：{exc}")
        except Exception as exc:
            self.status_value.setText("加载失败")
            self.log(f"模型加载失败：{exc}", "error")
            QtWidgets.QMessageBox.critical(self, "模型加载失败", str(exc))
        self.update_controls()

    def update_controls(self):
        has_model = self.detector.is_loaded
        busy = self.infer_busy
        self.image_button.setEnabled(has_model and not busy)
        self.video_button.setEnabled(has_model and not busy)
        self.camera_button.setEnabled(has_model and not busy)
        self.batch_button.setEnabled(has_model and not busy)
        self.compare_button.setEnabled(not busy)
        self.stop_button.setEnabled(self.timer.isActive() or busy or self.batch_mode)
        has_records = bool(self.records)
        self.export_button.setEnabled(has_records and not busy)
        self.clear_button.setEnabled(has_records and not busy)

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片", str(APP_DIR), IMAGE_FILTER)
        if not path:
            return
        self.stop_detection(reset_views=False)
        self.source_type = "image"
        self.source_path = Path(path)
        frame = cv2.imread(path)
        if frame is None:
            self.log(f"图片读取失败：{path}", "error")
            QtWidgets.QMessageBox.warning(self, "图片读取失败", f"无法读取图片：{path}")
            return
        self.show_frame(self.original_view, frame)
        self.frame_index = 1
        self.frame_value.setText("1")
        self.status_value.setText("图片推理中")
        self.start_inference(frame, source=str(self.source_path), frame_index=1)

    def open_batch(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片目录", str(APP_DIR))
        image_paths: list[Path] = []
        if directory:
            image_paths.extend(self.collect_images_from_dir(Path(directory)))
        else:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "选择多张图片", str(APP_DIR), BATCH_IMAGE_FILTER)
            image_paths.extend(Path(p) for p in paths)
        image_paths = [p for p in image_paths if p.exists()]
        if not image_paths:
            self.log("未选择可用的批量图片。", "warning")
            return

        self.stop_detection(reset_views=False)
        self.source_type = "batch_image"
        self.batch_queue = image_paths
        self.batch_mode = True
        self.frame_index = 0
        self.status_value.setText("批量检测中")
        self.log(f"开始批量检测，共 {len(self.batch_queue)} 张图片。")
        self.start_next_batch_image()

    def collect_images_from_dir(self, directory: Path) -> list[Path]:
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted([p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in suffixes])

    def start_next_batch_image(self):
        if not self.batch_queue:
            self.batch_mode = False
            self.status_value.setText("批量完成")
            self.log("批量检测完成。")
            self.update_controls()
            return

        path = self.batch_queue.pop(0)
        frame = cv2.imread(str(path))
        if frame is None:
            self.log(f"批量跳过无法读取的图片：{path}", "warning")
            self.start_next_batch_image()
            return

        self.source_path = path
        self.frame_index += 1
        self.frame_value.setText(str(self.frame_index))
        self.show_frame(self.original_view, frame)
        self.start_inference(frame, source=str(path), frame_index=self.frame_index)

    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频", str(APP_DIR), VIDEO_FILTER)
        if not path:
            return
        self.stop_detection(reset_views=False)
        self.source_type = "video"
        self.source_path = Path(path)
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            self.log(f"视频打开失败：{path}", "error")
            self.capture = None
            QtWidgets.QMessageBox.critical(self, "视频打开失败", f"无法打开视频：{path}")
            return
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 25
        target_fps = min(int(fps), self.max_fps_spin.value())
        self.frame_index = 0
        self.last_tick.restart()
        self.last_processed_time = 0.0
        self.timer.start(max(1, int(1000 / max(1, target_fps))))
        self.status_value.setText("视频检测中")
        self.log(f"开始视频检测：{path}")
        self.update_controls()

    def open_camera(self):
        self.stop_detection(reset_views=False)
        self.source_type = "camera"
        self.source_path = None
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.log("摄像头打开失败。请检查系统摄像头权限。", "error")
            self.capture = None
            QtWidgets.QMessageBox.critical(self, "摄像头打开失败", "请检查设备连接与系统权限。")
            return
        self.frame_index = 0
        self.last_tick.restart()
        self.last_processed_time = 0.0
        self.timer.start(max(1, int(1000 / self.max_fps_spin.value())))
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

        now = time.perf_counter()
        min_interval = 1.0 / max(1, self.max_fps_spin.value())
        if self.last_processed_time and now - self.last_processed_time < min_interval:
            return

        self.frame_index += 1
        self.frame_value.setText(str(self.frame_index))
        elapsed = max(1, self.last_tick.restart())
        self.fps_value.setText(f"{1000 / elapsed:.1f}")
        self.show_frame(self.original_view, frame)
        source = str(self.source_path) if self.source_path else "camera"

        if self.infer_busy:
            if self.frame_policy_combo.currentIndex() == 0:
                return
            self.pending_frame_data = (frame, source, self.frame_index)
            return

        self.last_processed_time = now
        self.start_inference(frame, source=source, frame_index=self.frame_index)

    def start_inference(self, frame, source: str, frame_index: int):
        if self.infer_busy:
            self.pending_frame_data = (frame, source, frame_index)
            return

        self.infer_busy = True
        self.update_controls()
        self.infer_thread = QtCore.QThread(self)
        self.infer_worker = InferenceWorker(
            self.detector,
            frame,
            self.detection_options(),
            source,
            frame_index,
        )
        self.infer_worker.moveToThread(self.infer_thread)
        self.infer_thread.started.connect(self.infer_worker.run)
        self.infer_worker.finished.connect(self.on_inference_finished)
        self.infer_worker.failed.connect(self.on_inference_failed)
        self.infer_worker.finished.connect(self.infer_thread.quit)
        self.infer_worker.failed.connect(self.infer_thread.quit)
        self.infer_thread.finished.connect(self.cleanup_inference_thread)
        self.infer_thread.start()

    def on_inference_finished(self, result, frame, source: str, frame_index: int, elapsed_ms: float):
        self.last_infer_ms = elapsed_ms
        self.show_frame(self.detected_view, result.plot())
        frame_records = records_from_result(
            result,
            source=source,
            source_type=self.source_type,
            frame_index=frame_index,
        )
        self.add_records(frame_records)
        if self.source_type == "image":
            self.status_value.setText("图片完成")
        elif self.source_type == "batch_image":
            self.status_value.setText("批量检测中")
            QtCore.QTimer.singleShot(1, self.start_next_batch_image)

    def on_inference_failed(self, message: str, source: str, frame_index: int):
        self.log(f"检测失败：source={source}, frame={frame_index}, error={message}", "error")
        self.stop_detection(reset_views=False)
        QtWidgets.QMessageBox.critical(self, "检测失败", message)

    def cleanup_inference_thread(self):
        self.infer_worker = None
        self.infer_thread = None
        self.infer_busy = False
        self.update_controls()
        if self.pending_frame_data and not self.batch_mode:
            frame, source, frame_index = self.pending_frame_data
            self.pending_frame_data = None
            self.start_inference(frame, source, frame_index)

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
        self.batch_mode = False
        self.batch_queue = []
        self.pending_frame_data = None
        if self.timer.isActive():
            self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.infer_thread is not None and self.infer_thread.isRunning():
            self.infer_thread.quit()
            self.infer_thread.wait(1500)
        if reset_views:
            self.original_view.clear_frame()
            self.detected_view.clear_frame()
        if self.status_value.text() in {"视频检测中", "实时检测中", "批量检测中"}:
            self.status_value.setText("已停止")
            self.log("检测已停止。")
        self.update_controls()

    def clear_records(self):
        self.records.clear()
        self.current_counts.clear()
        self.total_counts.clear()
        self.session_id = self.new_session_id()
        self.session_started_at = datetime.now()
        self.frame_index = 0
        self.frame_value.setText("0")
        self.current_value.setText("0")
        self.total_value.setText("0")
        self.fps_value.setText("0.0")
        self.count_table.setRowCount(0)
        self.log("统计结果已清空。")
        self.update_controls()

    def session_metadata(self) -> dict:
        return {
            "session_id": self.session_id,
            "model": str(self.detector.model_path) if self.detector.model_path else "",
            "confidence": self.conf_spin.value(),
            "image_size": self.imgsz_spin.value(),
            "device": self.device_combo.currentText(),
            "source_type": self.source_type,
            "started_at": self.session_started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame_count": self.frame_index,
            "frame_policy": self.frame_policy_combo.currentText(),
            "max_fps": self.max_fps_spin.value(),
            "last_infer_ms": round(self.last_infer_ms, 3),
        }

    def export_records(self):
        if not self.records:
            self.log("暂无检测结果可导出。", "warning")
            return

        EXPORT_DIR.mkdir(exist_ok=True)
        default_name = EXPORT_DIR / f"yolo_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出检测结果", str(default_name), EXPORT_FILTER
        )
        if not path:
            return

        if "." not in Path(path).name:
            if "CSV" in selected_filter:
                path += ".csv"
            elif "TSV" in selected_filter:
                path += ".tsv"
            elif "TXT" in selected_filter:
                path += ".txt"
            else:
                path += ".json"

        metadata = self.session_metadata()
        try:
            export_type = detect_export_type(path)
            if export_type == "csv":
                export_csv(path, self.records, metadata)
            elif export_type == "json":
                model_path = str(self.detector.model_path) if self.detector.model_path else ""
                export_json(path, self.records, model_path, self.total_counts, metadata)
            elif export_type == "tsv":
                export_tsv(path, self.records, metadata)
            else:
                export_txt_summary(path, self.records, self.total_counts, metadata)
            append_history(path, export_type, self.total_counts, metadata)
            self.log(f"检测结果已导出：{path}")
        except PermissionError as exc:
            self.log(f"导出失败（权限不足）：{exc}", "error")
            QtWidgets.QMessageBox.critical(self, "导出失败", f"没有写入权限：{exc}")
        except OSError as exc:
            self.log(f"导出失败（文件系统异常）：{exc}", "error")
            QtWidgets.QMessageBox.critical(self, "导出失败", f"文件系统异常：{exc}")
        except Exception as exc:
            self.log(f"导出失败：{exc}", "error")
            QtWidgets.QMessageBox.critical(self, "导出失败", str(exc))

    def save_task_config(self):
        CONFIG_DIR.mkdir(exist_ok=True)
        default = CONFIG_DIR / f"task_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存任务配置", str(default), "JSON (*.json)")
        if not path:
            return
        config = {
            "model_path": self.model_combo.currentData() or "",
            "confidence": self.conf_spin.value(),
            "image_size": self.imgsz_spin.value(),
            "device": self.device_combo.currentText(),
            "frame_policy_index": self.frame_policy_combo.currentIndex(),
            "max_fps": self.max_fps_spin.value(),
        }
        try:
            Path(path).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            self.log(f"任务配置已保存：{path}")
        except Exception as exc:
            self.log(f"保存配置失败：{exc}", "error")
            QtWidgets.QMessageBox.critical(self, "保存配置失败", str(exc))

    def load_task_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "加载任务配置", str(CONFIG_DIR), "JSON (*.json)")
        if not path:
            return
        try:
            config = json.loads(Path(path).read_text(encoding="utf-8"))
            model_path = config.get("model_path", "")
            if model_path:
                index = self.model_combo.findData(model_path)
                if index < 0:
                    self.model_combo.addItem(Path(model_path).name, model_path)
                    index = self.model_combo.count() - 1
                self.model_combo.setCurrentIndex(index)
            self.conf_spin.setValue(float(config.get("confidence", self.conf_spin.value())))
            self.imgsz_spin.setValue(int(config.get("image_size", self.imgsz_spin.value())))
            dev = str(config.get("device", self.device_combo.currentText()))
            dev_index = self.device_combo.findText(dev)
            if dev_index >= 0:
                self.device_combo.setCurrentIndex(dev_index)
            self.frame_policy_combo.setCurrentIndex(int(config.get("frame_policy_index", 0)))
            self.max_fps_spin.setValue(int(config.get("max_fps", self.max_fps_spin.value())))
            self.log(f"任务配置已加载：{path}")
        except Exception as exc:
            self.log(f"加载配置失败：{exc}", "error")
            QtWidgets.QMessageBox.critical(self, "加载配置失败", str(exc))

    def compare_models(self):
        model_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "选择用于对比的多个模型", str(APP_DIR), MODEL_FILE_FILTER
        )
        if len(model_paths) < 2:
            self.log("多模型对比至少需要选择两个模型。", "warning")
            return
        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择用于对比的图片", str(APP_DIR), IMAGE_FILTER)
        if not image_path:
            return

        frame = cv2.imread(image_path)
        if frame is None:
            self.log(f"对比图片读取失败：{image_path}", "error")
            return

        results = []
        for model_path in model_paths:
            try:
                det = YoloDetector()
                det.load(model_path)
                start = time.perf_counter()
                result = det.predict(frame, self.detection_options())
                ms = (time.perf_counter() - start) * 1000
                records = records_from_result(result, source=image_path, source_type="compare", frame_index=1)
                avg_conf = round(sum(r["confidence"] for r in records) / len(records), 4) if records else 0.0
                results.append((Path(model_path).name, ms, len(records), avg_conf))
            except Exception as exc:
                results.append((Path(model_path).name, -1.0, 0, 0.0))
                self.log(f"模型对比失败：{model_path} -> {exc}", "error")

        lines = ["模型\t耗时(ms)\t检测数\t平均置信度"]
        for name, ms, count, avg_conf in sorted(results, key=lambda x: (x[1] < 0, x[1])):
            ms_text = "失败" if ms < 0 else f"{ms:.2f}"
            lines.append(f"{name}\t{ms_text}\t{count}\t{avg_conf:.4f}")
        text = "\n".join(lines)
        self.log("多模型对比完成。")
        QtWidgets.QMessageBox.information(self, "多模型对比结果", text)

    def show_history_summary(self):
        items = load_history_items(limit=100)
        if not items:
            self.log("暂无历史导出记录。", "warning")
            return

        overall_counts = Counter()
        by_day = Counter()
        for item in items:
            overall_counts.update(item.get("counts", {}))
            day = str(item.get("time", ""))[:10]
            by_day[day] += int(item.get("total", 0))

        lines = [f"最近历史记录：{len(items)} 条", "", "按类别统计："]
        for cls, cnt in overall_counts.most_common():
            lines.append(f"- {cls}: {cnt}")
        lines.extend(["", "按日期统计："])
        for day, cnt in sorted(by_day.items()):
            lines.append(f"- {day}: {cnt}")
        lines.extend(["", "最近 5 条导出文件："])
        for item in items[:5]:
            lines.append(f"- {item.get('time')} | {item.get('type')} | {item.get('path')}")

        QtWidgets.QMessageBox.information(self, "历史回放与统计", "\n".join(lines))

    def run_first_self_check(self):
        marker = APP_DIR / ".first_run_checked"
        if marker.exists():
            return

        checks = []
        try:
            import ultralytics  # noqa: F401
            checks.append("依赖检查: ultralytics 可用")
        except Exception as exc:
            checks.append(f"依赖检查: ultralytics 不可用 ({exc})")

        try:
            import torch
            checks.append(f"GPU检查: cuda={'是' if torch.cuda.is_available() else '否'}")
        except Exception as exc:
            checks.append(f"GPU检查失败: {exc}")

        models = find_models()
        checks.append(f"模型路径检查: 发现 {len(models)} 个可用模型")
        for line in checks:
            self.log(line)

        QtWidgets.QMessageBox.information(self, "首次运行自检", "\n".join(checks))
        marker.write_text(datetime.now().isoformat(), encoding="utf-8")

    def closeEvent(self, event):
        self.stop_detection(reset_views=False)
        super().closeEvent(event)

    def log(self, message: str, level: str = "info"):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{stamp}] {message}")
        if level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)
