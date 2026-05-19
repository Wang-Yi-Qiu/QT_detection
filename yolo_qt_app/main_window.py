import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt

from .config import (
    APP_DIR,
    CONFIG_DIR,
    EXPORT_DIR,
    IMAGE_EXTENSIONS,
    IMAGE_FILTER,
    LOG_DIR,
    TASK_CONFIG_FILE,
    VIDEO_FILTER,
)
from .detector import (
    DetectionOptions,
    YoloDetector,
    available_devices,
    compare_models,
    records_from_result,
    resolve_device,
)
from .exporter import export_csv, export_json
from .history import summarize_by_class, summarize_by_hour
from .image_utils import frame_to_pixmap
from .model_finder import find_models
from .first_run import is_first_run, mark_first_run_complete
from .self_check import run_startup_self_check
from .session import SessionMetadata
from .styles import APP_STYLE
from .task_config import load_task_config, save_task_config
from .widgets import VideoLabel, section_label, stat_value


class InferenceWorker(QtCore.QObject):
    done = QtCore.pyqtSignal(object, str, int)
    failed = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.detector = YoloDetector()
        self.loaded_model_path = ""

    @QtCore.pyqtSlot(object, object, str, str, int)
    def infer(self, frame, options: DetectionOptions, model_path: str, source: str, frame_index: int):
        try:
            if not model_path:
                raise RuntimeError("未选择模型。")
            if self.loaded_model_path != model_path or not self.detector.is_loaded:
                self.detector.load(model_path)
                self.loaded_model_path = model_path
            result = self.detector.predict(frame, options)
            self.done.emit(result, source, frame_index)
        except Exception as exc:
            self.failed.emit(str(exc))


class YoloMainWindow(QtWidgets.QMainWindow):
    infer_requested = QtCore.pyqtSignal(object, object, str, str, int)

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
        self.session_meta: SessionMetadata | None = None

        self.infer_busy = False
        self.batch_running = False
        self.last_infer_time = 0.0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.process_next_frame)

        self._setup_logger()
        self._setup_inference_thread()

        self.build_ui()
        self.refresh_model_list()
        self.run_startup_check()
        self.update_controls()

    def _setup_logger(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("yolo_qt_app")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _setup_inference_thread(self):
        self.infer_thread = QtCore.QThread(self)
        self.infer_worker = InferenceWorker()
        self.infer_worker.moveToThread(self.infer_thread)
        self.infer_requested.connect(
            self.infer_worker.infer,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        self.infer_worker.done.connect(self.on_inference_done)
        self.infer_worker.failed.connect(self.on_inference_failed)
        self.infer_thread.start()

    def build_ui(self):
        self.setWindowTitle("YOLO 实时检测与统计导出")
        self.resize(1420, 860)

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
        parent_layout.addWidget(section_label("模型与参数"))
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

        stream_row = QtWidgets.QHBoxLayout()
        self.frame_skip_spin = QtWidgets.QSpinBox()
        self.frame_skip_spin.setRange(0, 15)
        self.frame_skip_spin.setValue(0)
        self.max_fps_spin = QtWidgets.QSpinBox()
        self.max_fps_spin.setRange(1, 120)
        self.max_fps_spin.setValue(30)
        stream_row.addWidget(QtWidgets.QLabel("跳帧"))
        stream_row.addWidget(self.frame_skip_spin)
        stream_row.addWidget(QtWidgets.QLabel("最大FPS"))
        stream_row.addWidget(self.max_fps_spin)
        parent_layout.addLayout(stream_row)

        cfg_row = QtWidgets.QHBoxLayout()
        self.save_cfg_button = QtWidgets.QPushButton("保存配置")
        self.load_cfg_button = QtWidgets.QPushButton("加载配置")
        self.compare_button = QtWidgets.QPushButton("模型对比")
        self.save_cfg_button.clicked.connect(self.save_current_config)
        self.load_cfg_button.clicked.connect(self.load_saved_config)
        self.compare_button.clicked.connect(self.compare_models_on_image)
        cfg_row.addWidget(self.save_cfg_button)
        cfg_row.addWidget(self.load_cfg_button)
        cfg_row.addWidget(self.compare_button)
        parent_layout.addLayout(cfg_row)

    def build_input_panel(self, parent_layout):
        parent_layout.addWidget(section_label("输入"))
        input_grid = QtWidgets.QGridLayout()
        self.image_button = QtWidgets.QPushButton("图片检测")
        self.batch_image_button = QtWidgets.QPushButton("目录批量")
        self.video_button = QtWidgets.QPushButton("视频检测")
        self.camera_button = QtWidgets.QPushButton("摄像头实时")
        self.stop_button = QtWidgets.QPushButton("停止")
        self.image_button.clicked.connect(self.open_image)
        self.batch_image_button.clicked.connect(self.open_image_dir)
        self.video_button.clicked.connect(self.open_video)
        self.camera_button.clicked.connect(self.open_camera)
        self.stop_button.clicked.connect(self.stop_detection)
        input_grid.addWidget(self.image_button, 0, 0)
        input_grid.addWidget(self.batch_image_button, 0, 1)
        input_grid.addWidget(self.video_button, 1, 0)
        input_grid.addWidget(self.camera_button, 1, 1)
        input_grid.addWidget(self.stop_button, 2, 0, 1, 2)
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
        parent_layout.addWidget(section_label("导出与回放"))
        export_row = QtWidgets.QHBoxLayout()
        self.export_csv_button = QtWidgets.QPushButton("导出 CSV")
        self.export_json_button = QtWidgets.QPushButton("导出 JSON")
        self.history_button = QtWidgets.QPushButton("回放 JSON")
        self.clear_button = QtWidgets.QPushButton("清空统计")
        self.export_csv_button.clicked.connect(lambda: self.export_records("csv"))
        self.export_json_button.clicked.connect(lambda: self.export_records("json"))
        self.history_button.clicked.connect(self.replay_history)
        self.clear_button.clicked.connect(self.clear_records)
        export_row.addWidget(self.export_csv_button)
        export_row.addWidget(self.export_json_button)
        parent_layout.addLayout(export_row)
        parent_layout.addWidget(self.history_button)
        parent_layout.addWidget(self.clear_button)
    def run_startup_check(self):
        if not is_first_run():
            self.log("首次启动自检已完成，跳过自动检查。")
            return

        report = run_startup_self_check()
        dep_text = ", ".join(f"{name}:{'OK' if ok else '缺失'}" for name, ok in report["dependencies"].items())
        self.log(f"首次启动自检 - 依赖: {dep_text}")
        self.log(f"首次启动自检 - 设备: {', '.join(report['devices'])}")
        self.log(f"首次启动自检 - GPU: {', '.join(report['gpu_devices']) if report['gpu_devices'] else '未发现可用 GPU'}")
        self.log(f"首次启动自检 - 发现模型: {report['models_found']}")

        missing = [name for name, ok in report["dependencies"].items() if not ok]
        bad_models = [item["path"] for item in report["model_paths"] if not item["readable"]]
        issues = []
        if missing:
            issues.append(f"依赖缺失: {', '.join(missing)}")
        if report["models_found"] == 0:
            issues.append("未发现模型文件")
        if bad_models:
            issues.append(f"模型路径不可读: {', '.join(bad_models)}")

        if issues:
            QtWidgets.QMessageBox.warning(self, "首次运行自检", "\n".join(issues))
        else:
            QtWidgets.QMessageBox.information(
                self,
                "首次运行自检",
                f"依赖、设备与模型路径检查通过。\n"
                f"GPU: {', '.join(report['gpu_devices']) if report['gpu_devices'] else '未发现可用 GPU'}\n"
                f"模型数量: {report['models_found']}",
            )

        mark_first_run_complete()

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
            "YOLO Models (*.pt *.onnx *.engine)",
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
            QtWidgets.QMessageBox.critical(self, "模型加载失败", f"模型路径: {model_path}\n\n错误: {exc}")
        self.update_controls()

    def update_controls(self):
        has_model = self.detector.is_loaded
        can_run_actions = not self.infer_busy and not self.batch_running
        self.image_button.setEnabled(has_model and can_run_actions)
        self.batch_image_button.setEnabled(has_model and can_run_actions)
        self.video_button.setEnabled(has_model and can_run_actions)
        self.camera_button.setEnabled(has_model and can_run_actions)
        self.compare_button.setEnabled(has_model and can_run_actions)
        self.stop_button.setEnabled(self.timer.isActive() or self.batch_running)
        self.load_model_button.setEnabled(bool(self.model_combo.currentData()) and can_run_actions)
        self.browse_model_button.setEnabled(can_run_actions)
        self.refresh_model_button.setEnabled(can_run_actions)
        has_records = bool(self.records)
        self.export_csv_button.setEnabled(has_records)
        self.export_json_button.setEnabled(has_records)
        self.clear_button.setEnabled(has_records)
    def detection_options(self) -> DetectionOptions:
        return DetectionOptions(
            confidence=self.conf_spin.value(),
            image_size=self.imgsz_spin.value(),
            device=resolve_device(self.device_combo.currentText()),
        )

    def start_session(self, source_type: str, source: str):
        model_path = str(self.detector.model_path) if self.detector.model_path else ""
        options = self.detection_options()
        self.session_meta = SessionMetadata(
            model_path=model_path,
            source_type=source_type,
            source=source,
            device=options.device,
            confidence=options.confidence,
            image_size=options.image_size,
            frame_skip=self.frame_skip_spin.value(),
            max_fps=self.max_fps_spin.value(),
        )

    def close_session(self):
        if self.session_meta and not self.session_meta.ended_at:
            self.session_meta.close()

    def submit_inference(self, frame, source: str, frame_index: int):
        model_path = str(self.detector.model_path) if self.detector.model_path else ""
        if not model_path:
            return
        self.infer_busy = True
        self.infer_requested.emit(frame.copy(), self.detection_options(), model_path, source, frame_index)

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
            QtWidgets.QMessageBox.warning(self, "图片读取失败", f"无法读取图片: {path}")
            return
        self.start_session(self.source_type, str(self.source_path))
        self.frame_index = 1
        self.frame_value.setText("1")
        self.show_frame(self.original_view, frame)
        self.status_value.setText("检测中")
        self.submit_inference(frame, source=str(self.source_path), frame_index=1)
        self.update_controls()

    def open_image_dir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片目录", str(APP_DIR))
        if not folder:
            return
        paths = [
            path
            for path in sorted(Path(folder).glob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not paths:
            self.log("目录中没有可检测图片。")
            QtWidgets.QMessageBox.information(self, "目录批量检测", "未发现支持格式的图片。")
            return

        self.stop_detection(reset_views=False)
        self.source_type = "batch_image"
        self.source_path = Path(folder)
        self.start_session(self.source_type, str(self.source_path))
        self.status_value.setText("批量检测中")
        self.log(f"开始批量检测，共 {len(paths)} 张图片。")
        self.batch_running = True
        progress = QtWidgets.QProgressDialog("批量检测中...", "取消", 0, len(paths), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        self.update_controls()

        try:
            for index, path in enumerate(paths, start=1):
                if progress.wasCanceled():
                    self.log("批量检测已取消。")
                    break
                frame = cv2.imread(str(path))
                if frame is None:
                    self.log(f"跳过损坏图片：{path}")
                    progress.setValue(index)
                    continue
                self.frame_index = index
                self.frame_value.setText(str(index))
                self.show_frame(self.original_view, frame)
                try:
                    result = self.detector.predict(frame, self.detection_options())
                except Exception as exc:
                    self.log(f"批量检测失败：{path} -> {exc}")
                    progress.setValue(index)
                    continue
                self.show_frame(self.detected_view, result.plot())
                self.add_records(
                    records_from_result(
                        result,
                        source=str(path),
                        source_type=self.source_type,
                        frame_index=index,
                    )
                )
                progress.setValue(index)
                QtWidgets.QApplication.processEvents()
        finally:
            self.batch_running = False
            progress.close()
            self.update_controls()

        self.close_session()
        self.status_value.setText("批量完成")
        self.log("批量检测完成。")

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
            QtWidgets.QMessageBox.warning(self, "视频打开失败", f"无法打开视频: {path}")
            return
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 25
        self.frame_index = 0
        self.last_tick.restart()
        self.last_infer_time = 0.0
        self.start_session(self.source_type, str(self.source_path))
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
            QtWidgets.QMessageBox.warning(self, "摄像头打开失败", "请检查系统权限或摄像头占用。")
            return
        self.frame_index = 0
        self.last_tick.restart()
        self.last_infer_time = 0.0
        self.start_session(self.source_type, "camera")
        self.timer.start(20)
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

        skip = self.frame_skip_spin.value()
        if (self.frame_index - 1) % (skip + 1) != 0:
            return

        max_fps = self.max_fps_spin.value()
        now = perf_counter()
        if max_fps > 0 and (now - self.last_infer_time) < (1.0 / max_fps):
            return

        if self.infer_busy:
            return

        self.last_infer_time = now
        source = str(self.source_path) if self.source_path else "camera"
        self.submit_inference(frame, source=source, frame_index=self.frame_index)

    @QtCore.pyqtSlot(object, str, int)
    def on_inference_done(self, result, source: str, frame_index: int):
        self.infer_busy = False
        self.show_frame(self.detected_view, result.plot())
        frame_records = records_from_result(
            result,
            source=source,
            source_type=self.source_type,
            frame_index=frame_index,
        )
        self.add_records(frame_records)
        if self.source_type == "image":
            self.close_session()
            self.status_value.setText("图片完成")
        self.update_controls()

    @QtCore.pyqtSlot(str)
    def on_inference_failed(self, error: str):
        self.infer_busy = False
        self.log(f"检测失败：{error}")
        if self.source_type in {"video", "camera"}:
            self.stop_detection(reset_views=False)
        self.status_value.setText("检测失败")
        QtWidgets.QMessageBox.critical(self, "检测失败", error)
        self.update_controls()

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
        self.close_session()
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
                self.close_session()
                model_path = str(self.detector.model_path) if self.detector.model_path else ""
                session_payload = self.session_meta.to_payload() if self.session_meta else {}
                history_payload = {
                    "by_class": summarize_by_class(self.records),
                    "by_hour": summarize_by_hour(self.records),
                }
                export_json(
                    path,
                    self.records,
                    model_path,
                    self.total_counts,
                    session=session_payload,
                    history=history_payload,
                )
            self.log(f"检测结果已导出：{path}")
        except Exception as exc:
            self.log(f"导出失败：{exc}")
            QtWidgets.QMessageBox.critical(self, "导出失败", f"路径: {path}\n\n错误: {exc}")

    def save_current_config(self):
        payload = {
            "model_path": self.model_combo.currentData() or "",
            "confidence": self.conf_spin.value(),
            "image_size": self.imgsz_spin.value(),
            "device": self.device_combo.currentText(),
            "frame_skip": self.frame_skip_spin.value(),
            "max_fps": self.max_fps_spin.value(),
        }
        save_task_config(TASK_CONFIG_FILE, payload)
        self.log(f"配置已保存：{TASK_CONFIG_FILE}")

    def load_saved_config(self):
        if not TASK_CONFIG_FILE.exists():
            self.log("未发现保存配置。")
            QtWidgets.QMessageBox.information(self, "加载配置", "当前还没有保存的配置文件。")
            return
        try:
            cfg = load_task_config(TASK_CONFIG_FILE)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "加载配置失败", str(exc))
            return

        model_path = cfg.get("model_path", "")
        if model_path:
            index = self.model_combo.findData(model_path)
            if index < 0:
                self.model_combo.addItem(Path(model_path).name, model_path)
                index = self.model_combo.count() - 1
            self.model_combo.setCurrentIndex(index)
        self.conf_spin.setValue(float(cfg.get("confidence", 0.5)))
        self.imgsz_spin.setValue(int(cfg.get("image_size", 640)))
        device = cfg.get("device", "auto")
        device_index = self.device_combo.findText(device)
        if device_index >= 0:
            self.device_combo.setCurrentIndex(device_index)
        self.frame_skip_spin.setValue(int(cfg.get("frame_skip", 0)))
        self.max_fps_spin.setValue(int(cfg.get("max_fps", 30)))
        self.log(f"配置已加载：{TASK_CONFIG_FILE}")

    def compare_models_on_image(self):
        if not self.detector.model_path:
            QtWidgets.QMessageBox.warning(self, "模型对比", "请先加载当前模型。")
            return

        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择对比图片", str(APP_DIR), IMAGE_FILTER
        )
        if not image_path:
            return
        frame = cv2.imread(image_path)
        if frame is None:
            QtWidgets.QMessageBox.warning(self, "模型对比", "图片读取失败。")
            return

        model_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择第二个模型",
            str(APP_DIR),
            "YOLO Models (*.pt *.onnx *.engine)",
        )
        if not model_path:
            return

        try:
            results = compare_models(
                [str(self.detector.model_path), model_path],
                frame,
                self.detection_options(),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "模型对比失败", str(exc))
            return

        lines = [
            f"{Path(item['model']).name}: {item['inference_ms']} ms, detections={item['detections']}"
            for item in results
        ]
        summary = "\n".join(lines)
        self.log(f"模型对比结果: {summary.replace(chr(10), ' | ')}")
        QtWidgets.QMessageBox.information(self, "模型对比结果", summary)

    def replay_history(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择历史 JSON",
            str(EXPORT_DIR if EXPORT_DIR.exists() else APP_DIR),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "回放失败", str(exc))
            return

        records = payload.get("records", [])
        if not isinstance(records, list) or not records:
            QtWidgets.QMessageBox.warning(self, "回放", "JSON 中没有可用 records。")
            return

        self.clear_records()
        self.records = records
        self.total_counts = Counter(record.get("class_name", "unknown") for record in records)
        last_frame = max(int(record.get("frame", 0)) for record in records)
        self.current_counts = Counter(
            record.get("class_name", "unknown")
            for record in records
            if int(record.get("frame", 0)) == last_frame
        )
        self.frame_index = last_frame
        self.frame_value.setText(str(last_frame))
        self.current_value.setText(str(sum(self.current_counts.values())))
        self.total_value.setText(str(sum(self.total_counts.values())))
        self.status_value.setText("历史回放")
        self.update_count_table()
        by_class = summarize_by_class(records)
        by_hour = summarize_by_hour(records)
        self.log(f"历史回放完成：{path}")
        self.log(f"历史统计-类别：{by_class}")
        self.log(f"历史统计-时间：{by_hour}")
        self.update_controls()

    def closeEvent(self, event):
        self.stop_detection(reset_views=False)
        self.infer_thread.quit()
        if not self.infer_thread.wait(1000):
            self.log("警告：推理线程未在预期时间内退出。")
        super().closeEvent(event)

    def log(self, message: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{stamp}] {message}")
        self.logger.info(message)
