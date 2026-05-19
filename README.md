# QT_detection

这是一个独立的 YOLO Qt 检测界面项目，不依赖原来的 `QT.py`。

功能包括：

- 加载 YOLO 常见导出模型，包括 `.pt`、`.onnx`、`.engine`、`.rknn` 等
- 图片检测
- 批量图片/目录检测
- 视频检测
- 摄像头实时检测
- 推理后台线程（避免主界面卡顿）
- 视频限帧与丢帧/排队策略
- 多模型对比（同一输入下速度与检测结果对比）
- 任务配置保存/加载（一键复用参数）
- 实时显示原始画面和检测结果
- 按类别统计当前帧和累计目标数量
- 导出 CSV / JSON / TSV / TXT 检测结果
- 导出会话级元数据（模型、参数、设备、时间段等）
- 历史导出回放与统计汇总（按类别、按时间）
- 首次运行自检（依赖、GPU、模型路径）
- 基础日志落盘（`logs/app.log`）

## 运行

进入本目录后运行：

```bash
python run_yolo_qt.py
```

如果你使用 conda 的 `ai_model` 环境，可以先激活环境：

```bash
conda activate ai_model
python run_yolo_qt.py
```

也可以不激活，直接运行：

```bash
conda run -n ai_model python run_yolo_qt.py
```

## 测试资源

项目内已放入一个 YOLO26 基础模型和两张测试图片：

```text
models/yolo26/yolo26n.pt
test_images/bus.jpg
test_images/zidane.jpg
```

## 验证

```bash
python -m compileall .
python -m unittest discover -s tests -v
```

## 支持的模型搜索格式

自动模型搜索会识别文件型和目录型模型：

```text
文件：.pt, .torchscript, .onnx, .engine, .mlpackage, .pb, .tflite, .mnn, .rknn
目录：_openvino_model, _saved_model, _web_model, _paddle_model, _ncnn_model,
     _imx_model, _rknn_model, _executorch_model, _axelera_model, _deepx_model
```

其中 RKNN 同时支持 `.rknn` 文件和 `_rknn_model` 目录。

## 文件结构

```text
QT_detection/
  run_yolo_qt.py              # 程序入口
  yolo_qt_app/
    main_window.py            # 主窗口、按钮事件、视频流控制
    detector.py               # YOLO 模型加载、推理、检测记录解析
    exporter.py               # CSV / JSON / TSV / TXT 导出
    history.py                # 历史导出记录与回放统计
    app_logging.py            # 日志落盘
    image_utils.py            # OpenCV 图像转 Qt Pixmap
    model_finder.py           # 自动搜索 .pt / .onnx / .engine 模型
    widgets.py                # 可复用 Qt 控件
    styles.py                 # 界面样式
    config.py                 # 路径和文件类型配置
  tests/
    test_config.py            # config 单元测试
    test_detector.py          # detector 单元测试
    test_exporter.py          # exporter 单元测试
  .github/workflows/
    ci.yml                    # 自动编译检查与单元测试
    desktop-package.yml       # Windows/macOS 桌面打包
```
