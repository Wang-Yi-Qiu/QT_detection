# QT_detection

这是一个独立的 YOLO Qt 检测界面项目，不依赖原来的 `QT.py`。

功能包括：

- 加载 YOLO `.pt` / `.onnx` / `.engine` 模型
- 图片检测
- 视频检测
- 摄像头实时检测
- 实时显示原始画面和检测结果
- 按类别统计当前帧和累计目标数量
- 导出 CSV / JSON 检测记录

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

## 文件结构

```text
QT_detection/
  run_yolo_qt.py              # 程序入口
  yolo_qt_app/
    main_window.py            # 主窗口、按钮事件、视频流控制
    detector.py               # YOLO 模型加载、推理、检测记录解析
    exporter.py               # CSV / JSON 导出
    image_utils.py            # OpenCV 图像转 Qt Pixmap
    model_finder.py           # 自动搜索 .pt / .onnx / .engine 模型
    widgets.py                # 可复用 Qt 控件
    styles.py                 # 界面样式
    config.py                 # 路径和文件类型配置
```
