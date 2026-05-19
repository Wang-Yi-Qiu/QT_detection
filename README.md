# QT_detection

这是一个独立的 YOLO Qt 检测界面项目，不依赖原来的 `QT.py`。

## 功能

- 加载 YOLO 常见导出模型，包括 `.pt`、`.onnx`、`.engine`、`.rknn` 等
- 图片检测
- 目录批量图片检测
- 视频检测
- 摄像头实时检测
- 后台线程推理（避免 UI 卡顿）
- 视频跳帧与最大 FPS 限制
- 多模型对比（同图速度与检测数量）
- 按类别统计当前帧和累计目标数量
- 导出 CSV / JSON 检测记录
- JSON 导出会话元数据（模型、参数、设备、耗时）
- 历史 JSON 回放与按类别/时间统计
- 任务参数保存/加载（常用参数复用）
- 首次运行自检（依赖、GPU、模型路径）
- 日志落盘（`logs/app.log`）

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

## 测试

```bash
python -m compileall .
python -m unittest discover -s tests -v
```

## CI

已添加 GitHub Actions：`.github/workflows/ci.yml`

- Python 3.11
- 安装依赖
- 编译检查
- 单元测试

## 打包

Windows/macOS 打包见：`packaging/README.md`

## 测试资源

项目内已放入一个 YOLO26 基础模型和两张测试图片：

```text
models/yolo26/yolo26n.pt
test_images/bus.jpg
test_images/zidane.jpg
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
    exporter.py               # CSV / JSON 导出
    history.py                # 历史记录统计（按类别/时间）
    self_check.py             # 启动自检
    session.py                # 会话元数据
    task_config.py            # 任务配置保存/加载
    image_utils.py            # OpenCV 图像转 Qt Pixmap
    model_finder.py           # 自动搜索 .pt / .onnx / .engine 模型
    widgets.py                # 可复用 Qt 控件
    styles.py                 # 界面样式
    config.py                 # 路径和文件类型配置
  tests/                      # 单元测试
  packaging/README.md         # 打包说明
```
