# 打包说明（Windows/macOS）

本项目可使用 PyInstaller 打包为桌面可执行程序。

## 1. 安装打包工具

```bash
pip install pyinstaller
```

## 2. Windows 打包

```bash
pyinstaller --noconfirm --windowed --name QTDetection run_yolo_qt.py
```

输出目录：`dist/QTDetection/`

## 3. macOS 打包

```bash
pyinstaller --noconfirm --windowed --name QTDetection run_yolo_qt.py
```

输出目录：`dist/QTDetection.app`

## 4. 注意事项

- 建议在目标系统上本地打包（Windows 上产 Windows 包，macOS 上产 macOS 包）。
- 首次启动会执行依赖与设备自检，若缺依赖会在界面提示。
- 如需携带模型文件，请将模型放置在程序同级目录或在程序内手动选择模型路径。
