# 打包说明（Windows/macOS）

本项目可使用 PyInstaller 打包为桌面可执行程序。

## 1. 安装打包工具

```bash
pip install pyinstaller
```

## 2. Windows 打包

在 PowerShell 或 CMD 里执行：

```bash
packaging\build_windows.bat
```

输出目录：`dist/QTDetection/`

## 3. macOS 打包

在终端执行：

```bash
bash packaging/build_macos.sh
```

输出目录：`dist/QTDetection.app`

## 4. 首次运行自检

程序首次启动会自动执行自检：

- 依赖检查：`cv2`、`PyQt6`、`ultralytics`
- 设备检查：CPU/GPU（CUDA 或 MPS）可用性
- 模型路径检查：模型搜索根目录与已发现模型文件可读性

首次检查完成后会写入标记文件：`configs/.startup_self_check_done`，后续启动默认不再自动弹窗。

## 5. 发布注意事项

- 建议在目标系统上本地打包（Windows 上产 Windows 包，macOS 上产 macOS 包）。
- 如需携带模型文件，请将模型放在程序同级目录（如 `models/`）或由用户在程序内手动选择模型路径。
- macOS 分发前通常还需要签名与公证；未签名应用可能被系统拦截。
