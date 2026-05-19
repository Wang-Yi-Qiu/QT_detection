#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
rm -rf build dist QTDetection.spec

pyinstaller --noconfirm --clean --windowed --name QTDetection run_yolo_qt.py

echo "Build finished: dist/QTDetection.app"
