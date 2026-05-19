@echo off
setlocal

cd /d %~dp0\..
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist QTDetection.spec del /q QTDetection.spec

pyinstaller --noconfirm --clean --windowed --name QTDetection run_yolo_qt.py

echo Build finished: dist\QTDetection\
