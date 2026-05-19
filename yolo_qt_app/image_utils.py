import cv2
from PyQt6 import QtGui


def frame_to_pixmap(frame) -> QtGui.QPixmap:
    if frame.ndim == 2:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    image = QtGui.QImage(
        rgb.data,
        width,
        height,
        channels * width,
        QtGui.QImage.Format.Format_RGB888,
    ).copy()
    return QtGui.QPixmap.fromImage(image)
