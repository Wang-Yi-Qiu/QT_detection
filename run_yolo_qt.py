import sys

from PyQt6 import QtWidgets

from yolo_qt_app.main_window import YoloMainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = YoloMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
