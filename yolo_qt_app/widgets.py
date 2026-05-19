from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt


class VideoLabel(QtWidgets.QLabel):
    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.setMinimumSize(420, 320)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText(title)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setStyleSheet(
            """
            QLabel {
                background: #111827;
                color: #9ca3af;
                border: 1px solid #273244;
                border-radius: 8px;
                font-size: 18px;
            }
            """
        )

    def clear_frame(self):
        self.clear()
        self.setText(self.title)


def section_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setStyleSheet("font-size: 16px; font-weight: 700; color: #f9fafb;")
    return label


def stat_value(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    label.setStyleSheet("color: #58a6ff; font-weight: 700;")
    return label
