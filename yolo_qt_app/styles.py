APP_STYLE = """
QMainWindow, QWidget {
    background: #0b1220;
    color: #e5e7eb;
    font-family: "Microsoft YaHei", "PingFang SC", Arial;
    font-size: 14px;
}
QLabel#Title {
    font-size: 28px;
    font-weight: 700;
    color: #f9fafb;
}
QLabel#Subtitle {
    color: #9ca3af;
    font-size: 14px;
}
QWidget#SidePanel, QTextBrowser#LogBox {
    background: #111827;
    border: 1px solid #273244;
    border-radius: 8px;
}
QTextBrowser#LogBox {
    padding: 10px;
    color: #d1d5db;
}
QPushButton {
    background: #1f6feb;
    color: white;
    border: 0;
    border-radius: 6px;
    padding: 9px 10px;
    font-weight: 600;
}
QPushButton:hover {
    background: #2f81f7;
}
QPushButton:disabled {
    background: #374151;
    color: #8b949e;
}
QComboBox, QSpinBox, QDoubleSpinBox {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px;
    color: #e5e7eb;
}
QTableWidget {
    background: #0f172a;
    border: 1px solid #273244;
    border-radius: 6px;
    gridline-color: #273244;
    color: #e5e7eb;
}
QHeaderView::section {
    background: #1f2937;
    color: #e5e7eb;
    border: 0;
    padding: 6px;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #334155;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
    background: #58a6ff;
}
"""
