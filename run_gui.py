from __future__ import annotations
"""GUI launcher for mini_yolo_app.

中文: mini_yolo_app 图形化界面启动入口。
English: GUI startup entry for mini_yolo_app.
"""

import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from gui_window import MainWindow


def main() -> None:
    """Create Qt app and show main window.

    中文: 创建 Qt 应用并显示主窗口。
    English: Initialize Qt application and show main window.
    """
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
    except FileNotFoundError as exc:
        QMessageBox.critical(None, "Model Error", str(exc))
        raise
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


