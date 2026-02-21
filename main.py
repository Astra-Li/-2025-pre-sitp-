import sys
from PyQt6.QtWidgets import QApplication
from ui_part import BladeDefectMainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("叶片损伤检测系统")

    window = BladeDefectMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
