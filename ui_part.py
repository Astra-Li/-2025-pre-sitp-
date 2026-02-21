from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QComboBox, QTextEdit,
    QProgressBar, QMessageBox, QGroupBox, QTableWidget,
    QTableWidgetItem
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from model_manager import ModelManager
from prediction_thread import PredictionThread
from batch_thread import BatchThread
from log_manager import LogManager

from pathlib import Path


class BladeDefectMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("叶片损伤智能检测系统")
        self.resize(1100, 700)

        self.model_manager = ModelManager()
        self.log_manager = LogManager()

        self.current_image_path = None

        self.init_ui()

    # ================= UI =================
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        # ========= 左侧控制区 =========
        left_panel = QVBoxLayout()

        title = QLabel("叶片损伤智能检测系统")
        title.setStyleSheet("font-size:22px;font-weight:bold;")
        left_panel.addWidget(title)

        desc = QLabel(
            "系统说明：\n"
            "本系统基于深度学习模型实现叶片图像损伤自动识别。\n"
            "支持 ResNet50-Attention 与 YOLOv5 模型。\n"
        )
        desc.setWordWrap(True)
        left_panel.addWidget(desc)

        # ===== 模型选择 =====
        model_box = QGroupBox("模型选择")
        model_layout = QVBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItems(self.model_manager.get_available_models())
        self.model_combo.currentTextChanged.connect(self.switch_model)

        model_layout.addWidget(self.model_combo)
        model_box.setLayout(model_layout)
        left_panel.addWidget(model_box)

        # ===== 单图检测 =====
        single_box = QGroupBox("单张图像检测")
        single_layout = QVBoxLayout()

        self.select_btn = QPushButton("选择图像")
        self.select_btn.clicked.connect(self.select_image)

        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.clicked.connect(self.start_detection)

        single_layout.addWidget(self.select_btn)
        single_layout.addWidget(self.detect_btn)
        single_box.setLayout(single_layout)
        left_panel.addWidget(single_box)

        # ===== 批量检测 =====
        batch_box = QGroupBox("批量检测")
        batch_layout = QVBoxLayout()

        self.batch_btn = QPushButton("选择文件夹并开始")
        self.batch_btn.clicked.connect(self.start_batch_detection)

        self.progress_bar = QProgressBar()

        batch_layout.addWidget(self.batch_btn)
        batch_layout.addWidget(self.progress_bar)
        batch_box.setLayout(batch_layout)
        left_panel.addWidget(batch_box)

        # ===== 操作指引 =====
        guide_box = QGroupBox("操作指引")
        guide_layout = QVBoxLayout()

        guide_text = QLabel(
            "操作步骤：\n"
            "1️⃣ 选择模型\n"
            "2️⃣ 选择图像或文件夹\n"
            "3️⃣ 点击检测\n"
            "4️⃣ 查看结果\n"
        )
        guide_text.setWordWrap(True)

        guide_layout.addWidget(guide_text)
        guide_box.setLayout(guide_layout)
        left_panel.addWidget(guide_box)

        left_panel.addStretch()

        # ========= 右侧显示区 =========
        right_panel = QVBoxLayout()

        self.image_label = QLabel("图像显示区域")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border:1px solid gray;")
        right_panel.addWidget(self.image_label)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        right_panel.addWidget(self.result_text)

        # 日志表格
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(3)
        self.log_table.setHorizontalHeaderLabels(["时间", "模型", "预测结果"])
        right_panel.addWidget(self.log_table)

        main_layout.addLayout(left_panel, 3)
        main_layout.addLayout(right_panel, 5)

    # ================= 功能逻辑 =================

    def switch_model(self, model_name):
        self.model_manager.switch_model(model_name)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(
                pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
            )

    def start_detection(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "提示", "请先选择图像")
            return

        self.thread = PredictionThread(
            self.model_manager, self.current_image_path
        )
        self.thread.finished.connect(self.show_result)
        self.thread.error.connect(self.show_error)
        self.thread.start()

    def show_result(self, result):
        text = (
            f"预测结果：{result['prediction']}\n"
            f"置信度：{result['confidence']:.4f}\n"
        )
        self.result_text.setText(text)

        self.log_manager.write(
            self.current_image_path,
            result,
            self.model_manager.current_model
        )
        self.load_logs()

    def show_error(self, msg):
        QMessageBox.critical(self, "错误", msg)

    def start_batch_detection(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if not folder:
            return

        self.batch_thread = BatchThread(folder, self.model_manager)
        self.batch_thread.progress.connect(self.progress_bar.setValue)
        self.batch_thread.finished.connect(
            lambda msg: QMessageBox.information(self, "完成", msg)
        )
        self.batch_thread.start()

    def load_logs(self):
        log_dir = Path("logs")
        if not log_dir.exists():
            return

        latest_file = sorted(log_dir.glob("*.json"), reverse=True)
        if not latest_file:
            return

        import json
        with open(latest_file[0], "r", encoding="utf-8") as f:
            logs = json.load(f)

        self.log_table.setRowCount(len(logs))
        for row, log in enumerate(logs):
            self.log_table.setItem(row, 0, QTableWidgetItem(log["time"]))
            self.log_table.setItem(row, 1, QTableWidgetItem(log["model"]))
            self.log_table.setItem(
                row, 2,
                QTableWidgetItem(log["result"]["prediction"])
            )
