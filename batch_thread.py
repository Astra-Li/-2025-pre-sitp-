from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
import time


class BatchThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, folder_path, model_manager):
        super().__init__()
        self.folder_path = folder_path
        self.model_manager = model_manager

    def run(self):
        image_exts = {".jpg", ".png", ".jpeg", ".bmp"}
        files = [p for p in Path(self.folder_path).iterdir()
                 if p.suffix.lower() in image_exts]

        if not files:
            self.finished.emit("未找到图片")
            return

        start = time.time()
        total = len(files)

        for i, img in enumerate(files, 1):
            self.model_manager.predict(str(img))
            self.progress.emit(int(i / total * 100))

        self.finished.emit(f"批量完成，共 {total} 张，用时 {time.time() - start:.1f}s")
