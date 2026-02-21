from PyQt6.QtCore import QThread, pyqtSignal
import traceback


class PredictionThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model_manager, image_path):
        super().__init__()
        self.model_manager = model_manager
        self.image_path = image_path

    def run(self):
        try:
            result = self.model_manager.predict(self.image_path)

            if not isinstance(result, dict):
                self.error.emit("预测结果异常")
                return

            if "error" in result:
                self.error.emit(result["error"])
            else:
                self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
            print(traceback.format_exc())
