import json
import datetime
from pathlib import Path


class LogManager:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def write(self, image_path, result, model):
        date = datetime.date.today().isoformat()
        log_file = self.log_dir / f"{date}.json"

        entry = {
            "time": datetime.datetime.now().isoformat(),
            "image": image_path,
            "model": model,
            "result": result
        }

        logs = []
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)

        logs.append(entry)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
