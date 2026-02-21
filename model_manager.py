import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, Any
import os
import sys

# 只需导入 ResNet50 的结构，YOLOv5 我们用官方 hub 接口直接加载
from models_def import ResNet50WithAttention


def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class ModelManager:
    def __init__(self):
        self.models = {}
        self.current_model = None
        # 自动检测是否有 GPU 加速
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_paths = {
            "ResNet50": get_resource_path("applied_best_models/best_model_resnet50.pth"),
            # ✅ 修改点 1：这里明确改为 .pt 后缀
            "YOLOv5": get_resource_path("applied_best_models/best_model_yolo.pt"),
        }

        self.resnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        self._load_models()

    def _load_models(self):
        # ========= 加载 ResNet50 =========
        resnet_path = Path(self.model_paths["ResNet50"])
        if resnet_path.exists():
            try:
                model = ResNet50WithAttention(num_classes=2)
                model.load_state_dict(torch.load(resnet_path, map_location=self.device))
                model.to(self.device).eval()
                self.models["ResNet50"] = model
                print("✅ ResNet50 模型加载成功！")
            except Exception as e:
                print(f"❌ ResNet50 加载失败: {e}")

        # ========= 加载 YOLOv5 =========
        yolo_path = Path(self.model_paths["YOLOv5"])
        if yolo_path.exists():
            try:
                # ✅ 修改点 2：使用 torch.hub 原生加载 YOLOv5 权重
                # 'ultralytics/yolov5' 会自动从本地缓存或 GitHub 获取官方架构代码
                yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(yolo_path), force_reload=False)
                yolo_model.to(self.device).eval()
                self.models["YOLOv5"] = yolo_model
                print("✅ YOLOv5 模型加载成功！")
            except Exception as e:
                print(f"❌ YOLOv5 加载失败: {e}")

        # 如果有加载成功的模型，默认选中第一个
        if self.models:
            self.current_model = list(self.models.keys())[0]

    def get_available_models(self):
        return list(self.models.keys())

    def switch_model(self, model_name):
        if model_name in self.models:
            self.current_model = model_name
            return True
        return False

    def predict(self, image_path) -> Dict[str, Any]:
        if not self.current_model:
            return {"error": "未加载任何模型"}

        # 路由到对应的预测逻辑
        if self.current_model == "ResNet50":
            return self._predict_resnet(image_path)

        if self.current_model == "YOLOv5":
            return self._predict_yolo(image_path)

        return {"error": "未知模型"}

    def _predict_resnet(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.resnet_transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.models["ResNet50"](tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, idx = torch.max(probs, 1)

            labels = ["Healthy", "Faulty"]

            return {
                "prediction": labels[idx.item()],
                "confidence": conf.item(),
                "probabilities": {
                    "Healthy": probs[0][0].item(),
                    "Faulty": probs[0][1].item()
                }
            }
        except Exception as e:
            return {"error": f"ResNet预测失败: {str(e)}"}

    def _predict_yolo(self, image_path):
        try:
            # ✅ 修改点 3：YOLOv5 专属的推理与结果解析逻辑
            results = self.models["YOLOv5"](image_path)
            
            # 使用 pandas 格式提取预测框结果
            df = results.pandas().xyxy[0]

            # 如果没有检测到任何缺陷
            if len(df) == 0:
                return {
                    "prediction": "未检测到已知缺陷",
                    "confidence": 0.0,
                    "probabilities": {}
                }

            # 默认取置信度最高的缺陷作为主要反馈（第一行）
            best_det = df.iloc[0]

            return {
                "prediction": best_det['name'],
                "confidence": float(best_det['confidence']),
                "probabilities": {
                    "xmin": float(best_det['xmin']),
                    "ymin": float(best_det['ymin']),
                    "xmax": float(best_det['xmax']),
                    "ymax": float(best_det['ymax'])
                }
            }
        except Exception as e:
            return {"error": f"YOLO预测失败: {str(e)}"}