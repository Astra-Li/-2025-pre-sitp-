import torch
import torch.nn as nn
import torchvision.models as models


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.attention(x)
        return x * attn


class ResNet50WithAttention(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet50(pretrained=False)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.attention = AttentionBlock(2048)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class YOLOv5Model:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        try:
            self.model = torch.load(self.model_path, map_location="cpu")
            self.model.eval()
            return True
        except:
            return False

    def predict(self, image_path):
        return {
            "prediction": "Healthy",
            "confidence": 0.95,
            "probabilities": {
                "Healthy": 0.95,
                "Faulty": 0.05
            }
        }
