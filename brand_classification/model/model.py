import torch
import torch.nn as nn
import torch.nn.functional as F

from brand_classification.model.backbone import ViTBackbone

class CarBrandClassifier(nn.Module):
    """차량 브랜드 분류기"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = ViTBackbone(config.MODEL_NAME)

        # 특징 차원 가져오기
        if "base" in config.MODEL_NAME:
            hidden_size = 768
        elif "large" in config.MODEL_NAME:
            hidden_size = 1024
        else:
            hidden_size = 768  # 기본값

        # 분류 헤드
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, config.NUM_LABELS)
        )

    def forward(self, pixel_values, labels=None):
        features = self.backbone(pixel_values)
        features = self.dropout(features)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            if hasattr(self, 'loss_fn'):
                loss = self.loss_fn(logits, labels)
            else:
                loss = F.cross_entropy(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "features": features
        }