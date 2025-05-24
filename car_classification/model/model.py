import torch
import torch.nn as nn
import torch.nn.functional as F

from car_classification.model.backbone import ViTBackbone


class HierarchicalCarClassifier(nn.Module):
    """계층적 차량 분류기 - 1차(그룹) + 2차(최종 클래스) 분류"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = ViTBackbone(config.MODEL_NAME)

        # 백본에서 자동으로 hidden_size 가져오기
        self.hidden_size = self.backbone.get_hidden_size()
        print(f"Model hidden size: {self.hidden_size}")

        # 공통 특징 추출 레이어
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # 1차 분류기 (그룹 분류)
        self.group_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, config.NUM_GROUPS)
        )

        # 2차 분류기 (최종 클래스 분류)
        # 입력: features + group_logits
        self.class_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + config.NUM_GROUPS, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, config.NUM_LABELS)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, pixel_values, labels=None, group_labels=None):
        # 백본 특징 추출
        backbone_features = self.backbone(pixel_values)
        backbone_features = self.dropout(backbone_features)

        # 공통 특징 추출
        features = self.feature_extractor(backbone_features)

        # 1차 분류 (그룹)
        group_logits = self.group_classifier(features)

        # 2차 분류를 위한 특징 결합
        # group_logits를 features와 concatenate
        combined_features = torch.cat([features, group_logits], dim=-1)

        # 2차 분류 (최종 클래스)
        class_logits = self.class_classifier(combined_features)

        # 손실 계산
        loss = None
        group_loss = None
        class_loss = None

        if labels is not None:
            # 2차 분류 손실 (주 태스크)
            if hasattr(self, 'class_loss_fn'):
                class_loss = self.class_loss_fn(class_logits, labels)
            else:
                class_loss = F.cross_entropy(class_logits, labels)

            # 1차 분류 손실 (보조 태스크)
            if group_labels is not None:
                if hasattr(self, 'group_loss_fn'):
                    group_loss = self.group_loss_fn(group_logits, group_labels)
                else:
                    group_loss = F.cross_entropy(group_logits, group_labels)

                # 총 손실 (가중 합)
                loss = (self.config.GROUP_LOSS_WEIGHT * group_loss +
                        self.config.CLASS_LOSS_WEIGHT * class_loss)
            else:
                loss = class_loss

        return {
            "loss": loss,
            "class_loss": class_loss,
            "group_loss": group_loss,
            "logits": class_logits,  # 최종 클래스 예측
            "group_logits": group_logits,  # 그룹 예측
            "features": features,
            "combined_features": combined_features
        }

    def get_model_info(self):
        """모델 정보 출력"""
        info = {
            "backbone_type": self.backbone.model_type,
            "model_name": self.backbone.model_name,
            "hidden_size": self.hidden_size,
            "num_groups": self.config.NUM_GROUPS,
            "num_labels": self.config.NUM_LABELS,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        return info

    def freeze_backbone(self):
        """백본 고정 (선택적)"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """백본 학습 가능하게 (선택적)"""
        for param in self.backbone.parameters():
            param.requires_grad = True


# 기존 코드와의 호환성을 위한 별칭
CarBrandClassifier = HierarchicalCarClassifier