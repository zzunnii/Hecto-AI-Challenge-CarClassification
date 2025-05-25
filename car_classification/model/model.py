import torch
import torch.nn as nn
import torch.nn.functional as F

from car_classification.model.backbone import ViTBackbone


class ResidualBlock(nn.Module):
    """Residual connection이 있는 블록"""

    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        # Skip connection을 위한 projection (차원이 다른 경우)
        self.skip_connection = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)

    def forward(self, x):
        identity = self.skip_connection(x)

        out = self.fc(x)
        out = self.ln(out)
        out = F.gelu(out)
        out = self.dropout(out)

        return out + identity


class AttentionPooling(nn.Module):
    """Attention 기반 특징 집계"""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )

    def forward(self, x):
        # x: [batch, hidden_size]
        scores = self.attention(x)
        weights = F.softmax(scores, dim=-1)
        weighted = x * weights
        return weighted


class ImprovedFeatureExtractor(nn.Module):
    """개선된 특징 추출기"""

    def __init__(self, hidden_size, num_layers=3):
        super().__init__()

        # 초기 프로젝션
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Residual 블록들
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size * 2, hidden_size * 2, dropout=0.1 + i * 0.05)
            for i in range(num_layers)
        ])

        # Attention pooling
        self.attention_pool = AttentionPooling(hidden_size * 2)

        # 최종 프로젝션
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2)
        )

    def forward(self, x):
        # 초기 변환
        x = self.input_proj(x)

        # Residual 블록 통과
        for block in self.blocks:
            x = block(x)

        # Attention pooling
        x = self.attention_pool(x)

        # 최종 프로젝션
        x = self.output_proj(x)

        return x


class GroupAwareClassifier(nn.Module):
    """그룹 정보를 활용하는 개선된 분류기"""

    def __init__(self, feature_size, num_groups, num_classes, hidden_size):
        super().__init__()

        # 그룹별 특징 변환
        self.group_transform = nn.Sequential(
            nn.Linear(num_groups, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 2)
        )

        # 특징 결합
        combined_size = feature_size + hidden_size // 2

        # 분류 헤드 (더 깊게)
        self.classifier = nn.Sequential(
            ResidualBlock(combined_size, hidden_size * 2, dropout=0.2),
            ResidualBlock(hidden_size * 2, hidden_size * 2, dropout=0.15),
            ResidualBlock(hidden_size * 2, hidden_size, dropout=0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, features, group_logits):
        # 그룹 정보 변환
        group_features = self.group_transform(group_logits)

        # 특징 결합
        combined = torch.cat([features, group_features], dim=-1)

        # 분류
        return self.classifier(combined)


class HierarchicalCarClassifierImproved(nn.Module):
    """개선된 계층적 차량 분류기"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = ViTBackbone(config.MODEL_NAME)

        # 백본 hidden size
        self.hidden_size = self.backbone.get_hidden_size()
        print(f"Model hidden size: {self.hidden_size}")

        # Temperature scaling 파라미터
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

        # 개선된 특징 추출기
        self.feature_extractor = ImprovedFeatureExtractor(
            self.hidden_size,
            num_layers=3
        )

        # 1차 분류기 (그룹) - 더 강력하게
        self.group_classifier = nn.Sequential(
            ResidualBlock(self.hidden_size * 2, self.hidden_size, dropout=0.15),
            ResidualBlock(self.hidden_size, self.hidden_size // 2, dropout=0.1),
            nn.Linear(self.hidden_size // 2, config.NUM_GROUPS)
        )

        # 2차 분류기 (최종 클래스) - 그룹 정보 활용 개선
        self.class_classifier = GroupAwareClassifier(
            feature_size=self.hidden_size * 2,
            num_groups=config.NUM_GROUPS,
            num_classes=config.NUM_LABELS,
            hidden_size=self.hidden_size
        )

        # Dropout layers
        self.backbone_dropout = nn.Dropout(0.1)
        self.feature_dropout = nn.Dropout(0.15)

        # Layer normalization for stability
        self.backbone_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, pixel_values, labels=None, group_labels=None):
        # 백본 특징 추출
        backbone_features = self.backbone(pixel_values)
        backbone_features = self.backbone_norm(backbone_features)  # 정규화
        backbone_features = self.backbone_dropout(backbone_features)

        # 개선된 특징 추출
        features = self.feature_extractor(backbone_features)
        features = self.feature_dropout(features)

        # 1차 분류 (그룹)
        group_logits = self.group_classifier(features)

        # Temperature scaling (추론 시 사용)
        if not self.training:
            group_logits = group_logits / self.temperature

        # 2차 분류 (최종 클래스) - 그룹 정보 활용
        class_logits = self.class_classifier(features, group_logits)

        # Temperature scaling (추론 시 사용)
        if not self.training:
            class_logits = class_logits / self.temperature

        # 손실 계산
        loss = None
        group_loss = None
        class_loss = None

        if labels is not None:
            # 2차 분류 손실 (주 태스크)
            if hasattr(self, 'class_loss_fn'):
                class_loss = self.class_loss_fn(class_logits, labels)
            else:
                # Label smoothing 증가
                class_loss = F.cross_entropy(class_logits, labels, label_smoothing=0.15)

            # 1차 분류 손실 (보조 태스크)
            if group_labels is not None:
                if hasattr(self, 'group_loss_fn'):
                    group_loss = self.group_loss_fn(group_logits, group_labels)
                else:
                    group_loss = F.cross_entropy(group_logits, group_labels, label_smoothing=0.1)

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
            "temperature": self.temperature
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
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "temperature": self.temperature.item()
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

    def set_temperature(self, temperature):
        """Temperature 수동 설정 (추론용)"""
        self.temperature.data.fill_(temperature)


# 기존 코드와의 호환성을 위한 별칭
HierarchicalCarClassifier = HierarchicalCarClassifierImproved
CarBrandClassifier = HierarchicalCarClassifierImproved