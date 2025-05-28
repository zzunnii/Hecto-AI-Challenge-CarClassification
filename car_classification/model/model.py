import torch
import torch.nn as nn
import torch.nn.functional as F

from car_classification.model.backbone import ViTBackbone


class GatedFusionModule(nn.Module):
    """그룹 정보를 게이트 방식으로 융합하는 모듈"""

    def __init__(self, feature_size, num_groups):
        super().__init__()

        # 그룹 임베딩
        self.group_embedding = nn.Embedding(num_groups, feature_size // 4)

        # 게이트 네트워크
        self.gate = nn.Sequential(
            nn.Linear(feature_size + feature_size // 4, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, feature_size),
            nn.Sigmoid()
        )

        # 융합 네트워크
        self.fusion = nn.Sequential(
            nn.Linear(feature_size + feature_size // 4, feature_size),
            nn.LayerNorm(feature_size),
            nn.ReLU()
        )

    def forward(self, features, group_probs):
        batch_size = features.size(0)

        # 그룹 확률의 가중 평균으로 그룹 임베딩 계산
        group_weights = F.softmax(group_probs, dim=1)  # [batch_size, num_groups]

        # 모든 그룹 임베딩 가져오기
        all_group_embs = self.group_embedding.weight  # [num_groups, emb_dim]

        # 가중 평균으로 그룹 특징 계산
        group_features = torch.matmul(group_weights, all_group_embs)  # [batch_size, emb_dim]

        # 특징 결합
        combined = torch.cat([features, group_features], dim=1)

        # 게이트 계산
        gate_weights = self.gate(combined)

        # 게이트된 특징 융합
        fused_features = self.fusion(combined)
        gated_features = gate_weights * fused_features + (1 - gate_weights) * features

        return gated_features


class ProgressiveClassifier(nn.Module):
    """점진적 학습을 지원하는 분류기"""

    def __init__(self, feature_size, num_groups, num_classes):
        super().__init__()

        # 1차 분류기 (그룹) - 더 간단하게
        self.group_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_size // 2, num_groups)
        )

        # 그룹 정보 융합 모듈
        self.fusion_module = GatedFusionModule(feature_size, num_groups)

        # 2차 분류기 (최종 클래스) - 더 강력하게
        self.class_classifier = nn.Sequential(
            nn.Dropout(0.2),  # up from 0.15
            nn.Linear(feature_size, feature_size * 2),
            nn.LayerNorm(feature_size * 2),
            nn.ReLU(),
            nn.Dropout(0.25),  # up from 0.15
            nn.Linear(feature_size * 2, feature_size),
            nn.LayerNorm(feature_size),
            nn.ReLU(),
            nn.Dropout(0.15),  # up from 0.1
            nn.Linear(feature_size, num_classes)
        )

        # 점진적 학습을 위한 플래그
        self.training_stage = "both"  # "group_only", "both"

    def set_training_stage(self, stage):
        """학습 단계 설정"""
        self.training_stage = stage

        if stage == "group_only":
            # 그룹만 학습 시 클래스 분류기 고정
            for param in self.class_classifier.parameters():
                param.requires_grad = False
        else:
            # 모든 파라미터 학습 가능
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, features):
        # 1차 분류 (그룹)
        group_logits = self.group_classifier(features)

        if self.training_stage == "group_only":
            # 그룹만 학습하는 단계
            dummy_class_logits = torch.zeros(features.size(0), self.class_classifier[-1].out_features,
                                             device=features.device)
            return group_logits, dummy_class_logits

        # 그룹 정보를 활용한 특징 융합
        fused_features = self.fusion_module(features, group_logits)

        # 2차 분류 (최종 클래스)
        class_logits = self.class_classifier(fused_features)

        return group_logits, class_logits


class HierarchicalCarClassifierImproved(nn.Module):
    """개선된 계층적 차량 분류기 - 점진적 학습 및 개선된 융합 (온도 스케일링 제거)"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = ViTBackbone(config.MODEL_NAME)

        # 백본 hidden size
        self.hidden_size = self.backbone.get_hidden_size()
        print(f"Model hidden size: {self.hidden_size}")

        # 특징 추출기
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),  # 추가
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(0.1)  # 추가
        )

        # 점진적 분류기
        self.classifier = ProgressiveClassifier(
            feature_size=self.hidden_size,
            num_groups=config.NUM_GROUPS,
            num_classes=config.NUM_LABELS
        )

        # 🔥 온도 스케일링 관련 코드 완전 제거
        # self.register_buffer('temperature', torch.tensor(...))  # 삭제
        # self.final_temperature = ...  # 삭제

        # 현재 에폭 추적
        self.current_epoch = 0

        # Dropout layers
        self.backbone_dropout = nn.Dropout(0.2)

    def set_epoch(self, epoch):
        """현재 에폭 설정 및 학습 단계 조정"""
        self.current_epoch = epoch

        # 점진적 학습 설정
        if hasattr(self.config, 'USE_PROGRESSIVE_TRAINING') and self.config.USE_PROGRESSIVE_TRAINING:
            group_only_epochs = getattr(self.config, 'GROUP_ONLY_EPOCHS', 5)

            if epoch < group_only_epochs:
                self.classifier.set_training_stage("group_only")
                print(f"Epoch {epoch}: Training GROUP ONLY")
            else:
                self.classifier.set_training_stage("both")
                if epoch == group_only_epochs:
                    print(f"Epoch {epoch}: Starting BOTH group and class training")

        # 🔥 온도 스케일링 동적 조정 제거
        # if hasattr(self.config, 'INITIAL_TEMPERATURE'): ...  # 삭제

    def get_dynamic_loss_weights(self, epoch):
        """동적 손실 가중치 계산"""
        if not getattr(self.config, 'USE_DYNAMIC_LOSS_WEIGHTS', False):
            return self.config.GROUP_LOSS_WEIGHT, self.config.CLASS_LOSS_WEIGHT

        # 점진적 학습 단계별 가중치 조정
        group_only_epochs = getattr(self.config, 'GROUP_ONLY_EPOCHS', 5)
        dominance_epochs = getattr(self.config, 'GROUP_DOMINANCE_EPOCHS', 15)

        if epoch < group_only_epochs:
            # 그룹만 학습
            return 1.0, 0.0
        elif epoch < dominance_epochs:
            # 그룹 우세 학습
            progress = (epoch - group_only_epochs) / (dominance_epochs - group_only_epochs)
            group_weight = self.config.MAX_GROUP_WEIGHT * (1 - progress) + self.config.MIN_GROUP_WEIGHT * progress
            class_weight = 1.0 - group_weight
            return group_weight, class_weight
        else:
            # 클래스 중심 학습
            return self.config.MIN_GROUP_WEIGHT, 1.0 - self.config.MIN_GROUP_WEIGHT

    def forward(self, pixel_values, labels=None, group_labels=None):
        # 백본 특징 추출
        backbone_features = self.backbone(pixel_values)
        backbone_features = self.backbone_dropout(backbone_features)

        # 특징 추출
        features = self.feature_extractor(backbone_features)

        # 분류
        group_logits, class_logits = self.classifier(features)

        # 🔥 온도 스케일링 완전 제거
        # 원본 logits를 그대로 사용 (학습/추론 구분 없음)

        # 손실 계산
        loss = None
        group_loss = None
        class_loss = None

        if labels is not None:
            # 동적 가중치 계산
            group_weight, class_weight = self.get_dynamic_loss_weights(self.current_epoch)

            # 2차 분류 손실 (주 태스크)
            if self.classifier.training_stage != "group_only":
                if hasattr(self, 'class_loss_fn'):
                    class_loss = self.class_loss_fn(class_logits, labels)
                else:
                    class_loss = F.cross_entropy(class_logits, labels, label_smoothing=0.1)

            # 1차 분류 손실 (보조 태스크)
            if group_labels is not None:
                if hasattr(self, 'group_loss_fn'):
                    group_loss = self.group_loss_fn(group_logits, group_labels)
                else:
                    group_loss = F.cross_entropy(group_logits, group_labels, label_smoothing=0.05)

                # 총 손실 계산
                if class_loss is not None:
                    loss = group_weight * group_loss + class_weight * class_loss
                else:
                    loss = group_loss
            else:
                loss = class_loss if class_loss is not None else torch.tensor(0.0)

        return {
            "loss": loss,
            "class_loss": class_loss,
            "group_loss": group_loss,
            "logits": class_logits,  # 최종 클래스 예측
            "group_logits": group_logits,  # 그룹 예측
            "features": features,
            # 🔥 온도 관련 반환값 제거
            # "temperature": self.temperature,  # 삭제
            "current_weights": self.get_dynamic_loss_weights(self.current_epoch) if labels is not None else None
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
            # 🔥 온도 관련 정보 제거
            # "temperature": self.temperature.item(),  # 삭제
            "training_stage": self.classifier.training_stage,
            "current_epoch": self.current_epoch
        }
        return info


# 기존 코드와의 호환성을 위한 별칭
HierarchicalCarClassifier = HierarchicalCarClassifierImproved
CarBrandClassifier = HierarchicalCarClassifierImproved