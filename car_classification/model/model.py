import torch
import torch.nn as nn
import torch.nn.functional as F

from car_classification.model.backbone import ViTBackbone


class EnhancedAttentionModule(nn.Module):
    """자동차 분류에 최적화된 강화된 어텐션 모듈"""

    def __init__(self, feature_size, num_groups):
        super().__init__()

        # 1. 강화된 멀티헤드 어텐션 (헤드 수 증가)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_size,
            num_heads=8,  # 다양한 시각적 패턴 포착
            dropout=0.1  # 정보 흐름 향상을 위해 낮은 드롭아웃
        )

        # 2. 정규화 레이어 (트랜스포머 스타일)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        # 3. 강화된 피드포워드 네트워크
        self.ffn = nn.Sequential(
            nn.Linear(feature_size, feature_size * 4),  # 더 넓은 중간 레이어
            nn.GELU(),  # ReLU보다 부드러운 활성화 함수
            nn.Dropout(0.1),
            nn.Linear(feature_size * 4, feature_size)
        )

        # 4. 그룹-특징 통합을 위한 크로스 어텐션
        self.group_embedding = nn.Embedding(num_groups, feature_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_size,
            num_heads=8,
            dropout=0.1
        )

        # 5. 최종 특징 융합
        self.fusion = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.LayerNorm(feature_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, features, group_probs):
        batch_size = features.size(0)

        # 셀프 어텐션 적용 (트랜스포머 구조 따름)
        attn_input = features.unsqueeze(0)  # [1, batch, dim]
        attn_output, _ = self.self_attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.squeeze(0)  # [batch, dim]

        # 잔차 연결 및 정규화
        features = self.norm1(features + attn_output)

        # 피드포워드 네트워크
        ffn_output = self.ffn(features)
        features = self.norm2(features + ffn_output)  # 두 번째 잔차 연결

        # 그룹 정보와 통합
        group_weights = F.softmax(group_probs, dim=1)
        group_embeds = torch.matmul(group_weights, self.group_embedding.weight)

        # 크로스 어텐션 적용
        cross_query = group_embeds.unsqueeze(0)
        cross_key_value = features.unsqueeze(0)
        cross_output, _ = self.cross_attention(cross_query, cross_key_value, cross_key_value)
        cross_output = cross_output.squeeze(0)

        # 최종 특징 융합
        combined = torch.cat([features, cross_output], dim=1)
        enhanced_features = self.fusion(combined)

        return enhanced_features


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

    def __init__(self, feature_size, num_groups, num_classes, config):
        super().__init__()
        self.config = config

        # 1차 분류기 (그룹) - 더 간단하게
        self.group_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_size // 2, num_groups)
        )

        # 그룹 정보 융합 모듈 (향상된 어텐션 또는 기본 게이트 모듈)
        if getattr(config, 'USE_ENHANCED_ATTENTION', False):
            self.fusion_module = EnhancedAttentionModule(feature_size, num_groups)
        else:
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

        # 클래스-그룹 매핑 정보 (런타임에 설정)
        self.register_buffer('class_to_group_mapping', None)

    def set_class_to_group_mapping(self, mapping_tensor):
        """클래스-그룹 매핑 설정"""
        self.class_to_group_mapping = mapping_tensor

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
            num_classes=config.NUM_LABELS,
            config=config
        )

        # 현재 에폭 추적
        self.current_epoch = 0

        # Dropout layers
        self.backbone_dropout = nn.Dropout(0.2)

        # 그룹 정보 활용을 위한 신뢰도 임계값
        self.group_confidence_threshold = getattr(config, 'GROUP_CONFIDENCE_THRESHOLD', 0.8)

        # 클래스-그룹 매핑 초기화를 위한 플래그
        self.mapping_initialized = False

        # 클래스 가중치 등록
        self.register_buffer('class_weights', None)

    def initialize_class_group_mapping(self, mapping_info, train_dataset=None):
        """클래스-그룹 매핑 초기화"""
        num_classes = self.config.NUM_LABELS
        num_groups = self.config.NUM_GROUPS

        # 클래스-그룹 매핑 텐서 생성
        mapping = torch.zeros(num_classes, dtype=torch.long)

        # 클래스 인덱스 정보 소스 선택
        if 'idx_to_class' in mapping_info:
            # 매핑 정보에서 직접 가져오기
            idx_to_class = mapping_info['idx_to_class']
        elif train_dataset is not None and hasattr(train_dataset, 'idx_to_class'):
            # 데이터셋에서 가져오기
            idx_to_class = train_dataset.idx_to_class
        else:
            # 둘 다 없으면 매핑 정보를 생성할 수 없음
            print("Warning: No class index mapping found. Group-class mapping initialization skipped.")
            return

        # 매핑 정보 채우기
        for class_idx in range(num_classes):
            class_key = str(class_idx) if isinstance(idx_to_class, dict) else class_idx
            if class_key in idx_to_class:
                class_name = idx_to_class[class_key]
                group_name = mapping_info['original_to_group'].get(class_name)
                if group_name:
                    group_idx = mapping_info['group_to_idx'].get(group_name, 0)
                    mapping[class_idx] = group_idx

        # 분류기에 매핑 설정
        self.classifier.set_class_to_group_mapping(mapping)
        self.mapping_initialized = True
        print(f"Class-group mapping initialized: {mapping.sum().item()} mappings set")

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

    def set_class_weights(self, weights):
        """클래스 가중치 설정 (2차 분류에 적용)"""
        self.class_weights = weights

    def get_dynamic_loss_weights(self, epoch):
        """동적 손실 가중치 계산 (그룹 vs 클래스 가중치)"""
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

    def adjust_probs_with_group_info(self, class_probs, group_probs):
        """그룹 정보를 활용한 클래스 확률 조정"""
        # 원본 확률 저장
        original_probs = class_probs.clone()
        adjusted_probs = class_probs.clone()

        # 그룹 확률이 높은 경우에만 조정 적용 (안전장치)
        max_group_probs, max_group_indices = group_probs.max(dim=1)
        high_confidence_mask = (max_group_probs > self.group_confidence_threshold).unsqueeze(1)

        # 클래스-그룹 매핑이 초기화된 경우에만 사용
        if self.mapping_initialized and self.classifier.class_to_group_mapping is not None:
            for batch_idx in range(class_probs.size(0)):
                # 이 배치 항목의 가장 확률 높은 그룹
                pred_group = max_group_indices[batch_idx].item()
                group_confidence = max_group_probs[batch_idx].item()

                if group_confidence > self.group_confidence_threshold:
                    # 그룹에 속한 클래스들 찾기
                    group_classes = (self.classifier.class_to_group_mapping == pred_group).nonzero(as_tuple=True)[0]

                    # 부스트 팩터 계산 (그룹 확률에 비례)
                    boost_factor = 0.3 * group_confidence

                    # 해당 그룹의 클래스들 확률 증가
                    for class_idx in group_classes:
                        adjusted_probs[batch_idx, class_idx] *= (1 + boost_factor)

            # 확률 정규화
            row_sums = adjusted_probs.sum(dim=1, keepdim=True)
            adjusted_probs = adjusted_probs / row_sums

            # 높은 신뢰도의 그룹 예측에만 적용, 낮은 신뢰도는 원본 유지
            final_probs = torch.where(high_confidence_mask, adjusted_probs, original_probs)
            return final_probs

        return original_probs

    def forward(self, pixel_values, labels=None, group_labels=None):
        # 백본 특징 추출
        backbone_features = self.backbone(pixel_values)
        backbone_features = self.backbone_dropout(backbone_features)

        # 특징 추출
        features = self.feature_extractor(backbone_features)

        # 분류
        group_logits, class_logits = self.classifier(features)

        # 추론 시 그룹 정보 활용
        if not self.training and hasattr(self.config, 'USE_GATED_FUSION') and self.config.USE_GATED_FUSION:
            # 그룹 및 클래스 확률
            group_probs = F.softmax(group_logits, dim=1)
            class_probs = F.softmax(class_logits, dim=1)

            # 그룹 정보를 활용한 클래스 확률 조정
            adjusted_probs = self.adjust_probs_with_group_info(class_probs, group_probs)

            # 조정된 확률을 로짓으로 변환
            class_logits = torch.log(adjusted_probs + 1e-10)

        # 손실 계산
        loss = None
        group_loss = None
        class_loss = None

        if labels is not None:
            # 동적 그룹/클래스 손실 가중치 계산
            group_weight, class_weight = self.get_dynamic_loss_weights(self.current_epoch)

            # 2차 분류 손실 (클래스 분류) - 클래스별 가중치 적용
            if self.classifier.training_stage != "group_only":
                if hasattr(self, 'class_loss_fn'):
                    class_loss = self.class_loss_fn(class_logits, labels)
                else:
                    # 클래스별 가중치는 여기에 적용 (cross_entropy의 weight 인자)
                    class_loss = F.cross_entropy(class_logits, labels,
                                                 weight=self.class_weights,  # 클래스별 가중치
                                                 label_smoothing=0.1)

            # 1차 분류 손실 (그룹 분류)
            if group_labels is not None:
                if hasattr(self, 'group_loss_fn'):
                    group_loss = self.group_loss_fn(group_logits, group_labels)
                else:
                    group_loss = F.cross_entropy(group_logits, group_labels,
                                                 label_smoothing=0.05)

            # 총 손실 계산 (그룹-클래스 가중치 적용)
            if class_loss is not None and group_loss is not None:
                loss = group_weight * group_loss + class_weight * class_loss
            elif class_loss is not None:
                loss = class_loss
            elif group_loss is not None:
                loss = group_loss
            else:
                loss = torch.tensor(0.0, device=pixel_values.device)

        return {
            "loss": loss,
            "class_loss": class_loss,
            "group_loss": group_loss,
            "logits": class_logits,  # 최종 클래스 예측
            "group_logits": group_logits,  # 그룹 예측
            "features": features,
            "current_weights": (group_weight, class_weight) if labels is not None else None
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
            "training_stage": self.classifier.training_stage,
            "current_epoch": self.current_epoch,
            "enhanced_attention": getattr(self.config, 'USE_ENHANCED_ATTENTION', False),
            "adaptive_weights": getattr(self.config, 'USE_ADAPTIVE_CLASS_WEIGHTS', False)
        }
        return info


# 기존 코드와의 호환성을 위한 별칭
HierarchicalCarClassifier = HierarchicalCarClassifierImproved
CarBrandClassifier = HierarchicalCarClassifierImproved