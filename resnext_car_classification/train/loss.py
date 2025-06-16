import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """다중 클래스 Focal Loss"""

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_softmax = F.log_softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        log_pt = torch.sum(log_softmax * targets_one_hot, dim=1)
        pt = torch.exp(log_pt)

        # Focal weight 계산
        focal_weight = (1 - pt) ** self.gamma

        # Class balancing weight
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha[targets]
            else:
                alpha = self.alpha

            focal_weight = focal_weight * alpha

        loss = -focal_weight * log_pt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class MixupLoss(nn.Module):
    """Mixup Loss"""

    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, outputs, targets_a, targets_b, lam):
        # 각 타겟 세트에 대한 손실 계산
        loss_a = self.criterion(outputs, targets_a)
        loss_b = self.criterion(outputs, targets_b)

        # lam이 텐서인 경우, 요소별 곱셈을 위해 형태 조정
        if isinstance(lam, torch.Tensor) and lam.dim() > 0:
            if loss_a.dim() > 0:  # loss_a에 차원이 있는 경우
                # 적절히 브로드캐스팅되도록 lam 재구성
                lam = lam.view(-1, *([1] * (loss_a.dim() - 1)))

        # 손실 혼합
        mixed_loss = lam * loss_a + (1 - lam) * loss_b

        # 결과가 스칼라인지 확인
        if isinstance(mixed_loss, torch.Tensor) and mixed_loss.dim() > 0:
            mixed_loss = mixed_loss.mean()

        return mixed_loss


class AdaptiveClassWeightScheduler:
    """클래스별 성능에 따라 적응적으로 가중치를 조정하는 스케줄러"""

    def __init__(self, num_classes, initial_f1_scores=None,
                 base_weight=1.0, max_weight=2.0,
                 threshold=0.7, memory_factor=0.7,
                 max_degradation=3):
        self.num_classes = num_classes

        # 초기 F1 점수가 없으면 모두 1.0으로 초기화
        if initial_f1_scores is None:
            initial_f1_scores = [1.0] * num_classes

        self.current_f1_scores = initial_f1_scores
        self.previous_f1_scores = initial_f1_scores.copy()

        self.base_weight = base_weight
        self.max_weight = max_weight
        self.threshold = threshold
        self.memory_factor = memory_factor  # 이전 가중치의 영향력

        # 현재 적용 중인 가중치 초기화
        self.current_weights = torch.ones(num_classes)
        self.problematic_classes = self._identify_problem_classes()

        # 성능 감소 탐지를 위한 카운터
        self.degradation_counter = torch.zeros(num_classes)
        self.max_degradation = max_degradation  # 연속 감소 허용 횟수

        # 클래스 간 영향 매트릭스 (훈련하면서 학습됨)
        self.class_influence = torch.zeros((num_classes, num_classes))

        print(f"Initialized AdaptiveClassWeightScheduler with {len(self.problematic_classes)} problematic classes")
        print(f"Max weight: {max_weight}, Threshold: {threshold}")

    def _identify_problem_classes(self):
        """성능이 낮은 클래스 식별"""
        return [i for i, score in enumerate(self.current_f1_scores)
                if score < self.threshold]

    def update_f1_scores(self, new_f1_scores):
        """새로운 F1 점수로 업데이트"""
        self.previous_f1_scores = self.current_f1_scores.copy()
        self.current_f1_scores = new_f1_scores

        # 문제 클래스 목록 갱신
        self.problematic_classes = self._identify_problem_classes()

        # 클래스 간 영향 분석
        self._analyze_class_influence()

        # 성능 변화에 따른 가중치 조정
        self._adjust_weights_based_on_performance()

        return self.current_weights

    def _analyze_class_influence(self):
        """가중치 변화가 다른 클래스 성능에 미치는 영향 분석"""
        for cls_idx in range(self.num_classes):
            # 이 클래스의 가중치가 유의미하게 증가했는지 확인
            if self.current_weights[cls_idx] > self.base_weight + 0.3:
                # 다른 클래스들의 성능 변화 확인
                for other_cls in range(self.num_classes):
                    if other_cls != cls_idx:
                        # 성능 변화율 계산
                        perf_change = (self.current_f1_scores[other_cls] -
                                       self.previous_f1_scores[other_cls])

                        # 유의미한 성능 저하가 있는 경우 영향 관계 기록
                        if perf_change < -0.02:  # 2% 이상 성능 저하
                            self.class_influence[cls_idx, other_cls] += abs(perf_change)

    def _adjust_weights_based_on_performance(self):
        """성능 변화에 따라 가중치 적응적 조정"""
        new_weights = torch.ones(self.num_classes)

        for cls_idx in self.problematic_classes:
            # 기본 가중치 계산 (성능이 낮을수록 높은 가중치)
            base_adjustment = min(self.threshold - self.current_f1_scores[cls_idx],
                                  self.max_weight - self.base_weight)

            proposed_weight = self.base_weight + base_adjustment

            # 성능 변화 확인
            if cls_idx < len(self.previous_f1_scores):
                perf_change = self.current_f1_scores[cls_idx] - self.previous_f1_scores[cls_idx]

                if perf_change < -0.01:  # 성능 저하
                    # 연속 저하 카운터 증가
                    self.degradation_counter[cls_idx] += 1

                    if self.degradation_counter[cls_idx] >= self.max_degradation:
                        # 연속 저하가 지속되면 가중치 감소
                        proposed_weight = max(self.base_weight,
                                              float(self.current_weights[cls_idx]) * 0.9)
                        # 카운터 리셋
                        self.degradation_counter[cls_idx] = 0
                else:
                    # 성능 향상 또는 유지 - 카운터 리셋
                    self.degradation_counter[cls_idx] = 0

            # 다른 클래스에 미치는 부정적 영향 확인
            negative_influence = False
            for other_cls in range(self.num_classes):
                if (other_cls != cls_idx and
                        self.class_influence[cls_idx, other_cls] > 0.05):  # 유의미한 영향
                    negative_influence = True
                    break

            if negative_influence:
                # 다른 클래스에 부정적 영향이 큰 경우 가중치 제한
                proposed_weight = min(proposed_weight, float(self.current_weights[cls_idx]) * 1.1)

            # 이전 가중치와 새 가중치를 혼합 (급격한 변화 방지)
            new_weights[cls_idx] = (self.memory_factor * float(self.current_weights[cls_idx]) +
                                    (1 - self.memory_factor) * proposed_weight)

        # 문제가 아닌 클래스는 기본 가중치 유지
        for cls_idx in range(self.num_classes):
            if cls_idx not in self.problematic_classes:
                new_weights[cls_idx] = self.base_weight

        # 가중치 업데이트
        self.current_weights = new_weights

        return new_weights

    def get_current_weights(self):
        """현재 클래스 가중치 반환"""
        return self.current_weights

    def get_problem_classes_info(self):
        """문제 클래스 정보 반환 (로깅용)"""
        info = []
        for cls_idx in self.problematic_classes:
            info.append({
                'class_idx': cls_idx,
                'f1_score': self.current_f1_scores[cls_idx],
                'weight': float(self.current_weights[cls_idx]),
                'degradation_count': int(self.degradation_counter[cls_idx])
            })
        return info