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