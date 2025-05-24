import numpy as np
from sklearn.metrics import log_loss


def log_loss_calc(y_true, y_pred):
    """로그 손실 계산"""
    # 확률값 클리핑 (0과 1 사이 값으로 제한)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # 각 행의 합이 1이 되도록 정규화
    y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)

    # 로그 손실 계산
    return log_loss(y_true, y_pred)