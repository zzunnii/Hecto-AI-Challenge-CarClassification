import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def multiclass_log_loss(answer_df, submission_df):
    """
    멀티클래스 로그 로스 계산 (대회 방식)

    Args:
        answer_df: DataFrame with columns ['ID', 'label']
        submission_df: DataFrame with columns ['ID', 'class1', 'class2', ...]

    Returns:
        float: log loss value
    """
    class_list = sorted(answer_df['label'].unique())

    if submission_df.shape[0] != answer_df.shape[0]:
        raise ValueError("submission_df 행 개수가 answer_df와 일치하지 않습니다.")

    submission_df = submission_df.sort_values(by='ID').reset_index(drop=True)
    answer_df = answer_df.sort_values(by='ID').reset_index(drop=True)

    if not all(answer_df['ID'] == submission_df['ID']):
        raise ValueError("ID가 정렬되지 않았거나 불일치합니다.")

    missing_cols = [col for col in class_list if col not in submission_df.columns]
    if missing_cols:
        raise ValueError(f"클래스 컬럼 누락: {missing_cols}")

    if submission_df[class_list].isnull().any().any():
        raise ValueError("NaN 포함됨")

    for col in class_list:
        if not ((submission_df[col] >= 0) & (submission_df[col] <= 1)).all():
            raise ValueError(f"{col}의 확률값이 0~1 범위 초과")

    # 정답 인덱스 변환
    true_labels = answer_df['label'].tolist()
    true_idx = [class_list.index(lbl) for lbl in true_labels]

    # 확률 정규화 + clip
    probs = submission_df[class_list].values
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred = np.clip(probs, 1e-15, 1 - 1e-15)

    return log_loss(true_idx, y_pred, labels=list(range(len(class_list))))


def log_loss_calc_for_validation(y_true, y_pred, class_names=None):
    """
    검증용 로그 로스 계산 (기존 코드와 호환성을 위해)

    Args:
        y_true: 정답 레이블 (numpy array or list)
        y_pred: 예측 확률 (numpy array, shape: [n_samples, n_classes])
        class_names: 클래스 이름 리스트 (옵션)

    Returns:
        float: log loss value
    """
    # 확률값 클리핑 (0과 1 사이 값으로 제한)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # 각 행의 합이 1이 되도록 정규화
    y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)

    # 로그 손실 계산
    return log_loss(y_true, y_pred)


def create_answer_df_from_arrays(ids, labels):
    """
    배열로부터 answer_df 형태 생성

    Args:
        ids: ID 리스트 또는 배열
        labels: 레이블 리스트 또는 배열

    Returns:
        pd.DataFrame: answer_df 형태
    """
    return pd.DataFrame({
        'ID': ids,
        'label': labels
    })


def create_submission_df_from_arrays(ids, predictions, class_names):
    """
    배열로부터 submission_df 형태 생성

    Args:
        ids: ID 리스트 또는 배열
        predictions: 예측 확률 배열 (shape: [n_samples, n_classes])
        class_names: 클래스 이름 리스트

    Returns:
        pd.DataFrame: submission_df 형태
    """
    data = {'ID': ids}
    for i, class_name in enumerate(class_names):
        data[class_name] = predictions[:, i]

    return pd.DataFrame(data)


# 기존 함수명과의 호환성을 위한 별칭
log_loss_calc = log_loss_calc_for_validation