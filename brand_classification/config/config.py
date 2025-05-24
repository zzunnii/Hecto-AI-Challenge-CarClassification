import os
from typing import Optional, List, Dict


class Config:
    # 기본 경로
    DATA_DIR = r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open\rm_bg_train"
    MAPPING_CSV = r"C:\Users\tjdwn\GitHub\Hecto-AI-Challenge-CarClassification\car_class_mapping.csv"
    OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")

    # 모델 설정
    MODEL_NAME = "google/vit-base-patch16-224-in21k"
    NUM_LABELS = 10  # 레이블 수는 데이터셋에 따라 자동으로 업데이트됨

    # 학습 파라미터
    SEED = 42
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 200
    WARMUP_RATIO = 0.1

    # 데이터 증강
    IMG_SIZE = 224
    USE_COLOR_AUGMENTATION = True
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2

    # Focal Loss 파라미터
    FOCAL_LOSS_GAMMA = 2.0
    FOCAL_LOSS_ALPHA = None  # None이면 클래스 분포에 따라 자동 계산

    # 평가 설정
    EVAL_STEPS = 0  # 0이면 에폭 당 한 번만 평가
    SAVE_STEPS = 0  # 0이면 에폭 당 한 번만 저장
    LOGGING_STEPS = 50

    # 얼리스토핑 설정
    EARLY_STOPPING_PATIENCE = 5  # 몇 에폭 동안 개선이 없으면 중단할지
    MAX_SAVED_MODELS = 5  # 최대 몇 개의 체크포인트를 저장할지

    # 기타 추가 설정
    NUM_WORKERS = 4  # 데이터로더 워커 수

    # 기타 설정
    USE_FP16 = True
    USE_CPU_OFFLOAD = False  # 메모리가 부족한 경우 GPU 메모리 절약을 위해 활성화

    def update_num_labels(self, num_labels: int):
        self.NUM_LABELS = num_labels

    def to_dict(self):
        """설정을 사전 형태로 반환"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('__') and not callable(getattr(self, k))}


config = Config()