import os
from typing import Optional, List, Dict


class Config:
    # 기본 베이스 경로
    BASE_DIR = r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open"

    # 데이터 경로들
    DATA_DIR = os.path.join(BASE_DIR, "train")  # 원본 데이터
    MASK_DIR = os.path.join(BASE_DIR, "extracted_mask")  # 마스크 파일들
    AUGMENTED_DATA_DIR = os.path.join(BASE_DIR, "augmented_data")  # 증강 후 저장될 곳
    SPLITS_DIR = os.path.join(BASE_DIR, "split_info")  # 폴드 정보 저장
    MODEL_OUTPUT_DIR = r"C:\Users\tjdwn\GitHub\Hecto-AI-Challenge-CarClassification\car_classification\outputs"  # 모델 아웃풋

    AUGMENTED_TRAIN_DIR = r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open\augmented_data\fold_0\train_augmented"
    VAL_DIR = r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open\augmented_data\fold_0\val"

    # 계층적 분류를 위한 매핑 파일
    MAPPING_CSV_PATH = r"C:\Users\tjdwn\GitHub\Hecto-AI-Challenge-CarClassification\car_class_mapping.csv"

    OUTPUT_DIR = "../outputs"

    # 모델 설정
    MODEL_NAME = "google/vit-base-patch16-224"
    NUM_LABELS = 10  # original class 수 (자동 업데이트)
    NUM_GROUPS = 10  # new group 수 (자동 업데이트)

    # 계층적 분류 설정
    USE_HIERARCHICAL_CLASSIFICATION = True
    GROUP_LOSS_WEIGHT = 0.4  # 1차 분류 (보조)
    CLASS_LOSS_WEIGHT = 0.6  # 2차 분류 (주)

    # 학습 파라미터
    SEED = 42
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 300
    WARMUP_RATIO = 0.1

    # 이미지 전처리
    IMG_SIZE = 224
    USE_ASPECT_PRESERVING = True
    PADDING_COLOR = (0, 0, 0)

    # 데이터 증강 (학습 중)
    USE_COLOR_AUGMENTATION = False
    USE_MIXUP = True
    MIXUP_ALPHA = 0.3

    # 새로운 증강 설정
    USE_BACKGROUND_BRIGHTNESS = True
    BRIGHTNESS_RANGE = (0.7, 1.3)

    # 학습 중 기본 증강
    USE_RANDOM_CROP = True
    RANDOM_CROP_SCALE = (0.8, 1.0)
    USE_RANDOM_FLIP = True
    FLIP_PROBABILITY = 0.5
    USE_ROTATION = True
    ROTATION_DEGREES = 10
    USE_AFFINE = True
    AFFINE_TRANSLATE = (0.05, 0.05)
    AFFINE_SCALE = (0.95, 1.05)

    # 손실 함수 설정
    USE_FOCAL_LOSS = False
    FOCAL_LOSS_GAMMA = 2.0
    FOCAL_LOSS_ALPHA = None
    LABEL_SMOOTHING = 0.01

    # 평가 설정
    EVAL_STEPS = 0
    SAVE_STEPS = 0
    LOGGING_STEPS = 50

    # 얼리스토핑 설정
    EARLY_STOPPING_PATIENCE = 8
    MAX_SAVED_MODELS = 3

    # 교차검증 설정
    USE_CROSS_VALIDATION = True
    N_FOLDS = 5
    CURRENT_FOLD = 0

    # 기타 설정
    NUM_WORKERS = 4
    USE_FP16 = True
    USE_CPU_OFFLOAD = False

    # 혼동 분석 설정
    SAVE_PREDICTIONS = True
    ANALYZE_CONFUSION = True

    def __init__(self):
        # 필요한 디렉토리들 자동 생성
        self._create_directories()

    def _create_directories(self):
        """필요한 디렉토리들을 자동으로 생성"""
        directories = [
            self.AUGMENTED_DATA_DIR,
            self.SPLITS_DIR,
            self.MODEL_OUTPUT_DIR
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def update_num_labels(self, num_labels: int):
        """Original class 수 업데이트"""
        self.NUM_LABELS = num_labels

    def update_num_groups(self, num_groups: int):
        """New group 수 업데이트"""
        self.NUM_GROUPS = num_groups

    def set_fold(self, fold_idx: int):
        """현재 폴드 설정"""
        self.CURRENT_FOLD = fold_idx
        # 폴드별 출력 디렉토리 설정
        self.OUTPUT_DIR = os.path.join(self.MODEL_OUTPUT_DIR, f"hierarchical_fold_{fold_idx}")
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def get_fold_data_dir(self, fold_idx: int, split_type: str = "train"):
        """폴드별 데이터 디렉토리 반환

        Args:
            fold_idx: 폴드 인덱스 (0-4)
            split_type: 'train' 또는 'val'
        """
        if split_type == "train":
            return os.path.join(self.AUGMENTED_DATA_DIR, f"fold_{fold_idx}", "train_augmented")
        else:
            return os.path.join(self.AUGMENTED_DATA_DIR, f"fold_{fold_idx}", "val")

    def get_splits_file_path(self):
        """폴드 분할 정보 파일 경로 반환"""
        return os.path.join(self.SPLITS_DIR, "fold_splits.json")

    def get_class_mapping_path(self):
        """클래스 매핑 파일 경로 반환"""
        return os.path.join(self.SPLITS_DIR, "class_mapping.json")

    def to_dict(self):
        """설정을 사전 형태로 반환"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('__') and not callable(getattr(self, k))}

    def verify_paths(self):
        """경로 설정이 올바른지 확인"""
        paths_to_check = {
            "원본 데이터": self.DATA_DIR,
            "마스크 데이터": self.MASK_DIR,
            "CSV 매핑 파일": self.MAPPING_CSV_PATH,
        }

        print("=== 경로 확인 ===")
        for name, path in paths_to_check.items():
            exists = os.path.exists(path)
            print(f"{name}: {path} {'✓' if exists else '✗'}")

        print(f"증강 데이터 저장 경로: {self.AUGMENTED_DATA_DIR}")
        print(f"폴드 정보 저장 경로: {self.SPLITS_DIR}")
        print(f"모델 출력 경로: {self.MODEL_OUTPUT_DIR}")

    def print_hierarchical_config(self):
        """계층적 분류 설정 출력"""
        print("\n=== 계층적 분류 설정 ===")
        print(f"USE_HIERARCHICAL_CLASSIFICATION: {self.USE_HIERARCHICAL_CLASSIFICATION}")
        print(f"GROUP_LOSS_WEIGHT: {self.GROUP_LOSS_WEIGHT}")
        print(f"CLASS_LOSS_WEIGHT: {self.CLASS_LOSS_WEIGHT}")
        print(f"NUM_GROUPS: {self.NUM_GROUPS}")
        print(f"NUM_LABELS: {self.NUM_LABELS}")
        print(f"MAPPING_CSV_PATH: {self.MAPPING_CSV_PATH}")


config = Config()