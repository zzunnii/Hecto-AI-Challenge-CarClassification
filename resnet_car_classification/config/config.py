import os
from typing import Optional, List, Dict, Tuple


class Config:
    # 기본 베이스 경로
    BASE_DIR = r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open"

    # 데이터 경로들
    DATA_DIR = os.path.join(BASE_DIR, "train")  # 원본 데이터
    MASK_DIR = os.path.join(BASE_DIR, "extracted_mask")  # 마스크 파일들
    AUGMENTED_DATA_DIR = os.path.join(BASE_DIR, "augmented_data")  # 증강 후 저장될 곳
    SPLITS_DIR = os.path.join(BASE_DIR, "split_info")  # 폴드 정보 저장
    MODEL_OUTPUT_DIR = r"./resnet_car_classification/outputs"  # 모델 아웃풋

    AUGMENTED_TRAIN_DIR = r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open\augmented_data\fold_0\train_augmented"
    VAL_DIR = r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open\augmented_data\fold_0\val"

    # 계층적 분류를 위한 매핑 파일
    MAPPING_CSV_PATH = r"C:\Users\tjdwn\GitHub\Hecto-AI-Challenge-CarClassification\car_class_mapping.csv"

    OUTPUT_DIR = "./resnet_car_classification/outputs"

    # 모델 설정
    MODEL_NAME = "resnet50"  # ResNet 계열 사용
    NUM_LABELS = 10  # original class 수 (자동 업데이트)
    NUM_GROUPS = 10  # new group 수 (자동 업데이트)

    # 🆕 개선된 계층적 분류 설정
    USE_HIERARCHICAL_CLASSIFICATION = True

    # 🔧 수정된 손실 가중치 (1차 분류 가중치 더 줄임)
    GROUP_LOSS_WEIGHT = 0.1  # 그룹 손실 가중치
    CLASS_LOSS_WEIGHT = 0.9  # 클래스 손실 가중치

    # 🆕 점진적 학습 설정
    USE_PROGRESSIVE_TRAINING = True
    GROUP_ONLY_EPOCHS = 10  # 처음 7 에폭은 그룹만 학습
    GROUP_DOMINANCE_EPOCHS = 20  # 15 에폭까지는 그룹 가중치 높게

    # 🆕 동적 가중치 스케줄링
    USE_DYNAMIC_LOSS_WEIGHTS = True
    MIN_GROUP_WEIGHT = 0.01  # 최소 그룹 가중치
    MAX_GROUP_WEIGHT = 0.3  # 최대 그룹 가중치

    # 🆕 그룹 정보 활용 개선
    USE_GATED_FUSION = True  # 게이트된 특징 융합
    GROUP_ATTENTION_WEIGHT = 1.0  # 그룹 어텐션 가중치
    GROUP_CONFIDENCE_THRESHOLD = 0.8  # 그룹 정보 활용 신뢰도 임계값

    # 🆕 적응형 클래스 가중치
    USE_ADAPTIVE_CLASS_WEIGHTS = False  # 클래스별 적응형 가중치 사용
    MAX_CLASS_WEIGHT = 2.0  # 최대 클래스 가중치
    CLASS_WEIGHT_THRESHOLD = 0.7  # 가중치 적용 F1 점수 임계값
    CLASS_WEIGHT_MEMORY_FACTOR = 0.7  # 이전 가중치 영향력
    MAX_DEGRADATION_COUNT = 3  # 연속 성능 저하 허용 횟수

    # 🆕 어텐션 메커니즘 설정
    USE_ENHANCED_ATTENTION = True  # 향상된 어텐션 사용
    ATTENTION_HEADS = 8  # 어텐션 헤드 수
    ATTENTION_DROPOUT = 0.1  # 어텐션 드롭아웃

    # 🆕 학습률 차등 적용
    USE_DIFFERENT_LR = True
    GROUP_LEARNING_RATE = 1e-5  # 그룹 분류기 학습률 (더 낮게)
    CLASS_LEARNING_RATE = 2e-4  # 클래스 분류기 학습률 (더 높게)

    # 🆕 조기 정지 개선
    USE_ADAPTIVE_EARLY_STOPPING = True
    GROUP_STAGE_METRIC = "group_acc"  # 그룹 단계 모니터링 지표
    CLASS_STAGE_METRIC = "class_acc"  # 클래스 단계 모니터링 지표
    EARLY_STOPPING_METRIC = "class_acc"  # 최종 조기 정지 지표
    EARLY_STOPPING_PATIENCE = 15  # 조기 정지 인내심

    # 학습 파라미터
    SEED = 42
    BATCH_SIZE = 32  # ResNet은 더 큰 이미지로 인해 배치 크기 줄임
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 1e-4  # ResNet 파인튜닝을 위해 더 작은 learning rate
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 300
    WARMUP_RATIO = 0.1

    # 이미지 전처리 - 모델별 동적 설정
    _IMG_SIZE = (384, 384)  # 기본값 384x384로 변경
    USE_ASPECT_PRESERVING = True
    PADDING_COLOR = (0, 0, 0)

    # 데이터 증강 (학습 중)
    USE_COLOR_AUGMENTATION = False
    USE_MIXUP = False
    MIXUP_ALPHA = 0.1

    # 새로운 증강 설정
    USE_BACKGROUND_BRIGHTNESS = True
    BRIGHTNESS_RANGE = (0.7, 1.3)
    RANDOM_CROP_RATIO = 0.8

    USE_SHIFT_SCALE_ROTATE = True
    SHIFT_LIMIT = 0.05
    SCALE_LIMIT = 0.05
    ROTATE_LIMIT = 5

    # 학습 중 기본 증강
    USE_RANDOM_CROP = True
    RANDOM_CROP_SCALE = (0.8, 1.0)
    USE_RANDOM_FLIP = True
    FLIP_PROBABILITY = 0.2
    USE_ROTATION = True
    ROTATION_DEGREES = 10
    USE_AFFINE = True
    AFFINE_TRANSLATE = (0.05, 0.05)
    AFFINE_SCALE = (0.95, 1.05)

    # 손실 함수 설정
    USE_FOCAL_LOSS = True
    FOCAL_LOSS_GAMMA = 2.0
    FOCAL_LOSS_ALPHA = None
    LABEL_SMOOTHING = 0.0

    # 평가 설정
    EVAL_STEPS = 0
    SAVE_STEPS = 0
    LOGGING_STEPS = 50

    # 최대 저장 모델 수
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

    @property
    def IMG_SIZE(self) -> Tuple[int, int]:
        """모델에 따른 이미지 크기 반환"""
        if self._IMG_SIZE is not None:
            return self._IMG_SIZE

        model_name_lower = self.MODEL_NAME.lower()

        # ResNet 계열은 더 큰 이미지 크기 사용
        if 'resnet' in model_name_lower:
            return (384, 384)  # 수정: 512x512 -> 384x384
        elif 'efficientnet' in model_name_lower:
            if 'b7' in model_name_lower:
                return (456, 456)  # 수정: 600x600 -> 456x456
            elif 'b6' in model_name_lower:
                return (420, 420)  # 수정: 528x528 -> 420x420
            elif 'b5' in model_name_lower:
                return (384, 384)  # 수정: 456x456 -> 384x384
            elif 'b4' in model_name_lower:
                return (380, 380)
            else:
                return (224, 224)
        elif 'convnext' in model_name_lower:
            return (384, 384)
        else:
            # ViT, Swin 등 기본값
            return (224, 224)

    def set_img_size(self, size: Tuple[int, int]):
        """이미지 크기 수동 설정"""
        self._IMG_SIZE = size

    def get_model_type(self) -> str:
        """모델 타입 반환"""
        model_name_lower = self.MODEL_NAME.lower()

        if 'resnet' in model_name_lower:
            return 'resnet'
        elif 'efficientnet' in model_name_lower:
            return 'efficientnet'
        elif 'convnext' in model_name_lower:
            return 'convnext'
        elif 'swin' in model_name_lower:
            return 'swin'
        elif 'vit' in model_name_lower:
            return 'vit'
        else:
            return 'unknown'

    def is_resnet_model(self) -> bool:
        """ResNet 계열 모델인지 확인"""
        return self.get_model_type() == 'resnet'

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
        model_type = self.get_model_type()
        self.OUTPUT_DIR = os.path.join(self.MODEL_OUTPUT_DIR, f"{model_type}_hierarchical_fold_{fold_idx}")
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def get_fold_data_dir(self, fold_idx: int, split_type: str = "train"):
        """폴드별 데이터 디렉토리 반환"""
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
        print("\n=== 개선된 계층적 분류 설정 ===")
        print(f"USE_HIERARCHICAL_CLASSIFICATION: {self.USE_HIERARCHICAL_CLASSIFICATION}")
        print(f"GROUP_LOSS_WEIGHT: {self.GROUP_LOSS_WEIGHT}")
        print(f"CLASS_LOSS_WEIGHT: {self.CLASS_LOSS_WEIGHT}")
        print(f"USE_PROGRESSIVE_TRAINING: {self.USE_PROGRESSIVE_TRAINING}")
        print(f"USE_DYNAMIC_LOSS_WEIGHTS: {self.USE_DYNAMIC_LOSS_WEIGHTS}")
        print(f"USE_DIFFERENT_LR: {self.USE_DIFFERENT_LR}")
        print(f"EARLY_STOPPING_METRIC: {self.EARLY_STOPPING_METRIC}")
        print(f"NUM_GROUPS: {self.NUM_GROUPS}")
        print(f"NUM_LABELS: {self.NUM_LABELS}")
        print(f"MODEL_NAME: {self.MODEL_NAME}")
        print(f"MODEL_TYPE: {self.get_model_type()}")
        print(f"IMG_SIZE: {self.IMG_SIZE}")
        print(f"USE_ENHANCED_ATTENTION: {self.USE_ENHANCED_ATTENTION}")
        print(f"USE_ADAPTIVE_CLASS_WEIGHTS: {self.USE_ADAPTIVE_CLASS_WEIGHTS}")


config = Config()