import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

from car_classification.config import config
from car_classification.dataset.augmentation import get_transform
from car_classification.utils.mapping import load_class_mapping


class CarBrandDataset(Dataset):
    def __init__(self, root_dir, processor=None, transform=None, is_train=True, class_mapping=None,
                 csv_mapping_path=None):
        self.root_dir = root_dir
        self.processor = processor
        self.transform = transform
        self.is_train = is_train

        # 이미지 경로 및 레이블 저장
        self.image_paths = []
        self.labels = []  # original class labels
        self.group_labels = []  # new group labels
        self.class_weights = None

        # CSV 매핑 정보 로드
        if csv_mapping_path and os.path.exists(csv_mapping_path):
            self.mapping_info = load_class_mapping(csv_mapping_path)
        else:
            raise FileNotFoundError(f"CSV mapping file not found: {csv_mapping_path}")

        # 클래스 매핑이 제공되면 사용, 아니면 생성
        if class_mapping is not None:
            self.class_to_idx = class_mapping
            self.idx_to_class = {idx: name for name, idx in class_mapping.items()}
        else:
            # 클래스 디렉토리 기반 매핑
            self._init_class_mapping()

        # 중복 처리 방지를 위한 세트
        self._processed_files = set()

        # 데이터 로드
        self._load_data()

        # 클래스 가중치 계산 (훈련 데이터에서만)
        if self.is_train:
            self._compute_class_weights()

    def _init_class_mapping(self):
        """클래스 디렉토리 이름 기반 인덱스 매핑"""
        class_names = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

    def _load_data(self):
        """이미지 경로 및 레이블 수집 (중복 제거)"""
        class_counts = defaultdict(int)
        group_counts = defaultdict(int)
        missing_classes = []
        missing_mappings = []
        duplicate_count = 0

        print(f"Loading dataset from: {self.root_dir}")

        # 실제 존재하는 클래스 디렉토리만 처리
        actual_class_dirs = [
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]

        print(f"Found {len(actual_class_dirs)} class directories")

        for class_name in tqdm(actual_class_dirs, desc="Loading dataset"):
            class_path = os.path.join(self.root_dir, class_name)

            # 클래스가 매핑에 없으면 경고하고 건너뛰기
            if class_name not in self.class_to_idx:
                missing_classes.append(class_name)
                continue

            # CSV 매핑에서 group 찾기
            if class_name not in self.mapping_info['original_to_group']:
                missing_mappings.append(class_name)
                continue

            class_id = self.class_to_idx[class_name]
            group_name = self.mapping_info['original_to_group'][class_name]
            group_id = self.mapping_info['group_to_idx'][group_name]

            # 이미지 파일 처리
            image_files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            for img_name in image_files:
                img_path = os.path.join(class_path, img_name)

                # 중복 제거를 위한 고유 키 생성
                file_key = f"{class_name}/{img_name}"

                if file_key in self._processed_files:
                    duplicate_count += 1
                    continue

                # 파일 존재 및 유효성 검사
                if not os.path.exists(img_path):
                    continue

                try:
                    # 이미지 파일이 실제로 열리는지 확인
                    with Image.open(img_path) as test_img:
                        test_img.verify()
                except (IOError, OSError):
                    print(f"Warning: Corrupted image skipped: {img_path}")
                    continue

                # 유효한 파일 추가
                self.image_paths.append(img_path)
                self.labels.append(class_id)
                self.group_labels.append(group_id)
                class_counts[class_name] += 1
                group_counts[group_name] += 1
                self._processed_files.add(file_key)

        # 경고 메시지 출력
        if missing_classes:
            print(f"⚠️ Warning: Found classes not in class mapping: {missing_classes[:5]}...")
        if missing_mappings:
            print(f"⚠️ Warning: Found classes not in CSV mapping: {missing_mappings[:5]}...")
        if duplicate_count > 0:
            print(f"⚠️ Warning: Skipped {duplicate_count} duplicate files")

        # 통계 출력
        print(f"\nDataset loaded successfully:")
        print(f"  Total images: {len(self.image_paths):,}")
        print(f"  Unique classes: {len([c for c in class_counts if class_counts[c] > 0])}")
        print(f"  Unique groups: {len([g for g in group_counts if group_counts[g] > 0])}")

        # 클래스별 분포 (상위 10개만)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 classes by count:")
        for class_name, count in sorted_classes[:10]:
            print(f"  {class_name}: {count:,} images")

        # 그룹별 분포
        sorted_groups = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\nGroup distribution:")
        for group_name, count in sorted_groups:
            print(f"  {group_name}: {count:,} images")

    def _compute_class_weights(self):
        """클래스 불균형을 위한 가중치 계산"""
        if len(self.labels) == 0:
            print("Warning: No labels found, skipping class weight computation")
            return

        # Original class weights
        label_counts = np.bincount(self.labels)
        n_samples = len(self.labels)
        n_classes = len(self.class_to_idx)

        # 0으로 나누기 방지
        label_counts = np.where(label_counts == 0, 1, label_counts)

        self.class_weights = n_samples / (n_classes * label_counts)
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)

        # Group weights
        group_label_counts = np.bincount(self.group_labels)
        n_groups = len(self.mapping_info['unique_groups'])

        group_label_counts = np.where(group_label_counts == 0, 1, group_label_counts)

        self.group_weights = n_samples / (n_groups * group_label_counts)
        self.group_weights = torch.tensor(self.group_weights, dtype=torch.float)

        print(f"Computed class weights for {len(self.class_weights)} classes")
        print(f"Computed group weights for {len(self.group_weights)} groups")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]  # original class
        group_label = self.group_labels[idx]  # new group

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 오류 발생시 대체 이미지 로드
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))

        # 변환 적용
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to {img_path}: {e}")
                # 기본 변환 적용
                from torchvision import transforms
                basic_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image = basic_transform(image)
        elif self.processor:
            try:
                processed = self.processor(images=image, return_tensors="pt")
                image = processed["pixel_values"].squeeze()
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                # 기본 변환 적용
                from torchvision import transforms
                basic_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image = basic_transform(image)

        return {
            "pixel_values": image,
            "label": label,  # original class (최종 목표)
            "group_label": group_label  # new group (1차 분류)
        }

    def get_class_weights(self):
        return self.class_weights

    def get_group_weights(self):
        return self.group_weights if hasattr(self, 'group_weights') else None

    @property
    def num_classes(self):
        return len(self.class_to_idx)

    @property
    def num_groups(self):
        return len(self.mapping_info['unique_groups'])

    @property
    def class_names(self):
        return list(self.class_to_idx.keys())

    @property
    def group_names(self):
        return self.mapping_info['unique_groups']


class ValidationDataset(Dataset):
    """검증 데이터셋 (중복 제거 포함)"""

    def __init__(self, full_dataset, indices, transform):
        self.full_dataset = full_dataset
        self.indices = indices
        self.transform = transform

        self.image_paths = [full_dataset.image_paths[i] for i in indices]
        self.labels = [full_dataset.labels[i] for i in indices]
        self.group_labels = [full_dataset.group_labels[i] for i in indices]
        self.class_to_idx = full_dataset.class_to_idx
        self.idx_to_class = full_dataset.idx_to_class
        self.mapping_info = full_dataset.mapping_info

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        img_path = self.full_dataset.image_paths[orig_idx]
        label = self.full_dataset.labels[orig_idx]
        group_label = self.full_dataset.group_labels[orig_idx]

        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error in validation dataset {img_path}: {e}")
            # 기본 변환 적용
            from torchvision import transforms
            basic_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            image = basic_transform(image)

        return {
            "pixel_values": image,
            "label": label,
            "group_label": group_label
        }


def create_dataloaders(train_data_dir, val_data_dir, config, processor=None):
    """폴드별로 분리된 데이터로 데이터로더 생성 (중복 제거 포함)"""

    # 저장된 클래스 매핑 로드
    import json
    class_mapping_path = config.get_class_mapping_path()

    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r', encoding='utf-8') as f:
            saved_mapping = json.load(f)
            # 인덱스가 문자열로 저장되므로 int로 변환하고, name->idx 형태로 변환
            class_to_idx = {name: int(idx) for idx, name in saved_mapping.items()}
    else:
        raise FileNotFoundError(f"Class mapping file not found: {class_mapping_path}")

    print(f"Creating transforms for image size: {config.IMG_SIZE}")

    # 모델 타입에 따른 처리 방식 결정
    model_type = config.get_model_type()
    use_processor = processor is not None and model_type in ['vit', 'swin', 'beit', 'deit']

    if use_processor:
        print(f"Using HuggingFace processor for {model_type} model")
        train_transform = None
        val_transform = None
    else:
        print(f"Using Albumentations transforms for {model_type} model")
        # 변환 정의
        train_transform = get_transform(config, is_train=True)
        val_transform = get_transform(config, is_train=False)
        processor = None  # transform 사용 시 processor는 None

    print(f"Using approach: {'HuggingFace Processor' if use_processor else 'Custom Transforms'}")

    # 데이터셋 생성
    print("Creating training dataset...")
    train_dataset = CarBrandDataset(
        root_dir=train_data_dir,
        processor=processor,
        transform=train_transform,
        is_train=True,
        class_mapping=class_to_idx,
        csv_mapping_path=config.MAPPING_CSV_PATH
    )

    print("Creating validation dataset...")
    val_dataset = CarBrandDataset(
        root_dir=val_data_dir,
        processor=processor,
        transform=val_transform,
        is_train=False,
        class_mapping=class_to_idx,
        csv_mapping_path=config.MAPPING_CSV_PATH
    )

    # 클래스 수 업데이트
    config.update_num_labels(train_dataset.num_classes)
    config.update_num_groups(train_dataset.num_groups)

    # 데이터로더 생성
    num_workers = getattr(config, 'NUM_WORKERS', 4)

    # 배치 크기 조정 (큰 이미지의 경우)
    effective_batch_size = config.BATCH_SIZE
    img_size = config.IMG_SIZE
    if isinstance(img_size, (tuple, list)):
        total_pixels = img_size[0] * img_size[1]
    else:
        total_pixels = img_size * img_size

    # 큰 이미지의 경우 배치 크기 자동 조정
    if total_pixels > 384 * 384:
        effective_batch_size = max(1, config.BATCH_SIZE // 2)
        print(f"Large image detected, reducing batch size to {effective_batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 일관된 배치 크기 유지
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataLoader Summary:")
    print(f"  Train dataset: {len(train_dataset):,} images")
    print(f"  Validation dataset: {len(val_dataset):,} images")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Validation batches: {len(val_loader):,}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Classes: {train_dataset.num_classes}")
    print(f"  Groups: {train_dataset.num_groups}")
    print(f"  Processing method: {'HuggingFace Processor' if use_processor else 'Custom Transforms'}")

    return train_loader, val_loader, train_dataset