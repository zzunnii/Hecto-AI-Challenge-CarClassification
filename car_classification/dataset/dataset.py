import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm import tqdm
import pandas as pd

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

        # 🔧 클래스 매핑이 제공되면 사용, 아니면 생성
        if class_mapping is not None:
            self.class_to_idx = class_mapping
            self.idx_to_class = {idx: name for name, idx in class_mapping.items()}
        else:
            # 클래스 디렉토리 기반 매핑
            self._init_class_mapping()

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
        """이미지 경로 및 레이블 수집"""
        class_counts = {name: 0 for name in self.class_to_idx}
        group_counts = {group: 0 for group in self.mapping_info['unique_groups']}
        missing_classes = []
        missing_mappings = []

        for class_name in tqdm(os.listdir(self.root_dir), desc="Loading dataset"):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            # 🔧 클래스가 매핑에 없으면 경고하고 건너뛰기
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

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_id)
                    self.group_labels.append(group_id)
                    class_counts[class_name] += 1
                    group_counts[group_name] += 1

        # 🔧 누락된 클래스가 있으면 경고
        if missing_classes:
            print(f"⚠️ Warning: Found classes not in class mapping: {missing_classes}")
        if missing_mappings:
            print(f"⚠️ Warning: Found classes not in CSV mapping: {missing_mappings}")

        print("\nClass distribution:")
        for name in sorted(self.class_to_idx.keys()):
            count = class_counts.get(name, 0)
            if count > 0:  # 0이 아닌 것만 출력
                print(f"  {name}: {count} images")

        print(f"\nGroup distribution:")
        for group in sorted(self.mapping_info['unique_groups']):
            count = group_counts.get(group, 0)
            if count > 0:
                print(f"  {group}: {count} images")

        print(f"\nLoaded {len(self.image_paths)} images")
        print(f"  - {len(self.class_to_idx)} original classes")
        print(f"  - {len(self.mapping_info['unique_groups'])} groups")

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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]  # original class
        group_label = self.group_labels[idx]  # new group

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        elif self.processor:
            processed = self.processor(images=image, return_tensors="pt")
            image = processed["pixel_values"].squeeze()

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

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return {
            "pixel_values": image,
            "label": label,
            "group_label": group_label
        }


def create_dataloaders(train_data_dir, val_data_dir, config, processor=None):
    """폴드별로 분리된 데이터로 데이터로더 생성"""

    # 🔧 저장된 클래스 매핑 로드
    import json
    class_mapping_path = config.get_class_mapping_path()

    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r', encoding='utf-8') as f:
            saved_mapping = json.load(f)
            # 인덱스가 문자열로 저장되므로 int로 변환하고, name->idx 형태로 변환
            class_to_idx = {name: int(idx) for idx, name in saved_mapping.items()}
    else:
        raise FileNotFoundError(f"Class mapping file not found: {class_mapping_path}")

    # 변환 정의
    train_transform = get_transform(config, is_train=True)
    val_transform = get_transform(config, is_train=False)

    # 데이터셋 생성 시 동일한 클래스 매핑 사용
    train_dataset = CarBrandDataset(
        root_dir=train_data_dir,
        processor=processor,
        transform=train_transform,
        is_train=True,
        class_mapping=class_to_idx,
        csv_mapping_path=config.MAPPING_CSV_PATH  # CSV 경로 전달
    )

    val_dataset = CarBrandDataset(
        root_dir=val_data_dir,
        processor=processor,
        transform=val_transform,
        is_train=False,
        class_mapping=class_to_idx,
        csv_mapping_path=config.MAPPING_CSV_PATH  # CSV 경로 전달
    )

    # 클래스 수 업데이트
    config.update_num_labels(train_dataset.num_classes)
    config.update_num_groups(train_dataset.num_groups)  # 그룹 수도 업데이트

    # 데이터로더 생성
    num_workers = getattr(config, 'NUM_WORKERS', 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    print(f"Train dataloader: {len(train_loader)} batches")
    print(f"Validation dataloader: {len(val_loader)} batches")

    return train_loader, val_loader, train_dataset