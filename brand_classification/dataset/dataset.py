import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from brand_classification.config import config
from brand_classification.dataset.augmentation import get_transform


class CarBrandDataset(Dataset):
    def __init__(self, root_dir, mapping_df, processor=None, transform=None, is_train=True):
        self.root_dir = root_dir
        self.mapping_df = mapping_df
        self.processor = processor
        self.transform = transform
        self.is_train = is_train

        # 이미지 경로 및 레이블 저장
        self.image_paths = []
        self.labels = []
        self.class_weights = None

        # 데이터 로드
        self._load_data()

        # 클래스 가중치 계산
        self._compute_class_weights()

    def _load_data(self):
        """데이터 로드 및 클래스 매핑"""
        original_to_group = dict(zip(self.mapping_df['original_class'], self.mapping_df['new_group']))
        unique_groups = sorted(self.mapping_df['new_group'].unique())
        self.group_to_idx = {group: idx for idx, group in enumerate(unique_groups)}
        self.idx_to_group = {idx: group for group, idx in self.group_to_idx.items()}

        # 클래스별 카운터 초기화
        class_counts = {group: 0 for group in unique_groups}

        for class_name in tqdm(os.listdir(self.root_dir), desc="Loading dataset"):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            # 클래스의 그룹 찾기
            if class_name in original_to_group:
                group = original_to_group[class_name]
                group_id = self.group_to_idx[group]

                # 해당 클래스의 모든 이미지 추가
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(group_id)
                        class_counts[group] += 1
            else:
                print(f"Warning: Class {class_name} not found in mapping")

        # 클래스 분포 출력
        print("Class distribution:")
        for group, count in class_counts.items():
            print(f"  {group}: {count} images")

        # 총 이미지 및 클래스 수 출력
        print(f"Loaded {len(self.image_paths)} images with {len(unique_groups)} classes/groups")

    def _compute_class_weights(self):
        """클래스 불균형을 위한 가중치 계산"""
        # 레이블 빈도 계산
        label_counts = np.bincount(self.labels)
        n_samples = len(self.labels)
        n_classes = len(label_counts)

        # 가중치 계산: 빈도가 적을수록 가중치 증가
        self.class_weights = n_samples / (n_classes * label_counts)
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 이미지 로드
        image = Image.open(img_path).convert('RGB')

        # 변환 적용
        if self.transform:
            image = self.transform(image)
        elif self.processor:
            # Hugging Face 프로세서 사용
            processed = self.processor(images=image, return_tensors="pt")
            image = processed["pixel_values"].squeeze()

        return {"pixel_values": image, "label": label}

    def get_class_weights(self):
        """클래스 가중치 반환"""
        return self.class_weights

    @property
    def num_classes(self):
        return len(self.group_to_idx)

    @property
    def class_names(self):
        return list(self.group_to_idx.keys())


def create_dataloaders(config, mapping_df, processor=None):
    """데이터 로더 생성"""
    # 데이터셋 로드
    train_transform = get_transform(config, is_train=True)
    val_transform = get_transform(config, is_train=False)

    full_dataset = CarBrandDataset(
        root_dir=config.DATA_DIR,
        mapping_df=mapping_df,
        processor=processor,
        transform=train_transform,
        is_train=True
    )

    # 클래스 수 업데이트
    config.update_num_labels(full_dataset.num_classes)

    # 데이터셋 분할 (8:2)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # 시드 고정하여 데이터셋 분할
    generator = torch.Generator().manual_seed(config.SEED)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # 검증 데이터셋의 변환 함수 변경
    def val_dataset_with_transform(idx):
        item = full_dataset[val_dataset.indices[idx]]
        img_path = full_dataset.image_paths[val_dataset.indices[idx]]
        label = item["label"]

        image = Image.open(img_path).convert('RGB')
        image = val_transform(image)

        return {"pixel_values": image, "label": label}

    val_dataset.transform = val_transform
    val_dataset.__getitem__ = val_dataset_with_transform
    val_dataset.__len__ = lambda: len(val_dataset.indices)

    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train dataloader: {len(train_loader)} batches")
    print(f"Validation dataloader: {len(val_loader)} batches")

    return train_loader, val_loader, full_dataset