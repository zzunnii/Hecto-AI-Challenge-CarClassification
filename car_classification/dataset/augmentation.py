import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumentationsTransform:
    """Albumentations 변환을 PyTorch 호환 가능하게 래핑"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        # PIL Image를 numpy array로 변환
        if isinstance(image, Image.Image):
            image = np.array(image)

        # BGR -> RGB 변환 (cv2를 사용하는 경우)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 이미 RGB라고 가정 (PIL에서 온 경우)
            pass

        # Albumentations 적용
        transformed = self.transform(image=image)
        return transformed['image']


def get_albumentations_transform(config, is_train=True):
    """Albumentations를 사용한 변환 파이프라인"""

    img_size = config.IMG_SIZE  # (height, width) 튜플
    height, width = img_size

    # ImageNet 정규화 값
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        transform_list = [
            # 1) 비율 유지 리사이즈
            A.LongestMaxSize(max_size=max(height, width), p=1.0),
            A.PadIfNeeded(min_height=height, min_width=width,
                          border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
        ]


        # 3. 기본 증강 (확률적 적용)
        if config.USE_RANDOM_FLIP:
            transform_list.append(A.HorizontalFlip(p=config.FLIP_PROBABILITY))

        if config.USE_ROTATION:
            transform_list.append(
                A.Rotate(
                    limit=config.ROTATION_DEGREES,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    p=0.3
                )
            )

        if config.USE_RANDOM_CROP:
            crop_h, crop_w = int(height * config.RANDOM_CROP_RATIO), \
                              int(width * config.RANDOM_CROP_RATIO)
            transform_list.extend([
                A.RandomCrop(height=crop_h, width=crop_w, p=0.5),
                A.Resize(height=height, width=width, p=1.0),
            ])

        # 4. 기하학적 변환
        if config.USE_SHIFT_SCALE_ROTATE:
            transform_list.append(
                A.ShiftScaleRotate(
                    shift_limit=config.SHIFT_LIMIT,
                    scale_limit=config.SCALE_LIMIT,
                    rotate_limit=config.ROTATE_LIMIT,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    p=0.3
                )
            )

        # 5. 원근 변환 (매우 미세하게)
        transform_list.append(
            A.Perspective(scale=(0.03, 0.06),           # 🔄 0.02–0.05 → 0.03–0.06
                          keep_size=True,
                          border_mode=cv2.BORDER_CONSTANT,
                          fill=0,
                          p=0.2)
        )

        # 6. 색상 및 밝기 조절
        if config.USE_COLOR_AUGMENTATION:
            transform_list.append(
                A.ColorJitter(
                    brightness=0.1,  # 밝기 ±10%
                    contrast=0.1,  # 대비 ±10%
                    saturation=0.1,  # 채도 ±10%
                    hue=0.01,  # 색조 ±1%
                    p=0.4
                )
            )

        if config.USE_BACKGROUND_BRIGHTNESS:
            transform_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,               # 🔄 0.15 → 0.3 (config 범위 반영)
                    contrast_limit=0.2,
                    p=0.3
                )
            )

        # 7. 노이즈 및 블러
        transform_list.extend([
            A.OneOf([
                A.GaussNoise(std_range=(0.05, 0.2), mean_range=(0.0, 0.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.2),

            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
            ], p=0.1),
        ])

        # 8. 정규화 및 텐서 변환
        transform_list.extend([
            A.Normalize(mean=mean, std=std),   # 🔄
            ToTensorV2()
        ])

        # 9. 랜덤 지우기 (Albumentations에서는 Cutout 사용)
        cutout_transform = A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(0.1, 0.1),
            hole_width_range=(0.1, 0.1),
            fill=0,
            p=0.1
        )
        transform_list.insert(-2, cutout_transform)


    else:
        # 검증/테스트용 변환 (증강 없음)
        transform_list = [
            # 비율 유지하며 리사이징
            A.LongestMaxSize(max_size=max(height, width), interpolation=cv2.INTER_LINEAR, p=1.0),

            # 패딩으로 원하는 크기 맞추기
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0
            ),

            # 정규화 및 텐서 변환
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]

    return A.Compose(transform_list)


class Mixup:
    """Mixup 데이터 증강"""

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, batch):
        """배치에 Mixup 적용"""
        images, labels = batch
        batch_size = len(images)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha, batch_size)
        else:
            lam = np.ones(batch_size)

        lam = torch.from_numpy(lam).float().to(images.device)
        lam = lam.view(-1, 1, 1, 1)

        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam.squeeze()


def get_transform(config, is_train=True):
    """변환 파이프라인 선택 (Albumentations 우선 사용)"""

    # Albumentations 사용 가능 여부 확인
    try:
        # Albumentations 변환 생성
        albu_transform = get_albumentations_transform(config, is_train)
        return AlbumentationsTransform(albu_transform)
    except ImportError:
        print("Warning: Albumentations not available, falling back to TorchVision")
        return get_albumentations_transform(config, is_train)
    except Exception as e:
        print(f"Warning: Error creating Albumentations transform: {e}")
        print("Falling back to TorchVision transform")
        return get_albumentations_transform(config, is_train)


def get_validation_transform(config):
    """검증/테스트용 변환"""
    return get_transform(config, is_train=False)