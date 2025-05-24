import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


class AspectPreservingResize:
    """비율 유지하며 리사이징 + 패딩"""

    def __init__(self, target_size=256, padding_color=(0, 0, 0)):
        self.target_size = target_size
        self.padding_color = padding_color

    def __call__(self, img):
        w, h = img.size

        # 비율 계산
        ratio = min(self.target_size / w, self.target_size / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        # 리사이즈 (비율 유지)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # 중앙 정렬로 패딩
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2)
        )

        return ImageOps.expand(img, padding, self.padding_color)


class AdaptiveBrightnessAdjust:
    """적응적 밝기 조절 - 이미지 히스토그램 기반"""

    def __init__(self, probability=0.3, strength_range=(0.85, 1.15)):
        self.probability = probability
        self.strength_range = strength_range

    def __call__(self, img):
        if random.random() < self.probability:
            # 이미지의 평균 밝기 계산
            img_array = np.array(img)
            avg_brightness = np.mean(img_array)

            # 너무 어두우면 밝게, 너무 밝으면 어둡게
            if avg_brightness < 100:
                brightness_factor = random.uniform(1.05, 1.2)
            elif avg_brightness > 180:
                brightness_factor = random.uniform(0.8, 0.95)
            else:
                brightness_factor = random.uniform(*self.strength_range)

            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(brightness_factor)
        return img


class GentleColorJitter:
    """부드러운 색상 조절 - 자동차 이미지에 특화"""

    def __init__(self, probability=0.4):
        self.probability = probability

    def __call__(self, img):
        if random.random() < self.probability:
            # 자동차 이미지에 적합한 미세한 범위
            brightness = random.uniform(0.95, 1.05)  # 매우 미세한 밝기 조절
            contrast = random.uniform(0.95, 1.1)  # 약간의 대비 조절
            saturation = random.uniform(0.9, 1.1)  # 채도 조절

            # 🔧 수정: hue는 튜플로 전달하거나 양수로 전달
            # 방법 1: 튜플로 전달 (권장)
            hue_range = (-0.01, 0.01)

            # 방법 2: 양수로 전달 (자동으로 (-값, +값) 범위로 해석됨)
            # hue_abs = 0.01

            transform = transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue_range  # 🔧 수정된 부분
            )
            return transform(img)
        return img


class CarSpecificRotation:
    """자동차에 특화된 회전 변환 - 작은 각도만"""

    def __init__(self, max_angle=10, probability=0.3):
        self.max_angle = max_angle
        self.probability = probability

    def __call__(self, img):
        if random.random() < self.probability:
            # 작은 각도로만 회전 (자동차는 보통 수평을 유지)
            angle = random.uniform(-self.max_angle, self.max_angle)
            return transforms.functional.rotate(img, angle, fill=(0, 0, 0))
        return img


class GaussianNoise:
    """가우시안 노이즈 추가"""

    def __init__(self, probability=0.2, noise_std=0.02):
        self.probability = probability
        self.noise_std = noise_std

    def __call__(self, img):
        if random.random() < self.probability:
            img_array = np.array(img).astype(np.float32) / 255.0

            # 가우시안 노이즈 생성
            noise = np.random.normal(0, self.noise_std, img_array.shape).astype(np.float32)

            # 노이즈 추가
            noisy_img = img_array + noise
            noisy_img = np.clip(noisy_img, 0, 1)

            # PIL 이미지로 변환
            return Image.fromarray((noisy_img * 255).astype(np.uint8))
        return img


class LightBlur:
    """가벼운 블러 효과"""

    def __init__(self, probability=0.15, blur_radius_range=(0.3, 0.8)):
        self.probability = probability
        self.blur_radius_range = blur_radius_range

    def __call__(self, img):
        if random.random() < self.probability:
            radius = random.uniform(*self.blur_radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


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
    """개선된 데이터 증강 파이프라인 - 기본적인 증강만 포함"""

    base_transforms = []

    # 1. 비율 유지 리사이징
    if config.USE_ASPECT_PRESERVING:
        base_transforms.append(AspectPreservingResize(
            target_size=config.IMG_SIZE,
            padding_color=config.PADDING_COLOR
        ))
    else:
        base_transforms.append(transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)))

    if is_train:
        # 2. 학습 중 기본 증강만
        train_transforms = []

        # 랜덤 크롭 (가장 기본적인 증강)
        if config.USE_RANDOM_CROP:
            train_transforms.append(
                transforms.RandomResizedCrop(
                    config.IMG_SIZE,
                    scale=config.RANDOM_CROP_SCALE,
                    ratio=(0.8, 1.2)  # 자동차에 적합한 비율
                )
            )

        # 수평 뒤집기
        if config.USE_RANDOM_FLIP:
            train_transforms.append(
                transforms.RandomHorizontalFlip(p=config.FLIP_PROBABILITY)
            )

        # 작은 각도 회전
        if config.USE_ROTATION:
            train_transforms.append(
                CarSpecificRotation(
                    max_angle=config.ROTATION_DEGREES,
                    probability=0.3
                )
            )

        # 아핀 변환 (미세한 변형만)
        if config.USE_AFFINE:
            train_transforms.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=config.AFFINE_TRANSLATE,
                    scale=config.AFFINE_SCALE,
                    fill=tuple(config.PADDING_COLOR)
                )
            )

        # 원근 변환 (매우 미세하게)
        train_transforms.append(
            transforms.RandomPerspective(
                distortion_scale=0.05,  # 더 작은 변형
                p=0.2,
                fill=tuple(config.PADDING_COLOR)
            )
        )

        # 적응적 밝기 조절 (Post augmentation에서 이미 색상이 변경됨)
        train_transforms.append(AdaptiveBrightnessAdjust(
            probability=0.3,
            strength_range=(0.9, 1.1)
        ))

        # 부드러운 색상 조절
        train_transforms.append(GentleColorJitter(probability=0.4))

        # 가우시안 노이즈 (실제 카메라 노이즈 시뮬레이션)
        train_transforms.append(GaussianNoise(
            probability=0.2,
            noise_std=0.01
        ))

        # 가벼운 블러 (카메라 흔들림 시뮬레이션)
        train_transforms.append(LightBlur(
            probability=0.1,
            blur_radius_range=(0.2, 0.5)
        ))

        base_transforms.extend(train_transforms)

    # 3. 텐서 변환 및 정규화
    base_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 4. 랜덤 지우기 (마지막에 적용)
    if is_train:
        base_transforms.append(
            transforms.RandomErasing(
                p=0.1,
                scale=(0.02, 0.1),  # 작은 영역만
                ratio=(0.3, 3.3),
                value='random'
            )
        )

    return transforms.Compose(base_transforms)


def get_validation_transform(config):
    """검증/테스트용 변환 (증강 없음)"""
    transforms_list = []

    # 비율 유지 리사이징
    if config.USE_ASPECT_PRESERVING:
        transforms_list.append(AspectPreservingResize(
            target_size=config.IMG_SIZE,
            padding_color=config.PADDING_COLOR
        ))
    else:
        transforms_list.append(transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)))

    # 정규화만 적용
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transforms.Compose(transforms_list)