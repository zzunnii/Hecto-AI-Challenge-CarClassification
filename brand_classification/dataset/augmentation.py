import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance

class CarColorTransform:
    """차량 색상 변환 클래스"""

    def __init__(self, probability=0.7):
        self.probability = probability
        # 차량 색상별 HSV 조정값 (hue, saturation, value)
        self.color_params = {
            "original": (0.0, 1.0, 1.0),
            "white": (0.0, 0.0, 1.0),
            "black": (0.0, 0.0, 0.2),
            "silver": (0.0, 0.1, 0.8),
            "gray": (0.0, 0.1, 0.6),
            "red": (0.0, 1.0, 0.8),
            "blue": (0.6, 0.8, 0.8),
            "dark_blue": (0.6, 1.0, 0.6),
            "green": (0.35, 0.8, 0.7),
            "brown": (0.08, 0.8, 0.6),
            "yellow": (0.15, 1.0, 1.0),
            "orange": (0.07, 1.0, 1.0),
            "gold": (0.12, 0.8, 0.8),
            "purple": (0.8, 0.7, 0.7),
            "maroon": (0.97, 0.8, 0.5)
        }

    def __call__(self, img):
        if random.random() > self.probability:
            return img

        # 랜덤 색상 선택
        color_name = random.choice(list(self.color_params.keys()))
        if color_name == "original":
            return img

        # HSV 공간으로 변환
        img_hsv = img.convert("HSV")
        h, s, v = img_hsv.split()

        # 색상 변환 매개변수
        h_shift, s_factor, v_factor = self.color_params[color_name]

        # 색상 조정 (오버플로우 수정)
        h_data = np.array(h).astype(np.int16)
        h_data = (h_data + int(h_shift * 255)) % 256
        h = Image.fromarray(h_data.astype(np.uint8))

        # 채도 조정
        s_data = np.array(s).astype(np.float32)
        s_data = np.clip(s_data * s_factor, 0, 255).astype(np.uint8)
        s = Image.fromarray(s_data)

        # 명도 조정
        v_data = np.array(v).astype(np.float32)
        v_data = np.clip(v_data * v_factor, 0, 255).astype(np.uint8)
        v = Image.fromarray(v_data)

        # 조정된 채널 합치기
        img_hsv_new = Image.merge("HSV", (h, s, v))

        # RGB로 변환
        return img_hsv_new.convert("RGB")


class Mixup:
    """Mixup 데이터 증강"""

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, batch):
        """배치에 Mixup 적용"""
        images, labels = batch
        batch_size = len(images)

        # 혼합 가중치 생성
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha, batch_size)
        else:
            lam = np.ones(batch_size)

        lam = torch.from_numpy(lam).float().to(images.device)
        lam = lam.view(-1, 1, 1, 1)

        # 인덱스 셔플
        index = torch.randperm(batch_size).to(images.device)

        # 이미지 및 레이블 혼합
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam.squeeze()


def get_transform(config, is_train=True):
    """데이터 증강 파이프라인 구성"""
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            CarColorTransform(probability=0.7) if config.USE_COLOR_AUGMENTATION else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform