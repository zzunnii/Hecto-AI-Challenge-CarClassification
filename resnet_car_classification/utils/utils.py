import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import koreanize_matplotlib
from PIL import Image


def seed_everything(seed):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_predictions(model, val_dataset, id_to_group, num_samples=10):
    """검증 데이터에 대한 예측 시각화"""
    model.eval()
    device = next(model.parameters()).device
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)

    plt.figure(figsize=(20, 15))

    for i, idx in enumerate(indices):
        sample = val_dataset[idx]
        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
        label = sample["label"]

        with torch.no_grad():
            outputs = model(pixel_values)

        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)
        predicted_class_id = logits.argmax(-1).item()

        # 이미지 경로 찾기
        img_path = val_dataset.image_paths[idx] if hasattr(val_dataset, 'image_paths') else None

        if img_path is None:
            continue

        # 이미지 로드
        img = Image.open(img_path).convert('RGB')

        # 예측 및 실제 레이블
        true_class = id_to_group[label]
        predicted_class = id_to_group[predicted_class_id]

        plt.subplot(num_samples // 5 + 1, 5, i + 1)
        plt.imshow(img)
        color = "green" if true_class == predicted_class else "red"
        plt.title(f"True: {true_class}\nPred: {predicted_class}\nProb: {probs[0, predicted_class_id]:.2f}", color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("prediction_samples.png", dpi=200)
    plt.show()