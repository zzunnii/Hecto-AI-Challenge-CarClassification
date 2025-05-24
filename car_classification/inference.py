import os
import argparse
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor

from brand_classification.config import config
from brand_classification.model import CarBrandClassifier
from brand_classification.dataset.augmentation import get_transform

def load_model(model_path, config):
    """저장된 모델 로드"""
    model = CarBrandClassifier(config).to("cuda")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_image(image_path, model, processor, transform):
    """단일 이미지에 대한 예측"""
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to("cuda")

    # 예측
    with torch.no_grad():
        outputs = model(image)

    logits = outputs["logits"]
    probs = torch.softmax(logits, dim=1)[0]

    return probs.cpu().numpy()


def predict_directory(dir_path, model, processor, transform, id_to_group):
    """디렉토리 내 모든 이미지에 대한 예측"""
    results = []

    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in tqdm(image_files, desc="Predicting"):
        img_path = os.path.join(dir_path, img_file)
        probs = predict_image(img_path, model, processor, transform)

        # 결과 저장
        result = {
            "file": img_file,
            "predicted_class": id_to_group[np.argmax(probs)],
            "confidence": np.max(probs)
        }

        # 각 클래스별 확률 추가
        for i, prob in enumerate(probs):
            result[id_to_group[i]] = prob

        results.append(result)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Inference script for car brand classification')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file')
    parser.add_argument('--mapping_csv', type=str, default=config.MAPPING_CSV, help='CSV mapping file')

    args = parser.parse_args()

    # 클래스 매핑 로드
    mapping_df = pd.read_csv(args.mapping_csv)
    unique_groups = sorted(mapping_df['new_group'].unique())
    id_to_group = {idx: group for idx, group in enumerate(unique_groups)}

    # 모델 및 프로세서 로드
    processor = ViTImageProcessor.from_pretrained(config.MODEL_NAME)
    transform = get_transform(config, is_train=False)

    # 모델 로드
    config.NUM_LABELS = len(unique_groups)
    model = load_model(args.model_path, config)

    # 예측 실행
    results_df = predict_directory(args.test_dir, model, processor, transform, id_to_group)

    # 결과 저장
    results_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()