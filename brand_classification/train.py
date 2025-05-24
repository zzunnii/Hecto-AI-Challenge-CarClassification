import os
import pandas as pd
import torch
from transformers import ViTImageProcessor

from brand_classification.config import config
from brand_classification.dataset import create_dataloaders
from brand_classification.model import CarBrandClassifier
from brand_classification.train import Trainer
from brand_classification.utils import seed_everything, visualize_predictions

def main():
    # 시드 고정
    seed_everything(config.SEED)

    # CSV 파일에서 클래스 매핑 정보 로드
    mapping_df = pd.read_csv(config.MAPPING_CSV)
    print(f"Loaded mapping for {len(mapping_df)} classes")

    # 이미지 프로세서 로드
    processor = ViTImageProcessor.from_pretrained(config.MODEL_NAME)

    # 데이터 로더 생성
    train_loader, val_loader, full_dataset = create_dataloaders(config, mapping_df, processor)

    # 클래스 가중치 가져오기
    class_weights = full_dataset.get_class_weights().to("cuda") if hasattr(full_dataset, 'get_class_weights') else None

    # 모델 초기화
    model = CarBrandClassifier(config).to("cuda")

    # 트레이너 초기화
    trainer = Trainer(model, train_loader, val_loader, config, class_weights)

    # 학습 시작
    trainer.train()

    # ID-레이블 매핑
    id_to_group = {idx: group for idx, group in enumerate(full_dataset.class_names)}

    # 예측 시각화
    visualize_predictions(model, val_loader.dataset, id_to_group)

    print("Training completed!")


if __name__ == "__main__":
    main()