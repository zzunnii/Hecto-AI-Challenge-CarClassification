import os
import pandas as pd
import torch
from transformers import ViTImageProcessor

from car_classification.config import config
from car_classification.dataset import create_dataloaders
from car_classification.model import HierarchicalCarClassifier
from car_classification.train import HierarchicalTrainer
from car_classification.utils import seed_everything, visualize_predictions
from car_classification.utils.mapping import load_class_mapping, save_mapping_info

train_dir = config.AUGMENTED_TRAIN_DIR
val_dir = config.VAL_DIR


def main():
    """계층적 분류를 위한 메인 함수"""

    # 시드 고정
    seed_everything(config.SEED)

    # 설정 출력
    print("=" * 60)
    print("Hierarchical Car Classification Training")
    print("=" * 60)

    config.print_hierarchical_config()

    print(f"\nTraining with:")
    print(f"  Train data: {train_dir}")
    print(f"  Val data: {val_dir}")
    print(f"  CSV mapping: {config.MAPPING_CSV_PATH}")

    # CSV 매핑 파일 확인
    if not os.path.exists(config.MAPPING_CSV_PATH):
        raise FileNotFoundError(f"CSV mapping file not found: {config.MAPPING_CSV_PATH}")

    # 매핑 정보 로드 및 저장
    print("\nLoading class mapping information...")
    mapping_info = load_class_mapping(config.MAPPING_CSV_PATH)

    print(f"Found {mapping_info['num_groups']} groups and {len(mapping_info['original_to_group'])} original classes")
    print(f"Groups: {mapping_info['unique_groups'][:10]}...")  # 처음 10개만 표시

    # 매핑 정보 저장 (나중에 추론에 사용)
    mapping_save_path = os.path.join(config.OUTPUT_DIR, 'mapping_info.json')
    save_mapping_info(mapping_info, mapping_save_path)
    print(f"Mapping info saved to: {mapping_save_path}")

    # 이미지 프로세서 로드
    processor = ViTImageProcessor.from_pretrained(config.MODEL_NAME)

    # 폴드별 데이터 로더 생성
    print("\nCreating data loaders...")
    train_loader, val_loader, train_dataset = create_dataloaders(
        train_dir, val_dir, config, processor=processor
    )

    # 클래스 가중치 가져오기
    class_weights = train_dataset.get_class_weights()
    group_weights = train_dataset.get_group_weights()

    if class_weights is not None:
        class_weights = class_weights.to("cuda")
        print(f"Using class weights for {len(class_weights)} classes")

    if group_weights is not None:
        group_weights = group_weights.to("cuda")
        print(f"Using group weights for {len(group_weights)} groups")

    # 모델 초기화
    print(f"\nInitializing hierarchical model...")
    model = HierarchicalCarClassifier(config).to("cuda")

    # 모델 정보 출력
    model_info = model.get_model_info()
    print(f"Model info:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    # 트레이너 초기화
    trainer = HierarchicalTrainer(
        model,
        train_loader,
        val_loader,
        config,
        class_weights=class_weights,
        group_weights=group_weights
    )

    # 학습 시작
    print("\nStarting training...")
    trainer.train()

    # ID-레이블 매핑
    id_to_class = {idx: class_name for idx, class_name in enumerate(train_dataset.class_names)}
    id_to_group = mapping_info['idx_to_group']

    # 매핑 정보 저장
    import json
    mappings = {
        'id_to_class': id_to_class,
        'id_to_group': id_to_group,
        'class_to_group': mapping_info['original_to_group']
    }

    mappings_path = os.path.join(config.OUTPUT_DIR, 'label_mappings.json')
    with open(mappings_path, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)
    print(f"\nLabel mappings saved to: {mappings_path}")

    # 예측 시각화 (선택적)
    if hasattr(config, 'VISUALIZE_PREDICTIONS') and config.VISUALIZE_PREDICTIONS:
        print("\nVisualizing predictions...")
        visualize_predictions(model, val_loader.dataset, id_to_class)

    print("\nTraining completed!")
    print("=" * 60)


def train_all_folds():
    """모든 폴드에 대해 학습 실행"""
    print("=" * 60)
    print("5-Fold Cross Validation Training")
    print("=" * 60)

    for fold_idx in range(config.N_FOLDS):
        print(f"\n{'=' * 60}")
        print(f"Training Fold {fold_idx}")
        print(f"{'=' * 60}")

        # 폴드 설정
        config.set_fold(fold_idx)

        # 폴드별 데이터 경로 설정
        global train_dir, val_dir
        train_dir = config.get_fold_data_dir(fold_idx, "train")
        val_dir = config.get_fold_data_dir(fold_idx, "val")

        # 경로 확인
        if not os.path.exists(train_dir):
            print(f"Warning: Train directory not found for fold {fold_idx}: {train_dir}")
            continue
        if not os.path.exists(val_dir):
            print(f"Warning: Val directory not found for fold {fold_idx}: {val_dir}")
            continue

        # 폴드 학습 실행
        try:
            main()
        except Exception as e:
            print(f"Error in fold {fold_idx}: {str(e)}")
            continue

    print("\n" + "=" * 60)
    print("All folds training completed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hierarchical Car Classification Training')
    parser.add_argument('--fold', type=int, default=2,
                        help='Specific fold to train (0-4). If not specified, train current fold in config.')
    parser.add_argument('--all-folds', action='store_true',
                        help='Train all 5 folds sequentially')

    args = parser.parse_args()

    if args.all_folds:
        train_all_folds()
    else:
        if args.fold is not None:
            config.set_fold(args.fold)
            train_dir = config.get_fold_data_dir(args.fold, "train")
            val_dir = config.get_fold_data_dir(args.fold, "val")
        main()