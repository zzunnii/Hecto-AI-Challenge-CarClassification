import os
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import random
from car_classification.utils import seed_everything

def create_cross_validation_splits(data_dir, output_dir, n_folds=5, seed=42):
    """5폴드 교차검증 분할 생성"""

    seed_everything(seed)
    print(f"Creating {n_folds}-fold cross validation splits...")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 클래스별 이미지 파일 수집
    class_to_files = defaultdict(list)
    all_files = []
    all_labels = []

    # 클래스 디렉토리 확인
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    class_names = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])

    if not class_names:
        raise ValueError(f"No class directories found in: {data_dir}")

    print(f"Found {len(class_names)} classes: {class_names}")

    # 클래스별 이미지 수집
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Class {class_name}: {len(image_files)} images")

        for img_file in image_files:
            file_info = {
                'class_name': class_name,
                'class_idx': class_idx,
                'file_path': os.path.join(class_name, img_file),
                'full_path': os.path.join(class_dir, img_file)
            }
            class_to_files[class_name].append(file_info)
            all_files.append(file_info)
            all_labels.append(class_idx)

    print(f"Total images: {len(all_files)}")

    # StratifiedKFold로 분할
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_splits = {}

    print(f"\nCreating {n_folds} folds...")
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(all_files, all_labels)):
        train_files = [all_files[i] for i in train_indices]
        val_files = [all_files[i] for i in val_indices]

        fold_splits[f'fold_{fold_idx}'] = {
            'train': train_files,
            'val': val_files,
            'train_size': len(train_files),
            'val_size': len(val_files)
        }

        print(f"Fold {fold_idx}: Train {len(train_files)}, Val {len(val_files)}")

    # JSON 직렬화를 위해 numpy int 변환
    for fold_key in fold_splits:
        for file_info in fold_splits[fold_key]['train'] + fold_splits[fold_key]['val']:
            file_info['class_idx'] = int(file_info['class_idx'])

    # 분할 정보 저장
    splits_file = os.path.join(output_dir, 'fold_splits.json')
    with open(splits_file, 'w', encoding='utf-8') as f:
        json.dump(fold_splits, f, indent=2, ensure_ascii=False)

    # 클래스 매핑 저장
    class_mapping = {idx: name for idx, name in enumerate(class_names)}
    mapping_file = os.path.join(output_dir, 'class_mapping.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)

    print(f"\nSplits saved to: {splits_file}")
    print(f"Class mapping saved to: {mapping_file}")

    return fold_splits, class_mapping


def verify_splits(splits_file):
    """분할 검증"""
    if not os.path.exists(splits_file):
        print(f"Splits file not found: {splits_file}")
        return

    with open(splits_file, 'r', encoding='utf-8') as f:
        fold_splits = json.load(f)

    print("=== Fold Verification ===")

    # 전체 통계
    total_train = sum(fold_splits[f'fold_{i}']['train_size'] for i in range(5))
    total_val = sum(fold_splits[f'fold_{i}']['val_size'] for i in range(5))

    print(f"Total training samples across all folds: {total_train}")
    print(f"Total validation samples across all folds: {total_val}")
    print(f"Total samples: {total_train + total_val}")

    # 각 폴드별 통계
    print(f"\nFold-wise distribution:")
    for i in range(5):
        fold_data = fold_splits[f'fold_{i}']
        train_size = fold_data['train_size']
        val_size = fold_data['val_size']
        val_ratio = val_size / (train_size + val_size)
        print(f"  Fold {i}: Train {train_size}, Val {val_size} (Val ratio: {val_ratio:.3f})")

    # 클래스별 분포 확인 (첫 번째 폴드만)
    fold_0 = fold_splits['fold_0']
    train_classes = defaultdict(int)
    val_classes = defaultdict(int)

    for file_info in fold_0['train']:
        train_classes[file_info['class_name']] += 1

    for file_info in fold_0['val']:
        val_classes[file_info['class_name']] += 1

    print(f"\nFold 0 class distribution:")
    for class_name in sorted(train_classes.keys()):
        train_count = train_classes[class_name]
        val_count = val_classes[class_name]
        total_count = train_count + val_count
        ratio = train_count / total_count if total_count > 0 else 0
        print(f"  {class_name}: Train {train_count}, Val {val_count} (Train ratio: {ratio:.3f})")


def load_fold_data(fold_idx, splits_dir):
    """특정 폴드의 데이터 로드"""
    splits_file = os.path.join(splits_dir, 'fold_splits.json')

    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found: {splits_file}")

    with open(splits_file, 'r', encoding='utf-8') as f:
        fold_splits = json.load(f)

    if f'fold_{fold_idx}' not in fold_splits:
        raise ValueError(f"Fold {fold_idx} not found in splits")

    return fold_splits[f'fold_{fold_idx}']


def main():
    """메인 실행 함수"""
    from car_classification.config import config

    # 경로 확인
    print("=== 경로 확인 ===")
    config.verify_paths()

    print(f"\n=== 교차검증 분할 생성 ===")

    # 5폴드 분할 생성
    try:
        fold_splits, class_mapping = create_cross_validation_splits(
            data_dir=config.DATA_DIR,
            output_dir=config.SPLITS_DIR,
            n_folds=config.N_FOLDS,
            seed=config.SEED
        )

        # 검증
        splits_file = config.get_splits_file_path()
        verify_splits(splits_file)

        print("\n✓ 5-fold cross validation splits created successfully")

    except Exception as e:
        print(f"✗ Error creating splits: {e}")
        raise


if __name__ == "__main__":
    main()