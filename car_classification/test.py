import os
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import warnings
from copy import deepcopy

from car_classification.config import config
from car_classification.model import HierarchicalCarClassifierImproved
from car_classification.dataset.augmentation import get_transform
from car_classification.utils.mapping import load_class_mapping


def load_hierarchical_model(model_path, base_config):
    """계층적 분류 모델 로드 (설정 완전 복원)"""
    print(f"Loading model from: {model_path}")

    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location='cuda')

    # 기본 config 복사 (원본 보존)
    inference_config = deepcopy(base_config)

    # 저장된 config가 있으면 완전히 복원
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        print("Restoring training configuration...")

        # 핵심 설정들 복원
        inference_config.NUM_LABELS = saved_config.get('NUM_LABELS', inference_config.NUM_LABELS)
        inference_config.NUM_GROUPS = saved_config.get('NUM_GROUPS', inference_config.NUM_GROUPS)
        inference_config.MODEL_NAME = saved_config.get('MODEL_NAME', inference_config.MODEL_NAME)

        # 이미지 크기 복원 (매우 중요!)
        if 'IMG_SIZE' in saved_config or '_IMG_SIZE' in saved_config:
            saved_img_size = saved_config.get('_IMG_SIZE') or saved_config.get('IMG_SIZE')
            if saved_img_size:
                inference_config.set_img_size(saved_img_size)
                print(f"Restored IMG_SIZE: {saved_img_size}")

        # 기타 중요한 설정들 복원
        inference_config.USE_PROGRESSIVE_TRAINING = saved_config.get('USE_PROGRESSIVE_TRAINING',
                                                                     inference_config.USE_PROGRESSIVE_TRAINING)
        inference_config.USE_DYNAMIC_LOSS_WEIGHTS = saved_config.get('USE_DYNAMIC_LOSS_WEIGHTS',
                                                                     inference_config.USE_DYNAMIC_LOSS_WEIGHTS)
        inference_config.USE_GATED_FUSION = saved_config.get('USE_GATED_FUSION', inference_config.USE_GATED_FUSION)
        inference_config.USE_DIFFERENT_LR = saved_config.get('USE_DIFFERENT_LR', inference_config.USE_DIFFERENT_LR)

        print(f"Restored configuration:")
        print(f"  NUM_LABELS: {inference_config.NUM_LABELS}")
        print(f"  NUM_GROUPS: {inference_config.NUM_GROUPS}")
        print(f"  IMG_SIZE: {inference_config.IMG_SIZE}")
        print(f"  MODEL_NAME: {inference_config.MODEL_NAME}")
    else:
        warnings.warn("No saved config found in checkpoint. Using current config.")

    # 모델 생성
    model = HierarchicalCarClassifierImproved(inference_config).to("cuda")

    # state_dict에서 불필요한 키 제거
    state_dict = checkpoint['model_state_dict']
    keys_to_remove = [k for k in state_dict.keys() if 'loss_fn' in k]
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
            print(f"Removed unnecessary key: {key}")

    # 가중치 로드
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model weights loaded successfully (strict mode)")
    except RuntimeError as e:
        print(f"Strict loading failed, trying non-strict mode: {e}")
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # 추론 모드로 설정 (모든 분류기 활성화)
    model.current_epoch = 999  # 충분히 큰 값으로 설정
    model.classifier.set_training_stage("both")

    print(f"Model set to inference mode - Stage: {model.classifier.training_stage}")

    return model, inference_config


def load_training_metadata(model_dir):
    """학습 시 사용된 메타데이터 로드"""
    print("Loading training metadata...")

    # 1. 레이블 매핑 로드 (최우선)
    label_mappings_path = os.path.join(model_dir, 'label_mappings.json')
    if os.path.exists(label_mappings_path):
        print(f"Loading label mappings from: {label_mappings_path}")
        with open(label_mappings_path, 'r', encoding='utf-8') as f:
            label_mappings = json.load(f)

        id_to_class = {int(k): v for k, v in label_mappings['id_to_class'].items()}
        id_to_group = {int(k): v for k, v in label_mappings['id_to_group'].items()}
        class_to_group = label_mappings['class_to_group']

        print(f"Loaded mappings: {len(id_to_class)} classes, {len(id_to_group)} groups")
        return id_to_class, id_to_group, class_to_group

    # 2. mapping_info.json 백업 시도
    mapping_info_path = os.path.join(model_dir, 'mapping_info.json')
    if os.path.exists(mapping_info_path):
        print(f"Loading mapping info from: {mapping_info_path}")
        with open(mapping_info_path, 'r', encoding='utf-8') as f:
            mapping_info = json.load(f)

        # mapping_info 구조에서 변환
        id_to_class = {}
        for idx, group_name in mapping_info['idx_to_group'].items():
            # 이 부분은 mapping_info 구조에 따라 조정 필요
            pass

        # TODO: mapping_info에서 id_to_class 복원 로직 구현
        warnings.warn("mapping_info.json found but conversion logic needed")

    # 3. 마지막 수단: CSV에서 직접 로드 (위험!)
    print("⚠️ Warning: No training metadata found! Using CSV mapping (may not match training order)")

    mapping_info = load_class_mapping(config.MAPPING_CSV_PATH)

    # 클래스 매핑 파일에서 학습 시 순서 로드 시도
    class_mapping_path = config.get_class_mapping_path()
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r', encoding='utf-8') as f:
            saved_mapping = json.load(f)
        id_to_class = {int(idx): name for idx, name in saved_mapping.items()}
        print(f"Loaded class order from: {class_mapping_path}")
    else:
        # 기본 정렬된 순서 (위험!)
        unique_classes = sorted(list(set(mapping_info['original_to_group'].keys())))
        id_to_class = {i: cls for i, cls in enumerate(unique_classes)}
        warnings.warn("Using default sorted order - this may not match training!")

    id_to_group = mapping_info['idx_to_group']
    class_to_group = mapping_info['original_to_group']

    return id_to_class, id_to_group, class_to_group


def find_best_model(model_dir):
    """최적의 모델 파일 찾기"""
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError(f"No model found in {model_dir}")

    print(f"Found {len(model_files)} model files:")
    for f in model_files:
        print(f"  {f}")

    def extract_score(filename):
        try:
            # 다양한 파일명 패턴 지원
            if '_classacc' in filename:
                return float(filename.split('_classacc')[1].split('_')[0])
            elif '_grpacc' in filename:
                return float(filename.split('_grpacc')[1].split('_')[0])
            elif '_logloss' in filename:
                return -float(filename.split('_logloss')[1].split('_')[0])
            elif '_loss' in filename:
                return -float(filename.split('_loss')[1].split('_')[0])
            elif '_acc' in filename:
                return float(filename.split('_acc')[1].split('_')[0])
            elif '_epoch' in filename:
                return float(filename.split('_epoch')[1].split('_')[0])
            else:
                return 0.0
        except:
            return 0.0

    # 점수 기준으로 정렬
    model_files.sort(key=extract_score, reverse=True)
    best_model = model_files[0]

    print(f"Selected best model: {best_model} (score: {extract_score(best_model):.4f})")
    return os.path.join(model_dir, best_model)


def predict_batch_hierarchical(model, image_paths, transform):
    """계층적 모델을 사용한 배치 예측"""
    batch_images = []
    valid_paths = []

    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')

            # 전처리 적용 시 예외 처리 강화
            try:
                image = transform(image)
                batch_images.append(image)
                valid_paths.append(img_path)
            except Exception as transform_error:
                print(f"Transform error for {img_path}: {transform_error}")
                continue

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

    if not batch_images:
        return [], [], []

    batch_tensor = torch.stack(batch_images).to("cuda")

    with torch.no_grad():
        try:
            outputs = model(batch_tensor)

            # 2차 분류 (최종 목표)
            class_logits = outputs["logits"]
            class_probs = torch.softmax(class_logits, dim=1)

            # 1차 분류 (그룹) - 참고용
            group_logits = outputs["group_logits"]
            group_probs = torch.softmax(group_logits, dim=1)

        except Exception as e:
            print(f"Prediction error: {e}")
            return [], [], []

    return valid_paths, class_probs.cpu().numpy(), group_probs.cpu().numpy()


def validate_inference_setup(inference_config, trained_classes, sample_submission_classes):
    """추론 설정 검증"""
    print("\n=== 추론 설정 검증 ===")

    issues = []

    # 1. 이미지 크기 확인
    print(f"Image size: {inference_config.IMG_SIZE}")

    # 2. 클래스 개수 일치 확인
    print(f"Trained classes: {len(trained_classes)}")
    print(f"Config NUM_LABELS: {inference_config.NUM_LABELS}")
    if len(trained_classes) != inference_config.NUM_LABELS:
        issues.append(f"Class count mismatch: trained={len(trained_classes)}, config={inference_config.NUM_LABELS}")

    # 3. 서브미션과의 클래스 overlap 확인
    trained_set = set(trained_classes)
    submission_set = set(sample_submission_classes)

    overlap = trained_set & submission_set
    missing_in_submission = trained_set - submission_set
    missing_in_training = submission_set - trained_set

    print(f"Class overlap: {len(overlap)}/{len(submission_set)} submission classes covered")

    if missing_in_submission:
        print(f"⚠️ Trained classes not in submission: {len(missing_in_submission)}")
        if len(missing_in_submission) <= 5:
            print(f"  Examples: {list(missing_in_submission)}")

    if missing_in_training:
        print(f"⚠️ Submission classes not trained: {len(missing_in_training)}")
        if len(missing_in_training) <= 5:
            print(f"  Examples: {list(missing_in_training)}")

    # 4. 모델 설정 확인
    print(f"Progressive training: {inference_config.USE_PROGRESSIVE_TRAINING}")
    print(f"Dynamic loss weights: {inference_config.USE_DYNAMIC_LOSS_WEIGHTS}")
    print(f"Gated fusion: {inference_config.USE_GATED_FUSION}")

    if issues:
        for issue in issues:
            print(f"❌ {issue}")
        warnings.warn("Issues found in inference setup!")
    else:
        print("✅ Inference setup validation passed")


def main(fold_idx=0):
    """메인 추론 함수"""
    # 경로 설정
    test_dir = r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open\test"
    FOLD_IDX = fold_idx

    # 모델 디렉토리
    model_dir = os.path.join(config.MODEL_OUTPUT_DIR, f"resnet_hierarchical_fold_{FOLD_IDX}")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # 최적 모델 선택
    model_path = find_best_model(model_dir)
    print(f"Using model: {model_path}")

    # 학습 메타데이터 로드
    id_to_class, id_to_group, class_to_group = load_training_metadata(model_dir)

    # 학습된 클래스 리스트 (인덱스 순서대로)
    trained_classes = [id_to_class[i] for i in range(len(id_to_class))]
    print(f"학습된 클래스 수: {len(trained_classes)}")
    print(f"학습된 그룹 수: {len(set(id_to_group.values()))}")
    print(f"첫 10개 클래스: {trained_classes[:10]}")

    # 모델 로드 (설정 완전 복원)
    model, inference_config = load_hierarchical_model(model_path, config)

    # 서브미션 클래스 로드
    sample_submission = pd.read_csv(r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open\sample_submission.csv")
    submission_classes = [col for col in sample_submission.columns if col != 'ID']

    # 추론 설정 검증
    validate_inference_setup(inference_config, trained_classes, submission_classes)

    # 전처리 설정 (복원된 config 사용)
    transform = get_transform(inference_config, is_train=False)
    print(f"Using restored image size: {inference_config.IMG_SIZE}")

    # 테스트 이미지 목록
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    test_files.sort()
    print(f"처리할 테스트 이미지 수: {len(test_files)}")

    # 결과 저장용
    results = []
    batch_size = 32
    group_predictions = []

    print("테스트 이미지 추론 중...")
    for i in tqdm(range(0, len(test_files), batch_size)):
        batch_files = test_files[i:i + batch_size]
        batch_paths = [os.path.join(test_dir, f) for f in batch_files]

        valid_paths, class_probs, group_probs = predict_batch_hierarchical(
            model, batch_paths, transform
        )

        for img_path, class_prob, group_prob in zip(valid_paths, class_probs, group_probs):
            img_name = os.path.basename(img_path)
            img_id = os.path.splitext(img_name)[0]

            # 예측 분석
            predicted_group_idx = np.argmax(group_prob)
            predicted_group = id_to_group.get(predicted_group_idx, "Unknown")
            predicted_class_idx = np.argmax(class_prob)
            predicted_class = trained_classes[predicted_class_idx]
            predicted_class_prob = class_prob[predicted_class_idx]

            # 일관성 체크
            expected_group = class_to_group.get(predicted_class, None)
            is_consistent = (expected_group == predicted_group)

            group_predictions.append({
                'id': img_id,
                'predicted_class': predicted_class,
                'predicted_class_prob': float(predicted_class_prob),
                'predicted_group': predicted_group,
                'expected_group': expected_group,
                'consistent': is_consistent
            })

            # 결과 행 생성
            result_row = {'ID': img_id}

            # 서브미션 클래스별 확률 할당
            for submission_class in submission_classes:
                if submission_class in trained_classes:
                    class_idx = trained_classes.index(submission_class)
                    result_row[submission_class] = float(class_prob[class_idx])
                else:
                    result_row[submission_class] = 0.0

            results.append(result_row)

    # 결과 정리
    results_df = pd.DataFrame(results)
    results_df = results_df[sample_submission.columns]

    # 파일명에 검증 정보 포함
    model_name = os.path.basename(model_path).replace('.pth', '')
    output_path = f'submission_hierarchical_fold{FOLD_IDX}_{inference_config.EARLY_STOPPING_METRIC}_{model_name}.csv'
    results_df.to_csv(output_path, index=False)

    print(f"\n결과가 {output_path}에 저장되었습니다.")
    print(f"총 {len(results_df)} 개 이미지 처리 완료")

    # 일관성 분석
    consistency_df = pd.DataFrame(group_predictions)
    consistent_count = consistency_df['consistent'].sum()
    total_count = len(consistency_df)

    print(f"\n=== 계층적 일관성 분석 ===")
    print(f"일관된 예측: {consistent_count}/{total_count} ({consistent_count / total_count * 100:.2f}%)")
    print(f"평균 예측 확률: {consistency_df['predicted_class_prob'].mean():.4f}")

    # 추가 검증 정보
    print(f"\n=== 최종 검증 정보 ===")
    print(f"사용된 이미지 크기: {inference_config.IMG_SIZE}")
    print(f"Early stopping metric: {inference_config.EARLY_STOPPING_METRIC}")
    print(f"모델 파일: {os.path.basename(model_path)}")
    print(f"학습된 클래스 수: {len(trained_classes)}")
    print(f"서브미션 클래스 수: {len(submission_classes)}")

    # 일관성 분석 결과 저장
    consistency_path = f'hierarchical_consistency_fold{FOLD_IDX}_{model_name}.csv'
    consistency_df.to_csv(consistency_path, index=False)
    print(f"일관성 분석 결과: {consistency_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hierarchical Model Inference')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index to use for inference (default: 0)')

    args = parser.parse_args()
    main(fold_idx=args.fold)