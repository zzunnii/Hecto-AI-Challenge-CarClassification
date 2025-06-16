import os
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
import re

from resnet_car_classification.config import config as resnet_config
from resnet_car_classification.model import HierarchicalCarClassifierImproved as ResNetModel
from resnet_car_classification.dataset.augmentation import get_transform as resnet_transform

from resnext_car_classification.config import config as resnext_config
from resnext_car_classification.model import HierarchicalCarClassifierImproved as ResNeXtModel
from resnext_car_classification.dataset.augmentation import get_transform as resnext_transform


def extract_performance_from_filename(filename):
    """파일명에서 성능 정보 추출"""
    performance = {'class_acc': 0.0, 'group_acc': 0.0, 'epoch': 0}

    try:
        # classacc0.9630 형태 매치 (정확히 숫자.숫자 형태만)
        class_acc_match = re.search(r'classacc(\d+\.\d+)', filename)
        if class_acc_match:
            performance['class_acc'] = float(class_acc_match.group(1))

        # grp0.9943 형태 매치 (정확히 숫자.숫자 형태만)
        group_acc_match = re.search(r'grp(\d+\.\d+)', filename)
        if group_acc_match:
            performance['group_acc'] = float(group_acc_match.group(1))

        # epoch45 형태 매치
        epoch_match = re.search(r'epoch(\d+)', filename)
        if epoch_match:
            performance['epoch'] = int(epoch_match.group(1))

    except (ValueError, AttributeError) as e:
        print(f"Warning: Could not parse performance from filename '{filename}': {e}")
        # 기본값 유지

    return performance


def calculate_performance_weights(model_paths, strategy='composite'):
    """성능 기반 가중치 계산"""
    if len(model_paths) <= 1:
        return [1.0] * len(model_paths)

    model_performances = []
    for model_path in model_paths:
        filename = os.path.basename(model_path)
        perf = extract_performance_from_filename(filename)
        model_performances.append(perf)
        print(f"  파일: {filename}")
        print(f"    class_acc: {perf['class_acc']}, group_acc: {perf['group_acc']}, epoch: {perf['epoch']}")

    if strategy == 'class_acc':
        scores = [p['class_acc'] for p in model_performances]
    elif strategy == 'group_acc':
        scores = [p['group_acc'] for p in model_performances]
    elif strategy == 'composite':
        scores = [0.7 * p['class_acc'] + 0.3 * p['group_acc'] for p in model_performances]
    else:
        return [1.0 / len(model_paths)] * len(model_paths)

    # 모든 점수가 0이거나 동일한 경우 동일 가중치
    if max(scores) == 0 or max(scores) <= min(scores):
        print("  성능 정보가 없거나 모든 모델 성능이 동일 -> 동일 가중치 사용")
        return [1.0 / len(model_paths)] * len(model_paths)

    # 성능 차이가 있는 경우만 가중치 적용
    scores = np.array(scores)
    exp_scores = np.exp(scores * 10)
    weights = exp_scores / np.sum(exp_scores)

    return weights.tolist()


def create_swa_model(model_paths, config, model_class, weighting_strategy='equal'):
    """SWA 모델 생성"""
    if not model_paths:
        return None

    print(f"Creating SWA model with {len(model_paths)} models using {weighting_strategy} weighting")

    if weighting_strategy == 'equal':
        weights = [1.0 / len(model_paths)] * len(model_paths)
    else:
        weights = calculate_performance_weights(model_paths, weighting_strategy)

    print("SWA weights:")
    for i, (path, weight) in enumerate(zip(model_paths, weights)):
        filename = os.path.basename(path)
        print(f"  {filename}: {weight:.4f}")

    # 첫 번째 모델 로드
    first_checkpoint = torch.load(model_paths[0], map_location='cpu')
    averaged_state_dict = {}
    original_dtypes = {}

    for key in first_checkpoint['model_state_dict'].keys():
        if 'loss_fn' not in key:
            weight_tensor = first_checkpoint['model_state_dict'][key]
            averaged_state_dict[key] = weights[0] * weight_tensor.clone().float()
            original_dtypes[key] = weight_tensor.dtype

    # 나머지 모델들과 가중 평균
    for i, model_path in enumerate(model_paths[1:], 1):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            for key in averaged_state_dict.keys():
                if key in checkpoint['model_state_dict']:
                    weight_tensor = checkpoint['model_state_dict'][key]
                    averaged_state_dict[key] += weights[i] * weight_tensor.float()
        except Exception as e:
            print(f"Error loading {model_path} for SWA: {e}")
            continue

    # 원래 데이터 타입으로 복원
    for key in averaged_state_dict.keys():
        if original_dtypes[key] != torch.float32:
            try:
                averaged_state_dict[key] = averaged_state_dict[key].to(original_dtypes[key])
            except:
                pass

    # 모델 생성 및 가중치 로드
    checkpoint = torch.load(model_paths[0], map_location='cuda')
    inference_config = deepcopy(config)

    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        inference_config.NUM_LABELS = saved_config.get('NUM_LABELS', inference_config.NUM_LABELS)
        inference_config.NUM_GROUPS = saved_config.get('NUM_GROUPS', inference_config.NUM_GROUPS)
        inference_config.MODEL_NAME = saved_config.get('MODEL_NAME', inference_config.MODEL_NAME)

        if 'IMG_SIZE' in saved_config or '_IMG_SIZE' in saved_config:
            saved_img_size = saved_config.get('_IMG_SIZE') or saved_config.get('IMG_SIZE')
            if saved_img_size:
                inference_config.set_img_size(saved_img_size)

    model = model_class(inference_config).to("cuda")
    model.load_state_dict(averaged_state_dict, strict=False)
    model.eval()

    return model, inference_config


def discover_fold_models(base_config, architecture='resnet'):
    """폴드별 모델 발견"""
    fold_models = {}
    model_output_base = base_config.MODEL_OUTPUT_DIR

    for item in os.listdir(model_output_base):
        item_path = os.path.join(model_output_base, item)
        if os.path.isdir(item_path) and 'hierarchical_fold_' in item:
            # 폴드 번호 추출
            if 'fold_' in item:
                fold_num = item.split('fold_')[-1]
                try:
                    fold_idx = int(fold_num)
                    model_files = [f for f in os.listdir(item_path) if f.endswith('.pth')]
                    if model_files:
                        model_paths = [os.path.join(item_path, f) for f in model_files]
                        fold_models[fold_idx] = model_paths
                        print(f"Found fold {fold_idx}: {len(model_paths)} models")
                except ValueError:
                    continue

    return fold_models


def predict_with_model(model, config, test_dir, transform_func, trained_classes):
    """단일 모델로 예측"""
    test_images = []
    for img_file in os.listdir(test_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_images.append(img_file)

    test_images.sort()
    num_classes = len(trained_classes)
    predictions = {}
    batch_size = 16

    print(f"Predicting {len(test_images)} images...")

    for i in tqdm(range(0, len(test_images), batch_size), desc="Processing batches"):
        batch_images = test_images[i:i + batch_size]
        batch_tensors = []

        for img_name in batch_images:
            img_path = os.path.join(test_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform_func(config, is_train=False)(image)
                batch_tensors.append(image_tensor)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                dummy_tensor = torch.zeros((3, config.IMG_SIZE[0], config.IMG_SIZE[1]))
                batch_tensors.append(dummy_tensor)

        if batch_tensors:
            batch_tensor = torch.stack(batch_tensors).to("cuda")

            with torch.no_grad():
                try:
                    outputs = model(batch_tensor)
                    probs = torch.softmax(outputs["logits"], dim=1).cpu().numpy()

                    for j, img_name in enumerate(batch_images):
                        predictions[img_name] = probs[j]

                except Exception as e:
                    print(f"Error predicting batch: {e}")
                    dummy_probs = np.ones(num_classes) / num_classes
                    for img_name in batch_images:
                        predictions[img_name] = dummy_probs

    return predictions


def average_predictions(prediction_list):
    """예측 결과들의 평균"""
    if not prediction_list:
        return {}

    averaged = {}
    all_images = set()
    for pred_dict in prediction_list:
        all_images.update(pred_dict.keys())

    for img_name in all_images:
        probs_list = []
        for pred_dict in prediction_list:
            if img_name in pred_dict:
                probs_list.append(pred_dict[img_name])

        if probs_list:
            averaged[img_name] = np.mean(probs_list, axis=0)

    return averaged


def create_submission_from_predictions(predictions, trained_classes, submission_classes, output_path):
    """예측 결과로부터 서브미션 파일 생성"""
    submission_data = []

    for img_name in sorted(predictions.keys()):
        img_id = os.path.splitext(img_name)[0]
        result_row = {'ID': img_id}

        probabilities = predictions[img_name]

        for submission_class in submission_classes:
            if submission_class in trained_classes:
                class_idx = trained_classes.index(submission_class)
                result_row[submission_class] = float(probabilities[class_idx])
            else:
                result_row[submission_class] = 0.0

        submission_data.append(result_row)

    submission_df = pd.DataFrame(submission_data)
    ordered_columns = ['ID'] + submission_classes
    submission_df = submission_df[ordered_columns]
    submission_df.to_csv(output_path, index=False)

    print(f"Saved submission: {output_path}")
    print(f"Images: {len(submission_data)}, Classes: {len(submission_classes)}")


def main():
    print("SWA ENSEMBLE SUBMISSION GENERATOR")
    print("=" * 80)

    # 기본 설정
    test_data_dir = os.path.join(resnet_config.BASE_DIR, "test")
    output_dir = "swa_ensemble_submissions"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(test_data_dir):
        raise FileNotFoundError(f"Test directory not found: {test_data_dir}")

    # 서브미션 클래스 정보
    sample_submission_path = os.path.join(resnet_config.BASE_DIR, "sample_submission.csv")
    sample_submission = pd.read_csv(sample_submission_path)
    submission_classes = [col for col in sample_submission.columns if col != 'ID']

    # 클래스 매핑 정보
    class_mapping_path = resnet_config.get_class_mapping_path()
    with open(class_mapping_path, 'r', encoding='utf-8') as f:
        saved_mapping = json.load(f)
        trained_classes = [saved_mapping[str(i)] for i in range(len(saved_mapping))]

    print(f"Submission classes: {len(submission_classes)}")
    print(f"Trained classes: {len(trained_classes)}")

    # 모델 발견
    resnet_fold_models = discover_fold_models(resnet_config, 'resnet')
    resnext_fold_models = discover_fold_models(resnext_config, 'resnext')

    print(f"ResNet folds: {list(resnet_fold_models.keys())}")
    print(f"ResNeXt folds: {list(resnext_fold_models.keys())}")

    # 예측 결과 저장
    all_predictions = {}

    # 1. ResNet SWA 예측들
    print("\n" + "=" * 50)
    print("RESNET SWA PREDICTIONS")
    print("=" * 50)

    resnet_swa_equal_preds = []
    resnet_swa_weighted_preds = []

    for fold_idx, model_paths in resnet_fold_models.items():
        print(f"\nProcessing ResNet Fold {fold_idx}...")

        # Equal weighted SWA
        swa_model_equal, config = create_swa_model(
            model_paths, resnet_config, ResNetModel, 'equal'
        )
        if swa_model_equal:
            pred_equal = predict_with_model(
                swa_model_equal, config, test_data_dir, resnet_transform, trained_classes
            )
            resnet_swa_equal_preds.append(pred_equal)
            del swa_model_equal

        # Performance weighted SWA
        swa_model_weighted, config = create_swa_model(
            model_paths, resnet_config, ResNetModel, 'composite'
        )
        if swa_model_weighted:
            pred_weighted = predict_with_model(
                swa_model_weighted, config, test_data_dir, resnet_transform, trained_classes
            )
            resnet_swa_weighted_preds.append(pred_weighted)
            del swa_model_weighted

        torch.cuda.empty_cache()

    # 2. ResNeXt SWA 예측들
    print("\n" + "=" * 50)
    print("RESNEXT SWA PREDICTIONS")
    print("=" * 50)

    resnext_swa_equal_preds = []
    resnext_swa_weighted_preds = []

    for fold_idx, model_paths in resnext_fold_models.items():
        print(f"\nProcessing ResNeXt Fold {fold_idx}...")

        # Equal weighted SWA
        swa_model_equal, config = create_swa_model(
            model_paths, resnext_config, ResNeXtModel, 'equal'
        )
        if swa_model_equal:
            pred_equal = predict_with_model(
                swa_model_equal, config, test_data_dir, resnext_transform, trained_classes
            )
            resnext_swa_equal_preds.append(pred_equal)
            del swa_model_equal

        # Performance weighted SWA
        swa_model_weighted, config = create_swa_model(
            model_paths, resnext_config, ResNeXtModel, 'composite'
        )
        if swa_model_weighted:
            pred_weighted = predict_with_model(
                swa_model_weighted, config, test_data_dir, resnext_transform, trained_classes
            )
            resnext_swa_weighted_preds.append(pred_weighted)
            del swa_model_weighted

        torch.cuda.empty_cache()

    # 3. 앙상블 및 서브미션 생성
    print("\n" + "=" * 50)
    print("CREATING SUBMISSIONS")
    print("=" * 50)

    # 서브미션 1: ResNet SWA Equal 평균
    if resnet_swa_equal_preds:
        resnet_equal_avg = average_predictions(resnet_swa_equal_preds)
        all_predictions['resnet_swa_equal'] = resnet_equal_avg
        create_submission_from_predictions(
            resnet_equal_avg, trained_classes, submission_classes,
            os.path.join(output_dir, "submission_resnet_swa_equal.csv")
        )

    # 서브미션 2: ResNet SWA Weighted 평균
    if resnet_swa_weighted_preds:
        resnet_weighted_avg = average_predictions(resnet_swa_weighted_preds)
        all_predictions['resnet_swa_weighted'] = resnet_weighted_avg
        create_submission_from_predictions(
            resnet_weighted_avg, trained_classes, submission_classes,
            os.path.join(output_dir, "submission_resnet_swa_weighted.csv")
        )

    # 서브미션 3: ResNeXt SWA Equal 평균
    if resnext_swa_equal_preds:
        resnext_equal_avg = average_predictions(resnext_swa_equal_preds)
        all_predictions['resnext_swa_equal'] = resnext_equal_avg
        create_submission_from_predictions(
            resnext_equal_avg, trained_classes, submission_classes,
            os.path.join(output_dir, "submission_resnext_swa_equal.csv")
        )

    # 서브미션 4: ResNeXt SWA Weighted 평균
    if resnext_swa_weighted_preds:
        resnext_weighted_avg = average_predictions(resnext_swa_weighted_preds)
        all_predictions['resnext_swa_weighted'] = resnext_weighted_avg
        create_submission_from_predictions(
            resnext_weighted_avg, trained_classes, submission_classes,
            os.path.join(output_dir, "submission_resnext_swa_weighted.csv")
        )

    # 서브미션 5: ResNet성능가중SWA + ResNeXt성능가중SWA 아키텍처 앙상블
    if 'resnet_swa_weighted' in all_predictions and 'resnext_swa_weighted' in all_predictions:
        combined_weighted = average_predictions([
            all_predictions['resnet_swa_weighted'],
            all_predictions['resnext_swa_weighted']
        ])
        create_submission_from_predictions(
            combined_weighted, trained_classes, submission_classes,
            os.path.join(output_dir, "submission_combined_swa_weighted.csv")
        )

    # 서브미션 6: ResNet동일가중SWA + ResNeXt동일가중SWA 아키텍처 앙상블
    if 'resnet_swa_equal' in all_predictions and 'resnext_swa_equal' in all_predictions:
        combined_equal = average_predictions([
            all_predictions['resnet_swa_equal'],
            all_predictions['resnext_swa_equal']
        ])
        create_submission_from_predictions(
            combined_equal, trained_classes, submission_classes,
            os.path.join(output_dir, "submission_combined_swa_equal.csv")
        )

    print(f"\nAll submissions saved in: {output_dir}/")
    print("Generated submissions:")
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith('.csv'):
            print(f"  {filename}")


if __name__ == "__main__":
    main()