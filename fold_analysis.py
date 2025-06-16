import os
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
import itertools
import re

from resnet_car_classification.config import config as resnet_config
from resnet_car_classification.model import HierarchicalCarClassifierImproved as ResNetModel
from resnet_car_classification.dataset.augmentation import get_transform as resnet_transform

from resnext_car_classification.config import config as resnext_config
from resnext_car_classification.model import HierarchicalCarClassifierImproved as ResNeXtModel
from resnext_car_classification.dataset.augmentation import get_transform as resnext_transform


def log_loss_calc(y_true, y_pred, eps=1e-15):
    """Log loss 계산"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)


def calculate_entropy(probs, eps=1e-15):
    """엔트로피 계산"""
    probs = np.clip(probs, eps, 1 - eps)
    return -np.sum(probs * np.log(probs))


def apply_confidence_calibration(predictions, confidence_threshold, target_confidence=0.99):
    """신뢰도 기반 보정"""
    calibrated_preds = []

    for pred in predictions:
        probs = np.array(pred['probabilities'])
        max_prob = np.max(probs)
        max_idx = np.argmax(probs)

        if max_prob >= confidence_threshold:
            # 높은 신뢰도 클래스를 target_confidence로 보정
            new_probs = probs.copy()
            remaining_prob = 1.0 - target_confidence

            # 나머지 클래스들의 현재 확률 합
            other_prob_sum = 1.0 - max_prob

            if other_prob_sum > 0:
                # 비례적으로 줄이기
                scaling_factor = remaining_prob / other_prob_sum
                new_probs = new_probs * scaling_factor
                new_probs[max_idx] = target_confidence
            else:
                # 모든 확률이 하나 클래스에 집중된 경우
                new_probs = np.zeros_like(probs)
                new_probs[max_idx] = 1.0
        else:
            new_probs = probs

        calibrated_preds.append({
            **pred,
            'probabilities': new_probs.tolist(),
            'calibrated': max_prob >= confidence_threshold,
            'original_confidence': max_prob
        })

    return calibrated_preds


def apply_entropy_calibration(predictions, entropy_strategy='adaptive'):
    """엔트로피 기반 보정"""
    calibrated_preds = []

    # 모든 예측의 엔트로피 계산
    entropies = []
    for pred in predictions:
        probs = np.array(pred['probabilities'])
        entropy = calculate_entropy(probs)
        entropies.append(entropy)

    entropies = np.array(entropies)

    # 엔트로피 임계값 설정 (전체 분포 기준)
    low_entropy_threshold = np.percentile(entropies, 33)  # 낮은 33% (확신)
    high_entropy_threshold = np.percentile(entropies, 67)  # 높은 33% (불확실)

    for i, pred in enumerate(predictions):
        probs = np.array(pred['probabilities'])
        entropy = entropies[i]
        max_prob = np.max(probs)
        max_idx = np.argmax(probs)

        new_probs = probs.copy()
        calibration_applied = False
        calibration_type = "none"

        if entropy_strategy == 'adaptive':
            if entropy <= low_entropy_threshold:
                # 낮은 엔트로피 (확신) → aggressive 보정
                if max_prob >= 0.8:  # 추가 조건
                    target_confidence = 0.99
                    remaining_prob = 1.0 - target_confidence
                    other_prob_sum = 1.0 - max_prob

                    if other_prob_sum > 0:
                        scaling_factor = remaining_prob / other_prob_sum
                        new_probs = new_probs * scaling_factor
                        new_probs[max_idx] = target_confidence
                        calibration_applied = True
                        calibration_type = "aggressive"

            elif entropy <= high_entropy_threshold:
                # 중간 엔트로피 → moderate 보정
                if max_prob >= 0.9:
                    target_confidence = 0.95
                    remaining_prob = 1.0 - target_confidence
                    other_prob_sum = 1.0 - max_prob

                    if other_prob_sum > 0:
                        scaling_factor = remaining_prob / other_prob_sum
                        new_probs = new_probs * scaling_factor
                        new_probs[max_idx] = target_confidence
                        calibration_applied = True
                        calibration_type = "moderate"

            else:
                # 높은 엔트로피 (불확실) → smoothing 또는 보정 안함
                if entropy_strategy == 'adaptive':
                    # 약간의 smoothing 적용
                    alpha = 0.1  # smoothing factor
                    uniform_dist = np.ones_like(probs) / len(probs)
                    new_probs = (1 - alpha) * probs + alpha * uniform_dist
                    calibration_applied = True
                    calibration_type = "smoothing"

        elif entropy_strategy == 'conservative':
            # 보수적: 매우 낮은 엔트로피이고 높은 신뢰도일 때만 보정
            if entropy <= low_entropy_threshold and max_prob >= 0.95:
                target_confidence = 0.98
                remaining_prob = 1.0 - target_confidence
                other_prob_sum = 1.0 - max_prob

                if other_prob_sum > 0:
                    scaling_factor = remaining_prob / other_prob_sum
                    new_probs = new_probs * scaling_factor
                    new_probs[max_idx] = target_confidence
                    calibration_applied = True
                    calibration_type = "conservative"

        calibrated_preds.append({
            **pred,
            'probabilities': new_probs.tolist(),
            'calibrated': calibration_applied,
            'original_confidence': max_prob,
            'entropy': float(entropy),
            'calibration_type': calibration_type
        })

    return calibrated_preds


def extract_performance_from_filename(filename):
    """파일명에서 성능 정보 추출"""
    # 기본값
    performance = {'class_acc': 0.0, 'group_acc': 0.0, 'epoch': 0}

    # classacc 추출 (예: classacc0.9630)
    class_acc_match = re.search(r'classacc([\d.]+)', filename)
    if class_acc_match:
        performance['class_acc'] = float(class_acc_match.group(1))

    # group accuracy 추출 (예: grp0.9943)
    group_acc_match = re.search(r'grp([\d.]+)', filename)
    if group_acc_match:
        performance['group_acc'] = float(group_acc_match.group(1))

    # epoch 추출 (예: epoch45)
    epoch_match = re.search(r'epoch(\d+)', filename)
    if epoch_match:
        performance['epoch'] = int(epoch_match.group(1))

    return performance


def swa_weights(model_paths, weighting_strategy='equal'):
    """SWA 가중치 계산 (다양한 가중치 전략 지원)"""
    if not model_paths:
        return None

    # 성능 기반 가중치 계산
    if weighting_strategy != 'equal':
        model_performances = []
        for model_path in model_paths:
            filename = os.path.basename(model_path)
            perf = extract_performance_from_filename(filename)
            model_performances.append(perf)

        # 가중치 계산
        if weighting_strategy == 'class_acc':
            scores = [p['class_acc'] for p in model_performances]
        elif weighting_strategy == 'group_acc':
            scores = [p['group_acc'] for p in model_performances]
        elif weighting_strategy == 'composite':
            # 클래스 정확도와 그룹 정확도의 가중 평균
            scores = [0.7 * p['class_acc'] + 0.3 * p['group_acc'] for p in model_performances]
        else:
            scores = [1.0] * len(model_paths)  # equal weights

        # softmax로 가중치 정규화 (높은 성능에 더 큰 가중치)
        if max(scores) > min(scores):  # 성능 차이가 있는 경우만
            scores = np.array(scores)
            exp_scores = np.exp(scores * 10)  # temperature=10으로 차이 증폭
            weights = exp_scores / np.sum(exp_scores)
        else:
            weights = np.ones(len(model_paths)) / len(model_paths)

        print(f"  SWA weights ({weighting_strategy}):")
        for i, (path, weight, score) in enumerate(zip(model_paths, weights, scores)):
            filename = os.path.basename(path)
            print(f"    {filename}: weight={weight:.4f}, score={score:.4f}")
    else:
        weights = np.ones(len(model_paths)) / len(model_paths)

    # 첫 번째 모델 로드
    first_checkpoint = torch.load(model_paths[0], map_location='cpu')
    averaged_state_dict = {}
    original_dtypes = {}

    for key in first_checkpoint['model_state_dict'].keys():
        if 'loss_fn' not in key:
            weight = first_checkpoint['model_state_dict'][key]
            averaged_state_dict[key] = weights[0] * weight.clone().float()
            original_dtypes[key] = weight.dtype

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

    return averaged_state_dict


def load_models_for_fold(fold_idx, architecture='resnet'):
    """특정 폴드의 모델들 로드"""
    if architecture == 'resnet':
        config = resnet_config
        model_class = ResNetModel
        transform_func = resnet_transform
        expected_models = 3
        folder_pattern = f"resnet_hierarchical_fold_{fold_idx}"
    else:  # resnext
        config = resnext_config
        model_class = ResNeXtModel
        transform_func = resnext_transform
        expected_models = 5
        folder_pattern = f"resnet_hierarchical_fold_{fold_idx}"  # resnext도 같은 패턴인지 확인 필요

    # 모델 디렉토리 경로 확인
    fold_dir = os.path.join(config.MODEL_OUTPUT_DIR, folder_pattern)
    print(f"Looking for {architecture} models in: {fold_dir}")

    if not os.path.exists(fold_dir):
        print(f"Model directory not found: {fold_dir}")
        # 대안 경로들 확인
        base_dir = config.MODEL_OUTPUT_DIR
        print(f"Available directories in {base_dir}:")
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path) and f"fold_{fold_idx}" in item:
                    print(f"  Found fold {fold_idx} directory: {item}")
                    # 실제 폴드 디렉토리를 찾았다면 사용
                    fold_dir = item_path
                    break
            else:
                # 못 찾았다면 모든 디렉토리 출력
                for item in os.listdir(base_dir):
                    item_path = os.path.join(base_dir, item)
                    if os.path.isdir(item_path):
                        print(f"  {item}")
                return None, None, None

    model_files = [f for f in os.listdir(fold_dir) if f.endswith('.pth')]
    print(f"Found {len(model_files)} .pth files (expected {expected_models})")

    if len(model_files) == 0:
        print(f"No model files found in {fold_dir}")
        return None, None, None

    if len(model_files) != expected_models:
        print(f"Warning: Expected {expected_models} models in {fold_dir}, found {len(model_files)}")

    model_paths = [os.path.join(fold_dir, f) for f in model_files]
    print(f"Model files: {model_files}")

    return model_paths, config, (model_class, transform_func)


def predict_with_models(model_paths, model_info, test_images, test_labels, method='average'):
    """모델들로 예측 수행"""
    model_class, transform_func = model_info[2]
    config = model_info[1]

    if method.startswith('swa'):
        # SWA 방식들
        if method == 'swa':
            weighting_strategy = 'equal'
        elif method == 'swa_class_acc':
            weighting_strategy = 'class_acc'
        elif method == 'swa_group_acc':
            weighting_strategy = 'group_acc'
        elif method == 'swa_composite':
            weighting_strategy = 'composite'
        else:
            weighting_strategy = 'equal'

        averaged_weights = swa_weights(model_paths, weighting_strategy)
        if averaged_weights is None:
            return []

        # 하나의 모델 구조에 평균 가중치 로드
        checkpoint = torch.load(model_paths[0], map_location='cuda')
        inference_config = deepcopy(config)

        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            inference_config.NUM_LABELS = saved_config.get('NUM_LABELS', inference_config.NUM_LABELS)
            inference_config.NUM_GROUPS = saved_config.get('NUM_GROUPS', inference_config.NUM_GROUPS)

        model = model_class(inference_config).to("cuda")
        model.load_state_dict(averaged_weights, strict=False)
        model.eval()

        predictions = predict_single_model(model, test_images, test_labels, transform_func, config)
        del model

    else:
        # 개별 모델 예측 후 평균/가중 평균
        all_model_predictions = []

        for model_path in model_paths:
            try:
                checkpoint = torch.load(model_path, map_location='cuda')
                inference_config = deepcopy(config)

                if 'config' in checkpoint:
                    saved_config = checkpoint['config']
                    inference_config.NUM_LABELS = saved_config.get('NUM_LABELS', inference_config.NUM_LABELS)
                    inference_config.NUM_GROUPS = saved_config.get('NUM_GROUPS', inference_config.NUM_GROUPS)

                model = model_class(inference_config).to("cuda")

                state_dict = checkpoint['model_state_dict']
                keys_to_remove = [k for k in state_dict.keys() if 'loss_fn' in k]
                for key in keys_to_remove:
                    if key in state_dict:
                        del state_dict[key]

                model.load_state_dict(state_dict, strict=False)
                model.eval()

                model_predictions = predict_single_model(model, test_images, test_labels, transform_func, config)
                all_model_predictions.append(model_predictions)

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                continue

        if method == 'average':
            predictions = average_predictions(all_model_predictions)
        elif method == 'confidence_weighted':
            predictions = confidence_weighted_average(all_model_predictions)
        elif method == 'best_confidence':
            predictions = select_best_confidence(all_model_predictions)

    return predictions


def predict_single_model(model, test_images, test_labels, transform_func, config):
    """단일 모델로 예측"""
    predictions = []
    transform = transform_func(config, is_train=False)

    batch_size = 16
    for i in range(0, len(test_images), batch_size):
        batch_images = test_images[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]

        batch_tensors = []
        for img_path in batch_images:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                batch_tensors.append(image_tensor)
            except:
                dummy_tensor = torch.zeros((3, config.IMG_SIZE[0], config.IMG_SIZE[1]))
                batch_tensors.append(dummy_tensor)

        if batch_tensors:
            batch_tensor = torch.stack(batch_tensors).to("cuda")

            with torch.no_grad():
                outputs = model(batch_tensor)
                probs = torch.softmax(outputs["logits"], dim=1).cpu().numpy()

                for j, (img_path, true_label) in enumerate(zip(batch_images, batch_labels)):
                    predictions.append({
                        'image': os.path.basename(img_path),
                        'true_label': true_label,
                        'probabilities': probs[j].tolist(),
                        'confidence': float(np.max(probs[j]))
                    })

    return predictions


def average_predictions(all_model_predictions):
    """단순 평균"""
    if not all_model_predictions:
        return []

    num_images = len(all_model_predictions[0])
    num_classes = len(all_model_predictions[0][0]['probabilities'])

    averaged_predictions = []

    for i in range(num_images):
        avg_probs = np.zeros(num_classes)
        image_name = all_model_predictions[0][i]['image']
        true_label = all_model_predictions[0][i]['true_label']

        for model_preds in all_model_predictions:
            avg_probs += np.array(model_preds[i]['probabilities'])

        avg_probs /= len(all_model_predictions)

        averaged_predictions.append({
            'image': image_name,
            'true_label': true_label,
            'probabilities': avg_probs.tolist(),
            'confidence': float(np.max(avg_probs))
        })

    return averaged_predictions


def confidence_weighted_average(all_model_predictions):
    """신뢰도 가중 평균"""
    if not all_model_predictions:
        return []

    num_images = len(all_model_predictions[0])
    num_classes = len(all_model_predictions[0][0]['probabilities'])

    weighted_predictions = []

    for i in range(num_images):
        weighted_probs = np.zeros(num_classes)
        total_weight = 0

        image_name = all_model_predictions[0][i]['image']
        true_label = all_model_predictions[0][i]['true_label']

        for model_preds in all_model_predictions:
            confidence = model_preds[i]['confidence']
            weight = confidence ** 2  # 신뢰도 제곱으로 가중치
            weighted_probs += weight * np.array(model_preds[i]['probabilities'])
            total_weight += weight

        if total_weight > 0:
            weighted_probs /= total_weight

        weighted_predictions.append({
            'image': image_name,
            'true_label': true_label,
            'probabilities': weighted_probs.tolist(),
            'confidence': float(np.max(weighted_probs))
        })

    return weighted_predictions


def select_best_confidence(all_model_predictions):
    """최고 신뢰도 예측 선택"""
    if not all_model_predictions:
        return []

    num_images = len(all_model_predictions[0])
    best_predictions = []

    for i in range(num_images):
        best_confidence = -1
        best_pred = None

        for model_preds in all_model_predictions:
            if model_preds[i]['confidence'] > best_confidence:
                best_confidence = model_preds[i]['confidence']
                best_pred = model_preds[i]

        best_predictions.append(best_pred)

    return best_predictions


def evaluate_predictions(predictions, trained_classes):
    """예측 결과 평가"""
    if not predictions:
        return {}

    # 전체 log loss 계산
    y_true_all = []
    y_pred_all = []

    class_metrics = defaultdict(list)

    for pred in predictions:
        true_label = pred['true_label']
        probs = np.array(pred['probabilities'])

        # one-hot encoding
        y_true = np.zeros(len(trained_classes))
        if true_label in trained_classes:
            true_idx = trained_classes.index(true_label)
            y_true[true_idx] = 1

            y_true_all.append(y_true)
            y_pred_all.append(probs)

            # 클래스별 메트릭
            pred_confidence = np.max(probs)
            pred_class = trained_classes[np.argmax(probs)]
            is_correct = pred_class == true_label

            class_metrics[true_label].append({
                'confidence': pred_confidence,
                'correct': is_correct,
                'log_loss': log_loss_calc(y_true.reshape(1, -1), probs.reshape(1, -1))
            })

    if not y_true_all:
        return {}

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    overall_log_loss = log_loss_calc(y_true_all, y_pred_all)
    overall_accuracy = np.mean([np.argmax(y_true_all[i]) == np.argmax(y_pred_all[i]) for i in range(len(y_true_all))])

    # 클래스별 통계
    class_stats = {}
    for class_name, metrics in class_metrics.items():
        class_stats[class_name] = {
            'count': len(metrics),
            'accuracy': np.mean([m['correct'] for m in metrics]),
            'avg_confidence': np.mean([m['confidence'] for m in metrics]),
            'avg_log_loss': np.mean([m['log_loss'] for m in metrics])
        }

    return {
        'overall_log_loss': overall_log_loss,
        'overall_accuracy': overall_accuracy,
        'class_stats': class_stats,
        'num_samples': len(predictions)
    }


def load_validation_data(fold_idx, config):
    """검증 데이터 로드 (폴더 구조)"""
    val_dir = os.path.join(config.BASE_DIR, "augmented_data", f"fold_{fold_idx}", "val")

    if not os.path.exists(val_dir):
        print(f"Validation directory not found: {val_dir}")
        return [], []

    test_images = []
    test_labels = []

    # 클래스별 폴더 순회
    for class_name in os.listdir(val_dir):
        class_path = os.path.join(val_dir, class_name)
        if os.path.isdir(class_path):
            # 각 클래스 폴더의 이미지들
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    test_images.append(img_path)
                    test_labels.append(class_name)

    print(f"Found {len(test_images)} validation images in {len(set(test_labels))} classes")
    return test_images, test_labels


def run_experiment():
    """메인 실험 실행"""
    print("CONFIDENCE CALIBRATION EXPERIMENT")
    print("=" * 80)

    # 실험 설정
    architectures = ['resnet', 'resnext']
    ensemble_methods = [
        'swa',  # 기본 SWA (동일 가중치)
        'swa_class_acc',  # 클래스 정확도 기반 가중 SWA
        'swa_group_acc',  # 그룹 정확도 기반 가중 SWA
        'swa_composite',  # 복합 점수 기반 가중 SWA
        'average',  # 단순 평균
        'confidence_weighted',  # 신뢰도 가중 평균
        'best_confidence'  # 최고 신뢰도 선택
    ]

    # 보정 방법들
    calibration_methods = [
        {'type': 'none'},
        {'type': 'confidence', 'threshold': 0.85},
        {'type': 'confidence', 'threshold': 0.90},
        {'type': 'confidence', 'threshold': 0.95},
        {'type': 'entropy', 'strategy': 'adaptive'},
        {'type': 'entropy', 'strategy': 'conservative'}
    ]

    # 클래스 정보 로드 (resnext config 기준으로 통일)
    class_mapping_path = resnext_config.get_class_mapping_path()
    with open(class_mapping_path, 'r', encoding='utf-8') as f:
        saved_mapping = json.load(f)
        trained_classes = [saved_mapping[str(i)] for i in range(len(saved_mapping))]

    print(f"Trained classes: {len(trained_classes)}")

    # 결과 저장
    all_results = []

    # 각 아키텍처별로 실험
    for architecture in architectures:
        print(f"\n{'=' * 60}")
        print(f"TESTING ARCHITECTURE: {architecture.upper()}")
        print(f"{'=' * 60}")

        config = resnet_config if architecture == 'resnet' else resnext_config

        # 폴드별 실험
        for fold_idx in range(5):  # 0-4 폴드
            print(f"\nFold {fold_idx}:")

            # 모델과 검증 데이터 로드
            model_info = load_models_for_fold(fold_idx, architecture)
            if model_info[0] is None:
                print(f"  No models found for fold {fold_idx}")
                continue

            test_images, test_labels = load_validation_data(fold_idx, config)
            if not test_images:
                print(f"  No validation data for fold {fold_idx}")
                continue

            print(f"  Models: {len(model_info[0])}, Validation samples: {len(test_images)}")

            # 각 앙상블 방법별 실험
            for method in ensemble_methods:
                print(f"    Method: {method}")

                try:
                    # 기본 예측
                    predictions = predict_with_models(model_info[0], model_info, test_images, test_labels, method)

                    if not predictions:
                        print(f"      No predictions generated")
                        continue

                    # 각 보정 방법별 테스트
                    for calibration_config in calibration_methods:
                        cal_type = calibration_config['type']

                        if cal_type == 'none':
                            calibrated_predictions = predictions
                            method_name = "no_calibration"

                        elif cal_type == 'confidence':
                            threshold = calibration_config['threshold']
                            calibrated_predictions = apply_confidence_calibration(predictions, threshold)
                            method_name = f"confidence_{threshold}"

                        elif cal_type == 'entropy':
                            strategy = calibration_config['strategy']
                            calibrated_predictions = apply_entropy_calibration(predictions, strategy)
                            method_name = f"entropy_{strategy}"

                        # 평가
                        metrics = evaluate_predictions(calibrated_predictions, trained_classes)

                        if metrics:
                            # 엔트로피 통계 추가
                            if cal_type == 'entropy' and calibrated_predictions:
                                entropies = [p.get('entropy', 0) for p in calibrated_predictions]
                                calibration_stats = {
                                    'avg_entropy': np.mean(entropies),
                                    'calibrated_count': sum(
                                        1 for p in calibrated_predictions if p.get('calibrated', False)),
                                    'total_count': len(calibrated_predictions)
                                }
                                metrics.update(calibration_stats)

                            result = {
                                'architecture': architecture,
                                'fold': fold_idx,
                                'ensemble_method': method,
                                'calibration_type': cal_type,
                                'calibration_method': method_name,
                                **metrics
                            }

                            all_results.append(result)

                            print(f"      Method {method_name}: "
                                  f"LogLoss={metrics['overall_log_loss']:.4f}, "
                                  f"Acc={metrics['overall_accuracy']:.4f}")

                            if cal_type == 'entropy':
                                print(f"        AvgEntropy={metrics.get('avg_entropy', 0):.3f}, "
                                      f"Calibrated={metrics.get('calibrated_count', 0)}/{metrics.get('total_count', 0)}")

                except Exception as e:
                    print(f"      Error with method {method}: {e}")
                    continue

            # 메모리 정리
            torch.cuda.empty_cache()

    # 결과 저장
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("confidence_calibration_results.csv", index=False)

    with open("confidence_calibration_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # 결과 요약
    print(f"\n{'=' * 80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")

    if not results_df.empty:
        # 전체 평균 성능
        summary = results_df.groupby(['architecture', 'ensemble_method', 'calibration_method']).agg({
            'overall_log_loss': ['mean', 'std'],
            'overall_accuracy': ['mean', 'std']
        }).round(4)

        print("\nAverage Performance by Configuration:")
        print(summary)

        # 최고 성능 찾기
        best_logloss = results_df.loc[results_df['overall_log_loss'].idxmin()]
        print(f"\nBest Log Loss Configuration:")
        print(f"  Architecture: {best_logloss['architecture']}")
        print(f"  Ensemble Method: {best_logloss['ensemble_method']}")
        print(f"  Calibration Method: {best_logloss['calibration_method']}")
        print(f"  Log Loss: {best_logloss['overall_log_loss']:.4f}")
        print(f"  Accuracy: {best_logloss['overall_accuracy']:.4f}")

        # 보정 방법별 평균 성능
        calibration_summary = results_df.groupby('calibration_method').agg({
            'overall_log_loss': 'mean',
            'overall_accuracy': 'mean'
        }).round(4).sort_values('overall_log_loss')

        print(f"\nCalibration Methods Ranking (by avg log loss):")
        for method, row in calibration_summary.iterrows():
            print(f"  {method}: LogLoss={row['overall_log_loss']:.4f}, Acc={row['overall_accuracy']:.4f}")

    print(f"\nResults saved to:")
    print(f"  confidence_calibration_results.csv")
    print(f"  confidence_calibration_results.json")


if __name__ == "__main__":
    run_experiment()