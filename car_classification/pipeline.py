import os
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import warnings

warnings.filterwarnings('ignore')

from car_classification.config import config
from car_classification.model import HierarchicalCarClassifierImproved
from car_classification.dataset.augmentation import get_transform


def load_hierarchical_model(path, base_config):
    """계층적 분류 모델을 로드해서 eval 모드로 설정"""
    checkpoint = torch.load(path, map_location="cuda")
    inference_config = deepcopy(base_config)

    if "config" in checkpoint:
        saved_config = checkpoint["config"]
        inference_config.NUM_LABELS = saved_config.get("NUM_LABELS", inference_config.NUM_LABELS)
        inference_config.NUM_GROUPS = saved_config.get("NUM_GROUPS", inference_config.NUM_GROUPS)
        inference_config.MODEL_NAME = saved_config.get("MODEL_NAME", inference_config.MODEL_NAME)
        if "IMG_SIZE" in saved_config or "_IMG_SIZE" in saved_config:
            saved_img_size = saved_config.get("_IMG_SIZE") or saved_config.get("IMG_SIZE")
            if saved_img_size:
                inference_config.set_img_size(saved_img_size)

    model = HierarchicalCarClassifierImproved(inference_config).to("cuda")
    state_dict = checkpoint["model_state_dict"]

    keys_to_remove = [k for k in state_dict.keys() if "loss_fn" in k]
    for k in keys_to_remove:
        state_dict.pop(k, None)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.current_epoch = 999
    model.classifier.set_training_stage("both")
    return model


def swa_models(model_paths, base_config):
    """여러 모델의 파라미터를 평균화한 SWA 모델 생성"""
    assert len(model_paths) > 0, "최소 하나 이상의 모델 경로가 필요합니다."

    base_model = load_hierarchical_model(model_paths[0], base_config)
    base_sd = base_model.state_dict()
    swa_sd = {k: base_sd[k].cpu() * 0.0 for k in base_sd.keys()}

    for path in model_paths:
        m = load_hierarchical_model(path, base_config)
        sd = m.state_dict()
        for k in swa_sd.keys():
            swa_sd[k] += sd[k].cpu()

    n = float(len(model_paths))
    for k in swa_sd.keys():
        swa_sd[k] = swa_sd[k] / n

    swa_model = load_hierarchical_model(model_paths[0], base_config)
    swa_model.load_state_dict(swa_sd, strict=False)
    swa_model.eval()
    return swa_model


def predict_with_model(model, model_config, image_paths):
    """단일 모델로 이미지 리스트 예측"""
    transform = get_transform(model_config, is_train=False)
    predictions = []

    batch_size = 16
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []

        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = transform(image)
            except:
                image_tensor = torch.zeros((3, model_config.IMG_SIZE[0], model_config.IMG_SIZE[1]))
            batch_tensors.append(image_tensor)

        batch_tensor = torch.stack(batch_tensors).to("cuda")
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs["logits"], dim=1).cpu().numpy()
            predictions.append(probs)

    return np.concatenate(predictions, axis=0)


def load_validation_performance():
    """각 fold의 독립적인 validation 성능 로드"""
    print("Loading validation performance for each fold...")

    analysis_dir = "model_analysis_results"
    summary_path = os.path.join(analysis_dir, "analysis_summary.csv")
    summary_df = pd.read_csv(summary_path)

    print(f"Available columns in summary: {summary_df.columns.tolist()}")

    fold_val_performance = {}

    for fold_idx in sorted(summary_df["fold"].unique()):
        fold_data = summary_df[summary_df["fold"] == fold_idx]

        # 각 fold의 평균 validation 성능 (상위 3개 모델 기준)
        avg_accuracy = fold_data["overall_accuracy"].mean()
        avg_log_loss = fold_data["overall_log_loss"].mean()
        avg_confidence = fold_data["overall_confidence"].mean()

        # 복합 점수 계산 (정확도 높고, 로그로스 낮고, 신뢰도 높을수록 좋음)
        # 로그로스는 역수를 취해서 높을수록 좋게 변환
        normalized_log_loss = 1.0 / (avg_log_loss + 1e-8)
        composite_score = (avg_accuracy * 0.5 +
                           normalized_log_loss * 0.3 +
                           avg_confidence * 0.2)

        fold_val_performance[fold_idx] = {
            'accuracy': avg_accuracy,
            'log_loss': avg_log_loss,
            'confidence': avg_confidence,
            'composite_score': composite_score
        }

        print(f"  Fold {fold_idx}: Acc={avg_accuracy:.4f}, LogLoss={avg_log_loss:.4f}, "
              f"Conf={avg_confidence:.4f}, Composite={composite_score:.4f}")

    return fold_val_performance


def rank_averaging(predictions_list):
    """순위 기반 앙상블"""
    print("  Applying Rank Averaging...")

    rank_sums = np.zeros_like(predictions_list[0])

    for preds in predictions_list:
        ranks = np.argsort(np.argsort(-preds, axis=1), axis=1)
        rank_sums += ranks

    avg_ranks = rank_sums / len(predictions_list)

    final_probs = np.zeros_like(avg_ranks)
    for i in range(len(avg_ranks)):
        inv_ranks = len(avg_ranks[i]) - avg_ranks[i]
        final_probs[i] = inv_ranks / np.sum(inv_ranks)

    return final_probs


def power_mean_ensemble(predictions_list, power=2.0):
    """거듭제곱 평균 앙상블"""
    print(f"  Applying Power Mean Ensemble (power={power})...")

    powered_preds = []
    for preds in predictions_list:
        powered = np.power(preds + 1e-8, power)
        powered_preds.append(powered)

    mean_powered = np.mean(powered_preds, axis=0)
    final_preds = np.power(mean_powered, 1.0 / power)
    final_preds = final_preds / np.sum(final_preds, axis=1, keepdims=True)

    return final_preds


def confidence_dynamic_ensemble(predictions_list, top_k=3):
    """신뢰도 기반 동적 앙상블"""
    print(f"  Applying Confidence Dynamic Ensemble (top_k={top_k})...")

    predictions_array = np.array(predictions_list)
    n_models, n_samples, n_classes = predictions_array.shape

    confidences = np.max(predictions_array, axis=2)
    final_predictions = np.zeros((n_samples, n_classes))

    for sample_idx in range(n_samples):
        sample_confidences = confidences[:, sample_idx]

        if top_k >= n_models:
            top_indices = range(n_models)
        else:
            top_indices = np.argsort(sample_confidences)[-top_k:]

        selected_preds = predictions_array[top_indices, sample_idx, :]
        selected_weights = sample_confidences[top_indices]

        if np.sum(selected_weights) > 0:
            weighted_pred = np.average(selected_preds, axis=0, weights=selected_weights)
        else:
            weighted_pred = np.mean(selected_preds, axis=0)

        final_predictions[sample_idx] = weighted_pred

    return final_predictions


def simple_blending(fold_val_performance, predictions_list):
    """각 fold의 validation 성능 기반 가중치 계산"""
    print("  Applying Simple Blending based on validation performance...")

    weights = []
    fold_indices = sorted(fold_val_performance.keys())

    for fold_idx in fold_indices:
        performance = fold_val_performance[fold_idx]
        weight = performance['composite_score']
        weights.append(weight)

    weights = np.array(weights)
    weights = weights / np.sum(weights)

    print(f"    Fold weights: {[f'{w:.4f}' for w in weights]}")

    weighted_pred = np.average(predictions_list, axis=0, weights=weights)
    return weighted_pred


def nelder_mead_optimization(fold_val_performance, predictions_list):
    """Nelder-Mead 최적화로 가중치 계산"""
    print("  Applying Nelder-Mead optimization...")

    fold_indices = sorted(fold_val_performance.keys())
    val_scores = np.array([fold_val_performance[idx]['composite_score'] for idx in fold_indices])

    def objective(weights):
        weights = np.abs(weights)
        weights = weights / np.sum(weights)

        # 가중치와 성능의 상관관계를 최대화
        correlation = np.corrcoef(weights, val_scores)[0, 1]
        if np.isnan(correlation):
            correlation = 0

        # entropy 정규화 (가중치가 너무 한쪽으로 치우치지 않도록)
        entropy = -np.sum(weights * np.log(weights + 1e-8))

        # 목적함수: correlation 최대화 + entropy 정규화
        return -(correlation + 0.1 * entropy)

    init_weights = np.ones(len(predictions_list)) / len(predictions_list)

    try:
        result = minimize(objective, init_weights, method='Nelder-Mead',
                          options={'maxiter': 1000, 'disp': False})
        optimal_weights = np.abs(result.x)
        optimal_weights = optimal_weights / np.sum(optimal_weights)

        print(f"    Optimized weights: {[f'{w:.4f}' for w in optimal_weights]}")

    except Exception as e:
        print(f"    Optimization failed: {e}, using performance-based weights")
        optimal_weights = val_scores / np.sum(val_scores)

    weighted_pred = np.average(predictions_list, axis=0, weights=optimal_weights)
    return weighted_pred


def stacking_ensemble(fold_val_performance, predictions_list):
    """validation 성능을 활용한 스태킹"""
    print("  Applying Stacking Ensemble...")

    try:
        fold_indices = sorted(fold_val_performance.keys())

        # Meta-features: 각 fold의 성능 지표들
        meta_features = []
        for fold_idx in fold_indices:
            perf = fold_val_performance[fold_idx]
            meta_feature = [
                perf['accuracy'],
                perf['confidence'],
                1.0 / (perf['log_loss'] + 1e-8),  # 로그로스 역수
                perf['composite_score']
            ]
            meta_features.append(meta_feature)

        meta_features = np.array(meta_features)  # (n_folds, n_features)

        # 단순한 linear combination 학습
        # 각 fold의 성능을 바탕으로 가중치 계산
        performance_scores = np.array([fold_val_performance[idx]['composite_score']
                                       for idx in fold_indices])

        # 성능 점수를 softmax로 변환하여 가중치로 사용
        exp_scores = np.exp(performance_scores * 2)  # temperature=0.5
        weights = exp_scores / np.sum(exp_scores)

        print(f"    Stacking weights: {[f'{w:.4f}' for w in weights]}")

        final_pred = np.average(predictions_list, axis=0, weights=weights)
        return final_pred

    except Exception as e:
        print(f"    Stacking failed: {e}, using simple average")
        return np.mean(predictions_list, axis=0)


def create_submission_file(predictions, img_ids, class_names, output_path, method_name):
    """제출 파일 생성"""
    submission_data = []

    for i, img_id in enumerate(img_ids):
        result = {"ID": img_id}
        for cls_idx, cls_name in enumerate(class_names):
            result[cls_name] = float(predictions[i, cls_idx])
        submission_data.append(result)

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_path, index=False)

    max_probs = np.max(predictions, axis=1)
    print(f"  {method_name} submission saved: {os.path.basename(output_path)}")
    print(f"    Average confidence: {np.mean(max_probs):.4f}")
    print(f"    Min/Max confidence: {np.min(max_probs):.4f}/{np.max(max_probs):.4f}")


def main():
    print("=" * 80)
    print("ADVANCED ENSEMBLE METHODS WITH SWA")
    print("=" * 80)

    analysis_results_dir = "model_analysis_results"
    test_data_dir = os.path.join(config.BASE_DIR, "test")
    output_dir = "ensemble_results"
    os.makedirs(output_dir, exist_ok=True)

    # 1) analysis_summary.csv 로드
    summary_path = os.path.join(analysis_results_dir, "analysis_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary 파일이 없습니다: {summary_path}")
    summary_df = pd.read_csv(summary_path)

    # 2) 각 fold별 SWA 모델 생성
    print("Creating SWA models for each fold...")
    swa_models_dict = {}

    for fold_idx in sorted(summary_df["fold"].unique()):
        fold_rows = summary_df[summary_df["fold"] == fold_idx]
        top3_names = fold_rows["model_name"].tolist()
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, f"resnet_hierarchical_fold_{fold_idx}")
        top3_paths = [os.path.join(model_dir, name) for name in top3_names]

        print(f"  Fold {fold_idx}: {[os.path.basename(p) for p in top3_paths]}")
        swa_model = swa_models(top3_paths, config)
        swa_models_dict[fold_idx] = {
            "model": swa_model,
            "config": deepcopy(config)
        }

    # 3) 테스트 데이터 로드
    test_images = [f for f in os.listdir(test_data_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    test_images.sort()
    test_image_paths = [os.path.join(test_data_dir, img) for img in test_images]
    print(f"총 테스트 이미지 수: {len(test_images)}")

    # 4) 클래스 매핑 로드
    class_mapping_path = config.get_class_mapping_path()
    with open(class_mapping_path, "r", encoding="utf-8") as f:
        saved_mapping = json.load(f)
    idx_to_class = {int(idx): name for idx, name in saved_mapping.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    img_ids = [os.path.splitext(img)[0] for img in test_images]

    # 5) 각 fold의 validation 성능 로드
    fold_val_performance = load_validation_performance()

    # 6) 각 SWA 모델별 테스트 예측 수집
    print("\nGenerating predictions from SWA models...")
    fold_predictions = []

    for fold_idx in sorted(swa_models_dict.keys()):
        print(f"  Predicting with SWA model {fold_idx}...")
        model_info = swa_models_dict[fold_idx]

        fold_preds = predict_with_model(
            model_info["model"],
            model_info["config"],
            test_image_paths
        )
        fold_predictions.append(fold_preds)

    # 7) 앙상블 기법들 적용
    print(f"\n{'=' * 60}")
    print("APPLYING ENSEMBLE METHODS")
    print(f"{'=' * 60}")

    ensemble_results = {}

    # Method 1: Simple Average (Original SWA)
    print("\n1. Simple Average (Original SWA)")
    simple_avg = np.mean(fold_predictions, axis=0)
    ensemble_results['simple_average'] = simple_avg

    # Method 2: Rank Averaging
    print("\n2. Rank Averaging")
    rank_avg = rank_averaging(fold_predictions)
    ensemble_results['rank_averaging'] = rank_avg

    # Method 3: Power Mean
    print("\n3. Power Mean Ensemble")
    power_avg = power_mean_ensemble(fold_predictions, power=2.0)
    ensemble_results['power_mean'] = power_avg

    # Method 4: Confidence Dynamic
    print("\n4. Confidence Dynamic Ensemble")
    conf_dynamic = confidence_dynamic_ensemble(fold_predictions, top_k=3)
    ensemble_results['confidence_dynamic'] = conf_dynamic

    # Method 5: Simple Blending
    print("\n5. Simple Blending")
    simple_blend = simple_blending(fold_val_performance, fold_predictions)
    ensemble_results['simple_blending'] = simple_blend

    # Method 6: Nelder-Mead Optimization
    print("\n6. Nelder-Mead Optimization")
    nelder_pred = nelder_mead_optimization(fold_val_performance, fold_predictions)
    ensemble_results['nelder_mead'] = nelder_pred

    # Method 7: Stacking
    print("\n7. Stacking Ensemble")
    stacking_pred = stacking_ensemble(fold_val_performance, fold_predictions)
    ensemble_results['stacking'] = stacking_pred

    # 8) 모든 결과를 서브미션 파일로 저장
    print(f"\n{'=' * 60}")
    print("SAVING SUBMISSION FILES")
    print(f"{'=' * 60}")

    for method_name, predictions in ensemble_results.items():
        output_path = os.path.join(output_dir, f"submission_swa_{method_name}.csv")
        create_submission_file(predictions, img_ids, class_names, output_path, method_name)

    # 9) 결과 요약
    print(f"\n{'=' * 80}")
    print("ENSEMBLE TESTING COMPLETED")
    print(f"{'=' * 80}")
    print(f"Generated {len(ensemble_results)} submission files:")

    for method_name in ensemble_results.keys():
        print(f"  submission_swa_{method_name}.csv")

    print(f"\nAll files saved in: {output_dir}/")
    print("Ready for submission testing!")

    # 메모리 정리
    for model_info in swa_models_dict.values():
        del model_info['model']
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()