import os
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import glob

from car_classification.config import config
from car_classification.model import HierarchicalCarClassifierImproved
from car_classification.dataset.augmentation import get_transform
from car_classification.utils.mapping import load_class_mapping
from car_classification.utils.metrics import log_loss_calc_for_validation


class EnsemblePredictor:
    """지능적 앙상블 예측기"""

    def __init__(self, analysis_results_dir, test_data_dir):
        self.analysis_dir = analysis_results_dir
        self.test_dir = test_data_dir
        self.models = []
        self.class_performance = {}  # 클래스별 성능 정보
        self.model_metadata = {}  # 모델 메타데이터

        # 분석 결과 로드
        self._load_analysis_results()

    def _load_analysis_results(self):
        """분석 결과 CSV 파일들 로드"""
        print("Loading analysis results...")

        # 요약 파일 로드
        summary_path = os.path.join(self.analysis_dir, "analysis_summary.csv")
        if os.path.exists(summary_path):
            self.summary_df = pd.read_csv(summary_path)
            print(f"Loaded summary: {len(self.summary_df)} models")
        else:
            raise FileNotFoundError(f"Summary file not found: {summary_path}")

        # 개별 분석 파일들 로드
        csv_files = glob.glob(os.path.join(self.analysis_dir, "fold_*.csv"))
        print(f"Found {len(csv_files)} analysis files")

        for csv_file in csv_files:
            filename = os.path.basename(csv_file)

            # 파일명에서 정보 추출 (fold_0_model_1_epoch_199_classacc0.8519_grpacc0.8889.csv)
            parts = filename.replace('.csv', '').split('_')
            fold_idx = int(parts[1])
            model_idx = int(parts[3])

            # CSV 데이터 로드
            df = pd.read_csv(csv_file)

            # OVERALL 행 제외하고 클래스별 성능만
            class_df = df[df['class_name'] != 'OVERALL'].copy()
            overall_row = df[df['class_name'] == 'OVERALL']

            model_key = f"fold_{fold_idx}_model_{model_idx}"

            # 클래스별 성능 저장
            self.class_performance[model_key] = {}
            for _, row in class_df.iterrows():
                self.class_performance[model_key][row['class_name']] = {
                    'accuracy': row['accuracy'],
                    'confidence': row['avg_confidence'],
                    'entropy': row['avg_entropy'],
                    'log_loss': row['log_loss'],
                    'sample_count': row['sample_count']
                }

            # 전체 성능 저장
            if not overall_row.empty:
                overall = overall_row.iloc[0]
                self.model_metadata[model_key] = {
                    'fold': fold_idx,
                    'model_idx': model_idx,
                    'filename': filename,
                    'overall_accuracy': overall['accuracy'],
                    'overall_confidence': overall['avg_confidence'],
                    'overall_entropy': overall['avg_entropy'],
                    'overall_log_loss': overall['log_loss'],
                    'total_samples': overall['sample_count']
                }

        print(f"Loaded performance data for {len(self.class_performance)} models")

    def calculate_ensemble_weights(self, strategy='class_specific_accuracy'):
        """다양한 전략으로 앙상블 가중치 계산"""

        strategies = {
            'simple_average': self._simple_average_weights,
            'accuracy_weighted': self._accuracy_weighted,
            'class_specific_accuracy': self._class_specific_accuracy_weights,
            'confidence_weighted': self._confidence_weighted,
            'log_loss_weighted': self._log_loss_weighted,
            'multi_metric': self._multi_metric_weights
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        print(f"Calculating weights using strategy: {strategy}")
        return strategies[strategy]()

    def _simple_average_weights(self):
        """단순 평균 (모든 모델 동일 가중치)"""
        num_models = len(self.model_metadata)
        weights = {}

        for model_key in self.model_metadata.keys():
            weights[model_key] = 1.0 / num_models

        return weights, "simple_average"

    def _accuracy_weighted(self):
        """전체 정확도 기반 가중치"""
        accuracies = []
        model_keys = []

        for model_key, metadata in self.model_metadata.items():
            accuracies.append(metadata['overall_accuracy'])
            model_keys.append(model_key)

        # 정확도 기반 가중치 (softmax 적용)
        accuracies = np.array(accuracies)
        exp_acc = np.exp(accuracies * 5)  # 5는 temperature parameter
        weights_array = exp_acc / np.sum(exp_acc)

        weights = {model_keys[i]: weights_array[i] for i in range(len(model_keys))}
        return weights, "accuracy_weighted"

    def _class_specific_accuracy_weights(self):
        """클래스별 정확도 기반 가중치"""
        # 모든 클래스 수집
        all_classes = set()
        for model_key in self.class_performance.keys():
            all_classes.update(self.class_performance[model_key].keys())
        all_classes = sorted(list(all_classes))

        # 클래스별 가중치 매트릭스 생성
        class_weights = {}

        for class_name in all_classes:
            class_accuracies = []
            model_keys = []

            for model_key in self.model_metadata.keys():
                if class_name in self.class_performance[model_key]:
                    acc = self.class_performance[model_key][class_name]['accuracy']
                    class_accuracies.append(acc)
                else:
                    class_accuracies.append(0.0)  # 해당 클래스 데이터 없음
                model_keys.append(model_key)

            # 클래스별 가중치 계산 (softmax)
            class_accuracies = np.array(class_accuracies)
            if np.sum(class_accuracies) > 0:
                exp_acc = np.exp(class_accuracies * 3)
                weights_array = exp_acc / np.sum(exp_acc)
            else:
                weights_array = np.ones(len(model_keys)) / len(model_keys)

            class_weights[class_name] = {
                model_keys[i]: weights_array[i] for i in range(len(model_keys))
            }

        return class_weights, "class_specific_accuracy"

    def _confidence_weighted(self):
        """신뢰도 기반 가중치"""
        confidences = []
        model_keys = []

        for model_key, metadata in self.model_metadata.items():
            confidences.append(metadata['overall_confidence'])
            model_keys.append(model_key)

        # 신뢰도 기반 가중치
        confidences = np.array(confidences)
        weights_array = confidences / np.sum(confidences)

        weights = {model_keys[i]: weights_array[i] for i in range(len(model_keys))}
        return weights, "confidence_weighted"

    def _log_loss_weighted(self):
        """로그로스 기반 가중치 (낮을수록 좋음)"""
        log_losses = []
        model_keys = []

        for model_key, metadata in self.model_metadata.items():
            log_losses.append(metadata['overall_log_loss'])
            model_keys.append(model_key)

        # 로그로스 역수 기반 가중치
        log_losses = np.array(log_losses)
        # inf 값 처리
        log_losses = np.where(np.isinf(log_losses), 10.0, log_losses)
        inverse_losses = 1.0 / (log_losses + 1e-8)
        weights_array = inverse_losses / np.sum(inverse_losses)

        weights = {model_keys[i]: weights_array[i] for i in range(len(model_keys))}
        return weights, "log_loss_weighted"

    def _multi_metric_weights(self):
        """다중 지표 기반 가중치"""
        scores = []
        model_keys = []

        for model_key, metadata in self.model_metadata.items():
            # 복합 점수 계산
            accuracy = metadata['overall_accuracy']
            confidence = metadata['overall_confidence']
            log_loss = metadata['overall_log_loss']

            # 로그로스 정규화
            log_loss = min(log_loss, 10.0)  # cap at 10
            normalized_log_loss = 1.0 / (log_loss + 1e-8)

            # 복합 점수 (가중 평균)
            composite_score = (
                    0.5 * accuracy +
                    0.3 * confidence +
                    0.2 * normalized_log_loss
            )

            scores.append(composite_score)
            model_keys.append(model_key)

        # 점수 기반 가중치
        scores = np.array(scores)
        exp_scores = np.exp(scores * 2)
        weights_array = exp_scores / np.sum(exp_scores)

        weights = {model_keys[i]: weights_array[i] for i in range(len(model_keys))}
        return weights, "multi_metric"


def load_models_for_ensemble(model_metadata, base_config):
    """앙상블용 모델들 로드"""
    models = {}

    print("Loading models for ensemble...")

    for model_key, metadata in tqdm(model_metadata.items(), desc="Loading models"):
        fold_idx = metadata['fold']

        # 모델 디렉토리 찾기
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, f"resnet_hierarchical_fold_{fold_idx}")

        if not os.path.exists(model_dir):
            print(f"Warning: Model directory not found: {model_dir}")
            continue

        # 모델 파일 찾기 (summary에서 가져온 정보 활용)
        summary_row = None
        if hasattr(base_config, 'summary_df'):
            matching_rows = base_config.summary_df[
                (base_config.summary_df['fold'] == fold_idx) &
                (base_config.summary_df['model_idx'] == metadata['model_idx'])
                ]
            if not matching_rows.empty:
                summary_row = matching_rows.iloc[0]

        if summary_row is not None:
            model_filename = summary_row['model_name']
            model_path = os.path.join(model_dir, model_filename)
        else:
            # 백업: 파일명 패턴으로 찾기
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if not model_files:
                continue
            # 가장 높은 점수의 모델 선택
            model_files.sort(reverse=True)
            model_path = os.path.join(model_dir, model_files[0])

        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
            continue

        try:
            # 모델 로드
            checkpoint = torch.load(model_path, map_location='cuda')
            inference_config = deepcopy(base_config)

            if 'config' in checkpoint:
                saved_config = checkpoint['config']
                inference_config.NUM_LABELS = saved_config.get('NUM_LABELS', inference_config.NUM_LABELS)
                inference_config.NUM_GROUPS = saved_config.get('NUM_GROUPS', inference_config.NUM_GROUPS)
                inference_config.MODEL_NAME = saved_config.get('MODEL_NAME', inference_config.MODEL_NAME)

                if 'IMG_SIZE' in saved_config or '_IMG_SIZE' in saved_config:
                    saved_img_size = saved_config.get('_IMG_SIZE') or saved_config.get('IMG_SIZE')
                    if saved_img_size:
                        inference_config.set_img_size(saved_img_size)

            model = HierarchicalCarClassifierImproved(inference_config).to("cuda")

            state_dict = checkpoint['model_state_dict']
            keys_to_remove = [k for k in state_dict.keys() if 'loss_fn' in k]
            for key in keys_to_remove:
                if key in state_dict:
                    del state_dict[key]

            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.current_epoch = 999
            model.classifier.set_training_stage("both")

            models[model_key] = {
                'model': model,
                'config': inference_config,
                'path': model_path
            }

            print(f"✓ Loaded {model_key}: {os.path.basename(model_path)}")

        except Exception as e:
            print(f"✗ Error loading {model_key}: {e}")
            continue

    print(f"Successfully loaded {len(models)} models")
    return models


def predict_test_data(models, test_dir, ensemble_weights, strategy_name):
    """테스트 데이터에 대한 앙상블 예측"""

    print(f"\nPredicting test data with {strategy_name} strategy...")

    # 테스트 이미지 수집
    test_images = []
    for img_file in os.listdir(test_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_images.append(img_file)

    test_images.sort()
    print(f"Found {len(test_images)} test images")

    # 클래스 정보 로드
    class_mapping_path = config.get_class_mapping_path()
    with open(class_mapping_path, 'r', encoding='utf-8') as f:
        saved_mapping = json.load(f)
        idx_to_class = {int(idx): name for idx, name in saved_mapping.items()}

    num_classes = len(idx_to_class)

    # 예측 결과 저장
    all_predictions = []

    batch_size = 16

    for i in tqdm(range(0, len(test_images), batch_size), desc="Processing batches"):
        batch_images = test_images[i:i + batch_size]
        batch_probs = []

        # 각 모델의 예측 수집
        model_predictions = {}

        for model_key, model_info in models.items():
            model = model_info['model']
            model_config = model_info['config']

            # 이미지 전처리
            transform = get_transform(model_config, is_train=False)

            batch_tensors = []
            for img_name in batch_images:
                img_path = os.path.join(test_dir, img_name)
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transform(image)
                    batch_tensors.append(image_tensor)
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")
                    # 더미 이미지
                    dummy_tensor = torch.zeros((3, model_config.IMG_SIZE[0], model_config.IMG_SIZE[1]))
                    batch_tensors.append(dummy_tensor)

            # 배치 텐서 생성
            if batch_tensors:
                batch_tensor = torch.stack(batch_tensors).to("cuda")

                with torch.no_grad():
                    try:
                        outputs = model(batch_tensor)
                        probs = torch.softmax(outputs["logits"], dim=1).cpu().numpy()
                        model_predictions[model_key] = probs
                    except Exception as e:
                        print(f"Error predicting with {model_key}: {e}")
                        # 더미 예측
                        dummy_probs = np.ones((len(batch_images), num_classes)) / num_classes
                        model_predictions[model_key] = dummy_probs

        # 앙상블 가중치 적용
        for batch_idx in range(len(batch_images)):
            img_name = batch_images[batch_idx]

            if strategy_name == "class_specific_accuracy":
                # 클래스별 가중치는 예측 후 적용하기 어려우므로 accuracy_weighted로 대체
                # 실제로는 더 복잡한 로직이 필요 (반복적 가중치 적용 등)
                ensemble_probs = np.zeros(num_classes)
                total_weight = 0

                for model_key, probs in model_predictions.items():
                    weight = ensemble_weights.get(model_key, 0)
                    ensemble_probs += weight * probs[batch_idx]
                    total_weight += weight

                if total_weight > 0:
                    ensemble_probs /= total_weight
                else:
                    ensemble_probs = np.ones(num_classes) / num_classes

            else:
                # 단순 가중 평균
                ensemble_probs = np.zeros(num_classes)
                total_weight = 0

                for model_key, probs in model_predictions.items():
                    weight = ensemble_weights.get(model_key, 0)
                    ensemble_probs += weight * probs[batch_idx]
                    total_weight += weight

                if total_weight > 0:
                    ensemble_probs /= total_weight
                else:
                    ensemble_probs = np.ones(num_classes) / num_classes

            # 예측 클래스
            pred_class_idx = np.argmax(ensemble_probs)
            pred_class_name = idx_to_class[pred_class_idx]
            confidence = np.max(ensemble_probs)

            all_predictions.append({
                'image': img_name,
                'predicted_class': pred_class_name,
                'confidence': confidence,
                'probabilities': ensemble_probs.tolist()
            })

    return all_predictions


def create_submission_file(predictions, output_path, strategy_name):
    """제출 파일 생성"""

    submission_data = []
    for pred in predictions:
        submission_data.append({
            'image': pred['image'],
            'label': pred['predicted_class']
        })

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_path, index=False)

    print(f"✓ Saved submission file: {output_path}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Predictions: {len(submission_data)}")

    # 통계 출력
    class_counts = submission_df['label'].value_counts()
    print(f"  Class distribution:")
    for class_name, count in class_counts.head(10).items():
        print(f"    {class_name}: {count}")


def main():
    """메인 앙상블 테스트 함수"""

    print("=" * 80)
    print("INTELLIGENT ENSEMBLE TESTING")
    print("=" * 80)

    # 설정
    analysis_results_dir = "model_analysis_results"
    test_data_dir = os.path.join(config.BASE_DIR, "test")
    output_dir = "ensemble_results"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(test_data_dir):
        raise FileNotFoundError(f"Test directory not found: {test_data_dir}")

    # 앙상블 예측기 초기화
    ensemble_predictor = EnsemblePredictor(analysis_results_dir, test_data_dir)

    # 모델들 로드
    models = load_models_for_ensemble(ensemble_predictor.model_metadata, config)

    if len(models) == 0:
        print("No models loaded! Exiting...")
        return

    # 다양한 앙상블 전략 테스트
    strategies = [
        'simple_average',
        'accuracy_weighted',
        'confidence_weighted',
        'log_loss_weighted',
        'multi_metric'
    ]

    results = {}

    for strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"TESTING STRATEGY: {strategy.upper()}")
        print(f"{'=' * 60}")

        try:
            # 가중치 계산
            weights, strategy_name = ensemble_predictor.calculate_ensemble_weights(strategy)

            # 가중치 출력
            print(f"Ensemble weights ({strategy_name}):")
            if isinstance(weights, dict) and not isinstance(list(weights.values())[0], dict):
                # 단순 가중치
                for model_key, weight in sorted(weights.items()):
                    print(f"  {model_key}: {weight:.4f}")
            else:
                print(f"  Using class-specific weights (too complex to display)")

            # 예측 수행
            predictions = predict_test_data(models, test_data_dir, weights, strategy_name)

            # 제출 파일 생성
            submission_path = os.path.join(output_dir, f"submission_{strategy_name}.csv")
            create_submission_file(predictions, submission_path, strategy_name)

            # 결과 저장
            results[strategy_name] = {
                'predictions': predictions,
                'weights': weights,
                'submission_file': submission_path
            }

        except Exception as e:
            print(f"Error with strategy {strategy}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 결과 요약
    print(f"\n{'=' * 80}")
    print("ENSEMBLE TESTING COMPLETED")
    print(f"{'=' * 80}")
    print(f"Strategies tested: {len(results)}")
    print(f"Models used: {len(models)}")
    print(f"Results saved in: {output_dir}/")

    for strategy_name, result in results.items():
        print(f"\n{strategy_name}:")
        print(f"  Submission file: {os.path.basename(result['submission_file'])}")

        # 신뢰도 통계
        confidences = [p['confidence'] for p in result['predictions']]
        print(f"  Avg confidence: {np.mean(confidences):.4f}")
        print(f"  Min confidence: {np.min(confidences):.4f}")
        print(f"  Max confidence: {np.max(confidences):.4f}")

    print(f"\n🎉 Ready for submission! Choose the best strategy based on validation performance.")

    # 메모리 정리
    for model_info in models.values():
        del model_info['model']
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()