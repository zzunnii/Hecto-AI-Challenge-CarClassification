import os
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy

from car_classification.config import config
from car_classification.model import HierarchicalCarClassifierImproved
from car_classification.dataset.augmentation import get_transform
from car_classification.utils.mapping import load_class_mapping
from car_classification.dataset import CarBrandDataset
from car_classification.utils.metrics import log_loss_calc_for_validation, create_answer_df_from_arrays, create_submission_df_from_arrays, \
    multiclass_log_loss


class ModelAnalyzer:
    """개별 모델 성능 분석 클래스"""

    def __init__(self, model, model_info, trained_classes, transform):
        self.model = model
        self.model_info = model_info
        self.trained_classes = trained_classes
        self.transform = transform
        self.num_classes = len(trained_classes)

    def calculate_entropy(self, probs):
        """엔트로피 계산"""
        epsilon = 1e-8
        return -np.sum(probs * np.log(probs + epsilon), axis=1)

    def calculate_class_specific_log_loss(self, true_class_idx, predicted_probs):
        """클래스별 로그로스 직접 계산 (이진 분류 방식)"""
        # 해당 클래스에 대한 예측 확률만 추출
        class_probs = predicted_probs[:, true_class_idx]

        # 실제 정답 여부 (해당 클래스면 1, 아니면 0)
        # 여기서는 이미 해당 클래스만 필터링했으므로 모두 1
        y_true_binary = np.ones(len(class_probs))

        # 로그로스 직접 계산
        epsilon = 1e-15
        class_probs = np.clip(class_probs, epsilon, 1 - epsilon)
        log_loss = -np.mean(y_true_binary * np.log(class_probs) +
                            (1 - y_true_binary) * np.log(1 - class_probs))

        return log_loss

    def analyze_validation_dataset(self, val_dataset):
        """검증 데이터셋에서 클래스별 성능 분석"""

        # 결과 저장용
        class_results = {}
        all_predictions = []
        all_true_labels = []
        all_confidences = []
        all_entropies = []
        all_probs = []  # 로그로스 계산용

        print(f"Analyzing {len(val_dataset)} validation samples...")

        # 배치 처리를 위한 DataLoader 생성
        from torch.utils.data import DataLoader
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # 배치별로 예측
        for batch in tqdm(val_loader, desc="Processing validation batches"):
            batch_images = batch["pixel_values"].to("cuda")
            batch_labels = batch["label"].cpu().numpy()

            with torch.no_grad():
                try:
                    outputs = self.model(batch_images)
                    class_logits = outputs["logits"]
                    class_probs = torch.softmax(class_logits, dim=1).cpu().numpy()

                    # 예측 결과 저장
                    for probs, true_label in zip(class_probs, batch_labels):
                        pred_class_idx = np.argmax(probs)
                        max_confidence = np.max(probs)
                        entropy = self.calculate_entropy(probs.reshape(1, -1))[0]

                        all_predictions.append(pred_class_idx)
                        all_true_labels.append(true_label)
                        all_confidences.append(max_confidence)
                        all_entropies.append(entropy)
                        all_probs.append(probs)

                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # 더미 예측으로 처리
                    batch_size = len(batch_labels)
                    for true_label in batch_labels:
                        dummy_probs = np.ones(self.num_classes) / self.num_classes
                        all_predictions.append(0)
                        all_true_labels.append(true_label)
                        all_confidences.append(1.0 / self.num_classes)
                        all_entropies.append(-np.log(1.0 / self.num_classes))
                        all_probs.append(dummy_probs)

        # 전체 로그로스 계산
        if all_probs:
            all_probs_array = np.array(all_probs)
            overall_log_loss = log_loss_calc_for_validation(all_true_labels, all_probs_array)
        else:
            overall_log_loss = float('inf')

        # 클래스별 성능 계산
        for class_idx, class_name in enumerate(self.trained_classes):
            class_mask = np.array(all_true_labels) == class_idx

            if not np.any(class_mask):
                # 해당 클래스 샘플이 없는 경우
                class_results[class_name] = {
                    'accuracy': 0.0,
                    'avg_confidence': 0.0,
                    'avg_entropy': 0.0,
                    'log_loss': float('inf'),
                    'sample_count': 0
                }
                continue

            class_predictions = np.array(all_predictions)[class_mask]
            class_true_labels = np.array(all_true_labels)[class_mask]
            class_confidences = np.array(all_confidences)[class_mask]
            class_entropies = np.array(all_entropies)[class_mask]
            class_probs = np.array(all_probs)[class_mask]

            # 정확도 계산
            accuracy = np.mean(class_predictions == class_true_labels)

            # 평균 신뢰도 및 엔트로피
            avg_confidence = np.mean(class_confidences)
            avg_entropy = np.mean(class_entropies)

            # 클래스별 로그로스 계산 (직접 계산)
            if len(class_probs) > 0:
                try:
                    class_log_loss = self.calculate_class_specific_log_loss(class_idx, class_probs)
                except Exception as e:
                    print(f"Warning: Could not calculate log loss for class {class_name}: {e}")
                    class_log_loss = float('inf')
            else:
                class_log_loss = float('inf')

            class_results[class_name] = {
                'accuracy': float(accuracy),
                'avg_confidence': float(avg_confidence),
                'avg_entropy': float(avg_entropy),
                'log_loss': float(class_log_loss),
                'sample_count': int(np.sum(class_mask))
            }

        # 전체 성능
        overall_accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))
        overall_confidence = np.mean(all_confidences)
        overall_entropy = np.mean(all_entropies)

        return class_results, {
            'overall_accuracy': float(overall_accuracy),
            'overall_confidence': float(overall_confidence),
            'overall_entropy': float(overall_entropy),
            'overall_log_loss': float(overall_log_loss),
            'total_samples': len(all_predictions)
        }


def load_hierarchical_model(model_path, base_config):
    """계층적 분류 모델 로드"""
    print(f"Loading model from: {os.path.basename(model_path)}")

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

    return model, inference_config


def load_validation_dataset(fold_idx):
    """특정 폴드의 검증 데이터셋 로드"""
    # 폴드별 검증 데이터 디렉토리
    val_dir = config.get_fold_data_dir(fold_idx, "val")

    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    # 클래스 매핑 정보 로드
    class_mapping_path = config.get_class_mapping_path()
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r', encoding='utf-8') as f:
            saved_mapping = json.load(f)
            class_to_idx = {name: int(idx) for idx, name in saved_mapping.items()}
    else:
        raise FileNotFoundError(f"Class mapping file not found: {class_mapping_path}")

    # 검증용 transform 생성
    val_transform = get_transform(config, is_train=False)

    # 검증 데이터셋 생성
    val_dataset = CarBrandDataset(
        root_dir=val_dir,
        processor=None,
        transform=val_transform,
        is_train=False,
        class_mapping=class_to_idx,
        csv_mapping_path=config.MAPPING_CSV_PATH
    )

    print(f"Loaded validation dataset for fold {fold_idx}: {len(val_dataset)} samples")
    return val_dataset


def find_top_models(model_dir, top_k=3):
    """폴드별 상위 K개 모델 찾기"""
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return []

    def extract_score(filename):
        try:
            if '_classacc' in filename:
                return float(filename.split('_classacc')[1].split('_')[0])
            elif '_grpacc' in filename:
                return float(filename.split('_grpacc')[1].split('_')[0])
            elif '_acc' in filename:
                return float(filename.split('_acc')[1].split('_')[0])
            else:
                return 0.0
        except:
            return 0.0

    model_scores = [(f, extract_score(f)) for f in model_files]
    model_scores.sort(key=lambda x: x[1], reverse=True)

    top_models = model_scores[:top_k]
    return [(os.path.join(model_dir, f), f, score) for f, score in top_models]


def load_training_metadata(model_dir):
    """학습 메타데이터 로드"""
    label_mappings_path = os.path.join(model_dir, 'label_mappings.json')
    if os.path.exists(label_mappings_path):
        with open(label_mappings_path, 'r', encoding='utf-8') as f:
            label_mappings = json.load(f)

        id_to_class = {int(k): v for k, v in label_mappings['id_to_class'].items()}
        return [id_to_class[i] for i in range(len(id_to_class))]

    # 백업 방법
    mapping_info = load_class_mapping(config.MAPPING_CSV_PATH)
    unique_classes = sorted(list(set(mapping_info['original_to_group'].keys())))
    return unique_classes


def main():
    """각 폴드별 모델 성능 분석 메인 함수"""
    print("=" * 80)
    print("MODEL PERFORMANCE ANALYSIS BY FOLD")
    print("=" * 80)
    print("Analyzing each model on its corresponding fold validation set")
    print("Generating 15 CSV files with class-wise performance metrics")
    print("=" * 80)

    # 설정
    folds = [0, 1, 2, 3, 4]
    models_per_fold = 3
    output_dir = "model_analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    # 전체 결과 저장
    all_results = []

    for fold_idx in folds:
        print(f"\n{'=' * 50}")
        print(f"ANALYZING FOLD {fold_idx}")
        print(f"{'=' * 50}")

        # 모델 디렉토리
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, f"resnet_hierarchical_fold_{fold_idx}")
        if not os.path.exists(model_dir):
            print(f"Model directory not found: {model_dir}")
            continue

        # 검증 데이터셋 로드
        try:
            print(f"Loading validation dataset for fold {fold_idx}...")
            val_dataset = load_validation_dataset(fold_idx)
        except Exception as e:
            print(f"Error loading validation dataset for fold {fold_idx}: {e}")
            continue

        # 클래스 정보 로드
        trained_classes = load_training_metadata(model_dir)
        print(f"Classes: {len(trained_classes)}")

        # 상위 모델들 찾기
        top_models = find_top_models(model_dir, models_per_fold)
        print(f"Top {models_per_fold} models found:")

        for model_idx, (model_path, model_name, score) in enumerate(top_models):
            print(f"\n--- Model {model_idx + 1}: {model_name} (score: {score:.4f}) ---")

            try:
                # 모델 로드
                model, model_config = load_hierarchical_model(model_path, config)
                transform = get_transform(model_config, is_train=False)

                # 모델 분석기 생성
                analyzer = ModelAnalyzer(model, {
                    'fold': fold_idx,
                    'name': model_name,
                    'path': model_path,
                    'score': score
                }, trained_classes, transform)

                # 성능 분석
                class_results, overall_results = analyzer.analyze_validation_dataset(val_dataset)

                # 결과 정리
                results_data = []
                for class_name, metrics in class_results.items():
                    results_data.append({
                        'class_name': class_name,
                        'accuracy': metrics['accuracy'],
                        'avg_confidence': metrics['avg_confidence'],
                        'avg_entropy': metrics['avg_entropy'],
                        'log_loss': metrics['log_loss'],
                        'sample_count': metrics['sample_count']
                    })

                # DataFrame 생성 및 저장
                results_df = pd.DataFrame(results_data)

                # 전체 성능 행 추가
                overall_row = {
                    'class_name': 'OVERALL',
                    'accuracy': overall_results['overall_accuracy'],
                    'avg_confidence': overall_results['overall_confidence'],
                    'avg_entropy': overall_results['overall_entropy'],
                    'log_loss': overall_results['overall_log_loss'],
                    'sample_count': overall_results['total_samples']
                }
                results_df = pd.concat([results_df, pd.DataFrame([overall_row])], ignore_index=True)

                # CSV 저장
                csv_filename = f"fold_{fold_idx}_model_{model_idx + 1}_{model_name.replace('.pth', '')}.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                results_df.to_csv(csv_path, index=False)

                print(f"✓ Saved: {csv_filename}")
                print(f"  Overall accuracy: {overall_results['overall_accuracy']:.4f}")
                print(f"  Overall log loss: {overall_results['overall_log_loss']:.4f}")
                print(f"  Average confidence: {overall_results['overall_confidence']:.4f}")
                print(f"  Classes analyzed: {len(class_results)}")

                # 전체 결과에 추가
                all_results.append({
                    'fold': fold_idx,
                    'model_idx': model_idx + 1,
                    'model_name': model_name,
                    'score': score,
                    'csv_file': csv_filename,
                    'overall_accuracy': overall_results['overall_accuracy'],
                    'overall_log_loss': overall_results['overall_log_loss'],
                    'overall_confidence': overall_results['overall_confidence']
                })

                # 메모리 정리
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"✗ Error analyzing {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 전체 요약 저장
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(output_dir, "analysis_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETED")
    print(f"{'=' * 80}")
    print(f"Total models analyzed: {len(all_results)}")
    print(f"Results saved in: {output_dir}/")
    print(f"Summary file: analysis_summary.csv")
    print(f"\nGenerated {len(all_results)} CSV files with class-wise performance metrics")
    print("These files can be used for intelligent ensemble weighting!")

    # 성능 순위 출력
    if all_results:
        print(f"\nTop 5 models by overall accuracy:")
        sorted_results = sorted(all_results, key=lambda x: x['overall_accuracy'], reverse=True)
        for i, result in enumerate(sorted_results[:5]):
            print(f"  {i + 1}. Fold {result['fold']} Model {result['model_idx']}: {result['overall_accuracy']:.4f}")


if __name__ == "__main__":
    main()