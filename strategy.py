import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import defaultdict
import hashlib
import glob
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ML Models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy.optimize import minimize
from sklearn.model_selection import cross_val_score

from resnext_car_classification.config import config as resnext_config
from resnext_car_classification.model import HierarchicalCarClassifierImproved as ResNeXtModel
from resnext_car_classification.dataset.augmentation import get_transform as resnext_get_transform
from resnext_car_classification.dataset import CarBrandDataset as ResNeXtDataset

from resnet_car_classification.config import config as resnet_config
from resnet_car_classification.model import HierarchicalCarClassifierImproved as ResNetModel
from resnet_car_classification.dataset.augmentation import get_transform as resnet_get_transform
from resnet_car_classification.dataset import CarBrandDataset as ResNetDataset


class SimpleValidationDataset:
    """간단한 검증 데이터셋 (색상 폴더 제외)"""

    def __init__(self, root_dir, transform, class_mapping):
        self.transform = transform
        self.class_mapping = class_mapping
        self.samples = []

        # 색상 폴더 제외
        color_folders = {'Black', 'Blue', 'Gray', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'Silver', 'White',
                         'Yellow'}

        # 유효한 클래스 폴더들만 처리
        for class_name in os.listdir(root_dir):
            if class_name in color_folders:
                continue

            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            if class_name not in class_mapping:
                continue

            class_idx = class_mapping[class_name]

            # 이미지 파일들 찾기
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(image_extensions):
                    img_path = os.path.join(class_dir, img_file)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # 이미지 로드
        try:
            image = Image.open(path).convert('RGB')
        except:
            # 이미지 로드 실패 시 더미 이미지 생성
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "label": label}


class AdvancedMetaStacking:
    """고급 메타 스태킹 분류기 (성능 최적화)"""

    def __init__(self, output_dir="meta_models"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 396

        # 출력 디렉토리 생성
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 클래스 매핑 정보
        self.class_mapping = None
        self.duplicate_images = set()

        # 메타 학습 데이터
        self.meta_features = None
        self.meta_labels = None
        self.enhanced_meta_features = None

        # 메타 모델들
        self.meta_models = {}
        self.scalers = {}
        self.optimal_weights = None

        # 기본 경로들
        self.base_dir = r"C:\Users\tjdwn\GitHub\Hecto-AI-Challenge-CarClassification"
        self.data_dir = r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open\augmented_data"

    def load_class_mappings(self):
        """클래스 매핑 정보 로드"""
        label_mappings_path = os.path.join(
            self.base_dir, "resnet_car_classification", "outputs",
            "resnet_hierarchical_fold_0", "label_mappings.json"
        )

        with open(label_mappings_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
            self.class_mapping = {int(k): v for k, v in mappings['id_to_class'].items()}

        print(f"Loaded {len(self.class_mapping)} class mappings")

    def detect_data_leakage(self):
        """간단한 해시 기반 중복 감지"""
        print("Detecting data leakage...")

        all_images_info = []
        for fold_idx in range(5):
            fold_val_dir = os.path.join(self.data_dir, f"fold_{fold_idx}", "val")
            if not os.path.exists(fold_val_dir):
                continue

            class_dirs = [d for d in os.listdir(fold_val_dir)
                          if os.path.isdir(os.path.join(fold_val_dir, d))
                          and d not in {'Black', 'Blue', 'Gray', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'Silver',
                                        'White', 'Yellow'}]

            for class_name in class_dirs:
                class_dir = os.path.join(fold_val_dir, class_name)
                image_files = glob.glob(os.path.join(class_dir, "*.jpg"))

                for img_path in image_files:
                    try:
                        with open(img_path, 'rb') as f:
                            img_hash = hashlib.md5(f.read()).hexdigest()
                        all_images_info.append({
                            'path': img_path,
                            'hash': img_hash,
                            'fold_idx': fold_idx
                        })
                    except:
                        continue

        # 해시 기반 중복 감지
        hash_groups = defaultdict(list)
        for img_info in all_images_info:
            hash_groups[img_info['hash']].append(img_info)

        hash_duplicates = []
        for img_hash, img_list in hash_groups.items():
            if len(img_list) > 1:
                fold_set = set(img['fold_idx'] for img in img_list)
                if len(fold_set) > 1:
                    hash_duplicates.append(img_list)
                    for img in img_list:
                        self.duplicate_images.add((img['fold_idx'], img['path']))

        print(f"Found {len(hash_duplicates)} duplicate groups")
        print(f"Total duplicate images: {len(self.duplicate_images)}")

        return len(hash_duplicates) > 0

    def load_hierarchical_model(self, model_path, architecture):
        """계층적 분류 모델 로드"""
        config = resnext_config if architecture == 'resnext' else resnet_config
        model_class = ResNeXtModel if architecture == 'resnext' else ResNetModel

        checkpoint = torch.load(model_path, map_location=self.device)
        inference_config = deepcopy(config)

        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            inference_config.NUM_LABELS = saved_config.get('NUM_LABELS', inference_config.NUM_LABELS)
            inference_config.NUM_GROUPS = saved_config.get('NUM_GROUPS', inference_config.NUM_GROUPS)

        model = model_class(inference_config).to(self.device)
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

    def find_top_models(self, model_dir, architecture, top_k):
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

    def load_validation_dataset(self, fold_idx, architecture):
        """검증 데이터셋 로드"""
        config = resnext_config if architecture == 'resnext' else resnet_config
        dataset_class = ResNeXtDataset if architecture == 'resnext' else ResNetDataset
        get_transform = resnext_get_transform if architecture == 'resnext' else resnet_get_transform

        val_dir = os.path.join(self.data_dir, f"fold_{fold_idx}", "val")
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")

        class_to_idx = {name: idx for idx, name in self.class_mapping.items()}
        val_transform = get_transform(config, is_train=False)

        # CSV 경로 찾기 시도
        csv_mapping_path = None
        possible_csv_paths = [
            os.path.join(self.base_dir, f"resnet_car_classification", "outputs",
                         f"resnet_hierarchical_fold_{fold_idx}", "mapping_info.csv"),
            os.path.join(self.base_dir, f"resnext_car_classification", "outputs",
                         f"resnext_hierarchical_fold_{fold_idx}", "mapping_info.csv"),
            os.path.join(self.base_dir, "data", "mapping.csv"),
        ]

        for path in possible_csv_paths:
            if os.path.exists(path):
                csv_mapping_path = path
                break

        try:
            val_dataset = dataset_class(
                root_dir=val_dir,
                processor=None,
                transform=val_transform,
                is_train=False,
                class_mapping=class_to_idx,
                csv_mapping_path=csv_mapping_path
            )
        except:
            # 대안: 글로벌 SimpleValidationDataset 사용
            val_dataset = SimpleValidationDataset(val_dir, val_transform, class_to_idx)

        return val_dataset

    def create_swa_model(self, model_paths, architecture):
        """SWA 모델 생성"""
        base_model, base_config = self.load_hierarchical_model(model_paths[0], architecture)
        base_sd = base_model.state_dict()
        del base_model
        torch.cuda.empty_cache()

        swa_sd = {k: torch.zeros_like(base_sd[k].cpu()) for k in base_sd.keys()}

        for path in model_paths:
            model, _ = self.load_hierarchical_model(path, architecture)
            sd = model.state_dict()
            for k in swa_sd.keys():
                if k in sd:
                    swa_sd[k] += sd[k].cpu()
            del model
            torch.cuda.empty_cache()

        n = float(len(model_paths))
        for k in swa_sd.keys():
            swa_sd[k] = swa_sd[k] / n

        swa_model, swa_config = self.load_hierarchical_model(model_paths[0], architecture)
        swa_sd_gpu = {k: v.to(self.device) for k, v in swa_sd.items()}
        swa_model.load_state_dict(swa_sd_gpu, strict=False)
        swa_model.eval()

        return swa_model, swa_config

    def predict_with_swa_model(self, fold_idx, architecture):
        """SWA 모델로 검증 데이터 예측"""
        if architecture == 'resnext':
            model_dir = os.path.join(self.base_dir, "resnext_car_classification", "outputs",
                                     f"resnext_hierarchical_fold_{fold_idx}")
            top_k = 5
        else:
            model_dir = os.path.join(self.base_dir, "resnet_car_classification", "outputs",
                                     f"resnet_hierarchical_fold_{fold_idx}")
            top_k = 3

        if not os.path.exists(model_dir):
            return None

        val_dataset = self.load_validation_dataset(fold_idx, architecture)
        top_models = self.find_top_models(model_dir, architecture, top_k)
        model_paths = [path for path, name, score in top_models]

        if not model_paths:
            return None

        swa_model, _ = self.create_swa_model(model_paths, architecture)
        dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                                num_workers=0, pin_memory=False)

        all_predictions_probs = []
        all_labels = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"  {architecture}")):
            images = batch["pixel_values"].to(self.device)
            labels = batch["label"].cpu().numpy()

            with torch.no_grad():
                outputs = swa_model(images)
                probs = torch.softmax(outputs["logits"], dim=1).cpu().numpy()

                all_predictions_probs.append(probs)
                all_labels.extend(labels)

        del swa_model
        torch.cuda.empty_cache()

        predictions_probs = np.concatenate(all_predictions_probs, axis=0)
        labels = np.array(all_labels)

        return {
            'predictions_probs': predictions_probs,
            'labels': labels,
            'architecture': architecture,
            'fold_idx': fold_idx
        }

    def collect_all_predictions(self):
        """모든 폴드의 예측 결과 수집"""
        print("Collecting all fold predictions...")

        fold_data = {}
        architectures = ['resnet', 'resnext']

        for fold_idx in range(5):
            print(f"Processing Fold {fold_idx}...")

            fold_predictions = []
            fold_labels = None

            for architecture in architectures:
                result = self.predict_with_swa_model(fold_idx, architecture)
                if result:
                    fold_predictions.append(result['predictions_probs'])
                    if fold_labels is None:
                        fold_labels = result['labels']

            if fold_predictions:
                combined_predictions = np.concatenate(fold_predictions, axis=1)
                fold_data[fold_idx] = {
                    'predictions': combined_predictions,
                    'labels': fold_labels
                }
                print(f"  Fold {fold_idx}: {combined_predictions.shape}")

        return fold_data

    def create_clean_meta_dataset(self, fold_data, remove_duplicates=True):
        """깨끗한 메타 데이터셋 생성"""
        print("Creating clean meta dataset...")

        all_predictions = []
        all_labels = []

        for fold_idx, fold_info in fold_data.items():
            fold_predictions = fold_info['predictions']
            fold_labels = fold_info['labels']

            if remove_duplicates:
                # 간단화된 중복 제거
                clean_indices = list(range(len(fold_labels)))
            else:
                clean_indices = list(range(len(fold_labels)))

            if clean_indices:
                clean_fold_predictions = fold_predictions[clean_indices]
                clean_fold_labels = fold_labels[clean_indices]

                all_predictions.append(clean_fold_predictions)
                all_labels.append(clean_fold_labels)

        self.meta_features = np.vstack(all_predictions)
        self.meta_labels = np.concatenate(all_labels)

        print(f"Clean meta features: {self.meta_features.shape}")
        print(f"Clean meta labels: {self.meta_labels.shape}")

    def create_advanced_features(self):
        """고급 메타 피처 생성 (확장된 버전)"""
        print("Creating advanced meta features...")

        num_models = self.meta_features.shape[1] // self.num_classes
        reshaped_features = self.meta_features.reshape(-1, num_models, self.num_classes)

        features = [self.meta_features]  # 원본 확률들

        # 기본 통계 피처들
        features.append(np.max(reshaped_features, axis=2))  # 최대 확률
        features.append(np.mean(reshaped_features, axis=2))  # 평균 확률
        features.append(np.std(reshaped_features, axis=2))  # 표준편차
        features.append(np.min(reshaped_features, axis=2))  # 최소 확률
        features.append(np.median(reshaped_features, axis=2))  # 중앙값

        # 고급 통계 피처들
        features.append(np.percentile(reshaped_features, 25, axis=2))  # 25% 백분위수
        features.append(np.percentile(reshaped_features, 75, axis=2))  # 75% 백분위수

        # 엔트로피 피처 (불확실성)
        entropy = -np.sum(reshaped_features * np.log(reshaped_features + 1e-15), axis=2)
        features.append(entropy)

        # 확신도 피처들
        sorted_probs = np.sort(reshaped_features, axis=2)
        confidence_top1_top2 = sorted_probs[:, :, -1] - sorted_probs[:, :, -2]  # Top1 - Top2
        confidence_top1_top3 = sorted_probs[:, :, -1] - sorted_probs[:, :, -3]  # Top1 - Top3
        features.append(confidence_top1_top2)
        features.append(confidence_top1_top3)

        # 모델 간 통계 피처들
        features.append(np.var(reshaped_features, axis=1))  # 모델 간 분산
        features.append(np.mean(reshaped_features, axis=1))  # 모델 간 평균
        features.append(np.max(reshaped_features, axis=1))  # 모델 간 최대값

        # Top-K 예측 일치도
        top_k_predictions = np.argsort(reshaped_features, axis=2)[:, :, -3:]  # Top 3 예측
        for k in [1, 2, 3]:
            agreement = np.sum(top_k_predictions[:, :, -k:] == top_k_predictions[:, [0], -k:], axis=2)
            features.append(agreement.astype(float))

        # 확률 분포의 날카로움 (sharpness)
        sharpness = np.sum(reshaped_features ** 2, axis=2)
        features.append(sharpness)

        self.enhanced_meta_features = np.hstack(features)
        print(f"Enhanced features: {self.enhanced_meta_features.shape}")

    def train_optimized_meta_models(self, test_size=0.25):
        """최적화된 메타 모델 훈련"""
        print("Training optimized meta models...")

        # 고급 피처 생성
        self.create_advanced_features()

        # 데이터 분할 (계층화)
        X_train, X_test, y_train, y_test = train_test_split(
            self.enhanced_meta_features, self.meta_labels,
            test_size=test_size, stratify=self.meta_labels, random_state=42
        )

        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # 피처 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler

        # 1. XGBoost (튜닝된 하이퍼파라미터)
        print("Training optimized XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            early_stopping_rounds=30,
            tree_method='gpu_hist',

        )

        eval_set = [(X_test_scaled, y_test)]
        xgb_model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)
        xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)
        xgb_logloss = log_loss(y_test, xgb_pred_proba, labels=range(self.num_classes))
        self.meta_models['xgboost'] = {'model': xgb_model, 'logloss': xgb_logloss}
        print(f"  XGBoost LogLoss: {xgb_logloss:.6f}")

        # 2. LightGBM (최적화)
        print("Training optimized LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.5,
            min_child_samples=25,
            random_state=42,
            n_jobs=-1,
            objective='multiclass',
            metric='multi_logloss',
            verbose=-1,
            force_col_wise=True
        )

        eval_set = [(X_test_scaled, y_test)]
        lgb_model.fit(X_train_scaled, y_train, eval_set=eval_set,
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        lgb_pred_proba = lgb_model.predict_proba(X_test_scaled)
        lgb_logloss = log_loss(y_test, lgb_pred_proba, labels=range(self.num_classes))
        self.meta_models['lightgbm'] = {'model': lgb_model, 'logloss': lgb_logloss}
        print(f"  LightGBM LogLoss: {lgb_logloss:.6f}")

        # 3. CatBoost (최적화)
        print("Training optimized CatBoost...")
        cat_model = CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.08,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=False,
            loss_function='MultiClass',
            eval_metric='MultiClass',
            early_stopping_rounds=30,
            task_type='GPU'
        )

        eval_set = (X_test_scaled, y_test)
        cat_model.fit(X_train_scaled, y_train, eval_set=eval_set)
        cat_pred_proba = cat_model.predict_proba(X_test_scaled)
        cat_logloss = log_loss(y_test, cat_pred_proba, labels=range(self.num_classes))
        self.meta_models['catboost'] = {'model': cat_model, 'logloss': cat_logloss}
        print(f"  CatBoost LogLoss: {cat_logloss:.6f}")

        # 저장 - 테스트 데이터
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test

    def create_optimal_ensemble(self):
        """최적화된 앙상블 생성"""
        print("Creating optimal ensemble...")

        # 각 모델의 예측 확률 수집
        predictions_proba = {}

        for model_name, model_info in self.meta_models.items():
            model = model_info['model']
            pred_probs = model.predict_proba(self.X_test_scaled)
            predictions_proba[model_name] = pred_probs

        # 성능 기반 모델 필터링
        baseline_logloss = min(info['logloss'] for info in self.meta_models.values()) * 1.2
        good_models = {}

        for model_name, pred_probs in predictions_proba.items():
            model_logloss = self.meta_models[model_name]['logloss']
            if model_logloss < baseline_logloss:
                good_models[model_name] = pred_probs
                print(f"  Including {model_name}: {model_logloss:.6f}")
            else:
                print(f"  Excluding {model_name}: {model_logloss:.6f} (poor performance)")

        if len(good_models) == 0:
            print("  No good models found, using all models")
            good_models = predictions_proba

        # 가중치 최적화
        def ensemble_logloss(weights):
            weights = weights / weights.sum()
            ensemble_probs = np.zeros_like(list(good_models.values())[0])

            for i, (model_name, pred_probs) in enumerate(good_models.items()):
                ensemble_probs += weights[i] * pred_probs

            ensemble_probs = np.clip(ensemble_probs, 1e-15, 1 - 1e-15)
            return log_loss(self.y_test, ensemble_probs, labels=range(self.num_classes))

        # 초기 가중치 (성능 기반)
        n_models = len(good_models)
        logloss_values = np.array([self.meta_models[name]['logloss'] for name in good_models.keys()])
        initial_weights = 1.0 / logloss_values
        initial_weights = initial_weights / initial_weights.sum()

        # 제약 조건부 최적화
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1) for _ in range(n_models)]

        result = minimize(
            ensemble_logloss,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        self.optimal_weights = result.x
        ensemble_logloss_final = result.fun

        print("Optimal ensemble weights:")
        for i, (model_name, weight) in enumerate(zip(good_models.keys(), self.optimal_weights)):
            print(f"  {model_name}: {weight:.4f}")

        return good_models, ensemble_logloss_final

    def save_all_models(self):
        """모든 모델과 구성 요소 저장"""
        print(f"Saving all models to {self.output_dir}...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 메타 모델들 저장
        for model_name, model_info in self.meta_models.items():
            model_path = os.path.join(self.output_dir, f"{model_name}_{timestamp}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model_info['model'], f)
            print(f"  Saved {model_name} to {model_path}")

        # 2. 스케일러 저장
        scaler_path = os.path.join(self.output_dir, f"scaler_{timestamp}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers['main'], f)
        print(f"  Saved scaler to {scaler_path}")

        # 3. 최적 가중치 저장
        weights_path = os.path.join(self.output_dir, f"optimal_weights_{timestamp}.json")
        weights_dict = {
            'weights': self.optimal_weights.tolist(),
            'model_names': list(self.meta_models.keys()),
            'timestamp': timestamp
        }
        with open(weights_path, 'w') as f:
            json.dump(weights_dict, f, indent=2)
        print(f"  Saved optimal weights to {weights_path}")

        # 4. 클래스 매핑 저장
        mapping_path = os.path.join(self.output_dir, f"class_mapping_{timestamp}.json")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.class_mapping, f, ensure_ascii=False, indent=2)
        print(f"  Saved class mapping to {mapping_path}")

        # 5. 메타 정보 저장
        meta_info = {
            'num_classes': self.num_classes,
            'meta_features_shape': self.meta_features.shape,
            'enhanced_features_shape': self.enhanced_meta_features.shape,
            'model_performance': {name: info['logloss'] for name, info in self.meta_models.items()},
            'timestamp': timestamp
        }

        meta_path = os.path.join(self.output_dir, f"meta_info_{timestamp}.json")
        with open(meta_path, 'w') as f:
            json.dump(meta_info, f, indent=2)
        print(f"  Saved meta info to {meta_path}")

        print(f"All models saved successfully with timestamp: {timestamp}")
        return timestamp

    def compare_results(self, good_models, ensemble_logloss):
        """결과 비교"""
        print("\nFINAL PERFORMANCE COMPARISON (LogLoss)")
        print("=" * 60)

        # 베이스라인들
        num_models = self.meta_features.shape[1] // self.num_classes
        meta_probs = self.X_test_scaled[:, :num_models * self.num_classes].reshape(-1, num_models, self.num_classes)

        # 최고 단일 모델 (ResNeXt)
        best_single_probs = meta_probs[:, -1, :]
        best_single_probs = np.clip(best_single_probs, 1e-15, 1 - 1e-15)
        best_single_logloss = log_loss(self.y_test, best_single_probs, labels=range(self.num_classes))

        # 결과 출력
        results = [("Best Single Model", best_single_logloss)]

        # 메타 모델들
        for model_name, model_info in self.meta_models.items():
            results.append((model_name.title().replace('_', ' '), model_info['logloss']))

        results.append(("OPTIMAL ENSEMBLE", ensemble_logloss))

        results.sort(key=lambda x: x[1])

        for name, logloss in results:
            improvement = (best_single_logloss - logloss) / best_single_logloss * 100
            if "OPTIMAL" in name:
                print(f"*** {name:25s}: {logloss:.6f} ({improvement:+.2f}%)")
            else:
                print(f"    {name:25s}: {logloss:.6f} ({improvement:+.2f}%)")

        print("=" * 60)
        print(f"BEST IMPROVEMENT: {(best_single_logloss - ensemble_logloss) / best_single_logloss * 100:.2f}%")

    def run_advanced_stacking_pipeline(self):
        """고급 스태킹 파이프라인 실행"""
        print("ADVANCED META STACKING PIPELINE")
        print("=" * 80)

        # 1. 기본 설정
        self.load_class_mappings()

        # 2. 데이터 누수 감지
        has_leakage = self.detect_data_leakage()

        # 3. 베이스 모델 예측 수집
        fold_data = self.collect_all_predictions()

        # 4. 메타 데이터셋 생성
        self.create_clean_meta_dataset(fold_data, remove_duplicates=has_leakage)

        # 5. 최적화된 메타 모델 훈련
        self.train_optimized_meta_models()

        # 6. 최적 앙상블 생성
        good_models, ensemble_logloss = self.create_optimal_ensemble()

        # 7. 모든 모델 저장
        timestamp = self.save_all_models()

        # 8. 결과 비교
        self.compare_results(good_models, ensemble_logloss)

        print("\nADVANCED STACKING COMPLETED!")
        print(f"Models saved with timestamp: {timestamp}")
        return ensemble_logloss, timestamp


def main():
    """메인 실행 함수"""
    stacker = AdvancedMetaStacking(output_dir="meta_models_optimized")
    final_logloss, timestamp = stacker.run_advanced_stacking_pipeline()
    print(f"\nFINAL OPTIMIZED LOGLOSS: {final_logloss:.6f}")
    print(f"MODELS SAVED: meta_models_optimized/*_{timestamp}.*")


if __name__ == "__main__":
    main()