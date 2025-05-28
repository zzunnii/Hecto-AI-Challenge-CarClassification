import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
import glob

from car_classification.config import config
from car_classification.train.loss import FocalLoss, MixupLoss
from car_classification.dataset.augmentation import Mixup
from car_classification.utils.metrics import log_loss_calc_for_validation, multiclass_log_loss, \
    create_answer_df_from_arrays, create_submission_df_from_arrays


class EarlyStopping:
    """Early stopping to prevent overfitting - 적응적 메트릭 지원"""

    def __init__(self, patience=5, min_delta=0.001, metric="class_acc"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, val_class_acc=None, val_group_acc=None, val_log_loss=None, training_stage="both"):
        # 학습 단계에 따른 동적 메트릭 선택
        if training_stage == "group_only":
            val_score = val_group_acc if val_group_acc is not None else 0.0
            higher_is_better = True
        else:
            # 메트릭에 따른 처리
            if self.metric == "class_acc":
                val_score = val_class_acc if val_class_acc is not None else 0.0
                higher_is_better = True
            elif self.metric == "group_acc":
                val_score = val_group_acc if val_group_acc is not None else 0.0
                higher_is_better = True
            elif self.metric == "log_loss":
                val_score = val_log_loss if val_log_loss is not None else val_loss
                higher_is_better = False
            else:  # "loss"
                val_score = val_loss
                higher_is_better = False

        if self.best_score is None:
            self.best_score = val_score
            return False

        # 개선 여부 확인
        if higher_is_better:
            improved = val_score > self.best_score + self.min_delta
        else:
            improved = val_score < self.best_score - self.min_delta

        if improved:
            self.best_score = val_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class ImprovedHierarchicalTrainer:
    """개선된 계층적 분류 모델 학습을 위한 클래스 - 점진적 학습 지원"""

    def __init__(self, model, train_loader, val_loader, config, class_weights=None, group_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.class_weights = class_weights
        self.group_weights = group_weights

        # 손실 함수 설정
        self._setup_loss_functions()

        # 옵티마이저 설정 (차등 학습률 지원)
        self._setup_optimizers()

        # 스케줄러 설정
        self._setup_schedulers()

        # Mixup 설정
        self.mixup = Mixup(alpha=config.MIXUP_ALPHA) if config.USE_MIXUP else None
        if self.mixup:
            self.mixup_class_criterion = MixupLoss(self.class_criterion)
            self.mixup_group_criterion = MixupLoss(self.group_criterion)

        # FP16 설정
        self.scaler = torch.amp.GradScaler(device='cuda') if config.USE_FP16 else None

        # 체크포인트 경로
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        self.checkpoint_dir = config.OUTPUT_DIR

        # Early stopping 메트릭에 따른 초기값 설정
        stopping_metric = getattr(config, 'EARLY_STOPPING_METRIC', 'class_acc')

        if stopping_metric in ['class_acc', 'group_acc']:
            self.best_val_score = 0.0  # accuracy는 높을수록 좋음
            self.higher_is_better = True
        else:  # loss, log_loss
            self.best_val_score = float('inf')  # loss는 낮을수록 좋음
            self.higher_is_better = False

        self.best_group_acc = 0.0

        # 얼리스토핑 설정
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            metric=stopping_metric,
            min_delta=0.001
        )

        # 최대 저장 모델 수
        self.max_saved_models = config.MAX_SAVED_MODELS
        self.saved_models = []

        # 🆕 클래스 이름 저장 (로그 로스 계산용)
        self.class_names = None

    def _setup_loss_functions(self):
        """손실 함수 설정"""
        # 2차 분류 손실 함수 (주 태스크)
        if self.config.USE_FOCAL_LOSS:
            if self.config.FOCAL_LOSS_ALPHA is None and self.class_weights is not None:
                self.alpha = self.class_weights
            else:
                self.alpha = self.config.FOCAL_LOSS_ALPHA
            self.class_criterion = FocalLoss(gamma=self.config.FOCAL_LOSS_GAMMA, alpha=self.alpha)
        else:
            if self.class_weights is not None:
                self.class_criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
            else:
                self.class_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 1차 분류 손실 함수 (보조 태스크)
        if self.group_weights is not None:
            self.group_criterion = nn.CrossEntropyLoss(weight=self.group_weights, label_smoothing=0.05)
        else:
            self.group_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        # 모델에 손실 함수 전달
        self.model.class_loss_fn = self.class_criterion
        self.model.group_loss_fn = self.group_criterion

    def _setup_optimizers(self):
        """옵티마이저 설정 (차등 학습률 지원)"""
        if getattr(self.config, 'USE_DIFFERENT_LR', False):
            # 그룹과 클래스 분류기에 다른 학습률 적용
            group_params = []
            class_params = []
            backbone_params = []

            for name, param in self.model.named_parameters():
                if 'group_classifier' in name:
                    group_params.append(param)
                elif 'class_classifier' in name or 'fusion_module' in name:
                    class_params.append(param)
                else:
                    backbone_params.append(param)

            self.optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': self.config.LEARNING_RATE},
                {'params': group_params, 'lr': self.config.GROUP_LEARNING_RATE},
                {'params': class_params, 'lr': self.config.CLASS_LEARNING_RATE}
            ], weight_decay=self.config.WEIGHT_DECAY)

            print(f"Using differential learning rates:")
            print(f"  Backbone: {self.config.LEARNING_RATE}")
            print(f"  Group classifier: {self.config.GROUP_LEARNING_RATE}")
            print(f"  Class classifier: {self.config.CLASS_LEARNING_RATE}")
        else:
            # 기본 옵티마이저
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )

    def _setup_schedulers(self):
        """스케줄러 설정"""
        num_training_steps = len(self.train_loader) * self.config.NUM_EPOCHS // self.config.GRADIENT_ACCUMULATION_STEPS
        num_warmup_steps = int(self.config.WARMUP_RATIO * num_training_steps)

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.LEARNING_RATE if not getattr(self.config, 'USE_DIFFERENT_LR', False) else [
                self.config.LEARNING_RATE,
                self.config.GROUP_LEARNING_RATE,
                self.config.CLASS_LEARNING_RATE
            ],
            total_steps=num_training_steps,
            pct_start=self.config.WARMUP_RATIO,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )

    def _get_current_metric_value(self, val_loss, val_acc, val_group_acc, val_logloss):
        """현재 모니터링 메트릭 값 반환"""
        metric = self.config.EARLY_STOPPING_METRIC

        if metric == "class_acc":
            return val_acc if val_acc > 0 else 0.0
        elif metric == "group_acc":
            return val_group_acc
        elif metric == "log_loss":
            return val_logloss if val_logloss > 0 else val_loss
        else:  # "loss"
            return val_loss

    def train_epoch(self, epoch):
        """한 에폭 훈련 (점진적 학습 지원)"""
        # 모델에 현재 에폭 설정
        self.model.set_epoch(epoch)

        self.model.train()
        epoch_loss = 0
        epoch_class_loss = 0
        epoch_group_loss = 0

        preds_list = []
        labels_list = []
        group_preds_list = []
        group_labels_list = []

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} [Train]")

        for step, batch in enumerate(progress_bar):
            # 배치 처리
            pixel_values = batch["pixel_values"].to("cuda")
            labels = batch["label"].to("cuda")  # original class
            group_labels = batch["group_label"].to("cuda")  # new group

            # Mixup 적용
            if self.mixup and self.config.USE_MIXUP:
                pixel_values, labels_a, labels_b, lam = self.mixup((pixel_values, labels))
                group_labels_a, group_labels_b = group_labels, group_labels[torch.randperm(len(group_labels))]

            # 그래디언트 누적을 위한 스케일링
            with torch.amp.autocast(device_type='cuda', enabled=self.config.USE_FP16):
                outputs = self.model(pixel_values, labels=labels, group_labels=group_labels)

                if self.mixup and self.config.USE_MIXUP:
                    # Mixup loss 계산
                    class_loss = self.mixup_class_criterion(outputs["logits"], labels_a, labels_b, lam) if outputs[
                                                                                                               "class_loss"] is not None else torch.tensor(
                        0.0)
                    group_loss = self.mixup_group_criterion(outputs["group_logits"], group_labels_a, group_labels_b,
                                                            lam)

                    # 동적 가중치 적용
                    if outputs["current_weights"] is not None:
                        group_weight, class_weight = outputs["current_weights"]
                        loss = group_weight * group_loss + class_weight * class_loss
                    else:
                        loss = outputs["loss"]
                else:
                    loss = outputs["loss"]
                    class_loss = outputs["class_loss"] if outputs["class_loss"] is not None else torch.tensor(0.0)
                    group_loss = outputs["group_loss"]

                # 그래디언트 누적
                loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS

            # FP16 훈련
            if self.scaler:
                self.scaler.scale(loss).backward()

                if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            else:
                loss.backward()

                if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            # 손실 누적
            epoch_loss += loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS
            if class_loss is not None:
                epoch_class_loss += class_loss.item()
            if group_loss is not None:
                epoch_group_loss += group_loss.item()

            # 예측 및 레이블 저장 (Mixup이 없고, 클래스 분류가 활성화된 경우만)
            if not (self.mixup and self.config.USE_MIXUP) and outputs["class_loss"] is not None:
                # 2차 분류 예측
                preds = torch.argmax(outputs["logits"], dim=1).detach().cpu().numpy()
                preds_list.extend(preds)
                labels_list.extend(labels.detach().cpu().numpy())

                # 1차 분류 예측
                group_preds = torch.argmax(outputs["group_logits"], dim=1).detach().cpu().numpy()
                group_preds_list.extend(group_preds)
                group_labels_list.extend(group_labels.detach().cpu().numpy())

            # 진행 상황 업데이트
            if step % self.config.LOGGING_STEPS == 0:
                current_weights = outputs.get("current_weights", (None, None))
                postfix = {
                    "loss": loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS,
                    "class_loss": class_loss.item() if class_loss is not None else 0,
                    "group_loss": group_loss.item() if group_loss is not None else 0
                }
                if current_weights[0] is not None:
                    postfix["grp_w"] = f"{current_weights[0]:.3f}"
                    postfix["cls_w"] = f"{current_weights[1]:.3f}"
                progress_bar.set_postfix(postfix)

        # 훈련 메트릭 계산
        train_loss = epoch_loss / len(self.train_loader)
        train_class_loss = epoch_class_loss / len(self.train_loader)
        train_group_loss = epoch_group_loss / len(self.train_loader)

        if len(labels_list) > 0:
            train_acc = accuracy_score(labels_list, preds_list)
            train_f1 = f1_score(labels_list, preds_list, average='weighted')
            train_group_acc = accuracy_score(group_labels_list, group_preds_list)
            return train_loss, train_class_loss, train_group_loss, train_acc, train_f1, train_group_acc
        else:
            return train_loss, train_class_loss, train_group_loss, None, None, None

    def validate(self, epoch):
        """검증 데이터로 평가 (새로운 로그 로스 사용)"""
        self.model.eval()
        val_loss = 0
        val_class_loss = 0
        val_group_loss = 0

        all_preds = []
        all_labels = []
        all_logits = []
        all_ids = []  # ID 추가

        all_group_preds = []
        all_group_labels = []
        all_group_logits = []

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} [Valid]")

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                pixel_values = batch["pixel_values"].to("cuda")
                labels = batch["label"].to("cuda")
                group_labels = batch["group_label"].to("cuda")

                with autocast(device_type="cuda", enabled=self.config.USE_FP16):
                    outputs = self.model(pixel_values, labels=labels, group_labels=group_labels)

                    loss = outputs["loss"]
                    class_loss = outputs["class_loss"]
                    group_loss = outputs["group_loss"]

                val_loss += loss.item() if loss is not None else 0
                if class_loss is not None:
                    val_class_loss += class_loss.item()
                if group_loss is not None:
                    val_group_loss += group_loss.item()

                # 2차 분류 예측 (클래스 분류가 활성화된 경우만)
                if outputs["class_loss"] is not None:
                    preds = torch.argmax(outputs["logits"], dim=1).detach().cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.detach().cpu().numpy())
                    all_logits.extend(torch.softmax(outputs["logits"], dim=1).detach().cpu().numpy())

                    # 🆕 가상 ID 생성 (배치 인덱스 기반)
                    batch_size = len(labels)
                    batch_ids = [f"val_{batch_idx}_{i}" for i in range(batch_size)]
                    all_ids.extend(batch_ids)

                # 1차 분류 예측
                group_preds = torch.argmax(outputs["group_logits"], dim=1).detach().cpu().numpy()
                all_group_preds.extend(group_preds)
                all_group_labels.extend(group_labels.detach().cpu().numpy())
                all_group_logits.extend(torch.softmax(outputs["group_logits"], dim=1).detach().cpu().numpy())

        # 검증 메트릭 계산
        val_loss /= len(self.val_loader)
        val_class_loss /= len(self.val_loader)
        val_group_loss /= len(self.val_loader)

        # 1차 분류 메트릭
        val_group_acc = accuracy_score(all_group_labels, all_group_preds)
        val_group_f1 = f1_score(all_group_labels, all_group_preds, average='weighted')

        # 2차 분류 메트릭 (활성화된 경우만)
        if len(all_preds) > 0:
            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='weighted')

            # 🆕 새로운 방식의 로그 로스 계산
            try:
                # 클래스 이름 생성 (인덱스 기반)
                if self.class_names is None:
                    self.class_names = [f"class_{i}" for i in range(len(all_logits[0]))]

                # DataFrame 생성
                answer_df = create_answer_df_from_arrays(all_ids, [self.class_names[label] for label in all_labels])
                submission_df = create_submission_df_from_arrays(all_ids, np.array(all_logits), self.class_names)

                # 새로운 방식으로 로그 로스 계산
                val_logloss = multiclass_log_loss(answer_df, submission_df)
            except Exception as e:
                print(f"New log loss calculation failed: {e}")
                # 백업으로 기존 방식 사용
                val_logloss = log_loss_calc_for_validation(all_labels, all_logits)

            # 분류 리포트 출력
            print("\n=== Original Class (2차) Classification Report ===")
            print(classification_report(all_labels, all_preds, target_names=None, digits=4))
        else:
            val_acc = 0.0
            val_f1 = 0.0
            val_logloss = 0.0
            print("\n=== Original Class (2차) Training Not Active ===")

        print("\n=== Group (1차) Classification Report ===")
        print(classification_report(all_group_labels, all_group_preds, target_names=None, digits=4))

        return (val_loss, val_class_loss, val_group_loss,
                val_acc, val_f1, val_logloss,
                val_group_acc, val_group_f1)

    def save_checkpoint(self, epoch, val_score, val_metrics):
        """모델 체크포인트 저장 (메트릭에 따라 적응적)"""
        metric_name = self.config.EARLY_STOPPING_METRIC

        if metric_name == "class_acc":
            filename = f"model_epoch{epoch + 1}_classacc{val_score:.4f}_grp{val_metrics['val_group_acc']:.4f}.pth"
        elif metric_name == "group_acc":
            filename = f"model_epoch{epoch + 1}_grpacc{val_score:.4f}_cls{val_metrics['val_acc']:.4f}.pth"
        elif metric_name == "log_loss":
            filename = f"model_epoch{epoch + 1}_logloss{val_score:.4f}_grp{val_metrics['val_group_acc']:.4f}.pth"
        else:  # loss
            filename = f"model_epoch{epoch + 1}_loss{val_score:.4f}_grp{val_metrics['val_group_acc']:.4f}.pth"

        model_path = os.path.join(self.checkpoint_dir, filename)

        # 저장할 데이터
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_score': val_score,
            'val_metrics': val_metrics,
            'config': self.config.to_dict(),
            'metric_name': metric_name,
            'class_names': self.class_names  # 🆕 클래스 이름 저장
        }

        # 모델 저장
        torch.save(checkpoint, model_path)

        # 저장된 모델 목록에 추가
        self.saved_models.append((val_score, model_path))

        # 메트릭에 따른 정렬 (accuracy는 내림차순, loss는 오름차순)
        if self.higher_is_better:
            self.saved_models.sort(reverse=True)  # 높은 값이 좋음
        else:
            self.saved_models.sort()  # 낮은 값이 좋음

        # 최대 모델 수 유지
        if len(self.saved_models) > self.max_saved_models:
            _, old_model_path = self.saved_models.pop()
            if os.path.exists(old_model_path):
                os.remove(old_model_path)
                print(f"Removed old model: {os.path.basename(old_model_path)}")

        return model_path

    def train(self):
        """전체 훈련 프로세스 (점진적 학습 지원)"""
        print(f"Starting hierarchical training for {self.config.NUM_EPOCHS} epochs...")
        print(f"Progressive training: {getattr(self.config, 'USE_PROGRESSIVE_TRAINING', False)}")
        print(f"Dynamic loss weights: {getattr(self.config, 'USE_DYNAMIC_LOSS_WEIGHTS', False)}")
        print(f"Early stopping metric: {self.config.EARLY_STOPPING_METRIC}")
        print(f"Higher is better: {self.higher_is_better}")
        print("🔥 Temperature scaling: DISABLED")

        start_time = time.time()

        for epoch in range(self.config.NUM_EPOCHS):
            # 훈련
            train_results = self.train_epoch(epoch)
            train_loss, train_class_loss, train_group_loss, train_acc, train_f1, train_group_acc = train_results

            # 검증
            val_results = self.validate(epoch)
            (val_loss, val_class_loss, val_group_loss,
             val_acc, val_f1, val_logloss,
             val_group_acc, val_group_f1) = val_results

            # 현재 학습 단계 정보
            current_weights = self.model.get_dynamic_loss_weights(epoch)
            training_stage = self.model.classifier.training_stage

            # 현재 모니터링 메트릭 값
            current_score = self._get_current_metric_value(val_loss, val_acc, val_group_acc, val_logloss)

            # 훈련 결과 출력
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS} (Stage: {training_stage}):")
            print(f"  Weights - Group: {current_weights[0]:.3f}, Class: {current_weights[1]:.3f}")
            print(f"  Train:")
            print(f"    Total Loss: {train_loss:.4f} (Class: {train_class_loss:.4f}, Group: {train_group_loss:.4f})")
            if train_acc is not None:
                print(f"    Class Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
                print(f"    Group Acc: {train_group_acc:.4f}")
            else:
                print(f"    Class metrics not available (Group-only or Mixup stage)")

            print(f"  Valid:")
            print(f"    Total Loss: {val_loss:.4f} (Class: {val_class_loss:.4f}, Group: {val_group_loss:.4f})")
            if val_acc > 0:
                print(f"    Class Acc: {val_acc:.4f}, F1: {val_f1:.4f}, LogLoss: {val_logloss:.4f}")
            print(f"    Group Acc: {val_group_acc:.4f}, F1: {val_group_f1:.4f}")
            print(f"    Monitoring {self.config.EARLY_STOPPING_METRIC}: {current_score:.4f}")

            # 검증 메트릭 저장
            val_metrics = {
                'val_loss': val_loss,
                'val_class_loss': val_class_loss,
                'val_group_loss': val_group_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_logloss': val_logloss,
                'val_group_acc': val_group_acc,
                'val_group_f1': val_group_f1,
                'training_stage': training_stage,
                'loss_weights': current_weights
            }

            # 현재 모델이 지금까지 중 최고인 경우 저장
            is_best = False
            if self.higher_is_better:
                is_best = current_score > self.best_val_score
            else:
                is_best = current_score < self.best_val_score

            if is_best:
                self.best_val_score = current_score
                self.best_group_acc = val_group_acc
                model_path = self.save_checkpoint(epoch, current_score, val_metrics)
                print(f"  🏆 Best model saved at {os.path.basename(model_path)}")
                print(f"     Best {self.config.EARLY_STOPPING_METRIC}: {current_score:.4f}")

            # Early stopping 검사 (그룹만 학습 단계에서는 제외)
            if training_stage != "group_only":
                if self.early_stopping(
                        val_loss=val_loss,
                        val_class_acc=val_acc,
                        val_group_acc=val_group_acc,
                        val_log_loss=val_logloss if val_logloss > 0 else None,
                        training_stage=training_stage
                ):
                    print(f"Early stopping triggered after epoch {epoch + 1}")
                    print(f"Best {self.config.EARLY_STOPPING_METRIC}: {self.best_val_score:.4f}")
                    break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 60:.2f} minutes")
        print(f"Best {self.config.EARLY_STOPPING_METRIC}: {self.best_val_score:.4f}")
        print(f"Best group accuracy: {self.best_group_acc:.4f}")

        # 최종적으로 저장된 모델 목록 출력
        print("\nSaved models (best to worst):")
        for idx, (score, path) in enumerate(self.saved_models):
            print(f"  {idx + 1}. {os.path.basename(path)} ({self.config.EARLY_STOPPING_METRIC}: {score:.4f})")


# 기존 코드와의 호환성
HierarchicalTrainer = ImprovedHierarchicalTrainer