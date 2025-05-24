import os
import time
import numpy as np
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
from car_classification.utils.metrics import log_loss_calc


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): Number of epochs to wait after validation metric stops improving
            min_delta (float): Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            return False

        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


class HierarchicalTrainer:
    """계층적 분류 모델 학습을 위한 클래스"""

    def __init__(self, model, train_loader, val_loader, config, class_weights=None, group_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.class_weights = class_weights
        self.group_weights = group_weights

        # 손실 함수 설정 - 2차 분류 (주 태스크)
        if config.USE_FOCAL_LOSS:
            if config.FOCAL_LOSS_ALPHA is None and class_weights is not None:
                self.alpha = class_weights
            else:
                self.alpha = config.FOCAL_LOSS_ALPHA
            self.class_criterion = FocalLoss(gamma=config.FOCAL_LOSS_GAMMA, alpha=self.alpha)
        else:
            if class_weights is not None:
                self.class_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            else:
                self.class_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 1차 분류 손실 함수 (보조 태스크)
        if group_weights is not None:
            self.group_criterion = nn.CrossEntropyLoss(weight=group_weights, label_smoothing=0.05)
        else:
            self.group_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        # 모델에 손실 함수 전달
        self.model.class_loss_fn = self.class_criterion
        self.model.group_loss_fn = self.group_criterion

        # Mixup 설정
        self.mixup = Mixup(alpha=config.MIXUP_ALPHA) if config.USE_MIXUP else None
        if self.mixup:
            self.mixup_class_criterion = MixupLoss(self.class_criterion)
            self.mixup_group_criterion = MixupLoss(self.group_criterion)

        # 옵티마이저 및 스케줄러 설정
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # 스케줄러: Cosine with warmup
        num_training_steps = len(train_loader) * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
        num_warmup_steps = int(config.WARMUP_RATIO * num_training_steps)

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            total_steps=num_training_steps,
            pct_start=config.WARMUP_RATIO,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )

        # FP16 설정
        self.scaler = torch.amp.GradScaler(device='cuda') if config.USE_FP16 else None

        # 체크포인트 경로
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        self.checkpoint_dir = config.OUTPUT_DIR

        # 최고 성능 모델 저장 변수
        self.best_val_score = 0.0
        self.best_group_acc = 0.0

        # 얼리스토핑 설정
        self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

        # 최대 저장 모델 수
        self.max_saved_models = config.MAX_SAVED_MODELS

        # 저장된 모델 관리
        self.saved_models = []

    def train_epoch(self, epoch):
        """한 에폭 훈련"""
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
                    class_loss = self.mixup_class_criterion(outputs["logits"], labels_a, labels_b, lam)
                    group_loss = self.mixup_group_criterion(outputs["group_logits"], group_labels_a, group_labels_b,
                                                            lam)
                    loss = (self.config.GROUP_LOSS_WEIGHT * group_loss +
                            self.config.CLASS_LOSS_WEIGHT * class_loss)
                else:
                    loss = outputs["loss"]
                    class_loss = outputs["class_loss"]
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

            # 예측 및 레이블 저장 (Mixup이 없을 때만)
            if not (self.mixup and self.config.USE_MIXUP):
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
                progress_bar.set_postfix({
                    "loss": loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS,
                    "class_loss": class_loss.item() if class_loss is not None else 0,
                    "group_loss": group_loss.item() if group_loss is not None else 0
                })

        # 훈련 메트릭 계산
        train_loss = epoch_loss / len(self.train_loader)
        train_class_loss = epoch_class_loss / len(self.train_loader)
        train_group_loss = epoch_group_loss / len(self.train_loader)

        if not (self.mixup and self.config.USE_MIXUP):
            train_acc = accuracy_score(labels_list, preds_list)
            train_f1 = f1_score(labels_list, preds_list, average='weighted')
            train_group_acc = accuracy_score(group_labels_list, group_preds_list)
            return train_loss, train_class_loss, train_group_loss, train_acc, train_f1, train_group_acc
        else:
            return train_loss, train_class_loss, train_group_loss, None, None, None

    def validate(self, epoch):
        """검증 데이터로 평가"""
        self.model.eval()
        val_loss = 0
        val_class_loss = 0
        val_group_loss = 0

        all_preds = []
        all_labels = []
        all_logits = []

        all_group_preds = []
        all_group_labels = []
        all_group_logits = []

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} [Valid]")

        with torch.no_grad():
            for batch in progress_bar:
                pixel_values = batch["pixel_values"].to("cuda")
                labels = batch["label"].to("cuda")
                group_labels = batch["group_label"].to("cuda")

                with autocast(device_type="cuda", enabled=self.config.USE_FP16):
                    outputs = self.model(pixel_values, labels=labels, group_labels=group_labels)

                    loss = outputs["loss"]
                    class_loss = outputs["class_loss"]
                    group_loss = outputs["group_loss"]

                val_loss += loss.item()
                if class_loss is not None:
                    val_class_loss += class_loss.item()
                if group_loss is not None:
                    val_group_loss += group_loss.item()

                # 2차 분류 예측
                preds = torch.argmax(outputs["logits"], dim=1).detach().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.detach().cpu().numpy())
                all_logits.extend(torch.softmax(outputs["logits"], dim=1).detach().cpu().numpy())

                # 1차 분류 예측
                group_preds = torch.argmax(outputs["group_logits"], dim=1).detach().cpu().numpy()
                all_group_preds.extend(group_preds)
                all_group_labels.extend(group_labels.detach().cpu().numpy())
                all_group_logits.extend(torch.softmax(outputs["group_logits"], dim=1).detach().cpu().numpy())

        # 검증 메트릭 계산
        val_loss /= len(self.val_loader)
        val_class_loss /= len(self.val_loader)
        val_group_loss /= len(self.val_loader)

        # 2차 분류 메트릭 (주 태스크)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_logloss = log_loss_calc(all_labels, all_logits)

        # 1차 분류 메트릭
        val_group_acc = accuracy_score(all_group_labels, all_group_preds)
        val_group_f1 = f1_score(all_group_labels, all_group_preds, average='weighted')

        # 분류 리포트 출력
        print("\n=== Original Class (2차) Classification Report ===")
        print(classification_report(all_labels, all_preds, target_names=None, digits=4))

        print("\n=== Group (1차) Classification Report ===")
        print(classification_report(all_group_labels, all_group_preds, target_names=None, digits=4))

        return (val_loss, val_class_loss, val_group_loss,
                val_acc, val_f1, val_logloss,
                val_group_acc, val_group_f1)

    def save_checkpoint(self, epoch, val_score, val_metrics):
        """모델 체크포인트 저장"""
        model_path = os.path.join(self.checkpoint_dir,
                                  f"model_epoch{epoch + 1}_acc{val_score:.4f}_grp{val_metrics['val_group_acc']:.4f}.pth")

        # 저장할 데이터
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': val_score,
            'val_metrics': val_metrics,
            'config': self.config.to_dict()
        }

        # 모델 저장
        torch.save(checkpoint, model_path)

        # 저장된 모델 목록에 추가
        self.saved_models.append((val_score, model_path))

        # 점수 기준으로 정렬 (내림차순)
        self.saved_models.sort(reverse=True)

        # 최대 모델 수 유지
        if len(self.saved_models) > self.max_saved_models:
            _, old_model_path = self.saved_models.pop()
            if os.path.exists(old_model_path):
                os.remove(old_model_path)
                print(f"Removed old model: {old_model_path}")

        return model_path

    def train(self):
        """전체 훈련 프로세스"""
        print(f"Starting hierarchical training for {self.config.NUM_EPOCHS} epochs...")
        print(f"Loss weights - Group: {self.config.GROUP_LOSS_WEIGHT}, Class: {self.config.CLASS_LOSS_WEIGHT}")
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

            # 훈련 결과 출력
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}:")
            print(f"  Train:")
            print(f"    Total Loss: {train_loss:.4f} (Class: {train_class_loss:.4f}, Group: {train_group_loss:.4f})")
            if train_acc is not None:
                print(f"    Class Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
                print(f"    Group Acc: {train_group_acc:.4f}")
            else:
                print(f"    (Metrics not available with Mixup)")

            print(f"  Valid:")
            print(f"    Total Loss: {val_loss:.4f} (Class: {val_class_loss:.4f}, Group: {val_group_loss:.4f})")
            print(f"    Class Acc: {val_acc:.4f}, F1: {val_f1:.4f}, LogLoss: {val_logloss:.4f}")
            print(f"    Group Acc: {val_group_acc:.4f}, F1: {val_group_f1:.4f}")

            # 검증 메트릭 저장
            val_metrics = {
                'val_loss': val_loss,
                'val_class_loss': val_class_loss,
                'val_group_loss': val_group_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_logloss': val_logloss,
                'val_group_acc': val_group_acc,
                'val_group_f1': val_group_f1
            }

            # 현재 모델이 지금까지 중 최고인 경우 저장 (주 태스크 기준)
            if val_acc > self.best_val_score:
                self.best_val_score = val_acc
                self.best_group_acc = val_group_acc
                model_path = self.save_checkpoint(epoch, val_acc, val_metrics)
                print(f"  🏆 Best model saved at {model_path}")
                print(f"     Class Acc: {val_acc:.4f}, Group Acc: {val_group_acc:.4f}")

            # Early stopping 검사 (주 태스크 기준)
            if self.early_stopping(val_acc):
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 60:.2f} minutes")
        print(f"Best validation scores:")
        print(f"  Class Acc: {self.best_val_score:.4f}")
        print(f"  Group Acc: {self.best_group_acc:.4f}")

        # 최종적으로 저장된 모델 목록 출력
        print("\nSaved models:")
        for idx, (score, path) in enumerate(self.saved_models):
            print(f"  {idx + 1}. {os.path.basename(path)} (val_acc: {score:.4f})")


# 기존 코드와의 호환성
Trainer = HierarchicalTrainer