import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler  # 이건 당분간 유지하되, 아래 참고
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
import glob

from brand_classification.config import config
from brand_classification.train.loss import FocalLoss, MixupLoss
from brand_classification.dataset.augmentation import Mixup
from brand_classification.utils.metrics import log_loss_calc


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


class Trainer:
    """모델 학습을 위한 클래스"""

    def __init__(self, model, train_loader, val_loader, config, class_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.class_weights = class_weights

        # 손실 함수 설정
        if config.FOCAL_LOSS_ALPHA is None and class_weights is not None:
            self.alpha = class_weights
        else:
            self.alpha = config.FOCAL_LOSS_ALPHA

        self.criterion = FocalLoss(gamma=config.FOCAL_LOSS_GAMMA, alpha=self.alpha)

        # Mixup 설정
        self.mixup = Mixup(alpha=config.MIXUP_ALPHA) if config.USE_MIXUP else None
        if self.mixup:
            self.mixup_criterion = MixupLoss(self.criterion)

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

        # 얼리스토핑 설정
        self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

        # 최대 저장 모델 수
        self.max_saved_models = config.MAX_SAVED_MODELS

        # 저장된 모델 관리
        self.saved_models = []  # (val_score, model_path) 형태로 저장

    def train_epoch(self, epoch):
        """한 에폭 훈련"""
        self.model.train()
        epoch_loss = 0
        preds_list = []
        labels_list = []

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} [Train]")

        for step, batch in enumerate(progress_bar):
            # 배치 처리
            pixel_values = batch["pixel_values"].to("cuda")
            labels = batch["label"].to("cuda")

            # Mixup 적용
            if self.mixup and self.config.USE_MIXUP:
                pixel_values, labels_a, labels_b, lam = self.mixup((pixel_values, labels))

            # 그래디언트 누적을 위한 스케일링
            with torch.amp.autocast(device_type='cuda', enabled=self.config.USE_FP16):
                outputs = self.model(pixel_values)
                logits = outputs["logits"]

                if self.mixup and self.config.USE_MIXUP:
                    loss = self.mixup_criterion(logits, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(logits, labels)

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

            # 예측 및 레이블 저장 (Mixup이 없을 때만)
            if not (self.mixup and self.config.USE_MIXUP):
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                preds_list.extend(preds)
                labels_list.extend(labels.detach().cpu().numpy())

            # 진행 상황 업데이트
            if step % self.config.LOGGING_STEPS == 0:
                progress_bar.set_postfix({"loss": loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS})

        # 훈련 메트릭 계산
        train_loss = epoch_loss / len(self.train_loader)

        if not (self.mixup and self.config.USE_MIXUP):
            train_acc = accuracy_score(labels_list, preds_list)
            train_f1 = f1_score(labels_list, preds_list, average='weighted')
            return train_loss, train_acc, train_f1
        else:
            return train_loss, None, None

    def validate(self, epoch):
        """검증 데이터로 평가"""
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_logits = []

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} [Valid]")

        with torch.no_grad():
            for batch in progress_bar:
                pixel_values = batch["pixel_values"].to("cuda")
                labels = batch["label"].to("cuda")

                with autocast(enabled=self.config.USE_FP16):
                    outputs = self.model(pixel_values)
                    logits = outputs["logits"]
                    loss = self.criterion(logits, labels)

                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.detach().cpu().numpy())
                all_logits.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())

        # 검증 메트릭 계산
        val_loss /= len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        # 로그 손실 계산
        val_logloss = log_loss_calc(all_labels, all_logits)

        # 분류 리포트 출력
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=None, digits=4))

        return val_loss, val_acc, val_f1, val_logloss

    def save_checkpoint(self, epoch, val_score, val_metrics):
        """모델 체크포인트 저장"""
        model_path = os.path.join(self.checkpoint_dir, f"model_epoch{epoch + 1}_acc{val_score:.4f}.pth")

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
        print(f"Starting training for {self.config.NUM_EPOCHS} epochs...")
        start_time = time.time()

        for epoch in range(self.config.NUM_EPOCHS):
            # 훈련
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)

            # 검증
            val_loss, val_acc, val_f1, val_logloss = self.validate(epoch)

            # 훈련 결과 출력
            if train_acc is not None and train_f1 is not None:
                print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}:")
                print(f"  Train Loss: {train_loss:.4f} (metrics not available with Mixup)")

            print(
                f"  Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.4f}, Valid F1: {val_f1:.4f}, Valid LogLoss: {val_logloss:.4f}")

            # 검증 메트릭 저장
            val_metrics = {
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_logloss': val_logloss,
            }

            # 현재 모델이 지금까지 중 최고인 경우 저장
            if val_acc > self.best_val_score:
                self.best_val_score = val_acc
                model_path = self.save_checkpoint(epoch, val_acc, val_metrics)
                print(f"  🏆 Best model saved at {model_path} (val_acc: {val_acc:.4f})")

            # Early stopping 검사
            if self.early_stopping(val_acc):
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

        total_time = time.time() - start_time
        print(f"Training completed in {total_time / 60:.2f} minutes")
        print(f"Best validation score: {self.best_val_score:.4f}")

        # 최종적으로 저장된 모델 목록 출력
        print("\nSaved models:")
        for idx, (score, path) in enumerate(self.saved_models):
            print(f"  {idx + 1}. {os.path.basename(path)} (val_acc: {score:.4f})")