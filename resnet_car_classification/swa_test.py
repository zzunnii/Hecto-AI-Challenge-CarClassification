import os
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from copy import deepcopy

from car_classification.config import config
from car_classification.model import HierarchicalCarClassifierImproved
from car_classification.dataset.augmentation import get_transform


def load_hierarchical_model(path, base_config):
    """
    계층적 분류 모델을 로드해서 eval 모드로 돌려주는 함수.
    저장된 state_dict에서 'loss_fn' 관련 키를 제거한 뒤 모델에 로드합니다.
    """
    checkpoint = torch.load(path, map_location="cuda")
    inference_config = deepcopy(base_config)

    # checkpoint 내부에 config 정보가 있으면 덮어쓰기
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

    # 'loss_fn' 이름이 들어간 키는 모두 제거
    keys_to_remove = [k for k in state_dict.keys() if "loss_fn" in k]
    for k in keys_to_remove:
        state_dict.pop(k, None)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.current_epoch = 999
    model.classifier.set_training_stage("both")
    return model


def swa_models(model_paths, base_config):
    """
    model_paths 리스트에 있는 여러 .pth 파일을 불러와 파라미터를 평균(SWA)한 모델을 반환합니다.
    모든 파라미터 연산을 CPU 위에서 수행하여 'cuda ↔ cpu' 불일치 문제를 방지합니다.
    """
    assert len(model_paths) > 0, "최소 하나 이상의 모델 경로가 필요합니다."

    # 첫 번째 모델 로드 후 state_dict를 CPU로 가져오기
    base_model = load_hierarchical_model(model_paths[0], base_config)
    base_sd = base_model.state_dict()
    swa_sd = {k: base_sd[k].cpu() * 0.0 for k in base_sd.keys()}

    # 나머지 모델들의 state_dict을 CPU로 누적
    for path in model_paths:
        m = load_hierarchical_model(path, base_config)
        sd = m.state_dict()
        for k in swa_sd.keys():
            swa_sd[k] += sd[k].cpu()

    # 평균화
    n = float(len(model_paths))
    for k in swa_sd.keys():
        swa_sd[k] = swa_sd[k] / n

    # 평균화된 state_dict을 GPU 모델에 로드
    swa_model = load_hierarchical_model(model_paths[0], base_config)
    swa_model.load_state_dict(swa_sd, strict=False)
    swa_model.eval()
    return swa_model


def main():
    print("=" * 80)
    print("FOLD별 SWA → 단순평균 앙상블 (ID + 원본 클래스 확률) 스크립트")
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

    # 2) 각 fold별 상위 3개 모델(.pth 이름) → 전체 경로 → SWA 모델 생성
    swa_models_dict = {}
    for fold_idx in sorted(summary_df["fold"].unique()):
        fold_rows = summary_df[summary_df["fold"] == fold_idx]
        top3_names = fold_rows["model_name"].tolist()  # .pth 파일 이름 3개
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, f"resnet_hierarchical_fold_{fold_idx}")
        # 전체 경로 리스트
        top3_paths = [os.path.join(model_dir, name) for name in top3_names]

        print(f"Fold {fold_idx} → SWA 대상 모델들: {top3_names}")
        swa_model = swa_models(top3_paths, config)
        swa_models_dict[fold_idx] = {
            "model": swa_model,
            "config": deepcopy(config)
        }

    # 3) 테스트 데이터 목록 (이미지 파일들)
    test_images = [f for f in os.listdir(test_data_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    test_images.sort()
    print(f"총 테스트 이미지 수: {len(test_images)}")

    # 4) 클래스 매핑 로드 (idx → 원본 클래스 이름)
    class_mapping_path = config.get_class_mapping_path()
    with open(class_mapping_path, "r", encoding="utf-8") as f:
        saved_mapping = json.load(f)
    idx_to_class = {int(idx): name for idx, name in saved_mapping.items()}
    num_classes = len(idx_to_class)
    # 순서대로 원본 클래스 리스트
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # 5) 배치 단위로 예측 → 5개 SWA 모델의 확률(softmax) 단순평균 앙상블
    all_predictions = []
    batch_size = 16

    for i in tqdm(range(0, len(test_images), batch_size), desc="테스트 배치"):
        batch_images = test_images[i : i + batch_size]

        # 각 fold SWA 모델별 확률 저장
        fold_probs = {}
        for fold_idx, info in swa_models_dict.items():
            model = info["model"]
            model_config = info["config"]
            transform = get_transform(model_config, is_train=False)

            batch_tensors = []
            for img_name in batch_images:
                img_path = os.path.join(test_data_dir, img_name)
                try:
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = transform(image)
                except:
                    # 로드 오류 시 더미 텐서 생성
                    image_tensor = torch.zeros((3, model_config.IMG_SIZE[0], model_config.IMG_SIZE[1]))
                batch_tensors.append(image_tensor)

            batch_tensor = torch.stack(batch_tensors).to("cuda")
            with torch.no_grad():
                outputs = model(batch_tensor)
                # 소프트맥스 확률
                probs = torch.softmax(outputs["logits"], dim=1).cpu().numpy()
                fold_probs[fold_idx] = probs

        # 5개 SWA 모델의 확률을 모두 더한 뒤 평균
        for idx_in_batch, img_name in enumerate(batch_images):
            ensemble_probs = np.zeros(num_classes, dtype=np.float32)
            for fold_idx in fold_probs.keys():
                ensemble_probs += fold_probs[fold_idx][idx_in_batch]
            ensemble_probs /= len(fold_probs)

            # 제출용 ID: 파일명에서 확장자 제거
            img_id = os.path.splitext(img_name)[0]  # e.g. "TEST_00000"

            # 결과 딕셔너리 생성: 첫 컬럼은 'id', 그 뒤에 클래스별 확률
            result = {"ID": img_id}
            for cls_idx, cls_name in enumerate(class_names):
                result[cls_name] = float(ensemble_probs[cls_idx])
            all_predictions.append(result)

    # 6) 원본 클래스 이름을 컬럼명으로 하는 softmax 제출 파일 생성
    submission_df = pd.DataFrame(all_predictions)
    submission_path = os.path.join(output_dir, "submission_swa_simpleavg_probs.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"✓ 제출 파일 저장: {submission_path}")

    # 7) 제출 파일 일부 미리보기 (상위 5개 행)
    print("제출 파일 샘플 (상위 5개 행):")
    print(submission_df.head(5))


if __name__ == "__main__":
    main()
