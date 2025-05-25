import os
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from car_classification.config import config
from car_classification.model import HierarchicalCarClassifier
from car_classification.dataset.augmentation import get_transform
from car_classification.utils.mapping import load_class_mapping

def adjust_logits(class_prob, mean=0.5684, std=0.1535):
    z = (class_prob - mean) / std
    adjusted = 1 / (1 + np.exp(-z))
    adjusted /= adjusted.sum()
    return adjusted

def load_hierarchical_model(model_path, config):
    checkpoint = torch.load(model_path, map_location='cuda')
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        config.NUM_LABELS = saved_config.get('NUM_LABELS', config.NUM_LABELS)
        config.NUM_GROUPS = saved_config.get('NUM_GROUPS', config.NUM_GROUPS)
    print(f"Loading model with {config.NUM_LABELS} classes and {config.NUM_GROUPS} groups")
    model = HierarchicalCarClassifier(config).to("cuda")
    state_dict = checkpoint['model_state_dict']
    keys_to_remove = [k for k in state_dict.keys() if 'loss_fn' in k]
    for key in keys_to_remove:
        del state_dict[key]
        print(f"Removed unnecessary key: {key}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def predict_batch_hierarchical(model, image_paths, transform):
    batch_images = []
    valid_paths = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            image = transform(image)
            batch_images.append(image)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    if not batch_images:
        return [], [], []
    batch_tensor = torch.stack(batch_images).to("cuda")
    with torch.no_grad():
        outputs = model(batch_tensor)
        class_logits = outputs["logits"]
        class_probs = torch.softmax(class_logits, dim=1)
        group_logits = outputs["group_logits"]
        group_probs = torch.softmax(group_logits, dim=1)
    return valid_paths, class_probs.cpu().numpy(), group_probs.cpu().numpy()

def main(fold_idx=1):
    test_dir = r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open\test"
    FOLD_IDX = fold_idx
    model_dir = os.path.join(config.MODEL_OUTPUT_DIR, f"hierarchical_fold_{FOLD_IDX}")
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError(f"No model found in {model_dir}")
    model_files.sort(key=lambda x: float(x.split('_acc')[1].split('_')[0]), reverse=True)
    model_path = os.path.join(model_dir, model_files[0])
    print(f"Using model: {model_path}")
    sample_submission = pd.read_csv(r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open\sample_submission.csv")
    submission_classes = [col for col in sample_submission.columns if col != 'ID']
    label_mappings_path = os.path.join(model_dir, 'label_mappings.json')
    if os.path.exists(label_mappings_path):
        with open(label_mappings_path, 'r', encoding='utf-8') as f:
            label_mappings = json.load(f)
        id_to_class = {int(k): v for k, v in label_mappings['id_to_class'].items()}
        id_to_group = {int(k): v for k, v in label_mappings['id_to_group'].items()}
        class_to_group = label_mappings['class_to_group']
    else:
        mapping_info = load_class_mapping(config.MAPPING_CSV_PATH)
        class_mapping_path = config.get_class_mapping_path()
        with open(class_mapping_path, 'r', encoding='utf-8') as f:
            saved_mapping = json.load(f)
        id_to_class = {int(idx): name for idx, name in saved_mapping.items()}
        id_to_group = mapping_info['idx_to_group']
        class_to_group = mapping_info['original_to_group']
    trained_classes = [id_to_class[i] for i in range(len(id_to_class))]
    model = load_hierarchical_model(model_path, config)
    transform = get_transform(config, is_train=False)
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    test_files.sort()
    results = []
    batch_size = 32
    group_predictions = []
    print("테스트 이미지 추론 중...")
    for i in tqdm(range(0, len(test_files), batch_size)):
        batch_files = test_files[i:i + batch_size]
        batch_paths = [os.path.join(test_dir, f) for f in batch_files]
        valid_paths, class_probs, group_probs = predict_batch_hierarchical(model, batch_paths, transform)
        for img_path, class_prob, group_prob in zip(valid_paths, class_probs, group_probs):
            img_name = os.path.basename(img_path)
            img_id = os.path.splitext(img_name)[0]
            predicted_group_idx = np.argmax(group_prob)
            predicted_group = id_to_group[predicted_group_idx]
            predicted_class_idx = np.argmax(class_prob)
            predicted_class = trained_classes[predicted_class_idx]
            expected_group = class_to_group.get(predicted_class, None)
            is_consistent = (expected_group == predicted_group)
            group_predictions.append({
                'id': img_id,
                'predicted_class': predicted_class,
                'predicted_group': predicted_group,
                'expected_group': expected_group,
                'consistent': is_consistent
            })
            result_row = {'ID': img_id}
            adjusted_probs = adjust_logits(class_prob)
            for submission_class in submission_classes:
                if submission_class in trained_classes:
                    class_idx = trained_classes.index(submission_class)
                    result_row[submission_class] = float(adjusted_probs[class_idx])
                else:
                    result_row[submission_class] = 0.0
            results.append(result_row)
    results_df = pd.DataFrame(results)
    results_df = results_df[sample_submission.columns]
    output_path = f'submission_hierarchical_fold{FOLD_IDX}.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n결과가 {output_path}에 저장되었습니다.")
    print(f"총 {len(results_df)} 개 이미지 처리 완료")
    consistency_df = pd.DataFrame(group_predictions)
    consistent_count = consistency_df['consistent'].sum()
    total_count = len(consistency_df)
    print(f"\n계층적 일관성 분석:")
    print(f"일관된 예측: {consistent_count}/{total_count} ({consistent_count / total_count * 100:.2f}%)")
    consistency_path = f'hierarchical_consistency_analysis_fold{FOLD_IDX}.csv'
    consistency_df.to_csv(consistency_path, index=False)
    print(f"일관성 분석 결과가 {consistency_path}에 저장되었습니다.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Hierarchical Model Inference')
    parser.add_argument('--fold', type=int, default=1, help='Fold index to use for inference (default: 1)')
    args = parser.parse_args()
    main(fold_idx=args.fold)