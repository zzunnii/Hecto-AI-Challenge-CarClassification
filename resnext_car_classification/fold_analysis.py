import os
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from copy import deepcopy

from resnext_car_classification.config import config
from resnext_car_classification.model import HierarchicalCarClassifierImproved
from resnext_car_classification.dataset.augmentation import get_transform


class EnsemblePredictor:

    def __init__(self, model_dirs):
        self.model_dirs = model_dirs
        self.available_models = self._discover_models()

    def _discover_models(self):
        print("Discovering available models...")
        available_models = {}

        for model_dir in self.model_dirs:
            if not os.path.exists(model_dir):
                print(f"Model directory not found: {model_dir}")
                continue

            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if model_files:
                fold_name = os.path.basename(model_dir)
                for i, model_file in enumerate(model_files):
                    model_key = f"{fold_name}_model_{i}"
                    available_models[model_key] = {
                        'path': os.path.join(model_dir, model_file),
                        'dir': model_dir,
                        'filename': model_file
                    }

        print(f"Found {len(available_models)} available models")
        return available_models

    def calculate_ensemble_weights(self):
        num_models = len(self.available_models)
        weights = {}

        for model_key in self.available_models.keys():
            weights[model_key] = 1.0 / num_models

        return weights


def load_models_for_ensemble(available_models, base_config):
    models = {}

    print("Loading models for ensemble...")

    for model_key, model_info in tqdm(available_models.items(), desc="Loading models"):
        model_path = model_info['path']

        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
            continue

        try:
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

            print(f"Loaded {model_key}: {os.path.basename(model_path)}")

        except Exception as e:
            print(f"Error loading {model_key}: {e}")
            continue

    print(f"Successfully loaded {len(models)} models")
    return models


def predict_test_data(models, test_dir, ensemble_weights, trained_classes):
    print("Predicting test data with simple average strategy...")

    test_images = []
    for img_file in os.listdir(test_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_images.append(img_file)

    test_images.sort()
    print(f"Found {len(test_images)} test images")

    num_classes = len(trained_classes)
    all_predictions = []
    batch_size = 16

    for i in tqdm(range(0, len(test_images), batch_size), desc="Processing batches"):
        batch_images = test_images[i:i + batch_size]
        model_predictions = {}

        for model_key, model_info in models.items():
            model = model_info['model']
            model_config = model_info['config']

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
                    dummy_tensor = torch.zeros((3, model_config.IMG_SIZE[0], model_config.IMG_SIZE[1]))
                    batch_tensors.append(dummy_tensor)

            if batch_tensors:
                batch_tensor = torch.stack(batch_tensors).to("cuda")

                with torch.no_grad():
                    try:
                        outputs = model(batch_tensor)
                        probs = torch.softmax(outputs["logits"], dim=1).cpu().numpy()
                        model_predictions[model_key] = probs
                    except Exception as e:
                        print(f"Error predicting with {model_key}: {e}")
                        dummy_probs = np.ones((len(batch_images), num_classes)) / num_classes
                        model_predictions[model_key] = dummy_probs

        for batch_idx in range(len(batch_images)):
            img_name = batch_images[batch_idx]

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

            pred_class_idx = np.argmax(ensemble_probs)
            pred_class_name = trained_classes[pred_class_idx]
            confidence = np.max(ensemble_probs)

            all_predictions.append({
                'image': img_name,
                'predicted_class': pred_class_name,
                'confidence': confidence,
                'probabilities': ensemble_probs.tolist(),
                'trained_classes': trained_classes
            })

    return all_predictions


def create_submission_file(predictions, output_path, submission_classes):
    submission_data = []

    for pred in predictions:
        img_id = os.path.splitext(pred['image'])[0]
        result_row = {'ID': img_id}

        probabilities = pred['probabilities']
        trained_classes = pred['trained_classes']

        for submission_class in submission_classes:
            if submission_class in trained_classes:
                class_idx = trained_classes.index(submission_class)
                result_row[submission_class] = float(probabilities[class_idx])
            else:
                result_row[submission_class] = 0.0

        submission_data.append(result_row)

    submission_df = pd.DataFrame(submission_data)
    ordered_columns = ['ID'] + submission_classes
    submission_df = submission_df[ordered_columns]
    submission_df.to_csv(output_path, index=False)

    print(f"Saved submission file: {output_path}")
    print(f"Predictions: {len(submission_data)}")
    print(f"Classes: {len(submission_classes)}")

    predicted_classes = []
    max_confidences = []

    for pred in predictions:
        probabilities = pred['probabilities']
        max_idx = np.argmax(probabilities)
        trained_classes = pred['trained_classes']
        predicted_class = trained_classes[max_idx]
        max_confidence = probabilities[max_idx]

        predicted_classes.append(predicted_class)
        max_confidences.append(max_confidence)

    class_counts = pd.Series(predicted_classes).value_counts()
    print("Top predicted classes:")
    for class_name, count in class_counts.head(10).items():
        print(f"  {class_name}: {count}")

    print(f"Average confidence: {np.mean(max_confidences):.4f}")


def main():
    print("SIMPLE AVERAGE ENSEMBLE TESTING")
    print("=" * 80)

    test_data_dir = os.path.join(config.BASE_DIR, "test")
    output_dir = "ensemble_results"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(test_data_dir):
        raise FileNotFoundError(f"Test directory not found: {test_data_dir}")

    sample_submission_path = os.path.join(config.BASE_DIR, "sample_submission.csv")
    if not os.path.exists(sample_submission_path):
        raise FileNotFoundError(f"Sample submission not found: {sample_submission_path}")

    sample_submission = pd.read_csv(sample_submission_path)
    submission_classes = [col for col in sample_submission.columns if col != 'ID']
    print(f"Sample submission classes: {len(submission_classes)}")

    class_mapping_path = config.get_class_mapping_path()
    with open(class_mapping_path, 'r', encoding='utf-8') as f:
        saved_mapping = json.load(f)
        trained_classes = [saved_mapping[str(i)] for i in range(len(saved_mapping))]

    print(f"Trained classes: {len(trained_classes)}")

    trained_set = set(trained_classes)
    submission_set = set(submission_classes)
    overlap = trained_set & submission_set
    print(f"Class overlap: {len(overlap)}/{len(submission_classes)} submission classes covered")

    # 모델 디렉토리들 찾기
    model_dirs = []
    model_output_base = config.MODEL_OUTPUT_DIR

    for item in os.listdir(model_output_base):
        item_path = os.path.join(model_output_base, item)
        if os.path.isdir(item_path) and item.startswith('resnet_hierarchical_fold_'):
            model_dirs.append(item_path)

    print(f"Found {len(model_dirs)} model directories")

    if len(model_dirs) == 0:
        print("No model directories found! Exiting...")
        return

    ensemble_predictor = EnsemblePredictor(model_dirs)
    models = load_models_for_ensemble(ensemble_predictor.available_models, config)

    if len(models) == 0:
        print("No models loaded! Exiting...")
        return

    weights = ensemble_predictor.calculate_ensemble_weights()

    print("Ensemble weights (simple average):")
    for model_key, weight in sorted(weights.items()):
        print(f"  {model_key}: {weight:.4f}")

    predictions = predict_test_data(models, test_data_dir, weights, trained_classes)

    submission_path = os.path.join(output_dir, "submission_simple_average.csv")
    create_submission_file(predictions, submission_path, submission_classes)

    print("\nEnsemble testing completed")
    print(f"Models used: {len(models)}")
    print(f"Results saved in: {output_dir}/")

    confidences = [p['confidence'] for p in predictions]
    print(f"Average confidence: {np.mean(confidences):.4f}")
    print(f"Min confidence: {np.min(confidences):.4f}")
    print(f"Max confidence: {np.max(confidences):.4f}")

    for model_info in models.values():
        del model_info['model']
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()