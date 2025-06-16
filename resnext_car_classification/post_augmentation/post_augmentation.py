import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings
import argparse
import random
from concurrent.futures import ThreadPoolExecutor
import time
from resnext_car_classification.utils import seed_everything
warnings.filterwarnings('ignore')


class UltraFastCarColorAugmentor:
    """ 초고속 최적화된 차량 색상 증강기 (K-means 제거)"""

    def __init__(self, seed=42):
        # 시드 고정
        seed_everything(seed)

        # 실제 데이터 분석 결과 기반 색상 정의 (간소화)
        self.class_colors = {
            'Black': {'rgb': [64, 34, 64], 'count': 23065},
            'Silver': {'rgb': [64, 173, 64], 'count': 4364},
            'Gray': {'rgb': [64, 97, 64], 'count': 3022},
            'White': {'rgb': [64, 203, 64], 'count': 1536},
            'Purple': {'rgb': [64, 77, 64], 'count': 382},
            'Red': {'rgb': [64, 157, 64], 'count': 326},
            'Blue': {'rgb': [64, 82, 64], 'count': 133},
            'Orange': {'rgb': [64, 169, 64], 'count': 128},
            'Yellow': {'rgb': [64, 200, 64], 'count': 120},
            'Pink': {'rgb': [64, 106, 64], 'count': 17},
            'Green': {'rgb': [64, 44, 64], 'count': 3}
        }

        # 11가지 명확한 증강 색상 (골드 제거)
        self.target_colors = {
            'black': [30, 30, 30],
            'white': [240, 240, 240],
            'red': [180, 30, 30],
            'blue': [30, 60, 180],
            'green': [30, 120, 30],
            'yellow': [220, 200, 30],
            'orange': [200, 100, 30],
            'purple': [100, 30, 120],
            'brown': [90, 60, 40],
            'navy': [20, 30, 100],
        }

        # 최적화된 증강 전략
        self._define_fast_augmentation_strategy()

    def _define_fast_augmentation_strategy(self):
        """ 빠른 증강 전략 정의"""
        total_samples = sum(cls_info['count'] for cls_info in self.class_colors.values())

        self.augmentation_strategy = {}

        for class_name, class_info in self.class_colors.items():
            count = class_info['count']
            ratio = count / total_samples

            # 최적화된 증강 수량
            if ratio > 0.5:  # Black
                num_augmentations = 1
            elif ratio > 0.1:  # Silver, Gray
                num_augmentations = 2
            elif ratio > 0.04:  # White
                num_augmentations = 3
            elif ratio > 0.01:  # Purple, Red
                num_augmentations = 4
            else:  # 소수 클래스
                num_augmentations = 5

            # 간단한 색상 선택 (거리 계산 생략)
            available_colors = self._get_fast_colors(class_name, num_augmentations)
            self.augmentation_strategy[class_name] = available_colors

    def _get_fast_colors(self, original_class, num_colors):
        """ 빠른 색상 선택 (복잡한 계산 없이)"""
        # 각 클래스별로 미리 정의된 최적 색상 조합 (골드 제거)
        fast_mappings = {
            'Black': ['white', 'red', 'blue', 'yellow', 'green'],
            'Silver': ['black', 'red', 'blue', 'white', 'green'],
            'Gray': ['white', 'red', 'blue', 'yellow', 'orange'],
            'White': ['black', 'red', 'blue', 'green', 'purple'],
            'Purple': ['white', 'black', 'red', 'blue', 'yellow'],
            'Red': ['black', 'white', 'blue', 'green', 'yellow'],
            'Blue': ['white', 'red', 'yellow', 'orange', 'green'],
            'Orange': ['black', 'white', 'blue', 'green', 'purple'],
            'Yellow': ['black', 'red', 'blue', 'purple', 'navy'],
            'Pink': ['black', 'white', 'blue', 'green', 'brown'],
            'Green': ['white', 'red', 'yellow', 'orange', 'purple']
        }

        colors = fast_mappings.get(original_class, ['black', 'white', 'red', 'blue', 'green'])
        return colors[:num_colors]

    def fast_detect_dominant_color(self, img_array, mask_array):
        """ 초고속 주요 색상 감지 (K-means 제거)"""
        try:
            # 마스크 영역의 픽셀만 추출
            mask_pixels = img_array[mask_array > 128]

            if len(mask_pixels) == 0:
                return None

            # 간단한 평균 색상 계산 (K-means 대신)
            mean_color = np.mean(mask_pixels, axis=0).astype(int)
            return mean_color

        except:
            return None

    def fast_detect_car_color_class(self, img_array, mask_array):
        """초고속 차량 색상 클래스 감지"""
        try:
            # 빠른 주요 색상 감지
            dominant_rgb = self.fast_detect_dominant_color(img_array, mask_array)
            if dominant_rgb is None:
                return None, None

            # 각 클래스 색상과의 거리 계산 (벡터화)
            class_rgbs = np.array([info['rgb'] for info in self.class_colors.values()])
            class_names = list(self.class_colors.keys())

            distances = np.sqrt(np.sum((class_rgbs - dominant_rgb) ** 2, axis=1))
            min_idx = np.argmin(distances)

            if distances[min_idx] <= 60:  # 허용 범위
                return class_names[min_idx], dominant_rgb
            else:
                return "Unknown", dominant_rgb

        except:
            return None, None

    def fast_create_color_mask(self, img_array, mask_array, target_rgb, tolerance=70):
        """초고속 색상 마스크 생성"""
        try:
            # 차량 영역만
            car_mask = mask_array > 128

            # 벡터화된 색상 거리 계산
            color_distances = np.sqrt(np.sum((img_array - target_rgb) ** 2, axis=2))
            color_match = color_distances < tolerance

            # 최종 마스크
            final_mask = np.logical_and(car_mask, color_match).astype(np.uint8) * 255
            return final_mask

        except:
            return np.zeros_like(mask_array)

    def fast_apply_color_augmentation(self, img_array, mask_array, target_rgb, new_color_name, tolerance=70):
        """초고속 색상 증강 적용"""
        try:
            # 빠른 색상 마스크 생성
            color_mask = self.fast_create_color_mask(img_array, mask_array, target_rgb, tolerance)
            color_mask_array = (color_mask > 50).astype(np.float32)

            if np.sum(color_mask_array) == 0:
                return img_array.astype(np.uint8)

            # 새 색상 적용
            new_rgb = np.array(self.target_colors[new_color_name])

            # 간단한 밝기 보존
            img_float = img_array.astype(np.float32)
            original_brightness = np.mean(img_float, axis=2, keepdims=True)
            brightness_factor = np.clip(original_brightness / 128.0, 0.6, 1.3)

            # 새 색상에 밝기 적용
            new_colored = new_rgb * brightness_factor
            new_colored = np.clip(new_colored, 0, 255)

            # 마스크 영역만 교체
            result = img_float * (1 - color_mask_array[..., np.newaxis]) + \
                     new_colored * color_mask_array[..., np.newaxis]

            return np.clip(result, 0, 255).astype(np.uint8)

        except:
            return img_array.astype(np.uint8)

    def process_single_image_fast(self, file_info, mask_dir, class_output_dir):
        """  단일 이미지 초고속 처리 - 실제 차량 클래스명 유지"""
        try:
            # 실제 차량 클래스명 사용 (디스커버리_5_2017_2020 등)
            actual_class_name = file_info['class_name']  # 실제 차량 모델명
            img_filename = os.path.basename(file_info['file_path'])
            img_path = file_info['full_path']

            # 마스크 찾기 (실제 클래스명으로)
            mask_path = find_matching_mask(img_filename, mask_dir, actual_class_name)
            if mask_path is None:
                return 0, f"No mask for {img_filename} in class {actual_class_name}"

            # 한 번만 이미지 로딩
            try:
                original_img = Image.open(img_path).convert('RGB')
                img_array = np.array(original_img)

                mask = Image.open(mask_path).convert('L')
                mask_array = np.array(mask)
            except Exception as e:
                return 0, f"Failed to load {img_filename}: {e}"

            # 빠른 색상 분석 (차량의 실제 색상 감지)
            detected_color_class, dominant_color = self.fast_detect_car_color_class(img_array, mask_array)

            if detected_color_class is None or dominant_color is None:
                # 색상 감지 실패시에도 원본은 저장
                base_name = os.path.splitext(img_filename)[0]
                original_output_path = os.path.join(class_output_dir, f"{base_name}_original.jpg")
                original_img.save(original_output_path, 'JPEG', quality=95)
                return 1, f"Color detection failed but saved original: {img_filename}"

            base_name = os.path.splitext(img_filename)[0]
            generated_count = 0

            # 원본 저장
            original_output_path = os.path.join(class_output_dir, f"{base_name}_original.jpg")
            original_img.save(original_output_path, 'JPEG', quality=95)
            generated_count += 1

            # 감지된 색상에 따른 증강 색상들 가져오기
            augmentation_colors = self.augmentation_strategy.get(
                detected_color_class, ['black', 'white', 'red', 'blue'][:2]
            )

            # 빠른 색상 증강 적용
            for color_name in augmentation_colors:
                try:
                    # 빠른 증강 적용
                    augmented_array = self.fast_apply_color_augmentation(
                        img_array, mask_array, dominant_color, color_name, tolerance=70
                    )

                    # 빠른 품질 검사
                    if self.fast_quality_check(augmented_array, img_array):
                        augmented_img = Image.fromarray(augmented_array)
                        aug_output_path = os.path.join(class_output_dir, f"{base_name}_{color_name}.jpg")
                        augmented_img.save(aug_output_path, 'JPEG', quality=95)
                        generated_count += 1

                except Exception as e:
                    continue

            return generated_count, f"Success: {img_filename} ({detected_color_class} detected)"

        except Exception as e:
            return 0, f"Error processing {file_info.get('file_path', 'unknown')}: {str(e)}"

    def fast_quality_check(self, aug_array, orig_array):
        """  빠른 품질 검사"""
        try:
            # 간단한 체크만
            aug_mean = np.mean(aug_array)
            orig_mean = np.mean(orig_array)

            # 너무 극단적인 변화 방지
            if abs(aug_mean - orig_mean) > 120:
                return False

            # 유효 범위 체크
            if np.any(aug_array < 0) or np.any(aug_array > 255):
                return False

            return True
        except:
            return False

    def print_strategy(self):
        print("=" * 50)
        print("초고속 색상 증강 전략")
        print("=" * 50)

        for class_name, colors in self.augmentation_strategy.items():
            count = self.class_colors[class_name]['count']
            multiplier = len(colors) + 1
            print(f"{class_name:12s}: {count:5,}대 → {count * multiplier:6,}장 ({multiplier}배)")
            print(f"             색상: {colors}")
        print("=" * 50)


def find_matching_mask(image_filename, mask_dir, class_name):
    """마스크 파일 찾기 (개선된 패턴 매칭)"""
    base_name = os.path.splitext(image_filename)[0]
    class_mask_dir = os.path.join(mask_dir, class_name)

    if not os.path.exists(class_mask_dir):
        return None

    # 다양한 마스크 패턴 시도
    patterns = [
        f"{base_name}_mask.png",
        f"{base_name}_mask.jpg",
        f"{base_name}.png",
        f"{base_name}.jpg",
        f"mask_{base_name}.png",
        f"mask_{base_name}.jpg",
        f"{base_name}_seg.png",
        f"{base_name}_seg.jpg",
    ]

    # 가장 일반적인 패턴 먼저 확인
    for pattern in patterns:
        mask_path = os.path.join(class_mask_dir, pattern)
        if os.path.exists(mask_path):
            return mask_path

    # 패턴 매칭 실패시 부분 문자열 매칭
    try:
        files_in_dir = os.listdir(class_mask_dir)
        matching_files = [f for f in files_in_dir if base_name in f and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if matching_files:
            return os.path.join(class_mask_dir, matching_files[0])
    except:
        pass

    return None


def load_fold_split(fold_idx, splits_dir):
    """폴드 분할 정보 로드"""
    splits_file = os.path.join(splits_dir, 'fold_splits.json')
    with open(splits_file, 'r', encoding='utf-8') as f:
        fold_splits = json.load(f)
    return fold_splits[f'fold_{fold_idx}']


def process_ultra_fast_fold_augmentation(fold_idx, train_dir, mask_dir, output_dir, splits_dir, augmentor):
    """ 초고속 폴드 증강 처리"""

    start_time = time.time()
    print(f"\n  Processing fold {fold_idx} with ULTRA-FAST optimization...")

    # 폴드 데이터 로드
    fold_data = load_fold_split(fold_idx, splits_dir)
    train_files = fold_data['train']
    val_files = fold_data['val']

    # 출력 디렉토리 생성
    fold_output_dir = os.path.join(output_dir, f'fold_{fold_idx}')
    train_output_dir = os.path.join(fold_output_dir, 'train_augmented')
    val_output_dir = os.path.join(fold_output_dir, 'val')

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)

    # 클래스별 출력 디렉토리 미리 생성 (실제 차량 클래스명으로)
    actual_class_names = set()
    for file_info in train_files + val_files:
        actual_class_names.add(file_info['class_name'])

    for actual_class_name in actual_class_names:
        os.makedirs(os.path.join(train_output_dir, actual_class_name), exist_ok=True)
        os.makedirs(os.path.join(val_output_dir, actual_class_name), exist_ok=True)

    #   Train 데이터 초고속 증강
    print(f"  Ultra-fast augmenting: {len(train_files)} images")
    total_generated = 0
    success_count = 0
    error_count = 0

    # 이미지별 처리
    for file_info in tqdm(train_files, desc=f"Fold {fold_idx} - Ultra Fast"):
        actual_class_name = file_info['class_name']  # 실제 차량 클래스명
        class_output_dir = os.path.join(train_output_dir, actual_class_name)

        generated_count, message = augmentor.process_single_image_fast(
            file_info, mask_dir, class_output_dir
        )
        total_generated += generated_count

        if generated_count > 0:
            success_count += 1
        else:
            error_count += 1
            if error_count <= 5:  # 처음 5개 에러만 출력
                print(f" {message}")

    # Validation 데이터 빠른 복사
    print(f"Fast copying validation data: {len(val_files)} images")
    processed_val = 0

    for file_info in tqdm(val_files, desc=f"Fold {fold_idx} - Val Copy"):
        try:
            class_name = file_info['class_name']
            img_path = file_info['full_path']
            img_filename = os.path.basename(file_info['file_path'])

            class_output_dir = os.path.join(val_output_dir, class_name)
            output_path = os.path.join(class_output_dir, img_filename)

            original_img = Image.open(img_path).convert('RGB')
            original_img.save(output_path, 'JPEG', quality=95)
            processed_val += 1

        except Exception as e:
            continue

    # 결과 출력
    elapsed_time = time.time() - start_time
    print(f"\n" + "=" * 50)
    print(f"Fold {fold_idx} 초고속 완료  ")
    print(f"처리 시간: {elapsed_time / 60:.1f}분")
    print(f"Train: {len(train_files)}장 → {total_generated}장 (성공: {success_count}, 실패: {error_count})")
    print(f"Val: {processed_val}장 복사")
    if elapsed_time > 0:
        print(f"속도: {len(train_files) / (elapsed_time / 60):.1f}장/분")
    print("=" * 50)

    return total_generated, processed_val


def main(fold=None):
    """  초고속 메인 프로세스"""
    from resnext_car_classification.config import config

    print("차량 색상 증강기 시작")
    print(" 벡터화 최적화 적용")
    print("=" * 50)

    # 경로 설정
    TRAIN_DIR = config.DATA_DIR
    MASK_DIR = config.MASK_DIR
    OUTPUT_DIR = config.AUGMENTED_DATA_DIR
    SPLITS_DIR = config.SPLITS_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 초고속 증강기 초기화
    augmentor = UltraFastCarColorAugmentor(seed=42)
    augmentor.print_strategy()

    if fold is not None:
        folds_to_process = [fold]
    else:
        folds_to_process = range(5)

    total_start_time = time.time()
    total_train_images = 0
    total_val_images = 0

    print(f"처리할 폴드: {folds_to_process}")

    for fold_idx in folds_to_process:
        try:
            train_count, val_count = process_ultra_fast_fold_augmentation(
                fold_idx, TRAIN_DIR, MASK_DIR, OUTPUT_DIR, SPLITS_DIR, augmentor
            )
            total_train_images += train_count
            total_val_images += val_count

        except Exception as e:
            print(f"Error processing fold {fold_idx}: {e}")

    # 최종 결과
    total_elapsed = time.time() - total_start_time
    print(f"\n" + "=" * 60)
    print(f"전체 증강 완료")
    print(f"총 처리 시간: {total_elapsed / 60:.1f}분")
    print(f"총 Train 이미지: {total_train_images:,}장")
    print(f"총 Val 이미지: {total_val_images:,}장")
    print(f"전체 속도: {(total_train_images + total_val_images) / (total_elapsed / 60):.1f}장/분")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='  Ultra-Fast Car Color Augmentation')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to process (0-4). If not specified, process all folds.')

    args = parser.parse_args()
    main(fold=args.fold)