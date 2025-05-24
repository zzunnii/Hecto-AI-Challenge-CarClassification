import numpy as np
from PIL import ImageOps
#timm kornia einops torch transformers
# Load BiRefNet with weights
from transformers import AutoModelForImageSegmentation
birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to('cuda')
birefnet.eval()
birefnet.half()


def find_dominant_car_color(image_path, mask):
    """마스크 영역에서 주요 차체 색상 찾기"""
    # 원본 이미지 로드
    original_img = Image.open(image_path).convert('RGB')
    img_array = np.array(original_img)
    mask_array = np.array(mask)

    # 마스크 영역의 픽셀만 추출
    mask_pixels = img_array[mask_array > 128]  # 마스크 임계값

    # K-means로 주요 색상 클러스터링 (3개 색상)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(mask_pixels)

    # 가장 많은 픽셀을 가진 색상 찾기
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    dominant_idx = unique[np.argmax(counts)]
    dominant_rgb = kmeans.cluster_centers_[dominant_idx].astype(int)

    print(f"주요 차체 색상 (RGB): {dominant_rgb}")
    return dominant_rgb


def create_color_mask(image_path, mask, target_rgb, tolerance=50):
    """특정 색상 영역만의 마스크 생성"""
    original_img = Image.open(image_path).convert('RGB')
    img_array = np.array(original_img)
    mask_array = np.array(mask)

    # 마스크 영역 내에서만 작업
    color_mask = np.zeros_like(mask_array)

    # 타겟 색상과 유사한 픽셀 찾기
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if mask_array[i, j] > 128:  # 차량 마스크 영역 내에서만
                pixel_rgb = img_array[i, j]
                # 색상 거리 계산
                color_distance = np.sqrt(np.sum((pixel_rgb - target_rgb) ** 2))
                if color_distance < tolerance:
                    color_mask[i, j] = 255

    return color_mask


# 🔥 더 빠른 벡터화 버전
def apply_vectorized_color_replacement(image_path, mask, target_rgb, new_color_name, tolerance=80):
    """벡터화된 빠른 색상 교체"""
    original_img = Image.open(image_path).convert('RGB')
    img_array = np.array(original_img).astype(np.float32)

    color_mask = create_color_mask(image_path, mask, target_rgb, tolerance)
    color_mask_array = (color_mask > 50).astype(np.float32)  # 이진 마스크

    # 새 색상 정의
    replacement_colors = {
        'White': [250, 250, 250],
        'Red': [200, 30, 30],
        'Blue': [30, 60, 200],
        'Silver': [180, 180, 180],
        'Pink': [255, 120, 180],
    }

    new_rgb = np.array(replacement_colors[new_color_name])

    # 원본 밝기 계산 (그레이스케일 변환)
    original_brightness = np.mean(img_array, axis=2, keepdims=True)
    brightness_factor = original_brightness / 128.0
    brightness_factor = np.clip(brightness_factor, 0.4, 1.3)

    # 새 색상에 밝기 적용
    new_colored = new_rgb * brightness_factor
    new_colored = np.clip(new_colored, 0, 255)

    # 마스크 영역만 교체
    result = img_array * (1 - color_mask_array[..., np.newaxis]) + new_colored * color_mask_array[..., np.newaxis]
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result), color_mask

def extract_mask_only(birefnet, imagepath):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath)
    input_images = transform_image(image).unsqueeze(0).to('cuda').half()

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)

    return mask  # 마스크만 반환


# 테스트 실행
image_path = r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open\train\디스커버리_5_2017_2020\디스커버리_5_2017_2020_0001.jpg"

# 마스크 추출 및 주요 색상 찾기
mask = extract_mask_only(birefnet, image_path)
dominant_color = find_dominant_car_color(image_path, mask)

# 🔥 강력한 색상 교체 테스트
colors = ['White', 'Red', 'Blue', 'Silver', 'Pink']

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for idx, color_name in enumerate(colors):
    augmented_img, _ = apply_vectorized_color_replacement(
        image_path, mask, dominant_color, color_name, tolerance=100
    )

    axes[idx].imshow(augmented_img)
    axes[idx].set_title(f'{color_name} Car (강화버전)')
    axes[idx].axis('off')

plt.tight_layout()
plt.show()