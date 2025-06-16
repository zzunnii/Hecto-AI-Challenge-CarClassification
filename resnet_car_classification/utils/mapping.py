import pandas as pd
import json
from collections import defaultdict


def load_class_mapping(csv_path):
    """CSV 파일에서 클래스 매핑 정보 로드"""
    df = pd.read_csv(csv_path)

    # 매핑 딕셔너리 생성
    original_to_brand = {}
    original_to_group = {}
    group_to_originals = defaultdict(list)

    for _, row in df.iterrows():
        original = row['original_class']
        brand = row['brand']
        group = row['new_group']

        original_to_brand[original] = brand
        original_to_group[original] = group
        group_to_originals[group].append(original)

    # 유니크한 브랜드와 그룹 목록
    unique_brands = sorted(list(set(original_to_brand.values())))
    unique_groups = sorted(list(set(original_to_group.values())))

    # 인덱스 매핑 생성
    brand_to_idx = {brand: idx for idx, brand in enumerate(unique_brands)}
    group_to_idx = {group: idx for idx, group in enumerate(unique_groups)}

    # 역매핑
    idx_to_brand = {idx: brand for brand, idx in brand_to_idx.items()}
    idx_to_group = {idx: group for group, idx in group_to_idx.items()}

    mapping_info = {
        'original_to_brand': original_to_brand,
        'original_to_group': original_to_group,
        'group_to_originals': dict(group_to_originals),
        'brand_to_idx': brand_to_idx,
        'group_to_idx': group_to_idx,
        'idx_to_brand': idx_to_brand,
        'idx_to_group': idx_to_group,
        'num_brands': len(unique_brands),
        'num_groups': len(unique_groups),
        'unique_brands': unique_brands,
        'unique_groups': unique_groups
    }

    return mapping_info


def save_mapping_info(mapping_info, output_path):
    """매핑 정보를 JSON으로 저장"""
    # defaultdict를 일반 dict로 변환
    save_dict = {
        k: dict(v) if isinstance(v, defaultdict) else v
        for k, v in mapping_info.items()
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, indent=2, ensure_ascii=False)


def load_mapping_info(json_path):
    """저장된 매핑 정보 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        mapping_info = json.load(f)

    # 인덱스 키를 정수로 변환 (JSON은 문자열로 저장됨)
    if 'idx_to_brand' in mapping_info:
        mapping_info['idx_to_brand'] = {
            int(k): v for k, v in mapping_info['idx_to_brand'].items()
        }
    if 'idx_to_group' in mapping_info:
        mapping_info['idx_to_group'] = {
            int(k): v for k, v in mapping_info['idx_to_group'].items()
        }

    return mapping_info


def get_group_mask(group_idx, mapping_info, num_classes):
    """특정 그룹에 속한 클래스들의 마스크 생성

    Args:
        group_idx: 그룹 인덱스
        mapping_info: 매핑 정보
        num_classes: 전체 클래스 수

    Returns:
        mask: [num_classes] 크기의 0/1 마스크
    """
    import numpy as np

    mask = np.zeros(num_classes, dtype=np.float32)
    group_name = mapping_info['idx_to_group'][group_idx]

    # 해당 그룹에 속한 original class들 찾기
    if group_name in mapping_info['group_to_originals']:
        for original_class in mapping_info['group_to_originals'][group_name]:
            # original_class의 인덱스 찾기 (이미 dataset에서 정의됨)
            # 여기서는 간단히 표시만
            pass

    return mask