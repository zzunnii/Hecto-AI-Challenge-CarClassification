"""
차량 분류 프로젝트 메인 파이프라인
1. 교차검증 분할 생성
2. 데이터 증강 (폴드별)
3. 모델 훈련 (폴드별)
"""
import os
import argparse
from car_classification.config import config
from car_classification.cross_validation.cross_validation import create_cross_validation_splits, verify_splits
from car_classification.post_augmentation.post_augmentation import main as augmentation_main
from car_classification.utils import seed_everything


def setup_project():
    """프로젝트 초기 설정"""
    print("=== 프로젝트 설정 확인 ===")

    # 경로 확인
    config.verify_paths()

    # 필요한 디렉토리가 없으면 안내
    required_paths = [config.DATA_DIR, config.MASK_DIR]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]

    if missing_paths:
        print(f"\n다음 경로들이 존재하지 않습니다:")
        for path in missing_paths:
            os.makedirs(path, exist_ok=True)
            print(f"  - {path}")
        print("필요한 폴더를 생성합니다.")
        return False

    print("✓ 모든 필수 경로가 확인되었습니다.")
    return True


def create_splits():
    """교차검증 분할 생성"""
    print("\n=== 1단계: 교차검증 분할 생성 ===")

    splits_file = config.get_splits_file_path()

    if os.path.exists(splits_file):
        print(f"기존 분할 파일이 발견되었습니다: {splits_file}")
        response = input("다시 생성하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("기존 분할을 사용합니다.")
            verify_splits(splits_file)
            return True

    try:
        fold_splits, class_mapping = create_cross_validation_splits(
            data_dir=config.DATA_DIR,
            output_dir=config.SPLITS_DIR,
            n_folds=config.N_FOLDS,
            seed=config.SEED
        )

        verify_splits(splits_file)
        print("✓ 교차검증 분할이 성공적으로 생성되었습니다.")
        return True

    except Exception as e:
        print(f"✗ 교차검증 분할 생성 실패: {e}")
        return False


def run_augmentation(fold=None):
    """데이터 증강 실행"""
    print(f"\n=== 2단계: 데이터 증강 {'(전체 폴드)' if fold is None else f'(폴드 {fold})'} ===")

    try:
        augmentation_main(fold=fold)
        print("데이터 증강이 성공적으로 완료되었습니다.")
        return True

    except Exception as e:
        print(f"데이터 증강 실패: {e}")
        return False


def run_training(fold_idx):
    """특정 폴드로 모델 훈련"""
    print(f"\n=== 3단계: 모델 훈련 (폴드 {fold_idx}) ===")

    try:
        # 폴드 설정
        config.set_fold(fold_idx)

        # 훈련 데이터 경로 설정
        train_data_dir = config.get_fold_data_dir(fold_idx, "train")
        val_data_dir = config.get_fold_data_dir(fold_idx, "val")

        # 경로 확인
        if not os.path.exists(train_data_dir):
            print(f"✗ 훈련 데이터 경로가 존재하지 않습니다: {train_data_dir}")
            print("먼저 데이터 증강을 실행해주세요.")
            return False

        if not os.path.exists(val_data_dir):
            print(f"✗ 검증 데이터 경로가 존재하지 않습니다: {val_data_dir}")
            print("먼저 데이터 증강을 실행해주세요.")
            return False

        print(f"훈련 데이터: {train_data_dir}")
        print(f"검증 데이터: {val_data_dir}")

        # 설정 업데이트
        original_data_dir = config.DATA_DIR
        config.DATA_DIR = train_data_dir  # 훈련용으로 임시 변경

        # 메인 훈련 함수 호출
        from car_classification.train import main as train_main
        train_main()

        # 원래 설정 복원
        config.DATA_DIR = original_data_dir

        print(f"✓ 폴드 {fold_idx} 훈련이 성공적으로 완료되었습니다.")
        return True

    except Exception as e:
        print(f"✗ 폴드 {fold_idx} 훈련 실패: {e}")
        return False


def run_all_folds():
    """모든 폴드에 대해 훈련 실행"""
    print(f"\n=== 전체 폴드 훈련 ({config.N_FOLDS}개 폴드) ===")

    results = []

    for fold_idx in range(config.N_FOLDS):
        print(f"\n{'=' * 50}")
        print(f"폴드 {fold_idx + 1}/{config.N_FOLDS} 시작")
        print(f"{'=' * 50}")

        success = run_training(fold_idx)
        results.append((fold_idx, success))

        if success:
            print(f"✓ 폴드 {fold_idx} 완료")
        else:
            print(f"✗ 폴드 {fold_idx} 실패")

    # 결과 요약
    print(f"\n{'=' * 50}")
    print("전체 훈련 결과 요약")
    print(f"{'=' * 50}")

    successful_folds = [fold for fold, success in results if success]
    failed_folds = [fold for fold, success in results if not success]

    print(f"성공한 폴드: {successful_folds}")
    if failed_folds:
        print(f"실패한 폴드: {failed_folds}")

    print(f"성공률: {len(successful_folds)}/{config.N_FOLDS} ({len(successful_folds) / config.N_FOLDS * 100:.1f}%)")


def main():
    seed_everything(42)
    parser = argparse.ArgumentParser(description='차량 분류 프로젝트 메인 파이프라인')
    parser.add_argument('--step', choices=['setup', 'splits', 'augment', 'train', 'all'],
                        default='augment', help='실행할 단계 선택')
    parser.add_argument('--fold', type=int, help='특정 폴드만 처리 (0-4)')
    parser.add_argument('--augment-only', action='store_true',
                        help='증강만 실행 (훈련 제외)')

    args = parser.parse_args()

    print("차량 분류 프로젝트 파이프라인")
    print("=" * 60)

    if args.step in ['setup', 'all']:
        if not setup_project():
            return

    if args.step in ['splits', 'all']:
        if not create_splits():
            return

    if args.step in ['augment', 'all'] or args.augment_only:
        if not run_augmentation(fold=args.fold):
            return

        if args.augment_only:
            print("\n✓ 데이터 증강만 완료되었습니다.")
            return

    if args.step in ['train', 'all']:
        if args.fold is not None:
            # 특정 폴드만 훈련
            run_training(args.fold)
        else:
            # 모든 폴드 훈련
            run_all_folds()

    print("\n파이프라인 실행 완료!")


if __name__ == "__main__":
    main()