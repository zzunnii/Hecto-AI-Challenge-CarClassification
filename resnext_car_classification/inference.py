import json
import pandas as pd
import numpy as np


def debug_class_mapping():
    """클래스 매핑 문제 디버깅"""

    # 1. Label mappings 로드
    label_mappings_path = r"/resnext_car_classification\outputs\hierarchical_fold_1\label_mappings.json"
    with open(label_mappings_path, 'r', encoding='utf-8') as f:
        label_mappings = json.load(f)

    id_to_class = {int(k): v for k, v in label_mappings['id_to_class'].items()}
    trained_classes = [id_to_class[i] for i in range(len(id_to_class))]

    # 2. Sample submission 로드
    sample_submission = pd.read_csv(r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open\sample_submission.csv")
    submission_classes = [col for col in sample_submission.columns if col != 'ID']

    print("=== 클래스 매핑 분석 ===")
    print(f"학습 클래스 수: {len(trained_classes)}")
    print(f"제출 클래스 수: {len(submission_classes)}")

    # 3. 첫 10개 클래스 비교
    print("\n첫 10개 클래스 비교:")
    print("학습 클래스:")
    for i in range(10):
        print(f"  {i}: {trained_classes[i]}")

    print("\n제출 클래스:")
    for i in range(10):
        print(f"  {i}: {submission_classes[i]}")

    # 4. 클래스 일치 여부 확인
    print("\n=== 클래스 일치 분석 ===")
    trained_set = set(trained_classes)
    submission_set = set(submission_classes)

    # 학습에는 있지만 제출에는 없는 클래스
    only_in_trained = trained_set - submission_set
    if only_in_trained:
        print(f"\n학습에만 있는 클래스 ({len(only_in_trained)}개):")
        for cls in list(only_in_trained)[:5]:
            print(f"  - {cls}")

    # 제출에는 있지만 학습에는 없는 클래스
    only_in_submission = submission_set - trained_set
    if only_in_submission:
        print(f"\n제출에만 있는 클래스 ({len(only_in_submission)}개):")
        for cls in list(only_in_submission)[:5]:
            print(f"  - {cls}")

    # 5. 순서 확인
    print("\n=== 순서 일치 확인 ===")
    order_match = True
    mismatch_count = 0
    for i in range(min(len(trained_classes), len(submission_classes))):
        if trained_classes[i] != submission_classes[i]:
            if mismatch_count < 5:  # 처음 5개만 출력
                print(f"인덱스 {i}: 학습[{trained_classes[i]}] != 제출[{submission_classes[i]}]")
            mismatch_count += 1
            order_match = False

    if order_match:
        print("✓ 클래스 순서가 완벽히 일치합니다!")
    else:
        print(f"✗ 총 {mismatch_count}개 위치에서 클래스 순서가 다릅니다!")

    # 6. 제출 파일 분석
    print("\n=== 제출 파일 분석 ===")
    submission_path = r"/resnext_car_classification\submission_hierarchical_fold2.csv"
    try:
        submission_df = pd.read_csv(submission_path)

        # 행 합 확인
        row_sums = submission_df.iloc[:, 1:].sum(axis=1)
        print(f"행 합 평균: {row_sums.mean():.4f} (1.0이어야 함)")
        print(f"행 합 표준편차: {row_sums.std():.4f} (0에 가까워야 함)")

        # 최대 확률 분포
        max_probs = submission_df.iloc[:, 1:].max(axis=1)
        print(f"\n최대 확률 통계:")
        print(f"  평균: {max_probs.mean():.4f}")
        print(f"  최소: {max_probs.min():.4f}")
        print(f"  최대: {max_probs.max():.4f}")
        print(f"  표준편차: {max_probs.std():.4f}")

        # 예측 분포
        print(f"\n예측 클래스 분포:")
        pred_classes = submission_df.iloc[:, 1:].idxmax(axis=1)
        class_counts = pred_classes.value_counts()
        print(f"  유니크 클래스 수: {len(class_counts)}")
        print(f"  상위 5개 클래스:")
        for cls, count in class_counts.head().items():
            print(f"    {cls}: {count}회")

    except FileNotFoundError:
        print(f"제출 파일을 찾을 수 없습니다: {submission_path}")

    return trained_classes, submission_classes


def create_mapping_fix():
    """올바른 클래스 매핑 생성"""

    # Sample submission의 클래스 순서 로드
    sample_submission = pd.read_csv(r"C:\Users\tjdwn\OneDrive\Desktop\hectoData\open\sample_submission.csv")
    submission_classes = [col for col in sample_submission.columns if col != 'ID']

    # 올바른 매핑 생성
    correct_mapping = {i: cls for i, cls in enumerate(submission_classes)}

    # 저장
    with open('correct_class_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(correct_mapping, f, ensure_ascii=False, indent=2)

    print(f"올바른 매핑을 correct_class_mapping.json에 저장했습니다.")
    print(f"총 {len(correct_mapping)}개 클래스")

    return correct_mapping


if __name__ == "__main__":
    print("클래스 매핑 디버깅 시작...\n")

    # 1. 현재 매핑 분석
    trained_classes, submission_classes = debug_class_mapping()

    # 2. 올바른 매핑 생성 제안
    print("\n" + "=" * 50)
    print("올바른 매핑을 생성하시겠습니까? (y/n)")
    # create_mapping_fix()  # 필요시 주석 해제