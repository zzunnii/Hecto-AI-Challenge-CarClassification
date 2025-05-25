import pandas as pd
import numpy as np

# 기존 파일 로드
df = pd.read_csv(r"C:\Users\tjdwn\GitHub\Hecto-AI-Challenge-CarClassification\car_classification\submission_hierarchical_fold1.csv")
id_col = df['ID']
probs = df.drop(columns=['ID']).values
columns = df.columns[1:]


# 사용자 정의 확률 재조정 함수
def confidence_boost_with_balance(probs, threshold=0.5, boost_range=(0.3, 0.4)):
    boosted_probs = []
    for row in probs:
        top_idx = np.argmax(row)
        top_prob = row[top_idx]

        if top_prob > threshold:
            # 0.2~0.3 랜덤 증가 (혹은 고정값 사용 가능)
            boost_amount = np.random.uniform(boost_range[0], boost_range[1])
            new_top_prob = min(top_prob + boost_amount, 1.0)
            diff = new_top_prob - top_prob

            # 나머지 확률에서 diff만큼 감산
            others = np.delete(row, top_idx)
            others = others * ((1 - new_top_prob) / others.sum())

            new_row = np.zeros_like(row)
            new_row[top_idx] = new_top_prob
            new_row[np.arange(len(row)) != top_idx] = others
            boosted_probs.append(new_row)
        else:
            # 변화 없이 그대로 유지
            boosted_probs.append(row)

    return np.array(boosted_probs)


# 확률 조정
adjusted_probs = confidence_boost_with_balance(probs, threshold=0.4, boost_range=(0.3, 0.4))

# 결과 DataFrame 생성
adjusted_df = pd.DataFrame(adjusted_probs, columns=columns)
adjusted_df.insert(0, 'ID', id_col)


# 저장
adjusted_df.to_csv("submission_hierarchical_fold1_boosted.csv", index=False)
