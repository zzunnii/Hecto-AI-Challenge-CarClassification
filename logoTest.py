import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 로드
csv_path = r"C:\Users\tjdwn\GitHub\Hecto-AI-Challenge-CarClassification\car_classification\submission_hierarchical_fold0_class_acc_model_epoch32_classacc0.9647_grp0.9888.csv"
df = pd.read_csv(csv_path)

# ID 분리
ids = df['ID']
probs = df.drop(columns=['ID']).values
class_names = df.columns[1:]

# 최대 확률값 (confidence)
max_probs = np.max(probs, axis=1)
pred_classes = np.argmax(probs, axis=1)

# 분석 출력
print(f"총 샘플 수: {len(df)}")
print(f"평균 최대 확률: {np.mean(max_probs):.4f}")
print(f"최소 최대 확률: {np.min(max_probs):.4f}")
print(f"95% 이상 확신 비율: {(max_probs >= 0.95).mean():.2%}")
print(f"90% 이상 확신 비율: {(max_probs >= 0.90).mean():.2%}")
print(f"70% 이하 확률로 예측한 비율: {(max_probs <= 0.70).mean():.2%}")

# 히스토그램 시각화
plt.figure(figsize=(10, 5))
plt.hist(max_probs, bins=50, color='skyblue', edgecolor='black')
plt.title("Max Softmax Confidence Histogram")
plt.xlabel("Max Probability")
plt.ylabel("Number of Samples")
plt.grid(True)
plt.tight_layout()
plt.show()

# Top-k 확률 분포 (예: top-1 ~ top-3)
sorted_probs = np.sort(probs, axis=1)[:, ::-1]
for k in range(1, 4):
    mean_topk = np.mean(sorted_probs[:, k - 1])
    print(f"평균 Top-{k} 확률: {mean_topk:.4f}")
