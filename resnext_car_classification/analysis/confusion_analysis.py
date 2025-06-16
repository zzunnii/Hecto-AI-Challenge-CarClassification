import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import defaultdict, Counter
from pathlib import Path
import pickle


class ConfusionAnalyzer:
    """혼동 패턴 분석기"""

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.fold_results = {}
        self.combined_results = None

    def load_fold_results(self):
        """각 폴드의 결과 로드"""
        print("Loading fold results...")

        for fold in range(5):
            fold_file = os.path.join(self.results_dir, f'fold_{fold}_results.pkl')
            if os.path.exists(fold_file):
                with open(fold_file, 'rb') as f:
                    self.fold_results[fold] = pickle.load(f)
                print(f"Loaded fold {fold}: {len(self.fold_results[fold]['y_true'])} samples")
            else:
                print(f"Warning: Fold {fold} results not found")

        # 모든 폴드 결과 합치기
        self._combine_fold_results()

    def _combine_fold_results(self):
        """모든 폴드 결과를 하나로 합치기"""
        all_y_true = []
        all_y_pred = []
        all_class_names = None

        for fold, results in self.fold_results.items():
            all_y_true.extend(results['y_true'])
            all_y_pred.extend(results['y_pred'])

            if all_class_names is None:
                all_class_names = results['class_names']

        self.combined_results = {
            'y_true': all_y_true,
            'y_pred': all_y_pred,
            'class_names': all_class_names
        }

        print(f"Combined results: {len(all_y_true)} total predictions")

    def analyze_confusion_patterns(self, top_k=20):
        """주요 혼동 패턴 분석"""
        if self.combined_results is None:
            print("No combined results available")
            return None

        y_true = self.combined_results['y_true']
        y_pred = self.combined_results['y_pred']
        class_names = self.combined_results['class_names']

        # 혼동 행렬 생성
        cm = confusion_matrix(y_true, y_pred)

        # 대각선 제거 (정답은 제외)
        cm_errors = cm.copy()
        np.fill_diagonal(cm_errors, 0)

        # 가장 많이 혼동되는 쌍 찾기
        confusion_pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm_errors[i, j] > 0:
                    confusion_pairs.append({
                        'true_class': class_names[i],
                        'pred_class': class_names[j],
                        'count': cm_errors[i, j],
                        'true_total': cm[i].sum(),
                        'error_rate': cm_errors[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
                    })

        # 혼동 빈도로 정렬
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

        print(f"=== Top {top_k} 혼동 패턴 ===")
        for idx, pair in enumerate(confusion_pairs[:top_k]):
            print(f"{idx + 1:2d}. {pair['true_class'][:30]:<30} → {pair['pred_class'][:30]:<30} "
                  f"({pair['count']:3d}회, {pair['error_rate']:.1f}%)")

        return confusion_pairs[:top_k]

    def analyze_brand_confusion(self, confusion_pairs):
        """브랜드별 혼동 패턴 분석"""
        print("\n=== 브랜드별 혼동 분석 ===")

        def extract_brand(class_name):
            """클래스명에서 브랜드 추출"""
            # 한글 브랜드명 처리
            korean_brands = ['현대', '기아', '제네시스', '쌍용', '르노', '쉐보레']
            for brand in korean_brands:
                if brand in class_name:
                    return brand

            # 영문 브랜드명 처리 (첫 번째 단어 또는 언더스코어 전)
            if '_' in class_name:
                return class_name.split('_')[0]
            else:
                return class_name.split()[0] if ' ' in class_name else class_name[:10]

        brand_errors = defaultdict(int)
        same_brand_errors = defaultdict(int)

        for pair in confusion_pairs:
            true_brand = extract_brand(pair['true_class'])
            pred_brand = extract_brand(pair['pred_class'])

            if true_brand == pred_brand:
                # 같은 브랜드 내 혼동
                same_brand_errors[true_brand] += pair['count']
            else:
                # 다른 브랜드간 혼동
                key = f"{true_brand} → {pred_brand}"
                brand_errors[key] += pair['count']

        print("\n다른 브랜드간 주요 혼동 (Top 10):")
        for brand_pair, count in sorted(brand_errors.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {brand_pair}: {count}회")

        print(f"\n같은 브랜드 내 혼동 (Top 10):")
        for brand, count in sorted(same_brand_errors.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {brand}: {count}회")

        return brand_errors, same_brand_errors

    def create_confusion_heatmap(self, save_path=None, top_classes=50):
        """혼동 행렬 히트맵 생성"""
        if self.combined_results is None:
            return

        y_true = self.combined_results['y_true']
        y_pred = self.combined_results['y_pred']
        class_names = self.combined_results['class_names']

        # 가장 많이 혼동되는 클래스들만 선택
        cm = confusion_matrix(y_true, y_pred)

        # 각 클래스의 총 에러 수 계산
        class_errors = []
        for i in range(len(class_names)):
            total_errors = np.sum(cm[i, :]) - cm[i, i]  # 대각선 제외
            class_errors.append((i, total_errors, class_names[i]))

        # 에러가 많은 순으로 정렬
        class_errors.sort(key=lambda x: x[1], reverse=True)
        top_indices = [x[0] for x in class_errors[:top_classes]]
        top_names = [x[2] for x in class_errors[:top_classes]]

        # 부분 혼동 행렬 생성
        cm_subset = cm[np.ix_(top_indices, top_indices)]

        # 정규화 (행별로)
        cm_normalized = cm_subset.astype('float') / cm_subset.sum(axis=1)[:, np.newaxis]

        # 히트맵 그리기
        plt.figure(figsize=(20, 16))

        # 클래스명 줄이기 (너무 길면)
        short_names = [name[:20] + '...' if len(name) > 20 else name for name in top_names]

        sns.heatmap(cm_normalized,
                    xticklabels=short_names,
                    yticklabels=short_names,
                    cmap='Blues',
                    fmt='.2f',
                    cbar_kws={'label': 'Normalized Confusion Rate'})

        plt.title(f'Confusion Matrix - Top {top_classes} Most Confused Classes')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion heatmap saved to: {save_path}")

        plt.show()

    def generate_report(self, output_dir):
        """종합 분석 보고서 생성"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 혼동 패턴 분석
        confusion_pairs = self.analyze_confusion_patterns(top_k=30)

        # 2. 브랜드 분석
        brand_errors, same_brand_errors = self.analyze_brand_confusion(confusion_pairs)

        # 3. 히트맵 생성
        heatmap_path = os.path.join(output_dir, 'confusion_heatmap.png')
        self.create_confusion_heatmap(save_path=heatmap_path)

        # 4. 결과를 CSV로 저장
        if confusion_pairs:
            df = pd.DataFrame(confusion_pairs)
            csv_path = os.path.join(output_dir, 'confusion_patterns.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Confusion patterns saved to: {csv_path}")

        # 5. 전체 성능 요약
        self._generate_performance_summary(output_dir)

    def _generate_performance_summary(self, output_dir):
        """성능 요약 생성"""
        print("\n=== 5-Fold Cross Validation 성능 요약 ===")

        fold_accuracies = []
        fold_f1_scores = []

        for fold, results in self.fold_results.items():
            if 'accuracy' in results:
                fold_accuracies.append(results['accuracy'])
            if 'f1_score' in results:
                fold_f1_scores.append(results['f1_score'])

        if fold_accuracies:
            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)
            print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

            for fold, acc in enumerate(fold_accuracies):
                print(f"  Fold {fold}: {acc:.4f}")

        if fold_f1_scores:
            mean_f1 = np.mean(fold_f1_scores)
            std_f1 = np.std(fold_f1_scores)
            print(f"F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")

        # 요약을 파일로 저장
        summary = {
            'accuracy_mean': mean_acc if fold_accuracies else None,
            'accuracy_std': std_acc if fold_accuracies else None,
            'f1_mean': mean_f1 if fold_f1_scores else None,
            'f1_std': std_f1 if fold_f1_scores else None,
            'fold_accuracies': fold_accuracies,
            'fold_f1_scores': fold_f1_scores
        }

        summary_path = os.path.join(output_dir, 'performance_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Performance summary saved to: {summary_path}")


def main():
    """메인 실행 함수"""

    # 결과 디렉토리 설정
    RESULTS_DIR = r"C:\Users\tjdwn\GitHub\Hecto-AI-Challenge-CarClassification\brand_classification\outputs\cv_results"
    OUTPUT_DIR = r"C:\Users\tjdwn\GitHub\Hecto-AI-Challenge-CarClassification\brand_classification\analysis\reports"

    # 분석기 초기화
    analyzer = ConfusionAnalyzer(RESULTS_DIR)

    # 결과 로드
    analyzer.load_fold_results()

    if analyzer.combined_results is None:
        print("No valid results found. Make sure to run cross-validation first.")
        return

    # 종합 분석 보고서 생성
    analyzer.generate_report(OUTPUT_DIR)

    print(f"\nAnalysis completed! Check reports in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()