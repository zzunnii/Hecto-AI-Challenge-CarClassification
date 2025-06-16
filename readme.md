# Hecto AI 자동차 분류 챌린지

이 저장소는 Hecto AI Challenge 참가를 위해 작성된 자동차 이미지 분류 코드입니다. ResNet과 ResNeXt 기반의 계층적 분류 모델을 사용하여 차량 브랜드와 세부 모델을 예측합니다.
데이터의 대한 저작권은 Dacon에 있습니다.
## 폴더 구조
- `resnet_car_classification/` : ResNet 기반 모델 구현
- `resnext_car_classification/` : ResNeXt 기반 모델 구현
- `car_class_mapping.csv` : 원본 클래스와 브랜드 그룹 매핑 정보
- `fold_analysis.py` : 교차 검증 결과 분석 스크립트
- `strategy.py` : 앙상블 전략 예시

## 환경 설정
데이터 위치와 결과 저장 폴더 등은 각 모듈의 `config/config.py` 파일에서 수정할 수 있습니다. 실행 전에 `BASE_DIR` 경로를 본인의 환경에 맞게 설정하세요.

## 설치
Python 3.10 이상과 PyTorch CUDA 환경이 필요합니다. 의존성은 다음과 같이 설치합니다.

```bash
pip install -r resnext_car_classification/requirements.txt
# 또는
pip install -r resnet_car_classification/requirements.txt
```

## 학습 방법
다음 명령으로 5폴드 교차 검증 학습을 수행할 수 있습니다.

```bash
python resnext_car_classification/train.py --all-folds
```

학습된 모델과 로그는 `outputs/` 폴더에 저장됩니다.

## 추론 예시
학습된 모델을 사용해 추론을 수행하려면 다음 스크립트를 실행합니다.

```bash
python resnext_car_classification/pipeline.py
```

예측 결과와 평가 지표는 `confidence_calibration_results.csv` 및 `confidence_calibration_results.json` 파일로 출력됩니다.

## 라이선스
본 프로젝트는 연구 목적으로만 사용 가능합니다.
