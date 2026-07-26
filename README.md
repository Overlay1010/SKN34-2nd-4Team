# 신용카드 고객 이탈 예측 프로젝트

Kaggle의 `Credit Card Customers` 데이터를 활용해 고객 이탈 위험을 예측하고,
거래 활동성과 고객 행동 유형을 함께 분석하는 머신러닝 프로젝트입니다.

## 분석 목표

1. **분류**: 고객별 이탈 확률을 예측합니다.
2. **회귀**: 고객 프로필 대비 기대 거래건수를 추정해 활동성 격차를 확인합니다.
3. **군집**: 행동 특성이 유사한 고객을 묶어 군집별 유지 전략을 설계합니다.
4. **의사결정 지원**: 이탈 위험도, 활동성, 고객 유형을 결합해 관리 우선순위를 제안합니다.

## 저장소 구조

```text
SKN34-2nd-4Team/
├── README.md
├── requirements.txt
├── requirements-original.txt
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── BankChurners.csv
│   └── processed/
│       └── bankchurners_clean.csv
├── notebooks/
│   ├── 00_project_roadmap.ipynb
│   ├── 01_data_load_clean.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_classification.ipynb
│   ├── 04_regression.ipynb
│   └── 05_clustering.ipynb
└── outputs/
    └── README.md
```

## 실행 순서

노트북은 `notebooks/` 디렉터리를 작업 디렉터리로 사용합니다.

```bash
python -m venv project_venv
source project_venv/bin/activate
pip install -r requirements.txt
cd notebooks
jupyter lab
```

아래 순서로 실행합니다.

1. `01_data_load_clean.ipynb`
2. `02_eda.ipynb`
3. `03_classification.ipynb`
4. `04_regression.ipynb`
5. `05_clustering.ipynb`

`00_project_roadmap.ipynb`는 역할 분담과 실행 계획을 정리한 문서입니다.

## 데이터 계보

- 원본: `data/raw/BankChurners.csv` — 10,127행, 23열
- 정제본: `data/processed/bankchurners_clean.csv` — 10,127행, 20열
- 정제 과정:
  - 식별자 `CLIENTNUM` 제거
  - 기존 모델 출력인 `Naive_Bayes_Classifier_..._1`, `_2` 제거
  - `Attrition_Flag`를 `Target`으로 변환
  - `Existing Customer=0`, `Attrited Customer=1`
  - `Unknown` 범주는 삭제·대체하지 않고 유지

자세한 데이터 설명과 무결성 정보는 `data/README.md`를 확인합니다.

## 평가 원칙

이탈 고객은 전체의 약 16.1%이므로 정확도만으로 모델을 평가하지 않습니다.

- 분류: Recall, Precision, F1, ROC-AUC, Lift/Gain
- 회귀: MAE, RMSE, R²
- 군집: Silhouette Score와 군집별 비즈니스 해석

## 주의사항

- `CLIENTNUM`과 두 개의 `Naive_Bayes_Classifier` 열은 모델 입력으로 사용하지 않습니다.
- 이 데이터는 시간 순서가 없는 공개·가공 데이터이므로 실제 운영 성능으로 직접 일반화하지 않습니다.
- 회귀 결과는 미래 LTV가 아니라 현재 데이터에 기반한 거래 활동성 분석으로 해석합니다.

## 데이터 출처

- https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers
