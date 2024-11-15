[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/znc2XbtA)
# ML Regression 부동산 실거래가 예측
## Team ML 4 < Walk into AI >

| ![박정준](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이다언](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김동완](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김묘정](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이현지](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박정준](https://github.com/UpstageAILab)             |            [이다언](https://github.com/danielinjesus/AI_Portfolio/tree/main/AI_Projects/House_Price_Prediction)             |            [김묘정](https://github.com/UpstageAILab)             |            [이현지](https://github.com/UpstageAILab)             |            [김동완A](https://github.com/UpstageAILab)             |
|                            팀장, Feature engineering, 
Feature Selection 
                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |
## 0. Overview

### Environment

- **IDE 및 코드 편집기**
    - Visual Studio Code (VS Code)
    - Cursor IDE
- **개발 환경**
    - Linux Docker 서버 및 로컬 환경에서 Python, Jupyter Notebook 사용
    - Anaconda로 패키지 및 환경 관리
- **버전 관리 및 협업 도구**
    - Git을 사용하여 코드 및 파일 버전 관리
    - GitHub를 통해 팀원들과 협업

### Requirements

- **프로그래밍 언어**
    - Python 3.x
- **주요 라이브러리 및 패키지**
    - Pandas
    - NumPy
    - Scikit-learn
    - XGBoost
    - WandB (Weights & Biases)
    - Sweetviz
    - Pygalker
    - Dataprep
    - Matplotlib
    - Seaborn
- **기타**
    - Jupyter Notebook
    - Docker (서버 환경 구축용)

---

## 1. Competition Info

### Overview

- **경진대회 주제:** 부동산 가격 예측 모델 개발
- **목표:** 주어진 데이터를 활용하여 부동산 가격을 정확하게 예측하는 모델을 구축하고 성능을 향상시키는 것
- **추가 세부 목표: **
-   프로젝트 수행 역량 기르기

### Timeline

- **프로젝트 기간:** 총 2주 (수강 1주 포함)
    - **1주차:** 팀원 역량 강화 및 베이스라인 코드 실행
    - **2주차:** 모델 개선 및 성능 향상 작업

---

## 2. Components

### Directory

```kotlin
kotlin
코드 복사
├── code
│   ├── jupyter_notebooks
│   │   ├── data_analysis.ipynb
│   │   ├── feature_engineering.ipynb
│   │   └── model_training.ipynb
│   ├── train.py
│   └── utils.py
├── docs
│   ├── pdf
│   │   └── Presentation.pdf
│   └── reports
│       └── EDA_report.pdf
├── data
│   ├── raw
│   │   ├── train.csv
│   │   └── test.csv
│   ├── processed
│   │   └── processed_data.csv
│   └── external
│       ├── bus_station_data.csv
│       └── subway_station_data.csv
└── README.md

```

---

## 3. Data Description

### Dataset Overview

- **메인 데이터셋:** 부동산 거래 데이터
    - **주요 변수:** 전용면적, 위치 정보, 거래 가격 등
- **추가 데이터:**
    - **버스 및 지하철 위치 데이터:** 역세권 여부 및 최단 거리 계산에 활용
    - **강남 대장 아파트 정보:** 지역적 특성 반영을 위해 추가

### EDA (Exploratory Data Analysis)

- **프로세스:**
    - **Pandas**를 사용하여 기본 통계량 분석 및 데이터 구조 파악
    - **Sweetviz**, **Pygalker**, **Dataprep** 등의 라이브러리로 데이터 시각화
    - 변수 간 상관관계 분석 및 분포 확인
- **결론:**
    - **전용면적의 중요성:** 반복적으로 중요한 변수로 선정됨
        - 분포가 우측으로 치우쳐 있어 상한을 완화하여 성능 개선
    - **이상치 처리의 영향:** IQR을 활용한 이상치 제거 시 성능 향상 미미
        - 이상치를 제거하지 않고 모델에 반영하였을 때 성능이 크게 향상됨
    - **추가 피처의 영향:** 외부 데이터를 활용한 피처 추가는 성능 개선에 큰 영향이 없었음

### Data Processing

- **데이터 정제:**
    - 결측치 처리 및 데이터 일관성 확보
- **피처 엔지니어링:**
    - **새로운 피처 생성:** 역세권 여부, 최단 거리 등
    - **추가 데이터 통합:** 강남 대장 아파트 정보 등
- **스케일링 및 변환:**
    - **Robust Scaling**, **Power Transform** 적용 시도
        - 성능 개선 효과는 미미하여 다른 방법 모색

---

## 4. Modeling

### Model Description

- **사용 모델:** XGBoost
- **선택 이유:**
    - **우수한 예측 성능:** 순차적 학습 방식을 통해 오차를 보완하며 Random Forest보다 성능 우수
    - **학습 속도 최적화:** 병렬 처리 및 캐시 최적화를 통한 빠른 학습 가능
    - **과적합 방지 기능:** L1, L2 정규화 및 다양한 규제 파라미터 제공

### Modeling Process

- **하이퍼파라미터 튜닝:**
    - **WandB**를 활용하여 학습률, 최대 깊이 등 최적의 하이퍼파라미터 탐색
    - 모델 자체의 변경만으로는 성능 향상에 한계가 있었음
- **피처 선택:**
    - **Feature Importance**를 고려하여 중요한 피처 선정
    - **수동 변수 선택:**
        - 팀원들의 직관과 경험을 바탕으로 변수 추가 및 제거하며 성능 변화 관찰
    - **Feature Selection 기법 적용:**
        - **Filter Method:** Variance Threshold, 상관관계 분석, VIF 활용
        - **Wrapper Method:** Sequential Feature Selection(SFS), Recursive Feature Elimination(RFE)
    - **결과:** 중요한 피처만 남기고 불필요한 피처를 제거하여 성능 향상
- **앙상블 모델 시도:**
    - 여러 모델의 예측 결과를 결합하여 성능 개선 시도
    - 기대한 만큼의 성능 향상은 이루어지지 않음

---

## 5. Result

### Leader Board

- **베이스라인 점수:** 46,433
- **최종 점수:** 베이스라인 대비 약 2배 향상
- **순위:** [순위 정보가 있을 경우 기재]

*(리더보드 캡처 이미지가 있을 경우 첨부)*

### Presentation

- **발표 자료:** Presentation.pdf

### Etc

### Meeting Log

- **회의록:** [Notion 링크 또는 Google Docs 링크](https://www.notion.so/wandb-16d1fc4f91164c5c92da1a11766fcfbd?pvs=21)

### Reference

- **모델 관련 문서:**
    - [XGBoost 공식 문서](https://xgboost.readthedocs.io/)
    - Scikit-learn Feature Selection
- **관련 블로그 및 자료:**
    - 부동산 가격 예측 프로젝트 관련 블로그
    - 데이터 분석 및 전처리 기법 자료
