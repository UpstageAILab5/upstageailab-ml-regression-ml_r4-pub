import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

class EDA:
    def __init__(self, data):
        """
        기본적인 EDA를 수행하기 위한 클래스입니다.

        Parameters:
        - data: 판다스 데이터프레임 형태의 입력 데이터
        """
        self.data = data

    def plot_histograms(self):
        """수치형 변수에 대한 히스토그램을 생성합니다."""
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numeric_cols].hist(figsize=(15, 10), bins=20)
        plt.tight_layout()
        plt.show()

    def plot_correlations(self):
        """피처 간 상관 관계 히트맵을 생성합니다."""
        corr = self.data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.show()

    def plot_pairplot(self, hue=None):
        """피처 간의 pairplot을 생성합니다.

        Parameters:
        - hue: 범주형 변수 이름 (옵션)
        """
        sns.pairplot(self.data, hue=hue)
        plt.show()

    def plot_missing_values(self):
        """결측치의 분포를 시각화합니다."""
        missing = self.data.isnull().sum()
        missing = missing[missing > 0]
        missing.sort_values(inplace=True)
        missing.plot.bar()
        plt.ylabel('Missing Value Count')
        plt.show()

class AutoMLPipeline:
    def __init__(self, data, target=None):
        """
        데이터와 타겟 변수를 받아 초기화합니다.

        Parameters:
        - data: 판다스 데이터프레임 형태의 입력 데이터
        - target: 문자열 형태의 타겟 변수 이름 (회귀 또는 분류)
        """
        self.data = data
        self.target = target
        self.features = data.drop(columns=[target]) if target else data
        self.target_series = data[target] if target else None
        self.report = {}

    def analyze_features(self):
        """컬럼의 결측치, 이상치, 기본 통계량, 상관 관계 등을 분석합니다."""
        self.report['missing_values'] = self.features.isnull().sum()
        self.report['descriptive_stats'] = self.features.describe()
        self.report['correlations'] = self.features.corr()

        # 이상치 탐지 (IQR 방식)
        Q1 = self.features.quantile(0.25)
        Q3 = self.features.quantile(0.75)
        IQR = Q3 - Q1
        self.report['outliers'] = ((self.features < (Q1 - 1.5 * IQR)) | (self.features > (Q3 + 1.5 * IQR))).sum()

    def recommend_feature_engineering(self):
        """피처 엔지니어링 방향을 추천합니다."""
        recommendations = []

        # 결측치 처리
        missing = self.report['missing_values']
        if missing.any():
            recommendations.append("결측치가 있는 컬럼이 있습니다. 적절한 방법으로 결측치를 처리하세요.")

        # 이상치 처리
        outliers = self.report['outliers']
        if outliers.any():
            recommendations.append("이상치가 있는 컬럼이 있습니다. 이상치를 제거하거나 처리하는 것을 고려하세요.")

        # 상관 관계 분석
        corr_matrix = self.report['correlations']
        high_corr = corr_matrix.abs().unstack().sort_values(kind="quicksort", ascending=False)
        high_corr = high_corr[high_corr < 1]  # 자기 자신과의 상관 관계 제외
        if not high_corr.empty and high_corr.iloc[0] > 0.8:
            recommendations.append("상관 관계가 높은 피처들이 있습니다. 다중공선성을 고려하세요.")

        self.report['recommendations'] = recommendations

    def feature_selection(self):
        """타겟 변수에 따라 적절한 피처 선택 방법을 적용합니다."""
        if self.target_series is None:
            print("타겟 변수가 없습니다. 피처 선택을 수행할 수 없습니다.")
            return

        # 수치형, 범주형 컬럼 분리
        numeric_features = self.features.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.features.select_dtypes(include=['object', 'category']).columns

        # 파이프라인 구성
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

        if self.target_series.dtype == 'object' or self.target_series.nunique() <= 10:
            # 분류 모델
            model = RandomForestClassifier()
            selector = SelectKBest(score_func=f_classif, k='all')
        else:
            # 회귀 모델
            model = RandomForestRegressor()
            selector = SelectKBest(score_func=f_regression, k='all')

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', selector),
            ('model', model)
        ])

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target_series, test_size=0.2, random_state=42)

        # 모델 학습
        pipeline.fit(X_train, y_train)

        # 중요한 피처 추출
        feature_scores = pipeline.named_steps['feature_selection'].scores_
        onehot_features = []
        if 'onehot' in pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps:
            onehot_features = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        feature_names = numeric_features.tolist() + onehot_features.tolist()
        feature_importances = pd.Series(feature_scores, index=feature_names).sort_values(ascending=False)
        self.report['feature_importances'] = feature_importances

    def recommend_models(self):
        """타겟 변수에 따라 적절한 모델을 추천합니다."""
        if self.target_series is None:
            print("타겟 변수가 없습니다. 모델 추천을 수행할 수 없습니다.")
            return

        if self.target_series.dtype == 'object' or self.target_series.nunique() <= 10:
            recommendations = ["로지스틱 회귀", "랜덤 포레스트 분류기", "서포트 벡터 머신", "XGBoost 분류기"]
        else:
            recommendations = ["선형 회귀", "랜덤 포레스트 회귀", "그라디언트 부스팅 회귀", "XGBoost 회귀"]

        self.report['model_recommendations'] = recommendations

    def generate_report(self):
        """분석 결과와 추천 사항을 종합하여 리포트를 생성합니다."""
        for key, value in self.report.items():
            print(f"===== {key} =====")
            print(value)
            print("\n")

    def run_all(self):
        """전체 파이프라인을 실행합니다."""
        self.analyze_features()
        self.recommend_feature_engineering()
        if self.target:
            self.feature_selection()
            self.recommend_models()
        self.generate_report()


def main():
    # 데이터 로드
    path_base = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(path_base, 'data', 'test.csv')

    data = pd.read_csv(path_data)

    # EDA 수행
    eda = EDA(data)
    eda.plot_missing_values()
    eda.plot_histograms()
    eda.plot_correlations()
    # 타겟 변수가 있는 경우 pairplot에서 hue 인자로 사용
    eda.plot_pairplot(hue='target')

    # AutoML 파이프라인 실행
    pipeline = AutoMLPipeline(data, target='target')
    pipeline.run_all()

if __name__ == '__main__':
    main()