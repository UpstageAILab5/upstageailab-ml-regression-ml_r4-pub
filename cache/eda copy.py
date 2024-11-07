import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import janitor
import dabl
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings('ignore')

class EDA:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = config.get('logger')

    def automated_eda(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """자동화된 EDA 수행의 메인 함수"""
        # 1. 기본 데이터 분석
        self._analyze_basic_info(df)
        # 2. 데이터 전처리
        df = self._preprocess_data(df, target_col)
        # 3. 변수 타입 분류 및 분석
        num_cols, cat_cols = self._group_columns(df)
        # 4. 수치형 변수 분석
        if num_cols:
            self._analyze_numerical_features(df, num_cols, target_col)
        # 5. 범주형 변수 분석
        if cat_cols:
            self._analyze_categorical_features(df, cat_cols, target_col)
        
        return df

    def _analyze_basic_info(self, df: pd.DataFrame) -> None:
        """기본 데이터 정보 분석"""
        self.logger.info("\n=== 기본 데이터 정보 ===")
        self.logger.info(f"\nDataFrame Info:\n{df.info()}")
        self.logger.info(f"\nBasic Statistics:\n{df.describe()}")
        # 결측치 분석
        missing_stats = self._analyze_missing_vals(df)
        self.logger.info(f"\nMissing Values:\n{missing_stats}")

    def _analyze_missing_vals(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 분석"""
        missing_stats = pd.DataFrame({
            'missing_count': df.isnull().sum(),
            'missing_ratio': df.isnull().sum() / len(df) * 100
        })
        return missing_stats.sort_values('missing_ratio', ascending=False)

    def _preprocess_data(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """데이터 전처리"""
        # 컬럼명 정리
        df = df.clean_names()
        
        # 결측치 처리
        if target_col:
            df = df.dropna(subset=[target_col])
        
        # 중복 제거
        initial_len = len(df)
        df = df.drop_duplicates()
        if initial_len != len(df):
            self.logger.info(f"\nRemoved {initial_len - len(df)} duplicate rows")
        
        # 데이터 타입 변환
        df = self._convert_to_numeric(df)
        
        return df

    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """가능한 컬럼을 숫자형으로 변환"""
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except ValueError:
                continue
        return df

    def _group_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """수치형과 범주형 컬럼 분류"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        self.logger.info(f"\nNumerical Columns ({len(numerical_cols)}): {numerical_cols}")
        self.logger.info(f"Categorical Columns ({len(categorical_cols)}): {categorical_cols}")
        
        return numerical_cols, categorical_cols

    def _analyze_numerical_features(self, df: pd.DataFrame, num_cols: List[str], 
                                  target_col: str = None) -> None:
        """수치형 변수 분석"""
        self.logger.info("\n=== 수치형 변수 분석 ===")
        
        # 분포 분석
        for col in num_cols:
            self._analyze_single_variable(df[col], col)
        
        # 상관관계 분석
        if len(num_cols) > 1:
            self._analyze_correlation(df[num_cols])
        
        # 목표변수 관련 분석
        if target_col and target_col in num_cols:
            self._analyze_target_relationships(df, num_cols, target_col)

    def _analyze_single_variable(self, data: pd.Series, col_name: str) -> None:
        """단일 변수 분석"""
        # 기술 통계량 계산
        stats = data.describe()
        self.logger.info(f"\n{col_name} Statistics:\n{stats}")
        
        # 이상치 분석
        Q1, Q3 = data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
        self.logger.info(f"Outliers count: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")
        
        # 분포 시각화
        self._plot_distribution(data, col_name)

    def _plot_distribution(self, data: pd.Series, title: str) -> None:
        """분포 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 히스토그램 + KDE
        sns.histplot(data=data, kde=True, ax=ax1)
        ax1.set_title(f'{title} - Distribution')
        
        # 박스플롯
        sns.boxplot(y=data, ax=ax2)
        ax2.set_title(f'{title} - Box Plot')
        
        plt.tight_layout()
        plt.show()

    def _analyze_correlation(self, df: pd.DataFrame) -> None:
        """상관관계 분석"""
        plt.figure(figsize=(10, 8))
        correlation = df.corr()
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def _analyze_categorical_features(self, df: pd.DataFrame, cat_cols: List[str], 
                                    target_col: str = None) -> None:
        """범주형 변수 분석"""
        self.logger.info("\n=== 범주형 변수 분석 ===")
        
        for col in cat_cols:
            if col != target_col:
                value_counts = df[col].value_counts()
                self.logger.info(f"\n{col} Value Counts:\n{value_counts}")
                
                # 범주형 변수 시각화
                plt.figure(figsize=(10, 5))
                sns.countplot(data=df, x=col)
                plt.title(f'Distribution of {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

    def _analyze_target_relationships(self, df: pd.DataFrame, num_cols: List[str], 
                                    target_col: str) -> None:
        """목표 변수와의 관계 분석"""
        self.logger.info("\n=== 목표 변수 관계 분석 ===")
        
        # 상관관계
        correlations = df[num_cols].corr()[target_col].sort_values(ascending=False)
        self.logger.info(f"\nCorrelations with {target_col}:\n{correlations}")
        
        # Dabl 시각화
        try:
            dabl.plot(df, target_col=target_col)
            plt.show()
        except Exception as e:
            self.logger.warning(f"Dabl plotting failed: {str(e)}")
    def _transform_data(delf, df, numerical_cols, categorical_cols):
        """
        Preprocess the data by encoding categorical columns and scaling numerical columns.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame.
        numerical_cols (list): List of numerical column names.
        categorical_cols (list): List of categorical column names.
        
        Returns:
        pd.DataFrame: Preprocessed DataFrame.
        """
        # One-hot encoding for categorical data
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        # Standard scaling for numerical data
        scaler = StandardScaler()

        # ColumnTransformer to apply transformations
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', scaler, numerical_cols),
                ('cat', ohe, categorical_cols)
            ],
            remainder='passthrough'  # Unaffected columns remain unchanged
        )

        # Fit and transform the data
        processed_data = preprocessor.fit_transform(df)
        
        # Reconstruct DataFrame with transformed data
        # Creating column names from OneHotEncoder categories and numerical columns
        ohe_categories = ohe.categories_ if categorical_cols else []
        ohe_column_names = []
        for idx, categories in enumerate(ohe_categories):
            ohe_column_names.extend([f"{categorical_cols[idx]}_{cat}" for cat in categories])
        column_names = numerical_cols + ohe_column_names
        df_processed = pd.DataFrame(processed_data, columns=column_names)
        
        print("\nData after processing (encoded and scaled):")
        print(df_processed.head())
        return df_processed

def main():
    eda = EDA()
    df = pd.read_csv('data/train.csv')
    processed_df = eda.automated_eda(
        df=df,
        target_col='target_column'  # 선택사항
    )
if __name__ == "__main__":
    main()
