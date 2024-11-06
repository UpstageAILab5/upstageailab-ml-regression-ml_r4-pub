
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
import matplotlib.font_manager as fm

fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf', # ttf 파일이 저장되어 있는 경로
    name='NanumBarunGothic')                        # 이 폰트의 원하는 이름 설정
fm.fontManager.ttflist.insert(0, fe)              # Matplotlib에 폰트 추가
plt.rcParams.update({'font.size': 10, 'font.family': 'NanumBarunGothic'}) # 폰트 설정
plt.rc('font', family='NanumBarunGothic')

import warnings;warnings.filterwarnings('ignore')

class EDA:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = config.get('logger')
    def analyze_missing_vals(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_stats = pd.DataFrame({
            'missing_count': df.isnull().sum(),
            'missing_ratio': df.isnull().sum() / len(df) * 100
        })
        return missing_stats.sort_values('missing_ratio', ascending=False)
    def analyze_dist(self, df: pd.DataFrame, numerical_cols: List[str]) -> None:
        """수치형 변수의 분포를 시각화하고 기초 통계량을 출력"""
        if not numerical_cols:
            self.logger.warning("수치형 변수가 없습니다!")
            return
        
        # 데이터 전처리: 문자열을 숫자로 변환
        df_numeric = df[numerical_cols].copy()
        for col in numerical_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df_numeric[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 기초 통계량 출력
        self.logger.info("\n=== 기초 통계량 ===")
        self.logger.info(f"\n{df_numeric.describe()}")
        
        # 그래프 생성
        n_cols = len(numerical_cols)
        fig, axes = plt.subplots(n_cols, 2, figsize=(15, 5*n_cols))
        
        # 변수가 1개인 경우 axes 처리
        if n_cols == 1:
            axes = axes.reshape(1, 2)
        
        for idx, col in enumerate(numerical_cols):
            data = df_numeric[col].dropna()  # 결측치 제외
            
            # 히스토그램 + KDE
            sns.histplot(data=data, kde=True, ax=axes[idx, 0])
            axes[idx, 0].set_title(f'{col} - 히스토그램 (결측치 제외)')
            
            # 박스플롯
            sns.boxplot(y=data, ax=axes[idx, 1])
            axes[idx, 1].set_title(f'{col} - 박스플롯 (결측치 제외)')
            
            # 이상치 및 결측치 정보 출력
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)].shape[0]
            missing_count = df_numeric[col].isnull().sum()
            
            self.logger.info(f"\n{col}:")
            self.logger.info(f"- 결측치: {missing_count}개 ({missing_count/len(df)*100:.2f}%)")
            self.logger.info(f"- 이상치: {outlier_count}개 ({outlier_count/len(data)*100:.2f}%)")
        
        plt.tight_layout()
        plt.show()
        
        # 상관관계 분석 (변수가 2개 이상인 경우)
        if n_cols > 1:
            plt.figure(figsize=(10, 8))
            correlation = df_numeric.corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
            plt.title('변수 간 상관관계')
            plt.tight_layout()
            plt.show()

    def analyze_corr(self, df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        plt.figure(figsize=(10,8))
        sns.heatmap(df[numerical_cols].corr(),
                    annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.show()


    def plot_histograms(self):
        """수치형 변수에 대한 히스토그램을 생성합니다."""
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numeric_cols].hist(figsize=(15, 10), bins=20)
        plt.tight_layout()
        plt.show()


    # def plot_pairplot(self, hue=None):
    #     """피처 간의 pairplot을 생성합니다.

    #     Parameters:
    #     - hue: 범주형 변수 이름 (옵션)
    #     """
    #     sns.pairplot(self.data, hue=hue)
    #     plt.show()

    # def plot_missing_values(self):
    #     """결측치의 분포를 시각화합니다."""
    #     missing = self.data.isnull().sum()
    #     missing = missing[missing > 0]
    #     missing.sort_values(inplace=True)
    #     missing.plot.bar()
    #     plt.ylabel('Missing Value Count')
    #     plt.show()
