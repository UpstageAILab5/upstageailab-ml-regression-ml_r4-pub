import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import janitor
import dabl
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.preprocessing import DataPrep
import warnings
import os
warnings.filterwarnings('ignore')
        
class EDA:
    def __init__(self, config: Dict):
        self.config = config
        self.time_delay = config.get('time_delay', 3)
        self.logger_instance = config.get('logger')
        self.logger_instance.setup_logger(log_file='eda')
        self.logger = self.logger_instance.logger
        self.out_path = config.get('out_path')
        self.time_delay = 3

    def automated_eda(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """자동화된 EDA 수행의 메인 함수"""
        before_profile = DataPrep.get_data_profile(df, stage="before")
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
        
        # 특성 통계량 계산 및 저장
        feature_stats = self._get_feature_statistics(df)
        self.logger.info("\n=== Feature Statistics ===")
        self.logger.info(f"\n{feature_stats}")
        
        # 특성 그룹화 및 시각화
        feature_groups = self._group_similar_features(df)
        self._plot_feature_groups(df, feature_groups)
        after_profile = DataPrep.get_data_profile(df, stage="after")
        # 전후 비교 테이블 생성
        comparison = pd.concat([before_profile, after_profile], 
                            keys=['Before', 'After'], 
                            axis=1)
        # 변화량 계산
        changes = pd.DataFrame()
        for col in comparison.index:
            if col in before_profile.index and col in after_profile.index:
                numeric_cols = ['missing_ratio', 'unique_ratio']
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.extend(['mean', 'std', 'outlier_ratio'])
                
                for metric in numeric_cols:
                    if metric in before_profile.loc[col] and metric in after_profile.loc[col]:
                        before_val = before_profile.loc[col, metric]
                        after_val = after_profile.loc[col, metric]
                        changes.loc[col, f'{metric}_change'] = after_val - before_val
        
        # 결과 저장 및 출력
        comparison_path = os.path.join(self.out_path, 'data_prep_comparison_eda_v2.csv')
        changes_path = os.path.join(self.out_path, 'data_prep_changes_eda_v2.csv')
        comparison.to_csv(comparison_path)
        changes.to_csv(changes_path)
        
        # 결과 출력
        self.logger.info('\n=== Data Preparation Comparison ===')
        self.logger.info('\nBefore vs After Statistics:')
        self.logger.info(f'\n{comparison}')
        self.logger.info('\nKey Changes:')
        self.logger.info(f'\n{changes}')
        
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
        
        # 분포 분석 - subplot으로 한 번에 그리기
        n_cols = min(3, len(num_cols))  # 한 행에 최대 3개
        n_rows = (len(num_cols) - 1) // n_cols + 1
        fig = plt.figure(figsize=(6*n_cols, 4*n_rows))
        fig.suptitle('Numerical Features Distribution', fontsize=16)
        
        for idx, col in enumerate(num_cols, 1):
            # 기술 통계량 계산
            stats = df[col].describe()
            self.logger.info(f"\n{col} Statistics:\n{stats}")
            
            # 이상치 분석
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            outlier_ratio = len(outliers)/len(df)*100
            
            # 분포 시각화
            ax = plt.subplot(n_rows, n_cols, idx)
            sns.histplot(data=df, x=col, kde=True, ax=ax)
            ax.set_title(f'{col}\nMean: {df[col].mean():.2f}\nOutliers: {outlier_ratio:.1f}%')
            
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(self.time_delay)  # 5초 동안 그래프 표시
        title='Numerical Features Distribution'
        plt.savefig(os.path.join(self.out_path, title +'.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
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
        #self._plot_distribution(data, col_name)

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
        plt.show(block=False)
        plt.pause(self.time_delay)  # 5초 동안 그래프 표시
        plt.close()

    def _analyze_correlation(self, df: pd.DataFrame) -> None:
        """상관관계 분석"""
        plt.figure(figsize=(10, 8))
        correlation = df.corr()
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show(block=False)
        title='Correlation Matrix'
        plt.savefig(os.path.join(self.out_path, title +'.png'), dpi=300, bbox_inches='tight')
        plt.pause(self.time_delay)  # 5초 동안 그래프 표시
        plt.close()

    def _analyze_categorical_features(self, df: pd.DataFrame, cat_cols: List[str], 
                                    target_col: str = None) -> None:
        """범주형 변수 분석"""
        self.logger.info("\n=== 범주형 변수 분석 ===")
        
        # subplot으로 한 번에 그리기
        n_cols = min(3, len(cat_cols))  # 한 행에 최대 3개
        n_rows = (len(cat_cols) - 1) // n_cols + 1
        fig = plt.figure(figsize=(6*n_cols, 4*n_rows))
        fig.suptitle('Categorical Features Distribution', fontsize=16)
        
        for idx, col in enumerate(cat_cols, 1):
            if col != target_col:
                # 값 분포 계산
                value_counts = df[col].value_counts()
                self.logger.info(f"\n{col} Value Counts:\n{value_counts}")
                
                # 분포 시각화
                ax = plt.subplot(n_rows, n_cols, idx)
                sns.countplot(data=df, x=col, ax=ax)
                ax.set_title(f'{col}\nUnique values: {df[col].nunique()}')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show(block=False)
        title='Categorical Features Distribution'
        plt.savefig(os.path.join(self.out_path, title +'.png'), dpi=300, bbox_inches='tight')
        plt.pause(self.time_delay)  # 5초 동안 그래프 표시
        plt.close()

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
            plt.show(block=False)
            title='Dabl Plot: target vs features'
            plt.savefig(os.path.join(self.out_path, title +'.png'), dpi=300, bbox_inches='tight')
            plt.pause(self.time_delay)  # 5초 동안 그래프 표시
            plt.close()
        except Exception as e:
            self.logger.warning(f"Dabl plotting failed: {str(e)}")

    def _get_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """각 컬럼별 주요 통계량 계산"""
        stats_dict = {}
        
        for col in df.columns:
            col_stats = {
                'dtype': df[col].dtype,
                'missing_count': df[col].isnull().sum(),
                'missing_ratio': df[col].isnull().sum() / len(df) * 100
            }
            
            if np.issubdtype(df[col].dtype, np.number):
                # 수치형 변수 통계량
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                
                col_stats.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'outlier_count': len(outliers),
                    'outlier_ratio': len(outliers) / len(df) * 100
                })
            else:
                # 범주형 변수 통계량
                col_stats.update({
                    'unique_count': df[col].nunique(),
                    'most_common': df[col].mode().iloc[0],
                    'most_common_ratio': df[col].value_counts().iloc[0] / len(df) * 100
                })
                
            stats_dict[col] = col_stats
        
        return pd.DataFrame.from_dict(stats_dict, orient='index')

    def _group_similar_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """특성의 유사도에 따라 컬럼 그룹화"""
        feature_groups = {
            'high_cardinality_cat': [],  # 높은 카디널리티 범주형
            'low_cardinality_cat': [],   # 낮은 카디널리티 범주형
            'binary': [],                # 이진 변수
            'continuous': [],            # 연속형 변수
            'discrete': [],              # 이산형 변수
            'datetime': [],              # 날짜/시간 변수
            'high_missing': []           # 높은 결측치 비율
        }
        
        for col in df.columns:
            missing_ratio = df[col].isnull().sum() / len(df)
            
            if missing_ratio > 0.5:
                feature_groups['high_missing'].append(col)
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_ratio = df[col].nunique() / len(df)
                if len(df[col].unique()) == 2:
                    feature_groups['binary'].append(col)
                elif unique_ratio > 0.05:  # 5% 이상이 유니크한 값
                    feature_groups['continuous'].append(col)
                else:
                    feature_groups['discrete'].append(col)
            else:
                if df[col].nunique() > 10:
                    feature_groups['high_cardinality_cat'].append(col)
                else:
                    feature_groups['low_cardinality_cat'].append(col)
        
        return {k: v for k, v in feature_groups.items() if v}  # 빈 그룹 제거

    def _plot_feature_groups(self, df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> None:
        """그룹별 특성 시각화"""
        for group_name, columns in feature_groups.items():
            if not columns:
                continue
                
            n_cols = min(3, len(columns))
            n_rows = (len(columns) - 1) // n_cols + 1
            fig = plt.figure(figsize=(6*n_cols, 4*n_rows))
            fig.suptitle(f'{group_name} Features Distribution', fontsize=16)
            
            for idx, col in enumerate(columns, 1):
                ax = plt.subplot(n_rows, n_cols, idx)
                
                if group_name in ['continuous', 'discrete']:
                    sns.histplot(data=df, x=col, kde=True, ax=ax)
                    ax.set_title(f'{col}\nMean: {df[col].mean():.2f}, Std: {df[col].std():.2f}')
                else:
                    value_counts = df[col].value_counts()
                    sns.barplot(x=value_counts.index[:10], y=value_counts.values[:10], ax=ax)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    ax.set_title(f'{col}\nUnique values: {df[col].nunique()}')
            
        plt.tight_layout()
        plt.show(block=False)
        title='Features Distribution'
        plt.savefig(os.path.join(self.out_path, title +'.png'), dpi=300, bbox_inches='tight')
        plt.pause(self.time_delay)  # 5초 동안 그래프 표시
        plt.close()

def main():
    eda = EDA()
    df = pd.read_csv('data/train.csv')
    processed_df = eda.automated_eda(
        df=df,
        target_col='target_column'  # 선택사항
    )
if __name__ == "__main__":
    main()
