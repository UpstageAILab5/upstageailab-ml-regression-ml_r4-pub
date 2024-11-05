import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

class EDAAnalyzer:
    """EDA 분석 유틸리티 클래스"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 분석"""
        missing_stats = pd.DataFrame({
            'missing_count': df.isnull().sum(),
            'missing_ratio': df.isnull().sum() / len(df) * 100
        })
        return missing_stats.sort_values('missing_ratio', ascending=False)
        
    def analyze_distributions(self, df: pd.DataFrame, 
                            numerical_cols: List[str]) -> None:
        """수치형 변수 분포 분석"""
        fig, axes = plt.subplots(len(numerical_cols), 2, 
                                figsize=(12, 4*len(numerical_cols)))
        
        for i, col in enumerate(numerical_cols):
            # 히스토그램
            sns.histplot(data=df, x=col, ax=axes[i,0])
            axes[i,0].set_title(f'{col} Distribution')
            
            # 박스플롯
            sns.boxplot(data=df, y=col, ax=axes[i,1])
            axes[i,1].set_title(f'{col} Boxplot')
            
        plt.tight_layout()
        
    def analyze_correlations(self, df: pd.DataFrame, 
                           numerical_cols: List[str]) -> None:
        """상관관계 분석"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numerical_cols].corr(), 
                   annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
