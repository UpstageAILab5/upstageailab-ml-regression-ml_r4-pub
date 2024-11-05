import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Tuple, List, Dict

class DataCollector:
    """데이터 수집 및 기본 검증 클래스"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> pd.DataFrame:
        """데이터 로드 및 기본 검증"""
        try:
            df = pd.read_csv(self.config['data']['raw_data_path'])
            self.logger.info(f"데이터 로드 완료: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"데이터 로드 실패: {str(e)}")
            raise
            
    def validate_data(self, df: pd.DataFrame) -> bool:
        """데이터 기본 검증
        - 필수 컬럼 존재 확인
        - 데이터 타입 확인
        - 기본적인 제약조건 확인
        """
        required_columns = (
            self.config['data']['categorical_features'] + 
            self.config['data']['numerical_features'] +
            [self.config['data']['target_column']]
        )
        
        # 컬럼 존재 확인
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            self.logger.error(f"필수 컬럼 누락: {missing_cols}")
            return False
            
        # 데이터 타입 및 기본 제약조건 확인
        if df[self.config['data']['target_column']].min() < 0:
            self.logger.error("타겟 변수에 음수 값 존재")
            return False
            
        return True

class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.label_encoders = {}
        self.logger = logging.getLogger(__name__)
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        df_processed = df.copy()
        
        # 결측치가 많은 컬럼 제거
        missing_ratio = df_processed.isnull().sum() / len(df_processed)
        cols_to_drop = missing_ratio[
            missing_ratio > self.config['preprocessing']['missing_threshold']
        ].index
        df_processed = df_processed.drop(columns=cols_to_drop)
        
        # 수치형 변수: 중앙값으로 대체
        numerical_features = self.config['data']['numerical_features']
        for col in numerical_features:
            if col in df_processed.columns:
                df_processed[col].fillna(df_processed[col].median(), 
                                      inplace=True)
        
        # 범주형 변수: 최빈값으로 대체
        categorical_features = self.config['data']['categorical_features']
        for col in categorical_features:
            if col in df_processed.columns:
                df_processed[col].fillna(df_processed[col].mode()[0], 
                                      inplace=True)
                
        return df_processed
        
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """이상치 처리 (IQR 방식)"""
        df_processed = df.copy()
        numerical_features = self.config['data']['numerical_features']
        
        for col in numerical_features:
            if col in df_processed.columns:
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                threshold = self.config['preprocessing']['outlier_threshold']
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # 이상치를 경계값으로 대체
                df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
                df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
                
        return df_processed
        
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """범주형 변수 인코딩"""
        df_processed = df.copy()
        categorical_features = self.config['data']['categorical_features']
        
        for col in categorical_features:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(
                        df_processed[col]
                    )
                else:
                    df_processed[col] = self.label_encoders[col].transform(
                        df_processed[col]
                    )
                    
        return df_processed
