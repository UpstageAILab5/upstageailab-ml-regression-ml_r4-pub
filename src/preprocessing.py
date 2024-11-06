import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Dict

class DataLoader:
    """데이터 수집 및 기본 검증 클래스"""
    def __init__(self, config: Dict):
        self.config = config
        self.logger = config.get('logger')
    def load_data(self, target) -> pd.DataFrame:
        try:
            print('Data loading...')
            data_list = self.config.get('data')['csv_files']
            target_list = [data for data in data_list if target in data]
            print(f'Target found: {target_list}\n{target_list[0]}' )
            df = pd.read_csv(target_list[0])
            self.logger.info(f'Data loaded: {df.shape}')
            display(df.head(3))
            print(f'Columns: {df.columns}')

            return df
        except Exception as e:
            self.logger.error(f'Failed to load data: {str(e)}')
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
            [self.config.get('target')]
        )
        
        # 컬럼 존재 확인
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            self.logger.error(f"필수 컬럼 누락: {missing_cols}")
            return False
            
        # 데이터 타입 및 기본 제약조건 확인
        if df[self.config.get('target')].min() < 0:
            self.logger.error("타겟 변수에 음수 값 존재")
            return False
        return True
  
    

class DataPrep:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = config.get('logger')

    def _is_numeric_convertible(self, series: pd.Series) -> bool:
        """결측치를 제외한 값들이 숫자로 변환 가능한지만 확인"""
        # 결측치가 아닌 값들만 검사
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return False
            
        try:
            pd.to_numeric(non_null_values, errors='raise')
            return True
        except (ValueError, TypeError):
            return False

    def infer_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        numerical_features = []
        categorical_features = []
        
        target_col = self.config.get('target')
        features = [col for col in df.columns if col != target_col]
        
        for col in features:
            self.logger.info(f"\n=== {col} 컬럼 분석 ===")
            self.logger.info(f"데이터 타입: {df[col].dtype}")
            self.logger.info(f"is_numeric_dtype 결과: {pd.api.types.is_numeric_dtype(df[col])}")
            
            # numpy나 pandas의 숫자형 타입 직접 확인
            is_numeric = df[col].dtype in ['int64', 'float64', 'int32', 'float32']
            self.logger.info(f"직접 타입 체크 결과: {is_numeric}")
            
            # 이미 숫자형인 경우
            if is_numeric:  # 수정된 부분
                unique_ratio = df[col].nunique() / len(df)
                self.logger.info(f"{col}: 숫자형, 유니크 비율 = {unique_ratio:.2%}")
                
                if unique_ratio < self.config.get('prep')['unique_thr']:
                    self.logger.info(f"{col}: 범주형으로 분류 (적은 유니크값)")
                    categorical_features.append(col)
                else:
                    self.logger.info(f"{col}: 수치형으로 분류")
                    numerical_features.append(col)
            else:
                if self._is_numeric_convertible(df[col]):
                    self.logger.info(f"{col}: 숫자형 변환 가능")
                    temp_series = df[col].copy()
                    non_null_mask = temp_series.notna()
                    temp_series.loc[non_null_mask] = pd.to_numeric(temp_series[non_null_mask], errors='coerce') #errors='coerce' will transform string values to NaN, that can then be replaced if desired
                    
                    unique_ratio = temp_series.nunique() / len(df)
                    if unique_ratio < self.config.get('prep')['unique_thr']:
                        categorical_features.append(col)
                    else:
                        numerical_features.append(col)
                else:
                    categorical_features.append(col)
        
        self.logger.info("\n=== 최종 결과 ===")
        self.logger.info(f"수치형 피처: {len(numerical_features)}\n{numerical_features}")
        self.logger.info(f"범주형 피처: {len(categorical_features)}\n{categorical_features}")
        self.config['data']['numerical_features'] = numerical_features
        self.config['data']['categorical_features'] = categorical_features
        return numerical_features, categorical_features

    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        df_processed = df.copy()
        
        # 결측치가 많은 컬럼 제거
        missing_ratio = df_processed.isnull().sum() / len(df_processed)
        cols_to_drop = missing_ratio[
            missing_ratio > self.config['prep']['missing_thr']
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
                threshold = self.config['prep']['outlier_thr']
                
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