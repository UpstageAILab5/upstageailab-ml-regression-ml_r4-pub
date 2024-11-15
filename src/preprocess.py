import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import List
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time
from scipy import sparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, PowerTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
class DataPrep:

    @staticmethod
    def transform_and_visualize(X_train, X_test, continuous_features, output_dir='plots', skew_threshold=0.5):
        """
        Transform and visualize continuous features in X_train and X_test datasets.
        
        Parameters:
        - X_train: pd.DataFrame - Training data
        - X_test: pd.DataFrame - Test data
        - continuous_features: list - List of continuous features
        - output_dir: str - Directory to save plots
        - skew_threshold: float - Skewness threshold for transformations
        
        Returns:
        - pd.DataFrame, pd.DataFrame - Transformed X_train and X_test
        """
        os.makedirs(output_dir, exist_ok=True)

        transformed_train = X_train.copy()
        transformed_test = X_test.copy()
        common_features = set(transformed_train.columns) & set(transformed_test.columns)
        continuous_features = [col for col in continuous_features if col in common_features]

        for feature in continuous_features:
            # Plot original distributions for X_train and X_test
            plt.figure(figsize=(12, 8))

            # X_train original
            plt.subplot(2, 2, 1)
            sns.histplot(X_train[feature], kde=True, color='blue', bins=30)
            plt.title(f"X_train Original Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")

            # X_test original
            plt.subplot(2, 2, 2)
            sns.histplot(X_test[feature], kde=True, color='blue', bins=30)
            plt.title(f"X_test Original Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")

            # Handle skewness and scaling based on X_train
            skewness = transformed_train[feature].skew()
            print(f"{feature} skewness before transformation: {skewness}")
            if abs(skewness) > skew_threshold:
                # Apply transformation to X_train
                transformed_train[feature] = np.log1p(transformed_train[feature] - transformed_train[feature].min() + 1) \
                    if skewness > 0 else PowerTransformer(method='yeo-johnson').fit_transform(transformed_train[[feature]])
                
                # Apply the same transformation to X_test
                transformed_test[feature] = np.log1p(transformed_test[feature] - X_train[feature].min() + 1) \
                    if skewness > 0 else PowerTransformer(method='yeo-johnson').fit_transform(transformed_test[[feature]])
            
            # Apply RobustScaler (fitting only on X_train)
            scaler = RobustScaler()
            transformed_train[feature] = scaler.fit_transform(transformed_train[[feature]])
            transformed_test[feature] = scaler.transform(transformed_test[[feature]])

            # Plot transformed distributions
            plt.subplot(2, 2, 3)
            sns.histplot(transformed_train[feature], kde=True, color='green', bins=30)
            plt.title(f"X_train Transformed Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")

            plt.subplot(2, 2, 4)
            sns.histplot(transformed_test[feature], kde=True, color='green', bins=30)
            plt.title(f"X_test Transformed Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")

            # Save plot
            plot_filename = os.path.join(output_dir, f"{feature}_train_test_distribution_comparison.png")
            plt.savefig(plot_filename)
            plt.close()
            
            print(f"Saved distribution plot for {feature} at {plot_filename}")

        return transformed_train, transformed_test
    # @staticmethod
    # def transform_features(df, continuous_features, categorical_encoded_features, skew_threshold=0.5):
    #     """
    #     Apply transformations to continuous features using RobustScaler and handle skewness.
        
    #     Parameters:
    #     - df: pd.DataFrame - Input data frame
    #     - continuous_features: list - List of continuous features
    #     - categorical_encoded_features: list - List of target encoded categorical features
    #     - skew_threshold: float - Skewness threshold for transformations
        
    #     Returns:
    #     - pd.DataFrame - Data frame with transformed features
    #     """
    #     transformed_df = df.copy()

    #     # Handle continuous features with RobustScaler
    #     scaler = RobustScaler()
    #     for feature in continuous_features:
    #         skewness = transformed_df[feature].skew()
    #         print(f"{feature} skewness: {skewness}")
    #         if abs(skewness) > skew_threshold:
    #             # Apply log transformation or PowerTransform for skewed features
    #             transformed_df[feature] = np.log1p(transformed_df[feature] - transformed_df[feature].min() + 1) \
    #                 if skewness > 0 else PowerTransformer(method='yeo-johnson').fit_transform(transformed_df[[feature]])
            
    #         # Apply RobustScaler to reduce the influence of outliers
    #         transformed_df[feature] = scaler.fit_transform(transformed_df[[feature]])
        
    #     # Handle target encoded categorical features (if necessary)
    #     for feature in categorical_encoded_features:
    #         # Normalization logic for encoded categorical features if needed
    #         transformed_df[feature] = df[feature]  # Placeholder for any specific encoding treatment
        
    #     return transformed_df

    # Example usage:
    # continuous_feats = ['feature1', 'feature2']
    # categorical_feats = ['encoded_cat1', 'encoded_cat2']
    # df_transformed = transform_features(df, continuous_feats, categorical_feats)

    @staticmethod
    def target_encoding_all(train_df, test_df, target_cols, target_col_y):
        train_df_copy = train_df.copy()
        test_df_copy = test_df.copy()
        for col in target_cols:
            train_encoded, test_encoded = DataPrep._target_encoding(train_df_copy, test_df_copy, col, target_col_y)
            train_df_copy.loc[:, col] = train_encoded
            test_df_copy.loc[:, col] = test_encoded
           
        return train_df_copy, test_df_copy
    @staticmethod
    def _target_encoding(train_df, test_df, column_name, target_col):
        # 학습 데이터에서 각 범주에 대한 목표 변수의 평균 계산
        mean_encoding = train_df.groupby(column_name)[target_col].mean()
        
        # 학습 데이터에 인코딩 적용
        train_encoded = train_df[column_name].map(mean_encoding)
        
        # 테스트 데이터에 인코딩 적용
        # 처음 등장하는 클래스에 대해 기본값 설정 (학습 데이터의 전체 평균)
        global_mean = train_df[target_col].mean()
        test_encoded = test_df[column_name].map(mean_encoding).fillna(global_mean)
        
        return train_encoded, test_encoded
    @staticmethod
    def remove_outliers_iqr(dt, column_name, modified=False):
        df = dt.query('is_test == 0')       # train data 내에 있는 이상치만 제거하도록 하겠습니다.
        df_test = dt.query('is_test == 1')

        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        if modified:
            print('Modified IQR')
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 290
        else:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR


        df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

        result = pd.concat([df, df_test])   # test data와 다시 합쳐주겠습니다.
        return result
    @staticmethod
    def convert_dtype(concat_select, columns_to_str, columns_to_num):
        for col in columns_to_str:  
            concat_select.loc[:, col] = concat_select.loc[:, col].astype('str')
        try:
            for col in columns_to_num:
                concat_select.loc[:, col] = concat_select.loc[:, col].astype('float')
        except:
            print(f'{columns_to_num} 변수는 없습니다.')
        return concat_select

    @staticmethod
    def prep_null(concat_select, continuous_columns, categorical_columns):
        print('\n#### Interpolation for Null values')
        print(f'보간 전 shape: {concat_select.shape}')
        print(f'보간 전 null 개수: {concat_select.isnull().sum()}')
        concat_select.info()

        # 디버깅 출력 추가
        print(f'Categorical columns: {categorical_columns}')
        print(f'Available columns in DataFrame: {concat_select.columns.tolist()}')

        # 범주형 변수에 대한 보간
        try:
            concat_select[categorical_columns] = concat_select[categorical_columns].fillna('NULL')
        except ValueError as e:
            print(f"Error: {e}")
            # 추가적인 디버깅 정보 출력
            for col in categorical_columns:
                if col not in concat_select.columns:
                    print(f"Column {col} is missing in DataFrame.")
                else:
                    print(f"Column {col} length: {len(concat_select[col])}")

        # 연속형 변수에 대한 보간 (선형 보간)
        concat_select[continuous_columns] = concat_select[continuous_columns].interpolate(method='linear', axis=0)
        print(f'보간된 모습을 확인해봅니다.\n{concat_select.isnull().sum()}')
        print(f'보간 후 shape: {concat_select.shape}')

        return concat_select

    @staticmethod
    def prep_null_advanced(
                            df: pd.DataFrame, 
                            continuous_cols: List[str],
                            categorical_cols: List[str],
                            group_cols: List[str] = ['도로명주소', '시군구', '도로명', '아파트명'],
                            ) -> pd.DataFrame:
        """결측치 채우기 함수"""
        df_filled = df.copy()
        
        # 결측치 현황 출력
        null_before = df_filled[categorical_cols + continuous_cols].isnull().sum()
        print(f"처리 전 결측치:\n{null_before}\n")
        
        # 상세 단위부터 큰 단위까지 순차적으로 결측치 채우기
        for i in range(len(group_cols), 0, -1):
            current_groups = group_cols[:i]
            
            # 그룹화 컬럼이 데이터에 있는지 확인
            valid_groups = [col for col in current_groups if col in df_filled.columns]
            
            if not valid_groups:
                continue
                
            for col in continuous_cols + categorical_cols:
                # 결측치가 있는 경우에만 처리
                if df_filled[col].isnull().any():
                    if col in categorical_cols:
                        # 범주형: 모든 값을 문자열로 변환하여 최빈값으로 채우기
                        fill_values = df_filled.groupby(valid_groups)[col].transform(
                            lambda x: x.astype(str).mode().iloc[0] if not x.astype(str).mode().empty else 'Unknown'
                        )
                        print(f'범주형: {col} 최빈값 채우기')
                    else:
                        # 수치형: 평균값으로 채우기
                        fill_values = df_filled.groupby(valid_groups)[col].transform('mean')
                        print(f'수치형: {col} 평균값 채우기')
                    
                    # NaN이 아닌 값만 업데이트
                    mask = df_filled[col].isnull() & fill_values.notna()
                    df_filled.loc[mask, col] = fill_values[mask]
                    
                    filled_count = null_before[col] - df_filled[col].isnull().sum()
                    if filled_count > 0:
                        print(f"{col}: {valid_groups}기준으로 {filled_count}개 채움")
        
        # 남은 결측치 처리
        for col in continuous_cols + categorical_cols:
            if df_filled[col].isnull().any():
                if col in categorical_cols:
                    fill_value = df_filled[col].mode().iloc[0] if not df_filled[col].mode().empty else 'Unknown'
                    method = "최빈값"
                else:
                    fill_value = df_filled[col].mean()
                    method = "평균값"
                    
                missing_count = df_filled[col].isnull().sum()
                df_filled[col] = df_filled[col].fillna(fill_value)
                print(f"{col}: 전체 {method}({fill_value})으로 {missing_count}개 채움")
        
        # 결과 확인
        null_after = df_filled[continuous_cols + categorical_cols].isnull().sum()
        print(f"\n처리 후 결측치:\n{null_after}")
        print(f"\n채워진 결측치 수:\n{null_before - null_after}")
        
        return df_filled
    
    @staticmethod   
    def remove_null(concat, threshold = 0.9):
        """결측치가 특정 비율 이상인 컬럼 제거"""
        # 각 컬럼별 결측치 비율 계산 (전체 행 수 대비)
        ratio_null = concat.isnull().sum() / len(concat)
        #count_null = concat.isnull().sum()
        # threshold 기준으로 컬럼 필터링
        columns_to_keep = ratio_null[ratio_null <= threshold].index
        columns_to_drop = ratio_null[ratio_null > threshold].index
        # 로깅
        print(f'* 결측치 비율이 {threshold} 초과인 변수들: {list(columns_to_drop)}')
        # 선택된 컬럼만 반환
        return concat[columns_to_keep]
    @staticmethod
    def encode_label(dt_train, dt_test, categorical_columns_v2):
        # 각 변수에 대한 LabelEncoder를 저장할 딕셔너리
        label_encoders = {}

        # Implement Label Encoding
        for col in tqdm( categorical_columns_v2 ):
            lbl = LabelEncoder()
        
            # Label-Encoding을 fit
            lbl.fit( dt_train.loc[:, col].astype(str) )
            dt_train.loc[:, col] = lbl.transform(dt_train.loc[:, col].astype(str))
            label_encoders[col] = lbl           # 나중에 후처리를 위해 레이블인코더를 저장해주겠습니다.

            # Test 데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가해줍니다.
            dt_test.loc[:, col] = dt_test[col].astype(str)
            for label in np.unique(dt_test[col]):
                if label not in lbl.classes_: # unseen label 데이터인 경우
                    lbl.classes_ = np.append(lbl.classes_, label) # 미처리 시 ValueError발생하니 주의하세요!
            dt_test.loc[:, col] = lbl.transform(dt_test.loc[:, col].astype(str))

            dt_train.head(1)        # 레이블인코딩이 된 모습입니다.

            #assert dt_train.shape[1] == dt_test.shape[1]          # train/test dataset의 shape이 같은지 확인해주겠습니다.
        return dt_train, dt_test, label_encoders
    @staticmethod
    def auto_adjust_min_frequency(df, base_threshold=0.05, strategy='percentile'):
        """
        각 컬럼별로 `min_frequency`를 자동으로 설정하는 함수.
        
        Parameters:
        - df (pd.DataFrame): 인코딩할 데이터프레임
        - base_threshold (float): 기본적으로 설정할 빈도 비율 (데이터 크기에 따라 결정)
        - strategy (str): 'percentile' 또는 'fixed'로 설정. 'percentile'은 백분위수 기반, 'fixed'는 기준값 기반
        
        Returns:
        - min_freq_dict (dict): 각 컬럼별 최적화된 `min_frequency` 값
        """
        min_freq_dict = {}
        
        for col in df.columns:
            value_counts = df[col].value_counts()
            total_count = len(df[col])
            
            if strategy == 'percentile':
                # 백분위수를 기준으로 min_frequency 결정
                min_freq = np.percentile(value_counts.values, base_threshold * 100)
            elif strategy == 'fixed':
                # 전체 데이터 크기에 base_threshold 비율만큼 설정
                min_freq = total_count * base_threshold
            else:
                raise ValueError("Invalid strategy. Choose 'percentile' or 'fixed'.")
            
            # 최소 1 이상의 빈도만 설정하도록 보장
            min_freq = max(1, int(min_freq))
            min_freq_dict[col] = min_freq
            
        return min_freq_dict
    @staticmethod
    def frequency_encode(train_df, test_df, min_freq_dict):
        """빈도 기반 인코딩과 기존 컬럼 값 교체"""
        print('\n#### Frequency Encoding')
        # start_time = time.time()
 

        for col in train_df.columns:
            # 빈도 계산
            value_counts = train_df[col].value_counts()
            min_frequency = min_freq_dict.get(col, 1)  # 기본값 설정
            frequent_categories = value_counts[value_counts >= min_frequency].index
            
            # Train 데이터 인코딩
            train_categories = train_df[col].apply(lambda x: str(x) if x in frequent_categories else 'OTHER')
            unique_cats = np.sort(train_categories.unique())
            cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
            
            # Test 데이터 인코딩
            test_categories = test_df[col].apply(lambda x: str(x) if x in frequent_categories else 'OTHER')
            
            # 희소 행렬 생성
            rows = range(len(train_categories))
            cols = [cat_to_idx[cat] for cat in train_categories]
            data = np.ones(len(train_categories))
            
            train_matrix = sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(len(train_categories), len(unique_cats))
            )
            #result_matrices.append(train_matrix)
            
            # 기존 컬럼을 인코딩된 값으로 교체
            train_df.loc[:, col] = train_categories.map(cat_to_idx)
            test_df.loc[:, col] = test_categories.map(cat_to_idx)
        
        # print(f'Frequency Encoding Time: {time.time() - start_time}')
        return train_df, test_df
