
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
import os

from sklearn.linear_model import Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
import pandas as pd

from scipy import sparse

from time import time
import psutil
import os
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from tqdm import tqdm
import numpy as np
import pickle
from scipy import sparse

from typing import List
# df.loc[some_condition, 'column'] = new_value

class FeatureSelect:
    def __init__(self):
        pass
    @staticmethod
    def filter_method(X_train, X_test, continuous_columns, categorical_columns):
        #print("Original DataFrame:\n", df.shape)
        #print("\nTransformed DataFrame:\n", df_transformed.shape)
        # FeatureSelect.calculate_vif(X_train, continuous_columns)
        # FeatureSelect.cramers_v_all(X_train, categorical_columns)
        cols_var = FeatureSelect._variance_threshold(X_train)

        #.to_csv(os.path.join(prep_path, 'df_combined_distance_feature_after_null_fill_reduced_var.csv'))
        cols_corr = FeatureSelect._corr_threshold(X_train) 

        # reduced_train_corr = reduced_train[cols_corr]
        # reduced_test_corr = reduced_test[cols_corr]

        # print(cols_corr, reduced_train_corr.shape)
        return cols_var, cols_corr
    @staticmethod
    def _variance_threshold(df, threshold = 0.1): 
        selector = VarianceThreshold(threshold=threshold)
        reduced_df = selector.fit_transform(df)

        print("Original DataFrame:\n", df.shape)
        print("Var based Reduced DataFrame:\n", pd.DataFrame(reduced_df).shape)
        selected_features = df.columns[selector.get_support()] 
        unselected_features = df.columns[~selector.get_support()]
        print(f'\n####### Selected Columns: {len(selected_features)}\n{selected_features}\n')
        print(f'########### Unselected Columns: {len(unselected_features)}\n{unselected_features}\n')
        return selected_features
    @staticmethod
    def _corr_threshold(df, threshold = 0.8):
        correlation_matrix = df.corr()
        # 상관계수 행렬을 절대값 기준으로 상관계수 기준 설정
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        # 상관계수가 threshold 이상인 열 이름 선택
        selected_features = [column for column in upper_triangle.columns if any(upper_triangle[column].abs() <= threshold)]
        # 선택된 변수 제거
        unselected_features = [column for column in df.columns if column not in selected_features]
        print(f'\n####### Selected Columns: {len(selected_features)}\n{selected_features}\n')
        print(f'########### Unselected Columns: {len(unselected_features)}\n{unselected_features}\n')
        return selected_features
    
    # 1. VIF 계산 (numerical columns)
    @staticmethod
    def calculate_vif(dataframe, numerical_columns, vif_threshold):
        numerical_columns = [col for col in numerical_columns if col in dataframe.columns]
        X = dataframe[numerical_columns].copy()

        if X.isnull().values.any():
            print("데이터에 결측값이 있습니다. 결측값을 처리하세요.")
        if np.isinf(X.values).any():
            print("데이터에 무한대 값이 있습니다. 무한대 값을 처리하세요.")
        
        X = X.assign(intercept=1)  # 상수항 추가
        vif_df = pd.DataFrame()
        vif_df['Feature'] = X.columns
        vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print(f'VIF: {vif_df}')
        high_vif_columns = vif_df[vif_df['VIF'] > vif_threshold]['Feature'].tolist()
        print(f'High VIF Columns: {high_vif_columns}')
        return high_vif_columns

    # 2. Cramer's V 계산 (categorical columns)
    @staticmethod
    def cramers_v(confusion_matrix):
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        r, k = confusion_matrix.shape
        return np.sqrt(chi2 / (n * min(k-1, r-1)))
    @staticmethod
    def cramers_v_all(df, categorical_columns, cramer_v_threshold):
        high_cramer_v_columns = []
        for i in range(len(categorical_columns)):
            for j in range(i + 1, len(categorical_columns)):
                col1 = categorical_columns[i]
                col2 = categorical_columns[j]
                contingency_table = pd.crosstab(df[col1], df[col2])
                cramer_v_value = FeatureSelect.cramers_v(contingency_table.values)
                print(f"Cramer's V between {col1} and {col2}: {cramer_v_value:.4f}")
                if cramer_v_value > cramer_v_threshold:
                    high_cramer_v_columns.append((col1, col2))
                    print(f"High Cramer's V between {col1} and {col2}: {cramer_v_value:.4f}")
        print(f'High Cramer\'s V Columns: {high_cramer_v_columns}')
        return high_cramer_v_columns
    @staticmethod
    def select_features_by_kbest(X_sampled, y_sampled, original_column_names, k=40):
        k = min(k, X_sampled.shape[1])  # 특성 수의 절반 또는 20개
        print(f'k: {k}')
        
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_filtered = selector.fit_transform(X_sampled, y_sampled)
        selected_features = original_column_names[selector.get_support()]
        X_train_sampled = pd.DataFrame(X_filtered, columns=selected_features)
        #X_test_sampled = pd.DataFrame(X_filtered, columns=original_column_names[selector.get_support()])
        print('#### Filtered columns by SelectKBest', selected_features)
        return X_train_sampled, selected_features#, X_test_sampled
    @staticmethod
    def compare_selected_features(selected_rfe, selected_sfs, list_features, original_column_names):
        set_rfe = set(selected_rfe)
        set_sfs = set(selected_sfs)
        # 공통 집합 계산
        common_features = set_rfe.intersection(set_sfs)
        union_features = set_rfe.union(set_sfs)

        # 결과 출력
        print(f"{list_features[0]}로 선택된 피처: \n{selected_rfe}\n")
        print(f"{list_features[1]}로 선택된 피처: \n{selected_sfs}\n")
        print("공통 피처:\n", common_features)
        print("합집합 피처:\n", union_features)
        rest_features = list(set(original_column_names) - set(union_features))
        print(f'Rest features: \n{rest_features}\n')

        return common_features, union_features, rest_features
    @staticmethod
    def wrapper_method(X_train, y_train, clf, fig_path):
        print(f'Starting Wrapper Feature Selection...\n')
        
        # SFS
        selected_features_sfs = FeatureSelect._optimized_sfs(clf, X_train, y_train, fig_path)
        # RFE
        selected_features_rfe = FeatureSelect._optimized_rfe(clf, X_train, y_train, fig_path)
        
        return selected_features_rfe, selected_features_sfs
    @staticmethod
    def _optimized_rfe(clf, X_train, y_train, fig_path):
        # 더 큰 step 사용
        step = max(1, X_train.shape[1] // 10)  # 특성 수의 10%씩 제거
        
        rfecv = RFECV(
            estimator=clf,
            step=step,
            scoring='neg_mean_absolute_error',
            cv=3,  # cv 수 줄임
            n_jobs=-1  # 병렬 처리
        )
        
        rfecv.fit(X_train, y_train)
        # cv_results_에서 평균 테스트 점수 추출
        mean_test_scores = rfecv.cv_results_['mean_test_score']     
        plt.figure(figsize=(8, 4))
        plt.xlabel('Number of Features Selected')
        plt.ylabel('Cross Validation Score (Accuracy)')
        
        plt.plot(range(1, len(mean_test_scores) + 1), mean_test_scores)
        plt.title('RFECV Performance')
    
        plt.show(block=False)
        plt.savefig(os.path.join(fig_path, 'rfecv_performance.png'))
        plt.pause(3)
        plt.close()
        selected_cols = X_train.columns[rfecv.support_].tolist()
        print(f'RFECV:{len(selected_cols)} {selected_cols}')
        return selected_cols

    @staticmethod
    def _optimized_sfs(clf, X_train, y_train, fig_path, k_features='best'):
        print('Sequential Feature Selection (SFS)')
        sfs = SFS(
            clf,
            k_features=k_features,
            forward=True,
            floating=False,
            scoring='neg_mean_absolute_error',
            cv=3,  # cv 수 줄임
            n_jobs=-1  # 병렬 처리
        )
        print(f'SFS: {sfs}')
        sfs = sfs.fit(X_train, y_train)
        print('Selected features:', sfs.k_feature_idx_)
        print('CV Score:', sfs.k_score_)
        # SFS 과정 시각화
        fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
        plt.title('Sequential Feature Selection (SFS) Performance')
        plt.grid()
        plt.show(block=False)
        plt.savefig(os.path.join(fig_path, 'sfs_performance.png'))
        plt.pause(3)
        plt.close()
        selected_features_sfs = X_train.columns[list(sfs.k_feature_idx_)].tolist()
        print(f'SFS: {len(selected_features_sfs)} {selected_features_sfs}')
        #sfs.k_feature_idx_
        
        return selected_features_sfs
   
class DataPrep:
    def __init__(self):
        pass
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
        # Interpolation
        # 연속형 변수는 선형보간을 해주고, 범주형변수는 알수없기에 “unknown”이라고 임의로 보간해 주겠습니다.
        concat_select.info()
        # 본번, 부번의 경우 float로 되어있지만 범주형 변수의 의미를 가지므로 object(string) 형태로 바꾸어주고 아래 작업을 진행하겠습니다.
        
        # 먼저, 연속형 변수와 범주형 변수를 위 info에 따라 분리해주겠습니다.
        # 범주형 변수에 대한 보간
        concat_select[categorical_columns] = concat_select[categorical_columns].fillna('NULL')
        # 연속형 변수에 대한 보간 (선형 보간)
        concat_select[continuous_columns] = concat_select[continuous_columns].interpolate(method='linear', axis=0)
        print(f'보간된 모습을 확인해봅니다.\n{concat_select.isnull().sum()}')
        print(f'보간 후 shape: {concat_select.shape}')

        return concat_select

    @staticmethod
    def prep_null_advanced(
                            df: pd.DataFrame, 
                            # target_cols: List[str],
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
                        # 범주형: 최빈값으로 채우기
                        fill_values = df_filled.groupby(valid_groups)[col].transform(
                            lambda x: x.mode().iloc[0] if not x.mode().empty else x
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
                    print(f'{col}: {fill_value}')
                else:
                    fill_value = df_filled[col].mean()
                    method = "평균값"
                    print(f'{col}: {fill_value}')
                    
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
        print(f'* 결측치 비율이 {threshold} 이하인 변수들: {list(columns_to_keep)}')
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
            lbl.fit( dt_train[col].astype(str) )
            dt_train[col] = lbl.transform(dt_train[col].astype(str))
            label_encoders[col] = lbl           # 나중에 후처리를 위해 레이블인코더를 저장해주겠습니다.

            # Test 데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가해줍니다.
            dt_test[col] = dt_test[col].astype(str)
            for label in np.unique(dt_test[col]):
                if label not in lbl.classes_: # unseen label 데이터인 경우
                    lbl.classes_ = np.append(lbl.classes_, label) # 미처리 시 ValueError발생하니 주의하세요!
            dt_test[col] = lbl.transform(dt_test[col].astype(str))

            dt_train.head(1)        # 레이블인코딩이 된 모습입니다.

            assert dt_train.shape[1] == dt_test.shape[1]          # train/test dataset의 shape이 같은지 확인해주겠습니다.
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
class Utils:
    @staticmethod
    def concat_train_test(dt, dt_test):
        Utils.remove_unnamed_columns(dt)
        Utils.remove_unnamed_columns(dt_test)
        dt['is_test'] = 0
        dt_test['is_test'] = 1
        dt_test['target'] = 0
        concat = pd.concat([dt, dt_test], axis=0).reset_index(drop=True)
        print(concat['is_test'].value_counts())
        return concat
    @staticmethod
    def unconcat_train_test(concat):
        Utils.remove_unnamed_columns(concat)
        dt = concat.query('is_test==0')
        # y_train = dt['target']
        dt.drop(columns=['is_test'], inplace=True)
        dt_test = concat.query('is_test==1')
        dt_test.drop(columns=['target', 'is_test'], inplace=True)
        return dt, dt_test
    @staticmethod
    def clean_df(df):
        df = Utils.remove_unnamed_columns(df)
        duplicated_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicated_cols:
            print("중복된 컬럼:", duplicated_cols)
            # 방법 1: 중복 컬럼 제거 (첫 번째 컬럼 유지)
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
        return df
    @staticmethod
    def remove_unnamed_columns(df):
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            print(f"Removing unnamed columns: {unnamed_cols}")  
            df = df.drop(columns=unnamed_cols)
        return df
    @staticmethod
    def categorical_numeric(df):
        # 파생변수 제작으로 추가된 변수들이 존재하기에, 다시한번 연속형과 범주형 칼럼을 분리해주겠습니다.
        continuous_columns_v2 = []
        categorical_columns_v2 = []

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                continuous_columns_v2.append(column)
            else:
                categorical_columns_v2.append(column)
        return continuous_columns_v2, categorical_columns_v2
    @staticmethod
    def chk_train_test_data(df):
        dt = df.query('is_test==0')
        dt_test = df.query('is_test==1')
        print(f'train data shape: {dt.shape},{dt.columns}\ntest data shape: {dt_test.shape},{dt_test.columns}')
        print(dt['target'].info())
        print(dt_test['target'].info())

def main():
    threshold_null = 0.9
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data')
    prep_path = os.path.join(data_path, 'preprocessed')
    fig_path = os.path.join(base_path, 'output', 'plots')
    #####
    path_raw = os.path.join(prep_path, 'df_raw.csv')
    
    path_feat = os.path.join(prep_path, 'feat_concat_raw.csv')

    path_null_prep = os.path.join(prep_path, 'df_feat_null-preped.csv')
    path_encoded = os.path.join(prep_path, 'df_feat_null-preped_freq-encoded.csv')

    cols_to_remove = ['등기신청일자', '거래유형', '중개사소재지'] +['홈페이지','k-전화번호', 'k-팩스번호', '고용보험관리번호']
    cols_to_str = ['본번', '부번'] + ['구', '동', '강남여부', '신축여부', 'cluster_dist_transport', 'cluster_dist_transport_count', 'cluster_select','subway_zone_type', 'bus_zone_type']
    cols_id = ['is_test', 'target']
    cols_date = ['단지신청일', '단지승인일','k-사용검사일-사용승인일']
    cols_to_num = []#['좌표X', '좌표Y', '위도', '경도']
    #cols_to_select = ['시군구', '전용면적', '계약년월', '건축년도']
    vif_threshold = 5
    cramer_v_threshold = 0.7
    path_feat = os.path.join(prep_path, 'feat_concat_raw.csv')
    df_feat = pd.read_csv(path_feat)
    df_feat = df_feat.loc[:, ~df_feat.columns.str.contains('^Unnamed')]
    #df = pd.read_csv(os.path.join(prep_path, 'df_raw_null_prep_coord.csv'))
    df_raw = pd.read_csv(path_raw)
    df_raw = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed')]

    #df_raw = df_raw.loc[:, cols_to_select + cols_id]
    print(f'\n##################df_raw.columns: {df_raw.columns}')
    df_raw = pd.concat([df_raw, df_feat], axis=1)
    print(f'\n##################df_raw.columns: {df_raw.columns}')
    Utils.chk_train_test_data(df_raw)
    for col in cols_date:
        df_raw[col] = pd.to_datetime(df_raw[col])
        df_raw[col] = df_raw[col].view('int64') / 10**9# 나노초 단위이므로 초 단위로 변환
    
##### Null prep: Interpolation for Null values
    continuous_columns, categorical_columns = Utils.categorical_numeric(df_raw)
    if os.path.exists(path_null_prep):
        df_interpolated = pd.read_csv(path_null_prep)
    else:
        cols_to_remove = [col for col in cols_to_remove if col in df_raw.columns]
        df_raw.drop(columns=cols_to_remove, inplace=True)
        print(f'#####\nRemove columns: {len(cols_to_remove)}\n{cols_to_remove}\n###')
        df_null_removed = DataPrep.remove_null(df_raw, threshold_null)
        cols_to_str = [col for col in cols_to_str if col in df_null_removed.columns]
        cols_to_num = [col for col in cols_to_num if col in df_null_removed.columns]
        df_null_removed = DataPrep.convert_dtype(df_null_removed, cols_to_str, cols_to_num)
        continuous_columns, categorical_columns = Utils.categorical_numeric(df_null_removed)
        #df_interpolated = DataPrep.prep_null(df_null_removed, continuous_columns, categorical_columns)
        group_cols = ['시군구', '도로명', '아파트명']
        df_columns = set(df_null_removed.columns)

        # 리스트에서 데이터프레임에 없는 컬럼 찾기
        missing_columns = [col for col in group_cols if col not in df_columns]
        if missing_columns:
            print("Missing columns:", missing_columns)
        df_interpolated = DataPrep.prep_null_advanced(df_null_removed, continuous_columns, categorical_columns, group_cols=missing_columns)
        df_interpolated.to_csv(path_null_prep)

    vif = FeatureSelect.calculate_vif(df_interpolated, continuous_columns, vif_threshold)
    cramers_v = FeatureSelect.cramers_v_all(df_interpolated, categorical_columns, cramer_v_threshold)
        
    continuous_columns, categorical_columns = Utils.categorical_numeric(df_interpolated)
## Encode categorical variables
    cols_exclude = ['target', 'is_test']
    df_train, df_test = Utils.unconcat_train_test(df_interpolated)
    y_train = df_train['target']
    X_train = df_train.drop(columns=['target'])
    X_test = df_test
    

    if os.path.exists(path_encoded):
        concat = pd.read_csv(path_encoded)
    else:
        min_freq_threshold = 0.05
        ##################X_train, X_test, encode_labels = encode_label(X_train, X_test, categorical_features)
        min_freq_dict = DataPrep.auto_adjust_min_frequency(X_train, base_threshold=min_freq_threshold)
        X_train_cat = X_train[categorical_columns]
        X_test_cat = X_test[categorical_columns]
        X_train_cat_encoded, X_test_cat_encoded = DataPrep.frequency_encode(X_train_cat, X_test_cat, min_freq_dict)
        X_train = pd.concat([X_train_cat_encoded, X_train.drop(columns=categorical_columns)], axis=1)
        X_test = pd.concat([X_test_cat_encoded, X_test.drop(columns=categorical_columns)], axis=1)
        X_train['target'] = y_train
        concat = Utils.concat_train_test(X_train, X_test)
        concat.to_csv(path_encoded)
        
## Select Features  
    print(f'##### Original dataset shape: {concat.shape}')                                                        
##### Filter Method
    concat = Utils.remove_unnamed_columns(concat)
    df_train, df_test = Utils.unconcat_train_test(concat)
    y_train = df_train['target']
    X_train = df_train.drop(columns=['target'])
    X_test = df_test
    original_column_names = X_train.columns#           
    cols_var, cols_corr = FeatureSelect.filter_method(X_train, X_test, continuous_columns, categorical_columns)

    filter_common_features, filter_union_features, filter_rest_features = FeatureSelect.compare_selected_features(cols_var, cols_corr, ['Variance Threshold', 'Correlation Threshold'], original_column_names)
##### Resampling for Feature Selection  
    n_resample = min(10000, len(X_train))
    X_sampled, y_sampled = resample(X_train, y_train, 
                                            n_samples=n_resample,
                                            random_state=2023)
    # #상위 K개 특성만 먼저 선택
    X_sampled, kbest_features = FeatureSelect.select_features_by_kbest(X_sampled, y_sampled, original_column_names, k=40)

    rf = RandomForestRegressor(random_state=2023)
    #rf = Ridge(alpha=1.0)
    selected_rfe, selected_sfs = FeatureSelect.wrapper_method(X_sampled, y_sampled, rf, fig_path)
    common_features, union_features, rest_features = FeatureSelect.compare_selected_features(selected_rfe, selected_sfs, ['RFE', 'SFS'], original_column_names)
  
    dict_result = {'vif': vif, 
                   'cramers_v': cramers_v,
                   'kbest_features': kbest_features,
                   'filter_common_features': filter_common_features,
                   'filter_union_features': filter_union_features,
                   'filter_rest_features': filter_rest_features,
                   'common_features': common_features,
                   'union_features': union_features,
                   'rest_features': rest_features,
                   'selected_rfe': selected_rfe,
                   'selected_sfs': selected_sfs
                   }
    print(dict_result)
    
    with open(os.path.join(prep_path, 'dict_feature_selection_result.pkl'), 'wb') as f:
        pickle.dump(dict_result, f)
if __name__ == '__main__':
    main()


# threshold_null = 0.9

# min_freq_threshold = 0.05
# threshold_corr = 0.8
# threshold_var = 0.1