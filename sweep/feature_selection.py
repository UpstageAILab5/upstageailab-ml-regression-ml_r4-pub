
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
import os
import json
from sklearn.linear_model import Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
import pandas as pd
import pprint
from scipy import sparse
from scipy.stats import chi2_contingency
from time import time
import psutil
import os
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from tqdm import tqdm
import numpy as np
import pickle
import xgboost as xgb
from typing import List
# SettingWithCopyWarning 경고 무시
pd.options.mode.chained_assignment = None  # default='warn'
import sys
from pathlib import Path

current = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f'\n##########current: {current}')


sys.path.append(str(current))
from sweep.sweep_config import config_baseline
# df.loc[some_condition, 'column'] = new_value


class FeatureEngineer():
    def __init__(self):
        print('#### Init Feature Engineering... ')
    @staticmethod
    def prep_feat(concat_select, year = 2009,  col_add=''):
        # 시군구, 년월 등 분할할 수 있는 변수들은 세부사항 고려를 용이하게 하기 위해 모두 분할해 주겠습니다.
        concat_select['구'] = concat_select['시군구'].map(lambda x : x.split()[1])
        concat_select['동'] = concat_select['시군구'].map(lambda x : x.split()[2])
        #

        concat_select['계약년'] = concat_select['계약년월'].astype('str').apply(lambda x: x[:4])
        concat_select['계약월'] = concat_select['계약년월'].astype('str').apply(lambda x: x[4:])
        #concat_select['계약년월'] = concat_select['계약년월'].astype('str').apply(lambda x: x[:4] + x[4:])
        
        #
        
        all = list(concat_select['구'].unique())
        gangnam = ['강서구', '영등포구', '동작구', '서초구', '강남구', '송파구', '강동구']
        gangbuk = [x for x in all if x not in gangnam]

        assert len(all) == len(gangnam) + len(gangbuk)       # 알맞게 분리되었는지 체크합니다.

        # 강남의 여부를 체크합니다.
        is_gangnam = []
        for x in concat_select['구'].tolist() :
            if x in gangnam :
                is_gangnam.append(1)
            else :
                is_gangnam.append(0)

        # 파생변수를 하나 만릅니다.
        concat_select['강남여부'] = is_gangnam
        
        # 건축년도 분포는 아래와 같습니다. 특히 2005년이 Q3에 해당합니다.
        # 2009년 이후에 지어진 건물은 10%정도 되는 것을 확인할 수 있습니다.
        concat_select['건축년도'].describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9])

        # 따라서 2009년 이후에 지어졌으면 비교적 신축이라고 판단하고, 신축 여부 변수를 제작해보도록 하겠습니다.
        concat_select['신축여부'] = concat_select['건축년도'].apply(lambda x: 1 if x >= int(year) else 0)
        try:
            if col_add == 'address':
                concat_select['시군구+번지'] = concat_select['시군구'].astype(str) + concat_select['번지'].astype(str)
        except:
            print('No address column')
        #concat_select.head(1)       # 최종 데이터셋은 아래와 같습니다.
        del concat_select['계약년월']
        del concat_select['시군구']
        print(concat_select.shape)
        print(concat_select.columns)
        return concat_select
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
    # @staticmethod
    # def cramers_v(confusion_matrix):
    #     chi2 = stats.chi2_contingency(confusion_matrix)[0]
    #     n = confusion_matrix.sum()
    #     r, k = confusion_matrix.shape
    #     return np.sqrt(chi2 / (n * min(k-1, r-1)))
    # @staticmethod
    # def cramers_v_all(df, categorical_columns, cramer_v_threshold):
    #     high_cramer_v_columns = []
    #     for i in range(len(categorical_columns)):
    #         for j in range(i + 1, len(categorical_columns)):
    #             col1 = categorical_columns[i]
    #             col2 = categorical_columns[j]
    #             contingency_table = pd.crosstab(df[col1], df[col2])
    #             cramer_v_value = FeatureSelect.cramers_v(contingency_table.values)
    #             print(f"Cramer's V between {col1} and {col2}: {cramer_v_value:.4f}")
    #             if cramer_v_value > cramer_v_threshold:
    #                 high_cramer_v_columns.append((col1, col2))
    #                 print(f"High Cramer's V between {col1} and {col2}: {cramer_v_value:.4f}")
    #     print(f'High Cramer\'s V Columns: {high_cramer_v_columns}')
    #     return high_cramer_v_columns
    @staticmethod
    def cramers_v_all(df, categorical_columns, threshold=0.7):
        """
        데이터프레임의 범주형 변수들 간의 Cramer's V 계산하고 
        threshold 이상의 연관성을 가진 컬럼 쌍 반환
        
        Parameters:
        -----------
        df : pandas DataFrame
            분석할 데이터프레임
        threshold : float, default=0.7
            Cramer's V 임계값 (이 값 이상의 연관성을 가진 쌍 추출)
        
        Returns:
        --------
        highly_correlated : list of tuples
            높은 연관성을 가진 컬럼 쌍의 리스트
        columns_to_drop : list
            제거 추천되는 컬럼 리스트
        """
        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            min_dim = min(confusion_matrix.shape) - 1
            return np.sqrt(chi2 / (n * min_dim))

        # 범주형 컬럼만 선택 (object 또는 category 타입)
        #categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        n_cols = len(categorical_columns)
        
        # 결과를 저장할 리스트
        highly_correlated = []
        correlation_scores = {}
        
        # 모든 가능한 컬럼 쌍에 대해 Cramer's V 계산
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                col1 = categorical_columns[i]
                col2 = categorical_columns[j]
                
                # Cramer's V 계산
                cramer_value = cramers_v(df[col1], df[col2])
                
                # 임계값 이상인 경우 저장
                if cramer_value >= threshold:
                    highly_correlated.append((col1, col2, cramer_value))
                    # 각 컬럼의 상관관계 점수 누적
                    correlation_scores[col1] = correlation_scores.get(col1, 0) + cramer_value
                    correlation_scores[col2] = correlation_scores.get(col2, 0) + cramer_value
        
        # 제거할 컬럼 선택 (각 쌍에서 더 높은 상관관계 점수를 가진 컬럼)
        columns_to_drop = set()
        for col1, col2, _ in highly_correlated:
            if correlation_scores[col1] > correlation_scores[col2]:
                columns_to_drop.add(col1)
            else:
                columns_to_drop.add(col2)
        
        # 결과 출력
        print(f"\n=== Cramer's V Analysis Results ===")
        print(f"분석된 범주형 변수 개수: {n_cols}")
        print(f"임계값: {threshold}")
        print("\n높은 연관성을 가진 변수 쌍:")
        for col1, col2, score in sorted(highly_correlated, key=lambda x: x[2], reverse=True):
            print(f"{col1} - {col2}: {score:.3f}")
        
        print("\n제거 추천되는 변수:")
        for col in sorted(columns_to_drop):
            print(f"- {col} (누적 상관계수: {correlation_scores[col]:.3f})")
        
        return highly_correlated, list(columns_to_drop)

    # 사용 예시
    """
    # 데이터프레임에서 높은 연관성을 가진 컬럼 찾기
    highly_correlated, drops = calculate_cramers_v_pairs(
        df,
        threshold=0.7  # Cramer's V 임계값 설정
    )

    # 제거 추천되는 컬럼 제거
    df_cleaned = df.drop(columns=drops)
    """
    @staticmethod
    def select_features_by_kbest(X_sampled, y_sampled, original_column_names, score_func, k=20):
        k = min(k, X_sampled.shape[1])  # 특성 수의 절반 또는 20개
        print(f'k: {k}')
        
        selector = SelectKBest(score_func=score_func, k=k)
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

        return list(common_features), list(union_features), list(rest_features)
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
        mean_test_scores = rfecv.cv_results_['mean_test_score']
        
        # 그래프 스타일 설정
        plt.style.use('seaborn')
        plt.figure(figsize=(12, 6))
        
        # 메인 플롯
        plt.plot(
            range(1, len(mean_test_scores) + 1), 
            mean_test_scores,
            marker='o',
            markersize=6,
            linewidth=2,
            color='#2E86C1',
            label='Cross Validation Score'
        )
        
        # 최적 포인트 강조
        optimal_num_features = np.argmax(mean_test_scores) + 1
        plt.plot(
            optimal_num_features, 
            np.max(mean_test_scores),
            'r*',
            markersize=15,
            label=f'Optimal Features: {optimal_num_features}'
        )
        
        # 그리드 추가
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 레이블과 제목
        plt.xlabel('Number of Features Selected', fontsize=12, fontweight='bold')
        plt.ylabel('Cross Validation Score (MAE)', fontsize=12, fontweight='bold')
        plt.title('Recursive Feature Elimination Cross Validation (RFECV)\nPerformance Analysis', 
                fontsize=14, fontweight='bold', pad=20)
        
        # 범례 추가
        plt.legend(loc='lower right', fontsize=10)
        
        # 여백 조정
        plt.tight_layout()
        
        # 그래프 저장 및 표시
        plt.savefig(
            os.path.join(fig_path, 'rfecv_performance.png'), 
            dpi=300, 
            bbox_inches='tight'
        )
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
        # 선택된 특성 반환
        selected_cols = X_train.columns[rfecv.support_].tolist()
        print(f'RFECV: Selected {len(selected_cols)} features: {selected_cols}')
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
            cv=3,
            n_jobs=-1
        )
        print(f'SFS Configuration: {sfs}')
        
        # 모델 학습
        sfs = sfs.fit(X_train, y_train)
        
        # 메트릭 데이터 추출
        metrics_dict = sfs.get_metric_dict()
        
        # 그래프 스타일 설정
        plt.style.use('seaborn')
        plt.figure(figsize=(12, 6))
        
        # 데이터 준비
        x_values = []
        y_means = []
        y_stds = []
        
        for k in sorted(metrics_dict.keys()):
            x_values.append(k)
            y_means.append(metrics_dict[k]['avg_score'])
            y_stds.append(metrics_dict[k]['std_dev'])
        
        # 메인 라인 플롯
        plt.plot(x_values, y_means, 
                marker='o', 
                markersize=8,
                linewidth=2,
                color='#2E86C1',
                label='Average Score')
        
        # 표준편차 영역 추가
        plt.fill_between(x_values,
                        [m - s for m, s in zip(y_means, y_stds)],
                        [m + s for m, s in zip(y_means, y_stds)],
                        alpha=0.2,
                        color='#2E86C1',
                        label='Standard Deviation')
        
        # 최적 포인트 강조
        best_k = max(metrics_dict.keys(), key=lambda k: metrics_dict[k]['avg_score'])
        best_score = metrics_dict[best_k]['avg_score']
        plt.plot(best_k, best_score,
                'r*',
                markersize=15,
                label=f'Optimal Features: {best_k}')
        
        # 그리드 및 레이블 설정
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Number of Features', fontsize=12, fontweight='bold')
        plt.ylabel('Cross Validation Score (MAE)', fontsize=12, fontweight='bold')
        plt.title('Sequential Feature Selection (SFS)\nPerformance Analysis',
                fontsize=14, fontweight='bold', pad=20)
        
        # 범례 설정
        plt.legend(loc='best', fontsize=10)
        
        # 여백 조정
        plt.tight_layout()
        
        # 그래프 저장 및 표시
        plt.savefig(
            os.path.join(fig_path, 'sfs_performance.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
        # 선택된 특성 처리
        selected_features_sfs = X_train.columns[list(sfs.k_feature_idx_)].tolist()
        print(f'Selected Features Score: {sfs.k_score_:.4f}')
        print(f'SFS: Selected {len(selected_features_sfs)} features: {selected_features_sfs}')
        
        return selected_features_sfs
   
class DataPrep:
    def __init__(self):
        pass
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
    def clean_column_names(df):
        # 하이픈을 언더바로 변경
        df.columns = df.columns.str.replace('-', '_', regex=False)
        # 알파벳, 숫자, 언더바를 제외한 특수 기호 제거
        df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)
        return df
    @staticmethod
    def prepare_test_data(X_test, model):
        """
        테스트 데이터를 예측을 위해 준비
        """
        # target 컬럼 제거
        if 'target' in X_test.columns:
            X_test = X_test.drop(['target'], axis=1)
        
        # 학습에 사용된 컬럼 확인
        train_columns = model.feature_names_in_
        
        # 누락된 컬럼 체크
        missing_cols = set(train_columns) - set(X_test.columns)
        if missing_cols:
            raise ValueError(f"테스트 데이터에 다음 컬럼이 없습니다: {missing_cols}")
        
        # 불필요한 컬럼 체크
        extra_cols = set(X_test.columns) - set(train_columns)
        if extra_cols:
            print(f"다음 컬럼은 예측에 사용되지 않습니다: {extra_cols}")
        
        # 학습에 사용된 컬럼만 선택하고 순서 맞추기
        X_test = X_test[train_columns]
        
        return X_test
    @staticmethod
    def get_unique_filename(filepath: str) -> str:
        """
        파일이 이미 존재할 경우 파일명_1, 파일명_2 등으로 변경
        
        Parameters:
        -----------
        filepath : str
            원본 파일 경로
        
        Returns:
        --------
        str : 유니크한 파일 경로
        """
        if not os.path.exists(filepath):
            return filepath
        
        # 파일 경로와 확장자 분리
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        
        # 새로운 파일명 생성
        counter = 1
        while True:
            new_filename = f"{name}_{counter}{ext}"
            new_filepath = os.path.join(directory, new_filename)
            
            if not os.path.exists(new_filepath):
                return new_filepath
            counter += 1
    @staticmethod
    def concat_train_test(dt, dt_test):
        Utils.remove_unnamed_columns(dt)
        Utils.remove_unnamed_columns(dt_test)
        dt = dt.reset_index(drop=True)
        dt_test = dt_test.reset_index(drop=True)
        dt['is_test'] = 0
        dt_test['is_test'] = 1
        
        # dt_test에 'target' 열 추가
        if 'target' not in dt_test.columns:
            dt_test['target'] = 0
        
        # 인덱스를 고유하게 설정
        dt.index = range(len(dt))
        dt_test.index = range(len(dt_test))
        
        print(f'dt.shape: {dt.shape}')
        print(f'dt_test.shape: {dt_test.shape}')
        
        concat = pd.concat([dt, dt_test], axis=0).reset_index(drop=True)
        print(concat['is_test'].value_counts())
        return concat
    @staticmethod
    def unconcat_train_test(concat):
        Utils.remove_unnamed_columns(concat)
        if 'is_test' not in concat.columns:
            raise ValueError("'is_test' 열이 데이터프레임에 없습니다.")
        # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
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
    vif_threshold = 10 
    cramer_v_threshold = 0.7
    min_freq_threshold = 0.05

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data')
    prep_path = os.path.join(data_path, 'preprocessed')
    fig_path = os.path.join(base_path, 'output', 'plots')
    #####

    group_arg = 'baseline'


    if group_arg == 'baseline':
        null_prep_method = 'baseline'
        encode_method = 'baseline'#'freq'
        print(f'##### Null prep: Baseline\nEncode: {encode_method}')
    elif group_arg == 'advanced':
        null_prep_method = 'advanced'
        encode_method = 'freq'
        print(f'##### Null prep: Advanced\nEncode: {encode_method}')
    prep_out_path = os.path.join(prep_path, group_arg)
    os.makedirs(prep_out_path, exist_ok=True)
    path_raw = os.path.join(prep_path, 'df_raw.csv')
    
    path_feat = os.path.join(prep_path, 'feat_concat_raw.csv')

    path_out_null_prep = os.path.join(prep_out_path, 'df_null-preped.csv')
    path_out_encoded = os.path.join(prep_out_path, 'df_null-preped-encoded.csv')

    
    cols_to_remove = ['등기신청일자', '거래유형', '중개사소재지'] +['홈페이지','k-전화번호', 'k-팩스번호', '고용보험관리번호']
    cols_to_str = ['본번', '부번'] + ['구', '동', '강남여부', '신축여부', 'cluster_dist_transport', 'cluster_dist_transport_count', 'cluster_select','subway_zone_type', 'bus_zone_type']
    cols_id = ['is_test', 'target']
    cols_add = ['시군구', '번지']

    cols_date = ['단지신청일', '단지승인일','k-사용검사일-사용승인일']
    cols_to_num = []#['좌표X', '좌표Y', '위도', '경도']
    #cols_to_select = ['시군구', '전용면적', '계약년월', '건축년도']
    
    path_feat = os.path.join(prep_path, 'feat_concat_raw.csv')
    df_feat = pd.read_csv(path_feat)
    df_feat = df_feat.loc[:, ~df_feat.columns.str.contains('^Unnamed')]
    #df = pd.read_csv(os.path.join(prep_path, 'df_raw_null_prep_coord.csv'))
    df_raw = pd.read_csv(path_raw)
    df_raw = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed')]
    df_raw['시군구+번지'] = df_raw['시군구'].astype(str) + df_raw['번지'].astype(str)
    print(f'시군구 번지 컬럼 생성: {df_raw.columns}')

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
    flag=False
    if flag:#os.path.exists(path_out_null_prep):
        df_interpolated = pd.read_csv(path_out_null_prep, index_col=0)
        df_interpolated = Utils.remove_unnamed_columns(df_interpolated)
        print(f'##### Load null prep: {path_out_null_prep}')
        print(df_interpolated.columns, df_interpolated.shape, df_interpolated.isnull().sum())
    else:
        print(f'##### Null prep: Interpolation for Null values')
        cols_to_remove = [col for col in cols_to_remove if col in df_raw.columns]
        df_raw.drop(columns=cols_to_remove, inplace=True)
        print(f'#####\nRemove columns: {len(cols_to_remove)}\n{cols_to_remove}\n###')
        df_null_removed = DataPrep.remove_null(df_raw, threshold_null)
        cols_to_str = [col for col in cols_to_str if col in df_null_removed.columns]
        cols_to_num = [col for col in cols_to_num if col in df_null_removed.columns]
        df_null_removed = DataPrep.convert_dtype(df_null_removed, cols_to_str, cols_to_num)
        continuous_columns, categorical_columns = Utils.categorical_numeric(df_null_removed)
        
        group_cols = ['시군구', '번지', '아파트명'] # 도로명
        df_columns = set(df_null_removed.columns)

        # 리스트에서 데이터프레임에 없는 컬럼 찾기
        missing_columns = [col for col in group_cols if col not in df_columns]
        if missing_columns:
            print("Missing columns:", missing_columns)
        if null_prep_method == 'advanced':
            df_null_removed = DataPrep.prep_null_advanced(df_null_removed, continuous_columns, categorical_columns, group_cols=missing_columns)
            path_out_null_prep = os.path.join(prep_out_path, 'df_feat_null-preped_advanced.csv')
        elif null_prep_method == 'baseline':
            path_out_null_prep = os.path.join(prep_out_path, 'df_feat_null-preped_baseline.csv')
        df_interpolated = DataPrep.prep_null(df_null_removed, continuous_columns, categorical_columns)
        print(df_interpolated.columns, df_interpolated.shape, df_interpolated.isnull().sum())
        df_interpolated.to_csv(path_out_null_prep)


    continuous_columns, categorical_columns = Utils.categorical_numeric(df_interpolated)
    vif = FeatureSelect.calculate_vif(df_interpolated, continuous_columns, vif_threshold)
    cramers_v_pairs, cramers_features_to_drop = FeatureSelect.cramers_v_all(df_interpolated, categorical_columns, cramer_v_threshold)
        
## Encode categorical variables
    cols_exclude = ['target', 'is_test']
    df_train, df_test = Utils.unconcat_train_test(df_interpolated)
    y_train = df_train['target']
    X_train = df_train.drop(columns=['target'])
    X_test = df_test
    

    if os.path.exists(path_out_encoded):
        concat = pd.read_csv(path_out_encoded, index_col=0)
        concat = Utils.remove_unnamed_columns(concat)
        print(f'##### Load encoded: {path_out_encoded}')
        continuous_columns, categorical_columns = Utils.categorical_numeric(concat)
    else:
        if encode_method == 'freq':
           
            min_freq_dict = DataPrep.auto_adjust_min_frequency(X_train, base_threshold=min_freq_threshold)
            X_train_cat = X_train[categorical_columns]
            X_test_cat = X_test[categorical_columns]
            X_train_cat_encoded, X_test_cat_encoded = DataPrep.frequency_encode(X_train_cat, X_test_cat, min_freq_dict)
        elif encode_method == 'baseline':
            X_train_cat_encoded, X_test_cat_encoded, label_encoders = DataPrep.encode_label(X_train, X_test, categorical_columns)

        X_train = pd.concat([X_train_cat_encoded, X_train.drop(columns=categorical_columns)], axis=1)
        X_test = pd.concat([X_test_cat_encoded, X_test.drop(columns=categorical_columns)], axis=1)
        X_train['target'] = y_train
        concat = Utils.concat_train_test(X_train, X_test)
        concat.to_csv(path_out_encoded)
        
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
    #n_sample = 10000
    ratio_sample = 0.05
    random_state =2023
    k = 20 #kbest #f_regression for num, mutual_info_classif for cat
    n_resample = int(len(X_train)* ratio_sample)
    print(f'N resample: {n_resample}, ratio {ratio_sample}')
    X_sampled, y_sampled = resample(X_train, y_train, 
                                            n_samples=n_resample,
                                            random_state=random_state)
    # #상위 K개 특성만 먼저 선택
    X_cat =X_sampled[categorical_columns]
    X_num = X_sampled[continuous_columns]
    
    X_cat, kbest_features_cat = FeatureSelect.select_features_by_kbest(X_cat, y_sampled, X_cat.columns, mutual_info_classif, k=20)
    X_num, kbest_features_num = FeatureSelect.select_features_by_kbest(X_num, y_sampled, X_num.columns, f_regression, k=20)
    
    #rf = RandomForestRegressor(random_state=2023)
    # XGBRegressor 모델 생성
    config = config_baseline.get('parameters')
    
    pprint.pprint(config)
    model = xgb.XGBRegressor(
                eta=config.xgboost_eta,
                max_depth=config.xgboost_max_depth,
                subsample=config.xgboost_subsample,
                colsample_bytree=config.xgboost_colsample_bytree,
                gamma=config.xgboost_gamma,
                reg_lambda=config.xgboost_reg_lambda,  
                reg_alpha=config.xgboost_alpha,
            )
    #rf = Ridge(alpha=1.0)
    selected_rfe, selected_sfs = FeatureSelect.wrapper_method(X_sampled, y_sampled, model, fig_path)
    common_features, union_features, rest_features = FeatureSelect.compare_selected_features(selected_rfe, selected_sfs, ['RFE', 'SFS'], original_column_names)
  
    dict_result = {'vif': vif, 
                   'cramers_v': cramers_features_to_drop,
                   #'kbest_features': kbest_features,
                   'filter_common_features': filter_common_features,
                   'filter_union_features': filter_union_features,
                   'filter_rest_features': filter_rest_features,
                   'wrapper_common_features': common_features,
                   'wrapper_union_features': union_features,
                   'wrapper_rest_features': rest_features,
                   'selected_rfe': selected_rfe,
                   'selected_sfs': selected_sfs
                   }
    print(dict_result)
    
    # JSON으로 저장
    with open(os.path.join(prep_path, 'dict_feature_selection_result.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_result, f, indent=4, ensure_ascii=False)

    # # 필요한 경우 JSON 파일 읽기
    # with open(os.path.join(prep_path, 'dict_feature_selection_result.json'), 'r', encoding='utf-8') as f:
    #     dict_result = json.load(f)
if __name__ == '__main__':
    main()


# threshold_null = 0.9

# min_freq_threshold = 0.05
# threshold_corr = 0.8
# threshold_var = 0.1