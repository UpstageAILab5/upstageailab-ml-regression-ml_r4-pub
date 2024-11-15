import os
import json

import scipy.stats as stats
import pandas as pd
import pprint

from time import time
import psutil
import os
from sklearn.utils import resample
from tqdm import tqdm
import xgboost as xgb
from typing import List
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_regression, mutual_info_classif
# SettingWithCopyWarning 경고 무시
pd.options.mode.chained_assignment = None  # default='warn'
import sys
from pathlib import Path
# current = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(f'\n##########current: {current}')
# sys.path.append(str(current))
# from config.config import config_baseline
from src.feature import FeatureEngineer, FeatureSelect
from src.utils import Utils
from src.preprocess import DataPrep
# df.loc[some_condition, 'column'] = new_value
def main():
    threshold_null = 0.9
    vif_threshold = 10 
    cramer_v_threshold = 0.7
    min_freq_threshold = 0.05
    ratio_sample = 0.02
    random_state =2023
    k = 20 #kbest #f_regression for num, mutual_info_classif for cat
    cols_id = ['is_test', 'target']
    group_cols = ['시군구', '번지', '아파트명'] # 도로명
    cols_to_remove = ['등기신청일자', '거래유형', '중개사소재지'] +['홈페이지','k-전화번호', 'k-팩스번호', '고용보험관리번호']
    cols_to_str = ['본번', '부번'] + ['구', '동', '강남여부', '신축여부', 'cluster_dist_transport', 'cluster_dist_transport_count', 'cluster_select','subway_zone_type', 'bus_zone_type']
    cols_date = ['단지신청일', '단지승인일','k-사용검사일-사용승인일']
    cols_to_num = []#['좌표X', '좌표Y', '위도', '경도']
    #cols_to_select = ['시군구', '전용면적', '계약년월', '건축년도']

    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'data')
    prep_path = os.path.join(data_path, 'preprocessed')
    fig_path = os.path.join(base_path, 'output', 'plots')
    #####
    group_arg = 'baseline'#'advanced'

    if group_arg == 'baseline':
        null_prep_method = 'baseline'
        encode_method = 'baseline' #label encoding
        print(f'##### Null prep: Baseline\nEncode: {encode_method}')
    elif group_arg == 'advanced':
        null_prep_method = 'advanced'
        encode_method = 'target_encoding' #'freq
        print(f'##### Null prep: Advanced\nEncode: {encode_method}')
    prep_out_path = os.path.join(prep_path, group_arg)
    os.makedirs(prep_out_path, exist_ok=True)
    path_raw = os.path.join(prep_path, 'df_raw.csv')
    path_feat = os.path.join(prep_path, 'feat_concat_raw.csv')
    path_out_null_prep = os.path.join(prep_out_path, f'df_null-preped_{null_prep_method}.csv')
    path_out_encoded = os.path.join(prep_out_path, f'df_null-preped-encoded_{encode_method}.csv')
    
    path_feat = os.path.join(prep_path, 'feat_concat_raw.csv')
    df_feat = pd.read_csv(path_feat)
    df_feat = df_feat.loc[:, ~df_feat.columns.str.contains('^Unnamed')]
    #df = pd.read_csv(os.path.join(prep_path, 'df_raw_null_prep_coord.csv'))
    df_raw = pd.read_csv(path_raw)
    df_raw = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed')]

    df_raw['시군구+번지'] = df_raw['시군구'].astype(str) + ' '+  df_raw['번지'].astype(str)
    print(f'시군구 번지 컬럼 생성: {df_raw.columns}')

    #df_raw = df_raw.loc[:, cols_to_select + cols_id]
    df_raw = pd.concat([df_raw, df_feat], axis=1)
    print(f'\n##################df_raw.columns: {df_raw.columns}')
    Utils.chk_train_test_data(df_raw)
    for col in cols_date:
        df_raw[col] = pd.to_datetime(df_raw[col])
        df_raw[col] = df_raw[col].view('int64') / 10**9# 나노초 단위이므로 초 단위로 변환
    
##### Null prep: Interpolation for Null values
    continuous_columns, categorical_columns = Utils.categorical_numeric(df_raw)
    flag=False
    if os.path.exists(path_out_null_prep):
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
        
        
        df_columns = set(df_null_removed.columns)

        # 리스트에서 데이터프레임에 없는 컬럼 찾기
        missing_columns = [col for col in group_cols if col not in df_columns]
        if missing_columns:
            print("Missing columns:", missing_columns)
        if df_null_removed.isnull().any().any():
            print(f'##### Null value found. prep: {null_prep_method}')
            if null_prep_method == 'advanced':
                df_null_removed = DataPrep.prep_null_advanced(df_null_removed, continuous_columns, categorical_columns, group_cols=missing_columns)
                path_out_null_prep = os.path.join(prep_out_path, 'df_feat_null-preped_advanced.csv')
            elif null_prep_method == 'baseline':
                path_out_null_prep = os.path.join(prep_out_path, 'df_feat_null-preped_baseline.csv')
            df_interpolated = DataPrep.prep_null(df_null_removed, continuous_columns, categorical_columns)
            print(df_interpolated.columns, df_interpolated.shape, df_interpolated.isnull().sum())
        df_interpolated.to_csv(path_out_null_prep)
    
    continuous_columns, categorical_columns = Utils.categorical_numeric(df_interpolated)
    cramers_v_pairs, cramers_features_to_drop = FeatureSelect.cramers_v_all(df_interpolated, categorical_columns, cramer_v_threshold)

    vif = FeatureSelect.calculate_vif(df_interpolated, continuous_columns, vif_threshold)
## Encode categorical variables
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
            print('\n##### Frequency Encoding\n')
            min_freq_dict = DataPrep.auto_adjust_min_frequency(X_train, base_threshold=min_freq_threshold)
            X_train_cat = X_train[categorical_columns]
            X_test_cat = X_test[categorical_columns]
            X_train_cat_encoded, X_test_cat_encoded = DataPrep.frequency_encode(X_train_cat, X_test_cat, min_freq_dict)
        elif encode_method == 'baseline':
            print('\n##### Label Encoding\n')
            X_train_cat_encoded, X_test_cat_encoded, label_encoders = DataPrep.encode_label(X_train, X_test, categorical_columns)
        elif encode_method == 'target_encoding':
            print('\n##### Target Encoding\n')
            X_train_cat_encoded, X_test_cat_encoded = DataPrep.target_encoding_all(X_train, X_test, categorical_columns, 'target')

        # 인덱스 재설정
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        # 열 일치 확인
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

    vif_total = FeatureSelect.calculate_vif(X_train, concat.columns, vif_threshold)
    cols_var, cols_corr = FeatureSelect.filter_method(X_train, X_test, continuous_columns, categorical_columns)

    filter_common_features, filter_union_features, filter_rest_features = FeatureSelect.compare_selected_features(cols_var, cols_corr, ['Variance Threshold', 'Correlation Threshold'], original_column_names)
##### Resampling for Feature Selection  
    #n_sample = 10000

    n_resample = int(len(X_train)* ratio_sample)
    print(f'N resample: {n_resample}, ratio {ratio_sample}')
    X_sampled, y_sampled = resample(X_train, y_train, 
                                            n_samples=n_resample,
                                            random_state=random_state)
    # #상위 K개 특성만 먼저 선택
    categorical_columns = [col for col in categorical_columns if col in X_sampled.columns]
    continuous_columns = [col for col in continuous_columns if col in X_sampled.columns]
    X_cat =X_sampled[categorical_columns]
    X_num = X_sampled[continuous_columns]
    
    X_cat, kbest_features_cat = FeatureSelect.select_features_by_kbest(X_cat, y_sampled, X_cat.columns, mutual_info_classif, k=k)
    X_num, kbest_features_num = FeatureSelect.select_features_by_kbest(X_num, y_sampled, X_num.columns, f_regression, k=k)
    
    #rf = RandomForestRegressor(random_state=2023)
    # XGBRegressor 모델 생성
  
    config = {
 
        'parameters': {
       
            'xgboost_eta': 0.3,
            'xgboost_max_depth': 10,
            'xgboost_n_estimators': 500,
            'xgboost_subsample': 0.6239,
            'xgboost_colsample_bytree': 0.63995,
            'xgboost_gamma': 2.46691,
            'xgboost_alpha': 1.12406,
            'xgboost_reg_alpha': 0.1,
            'xgboost_reg_lambda': 5.081,
            
        }
    }
    config = config['parameters']
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
    selected_rfe, selected_sfs = FeatureSelect.wrapper_method(X_cat, X_num, y_sampled, model, fig_path)
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