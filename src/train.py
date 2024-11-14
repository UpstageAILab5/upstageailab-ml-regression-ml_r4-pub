import wandb
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn import metrics
import os
import pandas as pd
import yaml
from src.feature import FeatureSelect, FeatureEngineer
from src.utils import Utils
from src.preprocess import DataPrep
import sys
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# 현재 파일의 경로를 기준으로 상위 디렉토리 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
print(f'###### Parent dir: {parent_dir}')

#from config.configs import config_xgb, config_lightgbm, config_catboost, config_random_forest
# module_path = os.path.dirname(os.path.abspath(__file__))
# print(module_path)
# # 모듈 검색 경로에 추가
# if module_path not in sys.path:
#     sys.path.append(module_path)

# 사용할 데이터셋을 로드하고 분할
# datasets = {
#     "boston": load_boston(),
#     "diabetes": load_diabetes()
# }
random_seed = 2024


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def cross_validate_and_evaluate(model, X, y, scale_data, random_seed):
        """
        교차 검증 및 모델 평가
        Parameters:
            model: 학습할 모델
            X: 특성 데이터프레임
            y: 타겟 시리즈
        """
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        rmse_scores = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
 
            # 데이터 분할
            X_train = X.iloc[train_index]
            X_val = X.iloc[val_index]
            y_train = y.iloc[train_index]
            y_val = y.iloc[val_index]
            
            # 분할된 데이터셋 shape 출력
            print(f"Train set shape - X: {X_train.shape}, y: {y_train.shape}")
            print(f"Val set shape   - X: {X_val.shape}, y: {y_val.shape}")
            
            # 모델 학습
            print("모델 학습 중...")
            model.fit(X_train, y_train)
            
            # 예측 및 RMSE 계산
            val_pred = model.predict(X_val)
            if scale_data == 'log_transform_target':
                val_pred = np.exp(val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            rmse_scores.append(rmse)
            
            print(f'Fold {fold} RMSE: {rmse:,.2f}')
            
            # 메모리 정리
            # del X_train, X_val, y_train, y_val, val_pred
            # import gc
            # gc.collect()
        
        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        
        print("\n=== 교차 검증 결과 ===")
        print(f'평균 RMSE: {mean_rmse:,.2f} (±{std_rmse:,.2f})')
        print(f'개별 RMSE: {[f"{score:,.2f}" for score in rmse_scores]}')
        
        return model, mean_rmse
def train_model(config=None):
    
    with wandb.init() as run:
    
        config_random_forest = {
                'parameters': {
                    'random_seed': -1,
                    'random_forest_n_estimators': 5,
                    'random_forest_criterion': 'squared_error',
                    'random_forest_random_state': 1,
                    'random_forest_n_jobs': -1
                }
            }
        config_xgb = {
        
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
        config_lightgbm = {
                'parameters': {

                    'lightgbm_n_estimators': 300,
                    'lightgbm_learning_rate': 0.1,
                    'lightgbm_num_leaves': 64,
                    'lightgbm_max_depth': 7,
                    'lightgbm_feature_fraction': 0.8,
                    'lightgbm_bagging_fraction': 0.8,
                    'lightgbm_colsample_bytree': 0.8,
                    'lightgbm_subsample': 0.8,
                    'lightgbm_min_data_in_leaf': 20,
                    'lightgbm_lambda_l1': 0.1,
                    'lightgbm_lambda_l2': 0.1,
                }
        }

        config_catboost = { 

                'parameters': {
                    'catboost_iterations': 500,
                    'catboost_depth': 6,
                    'catboost_learning_rate': 0.1,
                    'catboost_l2_leaf_reg': 3.0,
                    'catboost_bagging_temperature': 0.5,
                    'catboost_random_strength': 2.0,
                    'catboost_border_count': 128,
                    'catboost_boosting_type': 'Plain'
                }
        }
        config_xgb = config_xgb['parameters']
        config_lightgbm = config_lightgbm['parameters']
        config_catboost = config_catboost['parameters']
        config_random_forest = config_random_forest['parameters']
        

        config = wandb.config
        
        null_preped = config.null_preped 
        outlier_removal = config.outlier_removal
        features = config.features
        # feature_engineer = config.feature_engineer
        categorical_encoding = config.categorical_encoding
        split_type = config.split_type
        model_name = config.model
        scale_data = config.scale_data
        # 모델 선택
        print(f'\n#### Model: {model_name} ####')
        print(f'- Features: {features}\n- Null_preped: {null_preped}\n- Outlier_removal: {outlier_removal}\n- Categorical_encoding: {categorical_encoding}\n- Split type:{split_type}\n- Scale data: {scale_data}\n')
        # 모델 훈련 및 평가
        
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_path, 'data')
        out_path = os.path.join(base_path, 'output')
        plot_path = os.path.join(out_path, 'plots')
        prep_path = os.path.join(data_path, 'preprocessed')

        df = pd.read_csv(os.path.join(prep_path, 'df_raw.csv'))
        df_feat = pd.read_csv(os.path.join(prep_path, 'feat_concat_raw.csv'))
        # df_feat['시군구+번지'] = df['시군구'] + ' ' + df['번지']
        # df_feat.to_csv(os.path.join(prep_path, 'feat_concat_raw.csv'), index=False)
        df = pd.concat([df, df_feat], axis=1)
        # df_feat=Utils.remove_unnamed_columns(df_feat)
        df = Utils.remove_unnamed_columns(df)
        print(f'##### Features:{len(df.columns)} {df.columns}')
        cols_id = ['is_test', 'target']
        
        cols = ['시군구', '전용면적', '계약년월', '건축년도', '층', '도로명', '아파트명']
        cols_manual = ['전용면적', '계약년', '계약월', '구', '동', '층', '시군구+번지', '건축년도']
        cols_feat = ['신축여부', '구', '강남여부']
        cols_feat_common = ['k-난방방식', '전용면적', '좌표Y','좌표X','bus_direct_influence_count', 'subway_zone_type', '건축년도', '계약년월','k-수정일자', 'k-연면적']
        cols_feat2= ['주차대수','대장아파트_거리','subway_shortest_distance','bus_shortest_distance', 'cluster_dist_transport' , 'subway_direct_influence_count', 'subway_indirect_influence_count']
        
        cols_to_remove = ['등기신청일자', '거래유형', '중개사소재지'] 
        cols_to_remove_manual = ['홈페이지','k-전화번호', 'k-팩스번호', '고용보험관리번호']
        cols_to_remove_cramer =['bus_zone_type',
                            'cluster_dist_transport',
                            'k-건설사(시공사)',
                            'k-관리방식',
                            'k-난방방식',
                            'k-세대타입(분양형태)',
                            'k-수정일자',
                            'k-시행사',
                            '강남여부',
                            '관리비 업로드',
                            '구',
                            '도로명',
                            '번지',
                            '본번',
                            '사용허가여부',
                            '세대전기계약방법',
                            '시군구',
                            '아파트명',
                            '청소비관리형태']
        
        cols_to_remove_all = cols_to_remove + cols_to_remove_manual + cols_to_remove_cramer
        cols_to_remove_all = list(set(cols_to_remove_all))
        print(f'\n##### Select Features...\n')
        if features == 'baseline':
            cols_to_remove = [col for col in cols_to_remove if col in df.columns]
            cols_to_remove = list(set(cols_to_remove))
            
            df = df.drop(columns=cols_to_remove)
            print(f'Baseline number of feature: {len(df.columns)}/{len(df.columns)}')
        elif features == 'manual':
            cols_total = cols_id + cols_manual
            cols_total = [col for col in cols_total if col in df.columns]
            cols_total = list(set(cols_total))
            print(f'Manual number of feature: {len(cols_total)}/{len(df.columns)}')
            df = df[cols_total]
        elif features == 'minimum':
            cols_total = cols_id + cols + cols_feat +cols_feat_common
            cols_total = list(set(cols_total))
            cols_total = [col for col in cols_total if col in df.columns]
            print(f'Minimum number of feature: {len(cols_total)}/{len(df.columns)}')
            df = df[cols_total]
        elif features == 'features_all':
            cols_total = cols_id + cols + cols_feat + cols_feat2
            cols_total = [col for col in cols_total if col in df.columns]
            cols_total = list(set(cols_total))
            print(f'All number of feature: {len(cols_total)}/{len(df.columns)}')
            df = df[cols_total]
        elif features == 'remove_all':
            cols_to_remove_all = [col for col in cols_to_remove_all if col in df.columns]
            cols_to_remove_all = list(set(cols_to_remove_all))
            
            df = df.drop(columns=cols_to_remove_all)
            print(f'Remove all - Rest features: {len(df.columns)}/{len(df.columns)}')
        elif features =='wrapper':
            cols_total = cols_id + cols + cols_feat + cols_feat2
            cols_total = [col for col in cols_total if col in df.columns]
            cols_total = list(set(cols_total))
            print(f'Wrapper method - number of feature: {len(cols_total)}/{len(df.columns)}')
            df = df[cols_total]
        print(f'\n##### Feature selected: {df.shape}\n{df.columns}')
        print(f'##### Prep null...')
        df = DataPrep.remove_null(df)
        continuous_columns, categorical_columns = Utils.categorical_numeric(df)
        if null_preped =='baseline':
            df = DataPrep.prep_null(df, continuous_columns, categorical_columns)
        elif null_preped =='grouped':
            df = DataPrep.prep_null_advanced(df, continuous_columns, categorical_columns)
    #####
        if df.isnull().sum().any():
            print('##### Null values still exist. Basic prep_null...')
            df = DataPrep.prep_null(df, continuous_columns, categorical_columns)

        if outlier_removal == 'baseline':
            df = DataPrep.remove_outliers_iqr(df, '전용면적')
        elif outlier_removal == 'none':
            print('No outlier removal')
        elif outlier_removal == 'iqr_modified':
            df = DataPrep.remove_outliers_iqr(df, '전용면적', modified=True)
        feat_eng = FeatureEngineer()
        #####

        # if feature_engineer=='baseline':
        #     df = feat_eng.prep_feat(df)
        # elif feature_engineer == 'year_2020':
        #     df = feat_eng.prep_feat(df, year = 2020)
        # elif feature_engineer == 'address':
        #     df = feat_eng.prep_feat(df, year = 2020, col_add='address')

       
#####
        
        #df = Utils.clean_column_names(df) # 컬럼 문자열 기호 제거
        # 피처 이름 중복 확인 및 해결
        duplicated_columns = df.columns[df.columns.duplicated()]
        min_freq_threshold = 0.05
        # 중복된 피처 이름 출력
        if duplicated_columns.any():
            print("중복된 피처 이름이 있습니다:", duplicated_columns.tolist())
            df.columns = [f"feature_{i}" for i in range(df.shape[1])]
        continuous_columns, categorical_columns = Utils.categorical_numeric(df)
        Utils.remove_unnamed_columns(df)
        df_train, X_test= Utils.unconcat_train_test(df)
        y_train = df_train['target']
        X_train = df_train.drop(columns=['target'])

        if categorical_encoding == 'baseline':
            X_train_encoded, X_test_encoded, label_encoders = DataPrep.encode_label(X_train, X_test, categorical_columns)
        elif categorical_encoding == 'freq':
            min_freq_dict = DataPrep.auto_adjust_min_frequency(X_train, base_threshold=min_freq_threshold)
            X_train_encoded_cat = X_train[categorical_columns]
            X_test_encoded_cat = X_test[categorical_columns]
            X_train_encoded, X_test_encoded = DataPrep.frequency_encode(X_train_encoded_cat, X_test_encoded_cat, min_freq_dict)
        elif categorical_encoding == 'target_encoding':
            X_train['target'] = y_train
            X_train_encoded, X_test_encoded = DataPrep.target_encoding_all(X_train, X_test, categorical_columns, 'target')
            X_train.drop(columns=['target'], inplace=True)
        else:
            print('No categorical encoding')

        if scale_data == 'baseline':
            print('No scaling')
        elif scale_data == 'scaling':
            X_train_encoded, X_test_encoded = DataPrep.transform_and_visualize(X_train_encoded, X_test_encoded, continuous_columns, output_dir=plot_path, skew_threshold=0.5)
        elif scale_data == 'log_transform_target':
            print('Log scaling')
            y_train = np.log(y_train)

        if model_name == "xgboost":
            
            model = xgb.XGBRegressor(
                eta=config_xgb['xgboost_eta'],
                max_depth=config_xgb['xgboost_max_depth'],
                subsample=config_xgb['xgboost_subsample'],
                colsample_bytree=config_xgb['xgboost_colsample_bytree'],
                gamma=config_xgb['xgboost_gamma'],
                reg_lambda=config_xgb['xgboost_reg_lambda'],  
                reg_alpha=config_xgb['xgboost_alpha'],
                n_estimators = config_xgb['xgboost_n_estimators'],
                #early_stopping_rounds=50
            )

        elif model_name == "lightgbm":
            
            model = lgb.LGBMRegressor(
                learning_rate=config_lightgbm['lightgbm_learning_rate'],
                num_leaves=config_lightgbm['lightgbm_num_leaves'],
                max_depth=config_lightgbm['lightgbm_max_depth'],
                min_data_in_leaf=config_lightgbm['lightgbm_min_data_in_leaf'],
                feature_fraction=config_lightgbm['lightgbm_feature_fraction'],
                bagging_fraction=config_lightgbm['lightgbm_bagging_fraction'],
                lambda_l1=config_lightgbm['lightgbm_lambda_l1'],
                lambda_l2=config_lightgbm['lightgbm_lambda_l2'],
                #early_stopping_rounds=50
            )
        elif model_name == "random_forest":
            
            model = RandomForestRegressor(
                n_estimators=config_random_forest['random_forest_n_estimators'],
                n_jobs=config_random_forest['random_forest_n_jobs'],
                random_state=config_random_forest['random_forest_random_state'],
                criterion=config_random_forest['random_forest_criterion']
        
            )
        elif model_name == "catboost":
            
            model = CatBoostRegressor(
                iterations=config_catboost['catboost_iterations'],
                depth=config_catboost['catboost_depth'],
                learning_rate=config_catboost['catboost_learning_rate'],
                l2_leaf_reg=config_catboost['catboost_l2_leaf_reg'],
                bagging_temperature=config_catboost['catboost_bagging_temperature'],
                verbose=False, 
               # early_stopping_rounds=50
            )
        else:
            raise ValueError("Unsupported model type")
        X_train, feature_mapping = Utils.clean_feature_names(X_train_encoded)
        X_test = X_test.rename(columns=feature_mapping)
        if split_type == 'kfold':
            model, rmse_avg = cross_validate_and_evaluate(model, X_train, y_train, scale_data, random_seed)
            print(f'kfold: mean RMSE for val data set: {rmse_avg}')
            rmse = rmse_avg
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed)
            pred = model.predict(X_val)
        elif split_type == 'holdout':
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed)
            model.fit(X_train, y_train)#, early_stopping_rounds=50,verbose=100)
        
            pred = model.predict(X_val)
            if scale_data == 'log_transform_target':
                pred = np.exp(pred)
            rmse = np.sqrt(metrics.mean_squared_error(y_val, pred))
        wandb.log({"rmse": rmse})
        wandb.finish()
        
        # X_test_encoded = Utils.prepare_test_data(X_test, model)
        # real_test_pred = model.predict(X_test)

        # preds_df = pd.DataFrame(real_test_pred.astype(int), columns=["target"])
        # out_pred_path = Utils.get_unique_filename(os.path.join(out_path,f'output_{features}_{outlier_removal}_{categorical_encoding}_{feature_engineer}_{model_name}_{split_type}.csv'))
        # #preds_df.to_csv(out_pred_path, index=False)
