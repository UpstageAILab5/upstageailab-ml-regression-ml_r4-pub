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
from feature_selection import DataPrep, FeatureEngineer, Utils
random_seed = 2024

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def cross_validate_and_evaluate(model, X, y, random_seed):
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
        config = wandb.config
        dataset_name = config.dataset_name
        features = config.features
        split_type = config.split_type
        model_name = config.model
        outlier_removal = config.outlier_removal
        categorical_encoding = config.categorical_encoding
        feature_engineer = config.feature_engineer

        # 모델 선택
        print(f'\n#### Model: {model_name} ####\n')
        print(f'- Data: {dataset_name}\n- Features: {features}\n- Split type:{split_type}\n')
        

        # 모델 훈련 및 평가
        
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_path, 'data')
        out_path = os.path.join(base_path, 'output')
        prep_path = os.path.join(data_path, 'preprocessed')

        if dataset_name == 'baseline':
            df = pd.read_csv(os.path.join(prep_path, 'df_raw.csv')) 
        elif dataset_name == 'encoded':
            df = pd.read_csv(os.path.join(prep_path, 'df_encoded.csv')) 
        elif dataset_name == 'feat_null-preped_freq-encoded':
            df = pd.read_csv(os.path.join(prep_path, 'df_feat_null-preped_freq-encoded.csv')) 
        elif dataset_name == 'null-preped_freq-encoded':
            df = pd.read_csv(os.path.join(prep_path, 'df_null-preped_freq-encoded.csv')) 
        df = Utils.remove_unnamed_columns(df)
        cols_id = ['is_test', 'target']
        cols_to_remove = ['등기신청일자', '거래유형', '중개사소재지'] 
        
        cols = ['시군구', '전용면적', '계약년월', '건축년도', '층', '도로명', '아파트명']
        
        cols_feat = ['신축여부', '구', '강남여부']
        cols_feat_common = ['k-난방방식', '전용면적', '좌표Y','좌표X','bus_direct_influence_count', 'subway_zone_type', '건축년도', '계약년월','k-수정일자', 'k-연면적']
        cols_feat2= ['주차대수','대장아파트_거리','subway_shortest_distance','bus_shortest_distance', 'cluster_dist_transport' , 'subway_direct_influence_count', 'subway_indirect_influence_count']

        if features == 'baseline':
            cols_to_remove = [col for col in cols_to_remove if col in df.columns]
            print(f'Baseline feature: {len(cols_to_remove)}\n{cols_to_remove}')
            df = df.drop(columns=cols_to_remove)
        elif features == 'removed':
            cols_to_remove+=['홈페이지','k-전화번호', 'k-팩스번호', '고용보험관리번호']
            cols_to_remove = [col for col in cols_to_remove if col in df.columns]
            print(f'Remove_add_feature: {len(cols_to_remove)}\n{cols_to_remove}')
            df = df.drop(columns=cols_to_remove)
        elif features == 'minimum':
            cols_total = cols_id + cols + cols_feat +cols_feat_common
            print(f'Minimum_feature: {len(cols_to_remove)}\n{cols_to_remove}')
            cols_total = [col for col in cols_total if col in df.columns]
            df = df[cols_total]
        elif features == 'medium':
            cols_total = cols_id + cols + cols_feat + cols_feat2
            cols_total = [col for col in cols_total if col in df.columns]
            df = df[cols_total]

        if outlier_removal == 'baseline':
            df = DataPrep.remove_outliers_iqr(df, '전용면적')
        elif outlier_removal == 'none':
            print('No outlier removal')
        elif outlier_removal == 'iqr_modified':
            df = DataPrep.remove_outliers_iqr(df, '전용면적', modified=True)
        feat_eng = FeatureEngineer()
        if feature_engineer=='baseline':
            
            df = feat_eng.prep_feat(df)
        elif feature_engineer == 'year_2020':
            df = feat_eng.prep_feat(df, year = 2020)
        elif feature_engineer == 'address':
           
            df = feat_eng.prep_feat(df, year = 2020, col_add='address')
        
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
        else:
            print('No categorical encoding')

        if model_name == "xgboost":
            model = xgb.XGBRegressor(
                eta=config.xgboost_eta,
                max_depth=config.xgboost_max_depth,
                subsample=config.xgboost_subsample,
                colsample_bytree=config.xgboost_colsample_bytree,
                gamma=config.xgboost_gamma,
                reg_lambda=config.xgboost_reg_lambda,  
                reg_alpha=config.xgboost_alpha,
                #early_stopping_rounds=50
            )

        elif model_name == "lightgbm":
            model = lgb.LGBMRegressor(
                learning_rate=config.lightgbm_learning_rate,
                num_leaves=config.lightgbm_num_leaves,
                max_depth=config.lightgbm_max_depth,
                min_data_in_leaf=config.lightgbm_min_data_in_leaf,
                feature_fraction=config.lightgbm_feature_fraction,
                bagging_fraction=config.lightgbm_bagging_fraction,
                lambda_l1=config.lightgbm_lambda_l1,
                lambda_l2=config.lightgbm_lambda_l2,
                #early_stopping_rounds=50
            )
        elif model_name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=config.random_forest_n_estimators,
                n_jobs=config.random_forest_n_jobs,
                random_state=config.random_forest_random_state,
                criterion=config.random_forest_criterion
        
            )
        elif model_name == "catboost":
            model = CatBoostRegressor(
                iterations=config.catboost_iterations,
                depth=config.catboost_depth,
                learning_rate=config.catboost_learning_rate,
                l2_leaf_reg=config.catboost_l2_leaf_reg,
                bagging_temperature=config.catboost_bagging_temperature,
                verbose=False, 
               # early_stopping_rounds=50
            )
        else:
            raise ValueError("Unsupported model type")
        
        if split_type == 'kfold':
            # Ensure X and y are vertically stacked properly
            # X = np.vstack((X_train, X_val))
            # y = np.hstack((y_train, y_val))  # Use hstack since y is a 1D array
            
            model, rmse_avg = cross_validate_and_evaluate(model, X_train_encoded, y_train, random_seed)
            print(f'kfold: mean RMSE for val data set: {rmse_avg}')
            rmse = rmse_avg
            X_train, X_val, y_train, y_val = train_test_split(X_train_encoded, y_train, test_size=0.2, random_state=random_seed)
            pred = model.predict(X_val)
        elif split_type == 'holdout':
            X_train, X_val, y_train, y_val = train_test_split(X_train_encoded, y_train, test_size=0.2, random_state=random_seed)
            model.fit(X_train, y_train)#, early_stopping_rounds=50,verbose=100)
        
            pred = model.predict(X_val)
            rmse = np.sqrt(metrics.mean_squared_error(y_val, pred))

    
        # wandb에 메트릭 로깅
        wandb.log({"rmse": rmse})
        wandb.finish()
        
        X_test_encoded = Utils.prepare_test_data(X_test_encoded, model)
        real_test_pred = model.predict(X_test_encoded)
        # #real_test_pred          # 예측값들이 출력됨을 확인할 수 있습니다.
        # # 앞서 예측한 예측값들을 저장합니다.
        preds_df = pd.DataFrame(real_test_pred.astype(int), columns=["target"])
        out_pred_path = Utils.get_unique_filename(os.path.join(out_path,f'output_{dataset_name}_{features}_{outlier_removal}_{categorical_encoding}_{feature_engineer}_{model_name}_{split_type}.csv'))
        preds_df.to_csv(out_pred_path, index=False)
