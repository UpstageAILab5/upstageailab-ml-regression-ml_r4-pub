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
random_seed = 2024
def clean_column_names(df):
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)
    return df
def unconcat_train_test(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    dt = df.query('is_test==0')
    # y_train = dt['target']
    dt.drop(columns=['is_test'], inplace=True)
    dt_test = df.query('is_test==1')
    dt_test.drop(columns=['target', 'is_test'], inplace=True)
    return dt, dt_test

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
            del X_train, X_val, y_train, y_val, val_pred
            import gc
            gc.collect()
        
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
        # 모델 선택
        print(f'\n#### Model: {model_name} ####\n')
        print(f'- Data: {dataset_name}\n- Features: {features}\n- Split type:{split_type}\n')
        if model_name == "xgboost":
            model = xgb.XGBRegressor(
                eta=config.xgboost_eta,
                max_depth=config.xgboost_max_depth,
                subsample=config.xgboost_subsample,
                colsample_bytree=config.xgboost_colsample_bytree,
                gamma=config.xgboost_gamma,
                reg_lambda=config.xgboost_reg_lambda,  
                reg_alpha=config.xgboost_alpha,
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
                lambda_l2=config.lightgbm_lambda_l2
            )
        elif model_name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=config.random_forest_n_estimators,
                max_depth=config.random_forest_max_depth,
                min_samples_split=config.random_forest_min_samples_split,
                min_samples_leaf=config.random_forest_min_samples_leaf,
                max_features=config.random_forest_max_features
            )
        elif model_name == "catboost":
            model = CatBoostRegressor(
                iterations=config.catboost_iterations,
                depth=config.catboost_depth,
                learning_rate=config.catboost_learning_rate,
                l2_leaf_reg=config.catboost_l2_leaf_reg,
                bagging_temperature=config.catboost_bagging_temperature,
                verbose=False
            )
        else:
            raise ValueError("Unsupported model type")

        # 모델 훈련 및 평가
        
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(base_path, 'data')
        prep_path = os.path.join(data_path, 'preprocessed')

        if dataset_name == 'baseline':
            df = pd.read_csv(os.path.join(prep_path, 'df_feature.csv')) 
        elif dataset_name == 'encoded':
            df = pd.read_csv(os.path.join(prep_path, 'df_encoded.csv')) 
        elif dataset_name == 'feat_null-preped_freq-encoded':
            df = pd.read_csv(os.path.join(prep_path, 'df_feat_null-preped_freq-encoded.csv')) 
        elif dataset_name == 'null-preped_freq-encoded':
            df = pd.read_csv(os.path.join(prep_path, 'df_null-preped_freq-encoded.csv')) 
        cols_id = ['is_test', 'target']
        cols_to_remove = ['등기신청일자', '거래유형', '중개사소재지'] 
        
        cols = ['시군구', '전용면적', '계약년월', '건축년도', '층', '도로명', '아파트명']
        cols_feat = ['신축여부', '구', '강남여부']

        if features == 'baseline':
            cols_to_remove = [col for col in cols_to_remove if col in df.columns]
            df = df.drop(columns=cols_to_remove)
        elif features == 'removed':
            cols_to_remove+=['홈페이지','k-전화번호', 'k-팩스번호', '고용보험관리번호']
            cols_to_remove = [col for col in cols_to_remove if col in df.columns]
            df = df.drop(columns=cols_to_remove)
        elif features == 'minimum':
            cols_total = cols_id+cols+cols_feat
            cols_total = [col for col in cols_total if col in df.columns]
            df = df[cols_total]

        #continuous_columns, categorical_columns = categoric_numeric(df_encoded)
        df_train, _= unconcat_train_test(df)
        y = df_train['target']
        X = df_train.drop(columns=['target'])

        X = clean_column_names(X)
        # 피처 이름 중복 확인 및 해결

        duplicated_columns = X.columns[X.columns.duplicated()]

        # 중복된 피처 이름 출력
        if duplicated_columns.any():
            print("중복된 피처 이름이 있습니다:", duplicated_columns.tolist())
            X.columns = [f"feature_{i}" for i in range(X.shape[1])]
        try:
            print(X.shape)
            print(y.shape)
        except:
            print('shape print error.')
        if split_type == 'kfold':
            # Ensure X and y are vertically stacked properly
            # X = np.vstack((X_train, X_val))
            # y = np.hstack((y_train, y_val))  # Use hstack since y is a 1D array
            
            model, rmse_avg = cross_validate_and_evaluate(model, X, y, random_seed)
            print(f'kfold: mean RMSE for val data set: {rmse_avg}')
            rmse = rmse_avg
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_seed)
            pred = model.predict(X_val)
        elif split_type == 'holdout':
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_seed)
            model.fit(X_train, y_train)#, early_stopping_rounds=50,verbose=100)
        
            pred = model.predict(X_val)
            rmse = np.sqrt(metrics.mean_squared_error(y_val, pred))
        # wandb에 메트릭 로깅
        wandb.log({"rmse": rmse})
        wandb.finish()
