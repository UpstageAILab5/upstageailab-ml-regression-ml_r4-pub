import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from src.feature import  XAI
from src.utils import Utils
import warnings
import lightgbm as lgb
warnings.filterwarnings('ignore')

class Model():
    def __init__(self, config):
        self.config = config
        self.out_path = config.get('path').get('out')
        self.random_seed = config.get('random_seed')
        self.logger_instance = config.get('logger')
        self.logger_instance.setup_logger(log_file='preprocessing')
        self.logger = self.logger_instance.logger
        self.logger.info('#### Init Model Train... ')
        self.time_delay = config.get('time_delay')
    ### Model training
    @Utils.timeit
    def cross_validate_and_evaluate(self, model, X, y):
        """
        교차 검증 및 모델 평가
        Parameters:
            model: 학습할 모델
            X: 특성 데이터프레임
            y: 타겟 시리즈
        """
        self.logger.info(f"\n=== 교차 검증 시작 ===")
        self.logger.info(f"전체 데이터셋 shape - X: {X.shape}, y: {y.shape}")
        
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        rmse_scores = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
            self.logger.info(f"\n--- Fold {fold}/5 ---")
            
            # 데이터 분할
            X_train = X.iloc[train_index]
            X_val = X.iloc[val_index]
            y_train = y.iloc[train_index]
            y_val = y.iloc[val_index]
            
            # 분할된 데이터셋 shape 출력
            self.logger.info(f"Train set shape - X: {X_train.shape}, y: {y_train.shape}")
            self.logger.info(f"Val set shape   - X: {X_val.shape}, y: {y_val.shape}")
            
            # 모델 학습
            self.logger.info("모델 학습 중...")
            model.fit(X_train, y_train)
            
            # 예측 및 RMSE 계산
            val_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            rmse_scores.append(rmse)
            
            self.logger.info(f'Fold {fold} RMSE: {rmse:,.2f}')
            
            # 메모리 정리
            del X_train, X_val, y_train, y_val, val_pred
            import gc
            gc.collect()
        
        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        
        self.logger.info("\n=== 교차 검증 결과 ===")
        self.logger.info(f'평균 RMSE: {mean_rmse:,.2f} (±{std_rmse:,.2f})')
        self.logger.info(f'개별 RMSE: {[f"{score:,.2f}" for score in rmse_scores]}')
        
        return model, mean_rmse
    def model_train(self, X, y, model_name='xgboost', type='baseline'):
    # RandomForestRegressor를 이용해 회귀 모델을 적합시키겠습니다.
        #model = RandomForestRegressor(n_estimators=5, criterion='squared_error', random_state=1, n_jobs=-1)
        
        config = self.config.get('sweep_configs')[model_name]['parameters'] 
        if model_name == 'xgboost':
            model = xgb.XGBRegressor(
                    reg_alpha=config.get('alpha'),   
                colsample_bytree=config.get('colsample_bytree'),
                eta=config.get('eta'),
                gamma=config.get('gamma'),
                max_depth=config.get('max_depth'),
                n_estimators=config.get('n_estimators'),
                reg_lambda=config.get('reg_lambda'),
                subsample=config.get('subsample'), 
            )
        elif model_name == 'random_forest':
            # model = RandomForestRegressor(
            #         n_estimators=config.get('n_estimators'),
            #         max_depth=config.get('max_depth'),
            #         min_samples_split=config.get('min_samples_split'),
            #         min_samples_leaf=config.get('min_samples_leaf'),
            #         max_features=config.get('max_features')
            # )
            # RandomForestRegressor를 이용해 회귀 모델을 적합시키겠습니다.
            model = RandomForestRegressor(n_estimators=5, criterion='squared_error', random_state=1, n_jobs=-1)
        elif model_name == 'lightgbm':
            model = lgb.LGBMRegressor(
                objective='regression',
                num_leaves=config.get('num_leaves'),
                learning_rate=config.get('learning_rate'),
                subsample=config.get('subsample'),
                colsample_bytree=config.get('colsample_bytree'),
                lambda_l1=config.get('lambda_l1'),
                lambda_l2=config.get('lambda_l2'),
                min_data_in_leaf=config.get('min_data_in_leaf'),
                feature_fraction=config.get('feature_fraction'),    
                max_depth=config.get('max_depth')
            )
        # smote = SMOTE(random_state=42)
        # X_train, y_train = smote.fit_resample(X_train, y_train)
        # Recursive Feature Elimination 예시
        # rfe = RFE(estimator=model, n_features_to_select=10)
        # X_rfe = rfe.fit_transform(X_train, y)

        # if isinstance(X_train, pd.DataFrame):
        #     X_train = X_train.values
        # if isinstance(y_train, pd.Series):
        #     y_train = y_train.values
        # if isinstance(X_val, pd.DataFrame):
        #     X_val = X_val.values
        # if isinstance(y_train, pd.Series):
        #     y_val = y_val.values
    
        if type == 'kfold':
            # Ensure X and y are vertically stacked properly
            # X = np.vstack((X_train, X_val))
            # y = np.hstack((y_train, y_val))  # Use hstack since y is a 1D array
            model, rmse_avg = self.cross_validate_and_evaluate(model, X, y)
            self.logger.info(f'kfold: mean RMSE for val data set: {rmse_avg}')
            rmse = rmse_avg
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_seed)
            pred = model.predict(X_val)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_seed)
            model.fit(X_train, y_train)#, early_stopping_rounds=50,verbose=100)
        
            pred = model.predict(X_val)
            rmse = np.sqrt(metrics.mean_squared_error(y_val, pred))
        self.logger.info(f'RMSE {type}: {rmse}')
            # 회귀 관련 metric을 통해 train/valid의 모델 적합 결과를 관찰합니다.
        # 학습된 모델을 저장합니다. Pickle 라이브러리를 이용하겠습니다.
        out_path = os.path.join(self.out_path,f'saved_model_{model_name}_{type}.pkl' )
        with open(out_path, 'wb') as f:
            pickle.dump(model, f)
        self.logger.info(f'Model saved to {out_path}')
        self.logger.info(f'Feature Importances for {model_name}')
        # 위 feature importance를 시각화해봅니다.
        importances = pd.Series(model.feature_importances_, index=list(X_train.columns))
        importances = importances.sort_values(ascending=False)
        title_feat = "Feature Importances"
        plt.figure(figsize=(10,8))
        plt.title(title_feat)
        sns.barplot(x=importances, y=importances.index)
        plt.show(block=False)
        plt.pause(self.time_delay)  # 5초 동안 그래프 표시
        
        plt.savefig(os.path.join(self.out_path, title_feat +'.png'), dpi=300, bbox_inches='tight')
        plt.close()
        return model, pred, rmse

    # def k_fold_train(self, dt_train, k=5):
    #     y = dt_train['target']
    #     X = dt_train.drop(['target'], axis=1)
    #     kf = KFold(n_splits=k, shuffle=True, random_state=self.random_seed)
    #     # 또는 StratifiedKFold (타겟 불균형시)
    #     # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
    #     # models=[]
    #     # for train_index, val_index in kf.split(X, y):
    #     #     X_train, X_val = X[train_index], X[val_index]
    #     #     y_train, y_val = y[train_index], y[val_index]
    
    #     #     model, pred = self.model_train(X_train, X_val, y_train, y_val, self.out_path)
    #     #     models.append({'model':model,'pred':pred})
 
    #     return 

    # def inference(self, dt_test):
    #     print('inference start.')
    #     dt_test.head(2)      # test dataset에 대한 inference를 진행해보겠습니다.
    #     # 저장된 모델을 불러옵니다.
    #     out_model_path = os.path.join(self.out_path, 'saved_model.pkl')
    #     with open(out_model_path, 'rb') as f:
    #         model = pickle.load(f)

    #     X_test = dt_test.drop(['target'], axis=1)

    #     # Test dataset에 대한 inference를 진행합니다.
    #     real_test_pred = model.predict(X_test)
    #     #real_test_pred          # 예측값들이 출력됨을 확인할 수 있습니다.

    #     # 앞서 예측한 예측값들을 저장합니다.
    #     preds_df = pd.DataFrame(real_test_pred.astype(int), columns=["target"])
    #     preds_df.to_csv(os.path.join(self.out_path,'output.csv'), index=False)
    #     return preds_df

    def load_data_pkl(self, data_path):
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        except:
            print('err')
            data = None
        return data

    def save_data(self, prep_data):
        out_path = os.path.join(self.out_path,'prep_data.pkl')
        try:
            with open(out_path, 'wb') as f:
                    pickle.dump(prep_data, f)
            print('Dataset Saved to ', out_path)
        except:
            print('error.')
        return out_path