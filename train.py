import wandb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import numpy as np

def train_model(config=None, model_name="xgboost", X_train=None, y_train=None, X_test=None, y_test=None):
    wandb.init(config=config)
    config = wandb.config

    # 모델 선택
    if model_name == "xgboost":
        model = xgb.XGBRegressor(
            eta=config.eta,
            max_depth=config.max_depth,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            gamma=config.gamma,
            reg_lambda=config.reg_lambda,  
            reg_alpha=config.alpha,
        )
    elif model_name == "lightgbm":
        model = lgb.LGBMRegressor(
            learning_rate=config.learning_rate,
            num_leaves=config.num_leaves,
            max_depth=config.max_depth,
            min_data_in_leaf=config.min_data_in_leaf,
            feature_fraction=config.feature_fraction,
            bagging_fraction=config.bagging_fraction,
            lambda_l1=config.lambda_l1,
            lambda_l2=config.lambda_l2
        )
    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features
        )
    elif model_name == "catboost":
        model = CatBoostRegressor(
            iterations=config.iterations,
            depth=config.depth,
            learning_rate=config.learning_rate,
            l2_leaf_reg=config.l2_leaf_reg,
            bagging_temperature=config.bagging_temperature,
            verbose=False
        )
    else:
        raise ValueError("Unsupported model type")

    # 모델 훈련 및 평가
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))


    # wandb에 메트릭 로깅
    wandb.log({"rmse": rmse})
    wandb.finish()
