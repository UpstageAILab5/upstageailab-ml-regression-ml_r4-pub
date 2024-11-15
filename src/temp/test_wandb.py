import optuna
import yaml
import wandb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path
import os
import pprint
from tqdm import tqdm
import pandas as pd
# Load configuration from YAML
def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Config 파일 로드 중 오류 발생: {str(e)}")
        return None
def train_model(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        
        # 데이터셋 이름 로깅
        run.name = f"{config.dataset_name}_{config.model_name}"
        wandb.log({"dataset": config.dataset_name})
        
        if config.model_name == 'xgboost':
            model = xgb.XGBRegressor(
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                n_estimators=config.n_estimators,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree,
                early_stopping_rounds=20
            )
        elif config.model_name == 'lightgbm':
            model = lgb.LGBMRegressor(
                num_leaves=config.num_leaves,
                learning_rate=config.learning_rate,
                n_estimators=config.n_estimators,
                min_child_samples=config.min_child_samples,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree
            )

        # K-fold cross validation
        scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_fold_train, y_fold_train, 
                     eval_set=[(X_fold_val, y_fold_val)],
                     verbose=False)
            
            val_pred = model.predict(X_fold_val)
            rmse = mean_squared_error(y_fold_val, val_pred, squared=False)
            scores.append(rmse)
        
        avg_rmse = np.mean(scores)
        wandb.log({"avg_rmse": avg_rmse})
        return avg_rmse

def optimize_model(model_name, X, y, config_path):
    config = load_config(config_path)
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'avg_rmse', 'goal': 'minimize'},
        'parameters': {
            'model_name': {'value': model_name},
            'dataset_name': {'value': config['name']['dataset_name']},  # 데이터셋 이름 추가
            'max_depth': config['models'][model_name]['max_depth'],
            'learning_rate': config['models'][model_name]['learning_rate'],
            'n_estimators': config['models'][model_name]['n_estimators'],
            'subsample': config['models'][model_name]['subsample'],
            'colsample_bytree': config['models'][model_name]['colsample_bytree']
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project='wandb_example')
    wandb.agent(sweep_id, function=train_model, count=50)

# Function to train the final model with the best parameters and evaluate on the test set
def train_and_evaluate(model_name, X_train, y_train, X_test, best_params):
    if model_name == 'xgboost':
        model_class = xgb.XGBRegressor
    elif model_name == 'lightgbm':
        model_class = lgb.LGBMRegressor
    else:
        raise ValueError("Unsupported model name. Use 'xgboost' or 'lightgbm'.")

    # Initialize wandb run for final model
    wandb.init(project="optuna_wandb_example", name=f"{model_name}_final_model", config=best_params)

    # Train the final model
    model = model_class(**best_params)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    # # Evaluate the performance
    # mse = mean_squared_error(y_test, y_pred)
    # wandb.log({"test_mse": mse})  # Log the test MSE to wandb
    # print(f"Test MSE for {model_name}: {mse}")
    out_pred_path = os.path.join(config.get('path').get('out'),f'output_{model_name}.csv')

    X_test = Utils.prepare_test_data(X_test, model)
    y_pred = model.predict(X_test)
    preds_df = pd.DataFrame(y_pred.astype(int), columns=["target"])
    preds_df.to_csv(out_pred_path, index=False)
    print(f"Predictions saved to {out_pred_path}")
    wandb.finish()  # Close the wandb run
    return 
# # Data preparation (example using Boston dataset)
# data = load_boston()
# X = data.data
# y = data.target

def setup_project_path():
    """프로젝트 루트 경로를 찾아서 파이썬 경로에 추가"""
    current = Path.cwd()
    while current != current.parent:
        if (current / '.git').exists():
            if str(current) not in sys.path:
                sys.path.append(str(current))
                print(f'Project root found: {current}')
            return current
        current = current.parent
    return None
project_root = setup_project_path()
if project_root is None:
    # 프로젝트 루트를 찾지 못했다면 직접 지정
    project_root = Path("D:/dev/upstageailab5-ml-regression-ml_r4")
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


from src.utils import Utils
from src.logger import Logger
from src.preprocessing import DataPrep

def setup():
    logger_instance = Logger()
    logger = logger_instance.logger
    utils = Utils(logger)
    utils.setup_font_and_path_platform()
    current_platform = utils.current_platform
    #os.environ['PYTHONPATH'] = r'D:\dev\upstageailab5-ml-regression-ml_r4'
    current_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
    logger.info(f'#### Current workspalce: {current_path}')
    if current_platform == 'Windows':
        base_path = Path(r'D:\dev\upstageailab5-ml-regression-ml_r4')
        logger.info(f'{current_platform} platform. Path: {base_path}')
    elif current_platform == 'Darwin':          # Mac
        base_path = Path('/data/ephemeral/home/dev/upstageailab5-ml-regression-ml_r4')
        logger.info(f'{current_platform} platform. Path: {base_path}')
    else:
        base_path = Path('/data/ephemeral/home/dev/upstageailab5-ml-regression-ml_r4')    # Linux
        logger.info(f'{current_platform} platform. Path: {base_path}')
    config ={
            'path':{
                'base':base_path,
                'logs':os.path.join(base_path, 'logs'),
                'config':os.path.join(base_path, 'config'),
                'data':os.path.join(base_path, 'data'),
                'processed':os.path.join(base_path, 'data', 'processed'),
                'out':os.path.join(base_path,'output'),
            },
            'name':{
                'dataset_name': 'concat',#'concat_feat',#'feat_scaled',#'base_selected',#'feat_selected',   
                'split_type': 'hold_out',#'k_fold',
                'model_name': 'random_forest',#'xgboost',
            },
            'base_path':base_path,
            'subway_feature': os.path.join(base_path, 'data','subway_feature.csv'),
            'bus_feature': os.path.join(base_path, 'data','bus_feature.csv'),
            'logger': logger_instance,#logger,
        }
    config_file_path = os.path.join(config.get('path').get('config'), 'config.yaml')
    config_base = Utils.load_nested_yaml(config_file_path)
    config.update(config_base)
    pprint.pprint(config)
    
    return config, logger
# 프로젝트 경로 설정

config, logger = setup()

processed_path = config.get('path').get('processed')
concat_path = os.path.join(processed_path, 'concat_train_test.csv')
concat = pd.read_csv(concat_path, index_col=0)
train_df, test_df = DataPrep.unconcat_train_test(concat)
y_train = train_df['target']
X_train = train_df.drop(columns=['target'])
X_test = test_df

# #real_test_pred          # 예측값들이 출력됨을 확인할 수 있습니다.
# # 앞서 예측한 예측값들을 저장합니다.

# X_train, X_test, y_train, y_test = train_test_split(concat, concat['target'], test_size=0.2, random_state=2024)

model_name = 'xgboost'
# Run optimization and training for XGBoost
optimize_model('xgboost', X_train, y_train, os.path.join(config.get('path').get('config'), 'config.yaml'))

# Run optimization and training for LightGBM
model_name = 'lightgbm'
optimize_model('lightgbm', X_train, y_train, os.path.join(config.get('path').get('config'), 'config.yaml'))
