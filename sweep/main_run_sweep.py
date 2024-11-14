import wandb
from train import train_model
from sweep_config import config_baseline #sweep_config_xgb,
# from sklearn.datasets import load_boston, load_diabetes  # 예제 데이터셋
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import sys
module_path = os.path.dirname(os.path.abspath(__file__))
print(module_path)
# 모듈 검색 경로에 추가
if module_path not in sys.path:
    sys.path.append(module_path)

# 사용할 데이터셋을 로드하고 분할
# datasets = {
#     "boston": load_boston(),
#     "diabetes": load_diabetes()
# }
def save_sweep_id(sweep_id, filename="sweep_id.txt"):
    with open(filename, "w") as f:
        f.write(sweep_id)

def load_sweep_id(filename="sweep_id.txt"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return f.read().strip()
    return None
def run_sweep_for_model_and_dataset(project_name, count):
    # 각 모델에 대해 스위프 실행
    # for model_name, sweep_config in sweep_configs#.items():
    
    try:    
        sweep_id = load_sweep_id(f"sweep_id_{project_name}.txt")
    except FileNotFoundError:
        print('err')
        sweep_id = None
    # sweep_id = "g0kmrn6l"
    if not sweep_id:
        print('No sweep id. generating...')
        # 스위프 ID가 없으면 새로 생성
        
        #sweep_id = wandb.sweep(sweep_configs, project=project_name)
        sweep_id = wandb.sweep(config_baseline, project=project_name)
        save_sweep_id(sweep_id, f"sweep_id_{project_name}.txt")
    print(f'\n#### sweep_id: {sweep_id}\n')
    print(f'count: {count}')
    print(f'project_name: {project_name}')
    # wandb 스위프 생성
    # 모델-데이터셋 조합에 대한 스위프 실행
    wandb.agent(sweep_id, function=lambda: train_model(), project = project_name, count= count)  # 각 조합마다 count 회의 스위프 수행

def main_sweep():
# for dataset_name, dataset in datasets.items():
#     # X, y = dataset.data, dataset.target
    # #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #base_path = '/data/ephemeral/home/dev/upstageailab5-ml-regression-ml_r4'
    
    # print(f'X_train.columns: {X_train.columns}')
    # print(f'X_test.columns: {X_test.columns}')
    count = 500
    # 데이터셋과 모델 조합에 대해 스위프 실행
    project_name = 'House_price_prediction_main_baseline'#'House_price_prediction'
    run_sweep_for_model_and_dataset(project_name, count)
    # dataset_name = 'k_fold_cross_val'
    # run_sweep_for_model_and_dataset(dataset_name, X_train, y_train, X_test, y_test, count)
main_sweep()


# data
# 5-fold cross-val
# features