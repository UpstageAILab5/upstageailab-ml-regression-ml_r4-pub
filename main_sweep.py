import wandb
import os
from config.configs import config_baseline
from src.train import train_model

import yaml

def save_sweep_id(sweep_id, filename="sweep_id.txt"):
    with open(filename, "w") as f:
        f.write(sweep_id)

def load_sweep_id(filename="sweep_id.txt"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return f.read().strip()
    return None
def run_sweep_for_model_and_dataset(project_name, count):
    try:    
        sweep_id = load_sweep_id(f"sweep_id_{project_name}.txt")
    except FileNotFoundError:
        print('err')
        sweep_id = None
    # sweep_id = "g0kmrn6l"
    if not sweep_id:

        print('No sweep id. generating...')
        sweep_id = wandb.sweep(config_baseline, project=project_name)
        save_sweep_id(sweep_id, f"sweep_id_{project_name}.txt")
        
    #sweep_id = 't0c5c3tl'
    print(f'\n#### sweep_id: {sweep_id}\n')
    print(f'count: {count}')
    print(f'project_name: {project_name}')
    # wandb 스위프 생성
    # 모델-데이터셋 조합에 대한 스위프 실행
    wandb.agent(sweep_id, function=lambda: train_model(), project = project_name, count= count)  # 각 조합마다 count 회의 스위프 수행

def main_sweep():
    count = 500
    project_name = 'House_price_prediction_ml4_v0'#'House_price_prediction'
    run_sweep_for_model_and_dataset(project_name, count)

main_sweep()

# data
# 5-fold cross-val
# features