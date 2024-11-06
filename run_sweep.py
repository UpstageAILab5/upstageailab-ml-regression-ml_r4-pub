import wandb
from train import train_model
from sweep_config import sweep_configs
# from sklearn.datasets import load_boston, load_diabetes  # 예제 데이터셋
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import numpy as np
# 사용할 데이터셋을 로드하고 분할
# datasets = {
#     "boston": load_boston(),
#     "diabetes": load_diabetes()
# }

def run_sweep_for_model_and_dataset(dataset_name, X_train, y_train, X_test, y_test):
    # 각 모델에 대해 스위프 실행
    for model_name, sweep_config in sweep_configs.items():
        # wandb 스위프 생성
        sweep_id = wandb.sweep(sweep_config, project=f"{dataset_name}-{model_name}")
        
        # 모델-데이터셋 조합에 대한 스위프 실행
        wandb.agent(sweep_id, function=lambda: train_model(
            model_name=model_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        ), count= 30)  # 각 조합마다 count 회의 스위프 수행
def clean_column_names(df):
        df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)
        return df
def main_sweep():
    dataset_name = 'baseline'
# for dataset_name, dataset in datasets.items():
#     # X, y = dataset.data, dataset.target
    # #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    base_path = '/data/ephemeral/home/dev/upstageailab5-ml-regression-ml_r4'
    prep_data_path = os.path.join(base_path, 'output', 'prep_data.pkl')

    with open(prep_data_path, 'rb') as f:
        prep_data = pickle.load(f)

    X_train = prep_data.get('X_train')
    y_train = prep_data.get('y_train')
    X_test = prep_data.get('X_val')
    y_test = prep_data.get('y_val')
    cont_col = prep_data.get('continuous_columns')
    cat_col = prep_data.get('categorical_columns')
    print(X_train.head(3), X_test.head(3), y_train.head(3), y_test.head(3))
    import pandas as pd

    # 예시: X_train과 y_train이 pandas DataFrame 또는 Series인 경우
    if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
        print("Number of missing values in each column of X_train:")
        print(X_train.isna().sum())

    if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
        print("\nNumber of missing values in y_train:")
        print(y_train.isna().sum())



    # X_train = X_train[cont_col]
    # X_test = X_test[cont_col]
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.astype(float)
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.astype(float)
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.astype(float)
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.astype(float)
    cols = ['계약년', '전용면적', '강남여부', '구', '건축년도', '좌표X', '좌표Y', '동']
    X_train = X_train[cols]
    X_test = X_test[cols]

    X_train = clean_column_names(pd.DataFrame(X_train))
    y_train = clean_column_names(pd.DataFrame(y_train))
    X_test = clean_column_names(pd.DataFrame(X_test))
    y_test = clean_column_names(pd.DataFrame(y_test))
    print(X_test.head(3))
    
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    
    # 차원 확인
    # print(f"X_train shape: {X_train.shape}")
    # print(f"y_train shape: {y_train.shape}")
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_train, pd.Series):
        y_test = y_test.values
    print(X_test.shape, y_train.shape, X_test.shape, y_test.shape)
    # X_test가 1차원인 경우 2차원으로 변환
    if len(X_test.shape) == 1:
        X_test = X_test.reshape(-1, 1)
    
    try:
        print('Sanity check for the Data.\n', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    except:
        print('Data load error.')
    # 데이터셋과 모델 조합에 대해 스위프 실행
    run_sweep_for_model_and_dataset(dataset_name, X_train, y_train, X_test, y_test)

main_sweep()


# data
# 5-fold cross-val
# features