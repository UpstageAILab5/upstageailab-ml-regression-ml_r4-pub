import wandb
from train import train_model
from sweep_config import sweep_configs
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

def run_sweep_for_model_and_dataset(dataset_name, X_train, y_train, X_test, y_test, count):
    # 각 모델에 대해 스위프 실행
    for model_name, sweep_config in sweep_configs.items():
        # wandb 스위프 생성
        sweep_id = wandb.sweep(sweep_config, project=f"{dataset_name}-{model_name}")
        
        # 모델-데이터셋 조합에 대한 스위프 실행
        wandb.agent(sweep_id, function=lambda: train_model(dataset_name=dataset_name,
            model_name=model_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        ), count= count)  # 각 조합마다 count 회의 스위프 수행
def clean_column_names(df):
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)
    return df

def categoric_numeric(df):
        # 먼저, 연속형 변수와 범주형 변수를 위 info에 따라 분리해주겠습니다.
    continuous_columns = []
    categorical_columns = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continuous_columns.append(column)
        else:
            categorical_columns.append(column)

    print("연속형 변수:", continuous_columns)
    print("범주형 변수:", categorical_columns)
    return continuous_columns, categorical_columns

def unconcat_train_test(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    dt = df.query('is_test==0')
    # y_train = dt['target']
    dt.drop(columns=['is_test'], inplace=True)
    dt_test = df.query('is_test==1')
    dt_test.drop(columns=['target', 'is_test'], inplace=True)
    return dt, dt_test
def main_sweep():
    
# for dataset_name, dataset in datasets.items():
#     # X, y = dataset.data, dataset.target
    # #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #base_path = '/data/ephemeral/home/dev/upstageailab5-ml-regression-ml_r4'
    base_path = r'D:\dev\upstageailab5-ml-regression-ml_r4'
    prep_data_path = os.path.join(base_path, 'data','preprocessed', 'df_encoded.csv')
    df_encoded = pd.read_csv(prep_data_path)
    continuous_columns, categorical_columns = categoric_numeric(df_encoded)
    df_train, X_test = unconcat_train_test(df_encoded)
    y_train = df_train['target']
    X_train = df_train.drop(columns=['target'])

    X_train = clean_column_names(X_train)
    X_test = clean_column_names(X_test)
    print(f'X_train.columns: {X_train.columns}')
    print(f'X_test.columns: {X_test.columns}')
    
    # with open(prep_data_path, 'rb') as f:
    #     prep_data = pickle.load(f)
    
    # X_train = prep_data.get('X_train')
    # y_train = prep_data.get('y_train')
    # X_test = prep_data.get('X_val')
    # y_test = prep_data.get('y_val')
    # cont_col = prep_data.get('continuous_columns')
    # cat_col = prep_data.get('categorical_columns')

    # y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
    # y_test = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
    #cols = ['계약년', '전용면적', '강남여부', '구', '건축년도', '좌표X', '좌표Y', '동']+['층', 'k-전체동수', 'k-전체세대수', '주차대수'] +['신축여부', 'distance_sum_subway', 'distance_sum_bus']

    

    # # 예시: X_train과 y_train이 pandas DataFrame 또는 Series인 경우
    # if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
    #     print("Number of missing values in each column of X_train:")
    #     print(X_train.isna().sum())

    # if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
    #     print("\nNumber of missing values in y_train:")
    #     print(y_train.isna().sum())

    # X_train = X_train[cont_col]
    # X_test = X_test[cont_col]
    # if isinstance(X_train, pd.DataFrame):
    #     X_train = X_train.astype(float)
    # if isinstance(X_test, pd.DataFrame):
    #     X_test = X_test.astype(float)
    # if isinstance(y_test, pd.DataFrame):
    #     y_test = y_test.astype(float)
    # if isinstance(y_train, pd.DataFrame):
    #     y_train = y_train.astype(float)
    
    
    # X_train = X_train[cols]
    # X_test = X_test[cols]

    # X_train = clean_column_names(pd.DataFrame(X_train))
    # y_train = clean_column_names(pd.DataFrame(y_train))
    # X_test = clean_column_names(pd.DataFrame(X_test))
    # y_test = clean_column_names(pd.DataFrame(y_test))

    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    
    # 차원 확인
    # print(f"X_train shape: {X_train.shape}")
    # print(f"y_train shape: {y_train.shape}")
    # X_train = X_train[cols]
    # X_test = X_test[cols]

    # if isinstance(X_train, pd.DataFrame):
    #     X_train = X_train.values
    # if isinstance(y_train, pd.Series):
    #     y_train = y_train.values
    # if isinstance(X_test, pd.DataFrame):
    #     X_test = X_test.values
    # if isinstance(y_train, pd.Series):
    #     y_test = y_test.values
 
    # try:
    #     print('Sanity check for the Data.\nData shapes: X,y / train, test', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # except:
    #     print('Data load error.')
    # # X_test가 1차원인 경우 2차원으로 변환
    # if len(X_test.shape) == 1:
    #     print('X_test 1dim.')
    #     X_test = X_test.reshape(-1, 1)
    # if len(X_train.shape) == 1:
    #     print('X_train 1dim.')
        
    count = 50
    # 데이터셋과 모델 조합에 대해 스위프 실행
    dataset_name = 'test'
    run_sweep_for_model_and_dataset(dataset_name, X_train, y_train, X_test, y_test, count)
    # dataset_name = 'k_fold_cross_val'
    # run_sweep_for_model_and_dataset(dataset_name, X_train, y_train, X_test, y_test, count)

main_sweep()


# data
# 5-fold cross-val
# features