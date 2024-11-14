from pathlib import Path
import os
# import pygwalker as pyg
# import dabl
import pandas as pd
from src.logger import Logger
from src.preprocessing import DataPrep
from src.eda import EDA
from src.feature import FeatureEngineer, FeatureAdditional, Clustering
from src.train import Model
# from src.utils import setup_matplotlib_korean
import pickle
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
from src.utils import Utils
import platform
import pprint
import yaml
from sklearn.preprocessing import StandardScaler
# 메모리 정리
import gc
gc.collect()
get_unique_filename = Utils.get_unique_filename

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
                #'dataset_name': 'concat_scaled_selected',#'concat_feat',#'feat_scaled',#'base_selected',#'feat_selected',   
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
def save_config(config):
    # logger 객체만 제거한 복사본 생성
    config_to_save = {k: v for k, v in config.items() if k != 'logger'}
    dataset_name = config.get('name').get('dataset_name')
    config_log_path = os.path.join(config.get('path').get('config'), f'config_log_{dataset_name}.yaml')
    # YAML 파일로 저장
    with open(config_log_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            config_to_save,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False
        )


def main():
    ########################################################################################################################################
    # Setup
    config, logger = setup()
    ########################################################################################################################################
    ### Data Prep
    prep_path = config.get('path').get('processed')
    path_baseline = os.path.join(prep_path, 'df_base_prep.csv')
    path_auto = os.path.join(prep_path, 'df_auto_prep.csv')
    data_prep = DataPrep(config)
    ########################################################################################################################################
    
    # concat = pd.read_csv(os.path.join(prep_path, 'df_baseline_preprocessed.csv'))#'df_feat_selected_preprocessed_id.csv'))
    # concat = Utils.clean_df(concat)
    
    #concat=concat[col_to_select + col_id]
    # concat_df = concat.drop(columns=col_id, inplace=False)
    # print(concat_df.shape)
    # numeric_cols = concat.select_dtypes(include=['float64', 'int64']).columns
    # scaler = StandardScaler()
    # df_scaled = pd.DataFrame(scaler.fit_transform(concat[numeric_cols]), columns=numeric_cols)
    # high_vif_cols = FeatureEngineer.calculate_vif(df_scaled)
    # logger.info(f'높은 VIF 값(>10)을 가진 컬럼들: {high_vif_cols}')
  
    #concat = concat[col_to_select + col_id]

    dataset_name = config.get('name').get('dataset_name')
    print('##########################')
    #####
    if config.get('name').get('dataset_name') == 'baseline':
        concat = pd.read_csv(os.path.join(prep_path, 'df_concat_train_test.csv'), index_col=0)
    elif config.get('name').get('dataset_name') == 'base_scaled':
        concat = pd.read_csv(os.path.join(prep_path, 'df_base_scaled.csv'), index_col=0)
    elif config.get('name').get('dataset_name') == 'base_selected':
        concat = pd.read_csv(os.path.join(prep_path, 'df_base_selected.csv'), index_col=0)
    if config.get('name').get('dataset_name') == 'feat_scaled':
        concat = pd.read_csv(os.path.join(prep_path, 'df_feat_scaled.csv'), index_col=0)
    elif config.get('name').get('dataset_name') == 'feat_selected':
        concat = pd.read_csv(os.path.join(prep_path, 'df_feat_selected.csv'), index_col=0)
    elif config.get('name').get('dataset_name') == 'concat_feat':
        concat = pd.read_csv(os.path.join(prep_path, 'df_concat_feat.csv'), index_col=0)
    elif config.get('name').get('dataset_name') == 'concat':
        concat = pd.read_csv(os.path.join(prep_path, 'concat_train_test.csv'), index_col=0)
    
    elif config.get('name').get('dataset_name') == 'concat_scaled_selected':
        print('concat_scaled_selected')
        concat = pd.read_csv(os.path.join(prep_path, 'df_concat_scaled_selected.csv'), index_col=0)
    else:
        print('else')

    
    ########################################################################################################################################
    dt_train, dt_test = DataPrep.unconcat_train_test(concat)
    logger.info(f'>>>>Train shape: {dt_train.shape}, {dt_train.columns}\nTest shape: {dt_test.shape}, {dt_test.columns}')
    y_train = dt_train['target']
    X_train = dt_train.drop(['target'], axis=1)
    X_test =dt_test
   
    ########################################################################################################################################
    ##  Model Training
    model = Model(config)
    dataset_name = config.get('name').get('dataset_name')
    model_name = config.get('name').get('model_name')
    split_type = config.get('name').get('split_type')

    out_model_path = os.path.join(config.get('path').get('out'), f'saved_model_{model_name}_{dataset_name}_{split_type}.pkl')
    logger.info(f'out_model_path: {out_model_path}')
    if not os.path.exists(out_model_path):
        logger.info('>>>>Model Training 시작...')
        model, _ = model.model_train(X_train, y_train, model_name, split_type)#model.model_train(X_train, X_val, y_train, y_val, model_name, split_type)
        with open(out_model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        logger.info('>>>>Model Training 존재. Loading...')
        with open(out_model_path, 'rb') as f:
            model = pickle.load(f)
    
    ########################################################################################################################################
    # Infere
    logger.info('>>>>Test dataset에 대한 inference 시작...')
    X_test = Utils.prepare_test_data(X_test, model)
    real_test_pred = model.predict(X_test)
    # #real_test_pred          # 예측값들이 출력됨을 확인할 수 있습니다.
    # # 앞서 예측한 예측값들을 저장합니다.
    preds_df = pd.DataFrame(real_test_pred.astype(int), columns=["target"])

    out_pred_path = get_unique_filename(os.path.join(config.get('path').get('out'),f'output_{model_name}_{dataset_name}_{split_type}.csv'))
    preds_df.to_csv(out_pred_path, index=False)
    logger.info(f'Inference 결과 저장 완료 : {out_pred_path}')
    logger.info(f'{preds_df.head(3)}')
    save_config(config)
    # X_test = dt_test
    # X_test = dt_test.drop(['target'], axis=1)

    # from src.feature import XAI
    # xai = XAI(config)
    # xai.shap_summary(model, X_train, X_test)

    #visualizer = Visualizer(config)
if __name__ == '__main__':
    freeze_support()
    main()
