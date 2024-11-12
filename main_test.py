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
from src.utils import Utils, FileCache
import platform
import pprint
import yaml
from sklearn.preprocessing import StandardScaler
import re
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
                'prep':os.path.join(base_path, 'data', 'preprocessed'),
                'out':os.path.join(base_path,'output'),
            },
            'name':{
                'dataset_name': 'concat_scaled_selected',#'concat_feat',#'feat_scaled',#'base_selected',#'feat_selected',   
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
def seve_config(config):
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
def feat_add_transport(config, logger, concat):
    path_feat_transport = os.path.join(config.get('path').get('prep'), 'feat_transport.csv')
    if not os.path.exists(path_feat_transport):
        feat_add = FeatureAdditional(config)
        logger.info('>>>>feat add 존재하지 않음. 생성 시작...')
        df_coor = {'x': '좌표X', 'y': '좌표Y'}
        subway_coor = {'x': '경도', 'y':'위도' }
        bus_coor = {'x': 'X좌표', 'y': 'Y좌표'}
        subway_feature = pd.read_csv(config.get('subway_feature'))
        bus_feature = pd.read_csv(config.get('bus_feature'), index_col=0)
        #concat = concat.sample(frac=0.001)

        #  BallTree 방식 (더 빠르지만 메모리 많이 사용)
        concat, subway_cols = feat_add.distance_analysis_balltree(
            concat, subway_feature, df_coor, subway_coor, target='subway'
        )
        concat, bus_cols = feat_add.distance_analysis_balltree(concat, bus_feature, df_coor, bus_coor, target='bus')
        #df.drop(df.columns[0], axis=1, inplace=True)
        #concat.to_csv(path_feat_add)
        transport_cols = subway_cols + bus_cols
        logger.info(f'Total names: {len(concat.columns)} \nConcat data new column names : {transport_cols}')
        feat_transport = concat[transport_cols]
        feat_transport.to_csv(path_feat_transport, index=False)
    else:   
        logger.info('>>>>feat add 존재. Loading...')
        feat_transport = pd.read_csv(path_feat_transport, index_col=0)
        feat_transport = feat_transport.reset_index(drop=True)
        subway_cols = [col for col in feat_transport.columns if 'subway' in col.lower()]
        bus_cols = [col for col in feat_transport.columns if 'bus' in col.lower()]
    
    transport_cols = subway_cols + bus_cols

    logger.info(f'Feature Add 1. Transport new column names : {transport_cols}')
    return feat_transport, transport_cols
def feat_add_clustering(config, logger, concat, cols_for_clustering):
    clustering = Clustering(config)
    path_feat_cluster = os.path.join(config.get('path').get('prep'), 'feat_cluster.csv')
    if not os.path.exists(path_feat_cluster):    
        logger.info('>>>>Dist/Transport-based Clustering 시작...')
        concat = clustering.dbscan_clustering(concat, features = ['좌표X', '좌표Y'] + config.get('transport_cols'), target='dist_transport')
        logger.info(f'>>>>Clustering based on specific cols: 시작...\nClustering 대상 컬럼 : {cols_for_clustering}')
        concat = clustering.dbscan_clustering(concat, features = cols_for_clustering, target='select')
        cluster_cols = [col for col in concat.columns if 'cluster' in col]
        feat_cluster = concat[cluster_cols]
        feat_cluster.to_csv(path_feat_cluster, index=False)
    else:
        logger.info('>>>>Clustering Feature 존재. Loading...')
        feat_cluster = pd.read_csv(path_feat_cluster)
        cluster_cols = feat_cluster.columns
        concat = concat.reset_index(drop=True)
        feat_cluster = feat_cluster.reset_index(drop=True)
        concat = pd.concat([concat, feat_cluster], axis=1)
    
    return feat_cluster, cluster_cols
def concat_feature(concat, feat_transport, feat_cluster, logger):
    concat = concat.reset_index(drop=True)
    feat_transport = feat_transport.reset_index(drop=True)
    feat_cluster = feat_cluster.reset_index(drop=True)

    # 데이터프레임 크기 확인 로깅
    logger.info(f'Shape before concat - concat: {concat.shape}, feat_transport: {feat_transport.shape}, feat_cluster: {feat_cluster.shape}')

    # 중복 컬럼 확인
    duplicate_cols = set(concat.columns) & set(feat_transport.columns) & set(feat_cluster.columns)
    if duplicate_cols:
        logger.warning(f'중복된 컬럼 발견: {duplicate_cols}')
        # 중복 컬럼 처리
        feat_transport = feat_transport.drop(columns=list(duplicate_cols & set(feat_transport.columns)))
        feat_cluster = feat_cluster.drop(columns=list(duplicate_cols & set(feat_cluster.columns)))

    # 데이터프레임 합치기
    concat = pd.concat([concat, feat_transport, feat_cluster], axis=1)

    # 결과 확인
    logger.info(f'Shape after concat: {concat.shape}')
    logger.info(f'Final columns: {concat.columns.tolist()}')
    return concat
def main():
    ########################################################################################################################################
    # Setup
    config, logger = setup()
    file_cache = FileCache(logger)
    ########################################################################################################################################
    ### Data Prep
    prep_path = config.get('path').get('prep')
    path_base_prep = os.path.join(prep_path, 'df_interpolation.csv')
    path_auto = os.path.join(prep_path, 'df_auto_prep.csv')
    data_prep = DataPrep(config)
    ########################################################################################################################################
    # path_feat_add = os.path.join(prep_path, 'df_feat_add.csv')
    # path_prep_baseline = os.path.join(prep_path, 'prep_baseline.csv')
    remove_cols = ['등기신청일자','거래유형','중개사소재지'] + ['k-팩스번호',
                       'k-전화번호',
                       'k-홈페이지',
                       '고용보험관리번호',
                       '단지소개기존clob']

    df = file_cache.load_or_create(
        path_base_prep,
        lambda: data_prep.data_prep(remove_cols=remove_cols)
    )
    # eda = EDA(config)
    # df_auto = file_cache.load_or_create(
    #     path_auto,
    #     eda.automated_eda,
    #     df=df
    # )
    df_profile = DataPrep.get_data_profile(df)
    df_profile.to_csv(os.path.join(prep_path, 'profile_after_interpolation.csv'))

    ########################################################################################################################################
    ## Feature Engineering
    path_feat = os.path.join(prep_path, 'feat_engineering.pickle')
    feat_eng = FeatureEngineer(config)
    flag_add = True # 거리 분석 포함 여부
    if not os.path.exists(path_feat):
        logger.info('>>>>feat eng 존재하지 않음. 생성 시작...')
        ### Feature Engineering
        feat_eng_data = feat_eng.feature_engineering(df, flag_add=flag_add)
        feat_eng_data.get('concat_scaled')

        #     # VIF 분석 추가
        # numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        # scaler = StandardScaler()
        # df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
        # high_vif_cols = FeatureEngineer.calculate_vif(df_scaled)
        # logger.info(f'높은 VIF 값(>10)을 가진 컬럼들: {high_vif_cols}')
        with open(path_feat, 'wb') as f:
            pickle.dump(feat_eng_data, f)
    else:
        logger.info('>>>>feat eng 존재. Loading...')
        with open(path_feat, 'rb') as f:
            feat_eng_data = pickle.load(f)
    ########################################################################################################################################
    if flag_add:
        dt_train = feat_eng_data.get('dt_train')
    else:
        X_train = feat_eng_data.get('X_train')
        X_val = feat_eng_data.get('X_val')  
        y_train = feat_eng_data.get('y_train')
        y_val = feat_eng_data.get('y_val')

    dt_test = feat_eng_data.get('dt_test')
    path_concat = os.path.join(prep_path, 'df_feature.csv')
    
    if not os.path.exists(path_concat):
        concat = Utils.concat_train_test(dt_train, dt_test)
        concat.to_csv(path_concat, index=False)
    else:
        concat = pd.read_csv(path_concat, index_col=0)
    #   concat = DataPrep.concat_train_test(dt_train, dt_test)
    # DataPrep.unconcat_train_test(concat)
    # concat.to_csv()
    label_encoders = feat_eng_data.get('label_encoders')
    continuous_columns_v2 = feat_eng_data.get('continuous_columns_v2')
    categorical_columns_v2 = feat_eng_data.get('categorical_columns_v2')

    ########################################################################################################################################
    ## Feature Add 1. 거리 분석
    feat_transport, transport_cols = feat_add_transport(config, logger, concat)
    
    ########################################################################################################################################
    ## Feature Add 2. Clustering
    cols_for_clustering = ['아파트명','전용면적','층','건축년도','k-건설사(시공사)','주차대수','강남여부','신축여부','k-주거전용면적' ] + transport_cols
    col_id=['is_test','target']
    col_to_select = ['전용면적', '강남여부','subway_shortest_distance', 'cluster_dist_transport','k-관리비부과면적','계약년', '층', '도로명', '아파트명', '건축년도' ]#'k-주거전용면적'
    config.update({'transport_cols': transport_cols})
    feat_cluster, cluster_cols = feat_add_clustering(config, logger, concat, cols_for_clustering=cols_for_clustering)
    # logger.info('>>>>Feature Selection 존재. Loading...')
    concat = concat_feature(concat, feat_transport, feat_cluster, logger)
   
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
    print(concat.columns)
    dataset_name = config.get('name').get('dataset_name')
    print('##########################')
    #####
    if config.get('name').get('dataset_name') == 'baseline':
        concat = pd.read_csv(os.path.join(prep_path, 'df_baseline.csv'), index_col=0)
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
    
    elif config.get('name').get('dataset_name') == 'feat_scaled_selected':
        print('concat_scaled_selected')
        concat = pd.read_csv(os.path.join(prep_path, 'df_feat_scaled_selected.csv'), index_col=0)
    elif config.get('name').get('dataset_name') == 'encoded':
        concat = pd.read_csv(os.path.join(prep_path, 'df_encoded.csv'), index_col=0)
    elif config.get('name').get('dataset_name') == 'concat_scaled_selected':
        concat = pd.read_csv(os.path.join(prep_path, 'df_concat_scaled_selected.csv'), index_col=0)
    elif config.get('name').get('dataset_name') == 'df_scaled':
        concat = pd.read_csv(os.path.join(prep_path, 'df_scaled.csv'), index_col=0)
    elif config.get('name').get('dataset_name') == 'feat_all3':
        concat = pd.read_csv(os.path.join(prep_path, 'df_feat_all3_encoded.csv'), index_col=0)
    else:
        print('else')
    ########################################################################################################################################
    dt_train, dt_test = Utils.unconcat_train_test(concat)
    logger.info(f'>>>>Train shape: {dt_train.shape}, {dt_train.columns}\nTest shape: {dt_test.shape}, {dt_test.columns}')
    y_train = dt_train['target']
    X_train = dt_train.drop(['target'], axis=1)
    
    print(X_train.columns)
    X_test =dt_test
    
    X_train.columns = [re.sub(r'[^\w]', '_', col) for col in X_train.columns]
    X_test.columns = [re.sub(r'[^\w]', '_', col) for col in X_test.columns]
    print(X_train.columns)
    ########################################################################################################################################
    ##  Model Training
    model = Model(config)
    dataset_name = config.get('name').get('dataset_name')
    model_name = config.get('name').get('model_name')
    split_type = config.get('name').get('split_type')
    
    out_model_path = os.path.join(config.get('path').get('out'), f'saved_model_{model_name}_{dataset_name}_{split_type}.pkl')
    # X_train, y_train = feat_eng._prep_x_y_split_target(dt_train, flag_val=False)
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
    seve_config(config)
    # X_test = dt_test
    # X_test = dt_test.drop(['target'], axis=1)

    # from src.feature import XAI
    # xai = XAI(config)
    # xai.shap_summary(model, X_train, X_test)

    #visualizer = Visualizer(config)
if __name__ == '__main__':
    freeze_support()
    main()
