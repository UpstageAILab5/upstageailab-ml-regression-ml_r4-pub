
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
# 메모리 정리
import gc
gc.collect()

def main():
    get_unique_filename = Utils.get_unique_filename
    # import matplotlib.pyplot as plt
    # import matplotlib.font_manager as fm
    # setup_matplotlib_korean(logger)
    ########################################################################################################################################
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

    logs_path = os.path.join(base_path, 'logs')
    config_path = os.path.join(base_path, 'config')

    out_path = os.path.join(base_path,'output')
    config_file_path = os.path.join(config_path, 'config.yaml')
    print(config_file_path)
    config_base = Utils.load_nested_yaml(config_file_path)
    #config_base = Utils.get_nested_value(loaded_config, 'config')
    config ={   
            'out_path':out_path,
            'base_path':base_path,
            'subway_feature': os.path.join(base_path, 'data','subway_feature.csv'),
            'bus_feature': os.path.join(base_path, 'data','bus_feature.csv'),
            'logger': logger_instance,#logger,
            'wandb': {
                'project': 'project-regression_house_price',     # 필수: wandb 프로젝트명
                'entity': 'joon',          # 필수: wandb 사용자/조직명
                'group': 'group-ml4',    # 선택: 실험 그룹명
            }
        }
    config.update(config_base)
    pprint.pprint(config)

    # logger 객체만 제거한 복사본 생성
    config_to_save = {k: v for k, v in config.items() if k != 'logger'}

    config_log_path = os.path.join(base_path, 'config_log.yaml')
    # YAML 파일로 저장
    with open(config_log_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            config_to_save,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False
        )
    ########################################################################################################################################
    ### Data Prep
    prep_path = os.path.join(base_path, 'data', 'processed')
    path_baseline = os.path.join(prep_path, 'df_baseline_prep.csv')
    path_auto = os.path.join(prep_path, 'df_auto_prep.csv')
    ########################################################################################################################################
    ### EDA
    
    path_feat = os.path.join(prep_path, 'df_feat.csv')
    path_feat_add = os.path.join(prep_path, 'df_feat_add.csv')

    data_prep = DataPrep(config)
    if not os.path.exists(path_auto):
        logger.info('>>>>Data prep or auto eda 존재하지 않음. 생성 시작...')
        df = data_prep.data_prep()
        #df_raw = data_prep._load_data_concat_train_test()
        eda = EDA(config)
        df_auto = eda.automated_eda(df)
        df_auto.to_csv(path_auto)
        df.to_csv(path_baseline)
    else:
        logger.info('>>>>auto eda 존재. Loading...')
        df_auto = pd.read_csv(path_auto)
        df = pd.read_csv(path_baseline)
    ########################################################################################################################################
    ## Feature Engineering
    feat_eng = FeatureEngineer(config)
    flag_add = True # 거리 분석 포함 여부
    if not os.path.exists(path_feat):
        logger.info('>>>>feat eng 존재하지 않음. 생성 시작...')
        ### Feature Engineering
        feat_eng_data = feat_eng.feature_engineering(df, flag_add=flag_add)
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
    label_encoders = feat_eng_data.get('label_encoders')
    continuous_columns_v2 = feat_eng_data.get('continuous_columns_v2')
    categorical_columns_v2 = feat_eng_data.get('categorical_columns_v2')

    # logger.info(f'For concat.Train data shape : {dt_train.shape}, Test data shape : {dt_test.shape}\n{dt_train.head(1)}\n{dt_test.head(1)}')
    # concat = pd.concat([dt_train, dt_test], axis=0).reset_index(drop=True)

    # dt_train['is_test'] = 0
    # dt_test['is_test'] = 1
    # logger.info('is_test column added to train and test data.\nConcat train and test data.')
    # concat = pd.concat([dt_train, dt_test])     # 하나의 데이터로 만들어줍니다.
    # concat['is_test'].value_counts()      # train과 test data가 하나로 합쳐진 것을 확인할 수 있습니다.
    ########################################################################################################################################
    ## Feature Add 1. 거리 분석
    
    if not os.path.exists(path_feat_add):
        feat_add = FeatureAdditional(config)
        logger.info('>>>>feat add 존재하지 않음. 생성 시작...')
        df_coor = {'x': '좌표X', 'y': '좌표Y'}
        subway_coor = {'x': '경도', 'y':'위도' }
        bus_coor = {'x': 'X좌표', 'y': 'Y좌표'}
        subway_feature = pd.read_csv(config.get('subway_feature'))
        bus_feature = pd.read_csv(config.get('bus_feature'))

        logger.info(f'For concat.Train data shape : {dt_train.shape}, Test data shape : {dt_test.shape}\n{dt_train.head(1)}\n{dt_test.head(1)}')
        
    
        #concat = concat.sample(frac=0.001)
        #print(concat.shape)


        if flag_add:
            concat = pd.concat([dt_train, dt_test], axis=0).reset_index(drop=True)
            dt_train['is_test'] = 0
            dt_test['is_test'] = 1
            logger.info('is_test column added to train and test data.\nConcat train and test data.')
            concat = pd.concat([dt_train, dt_test])     # 하나의 데이터로 만들어줍니다.
            concat['is_test'].value_counts()      # train과 test data가 하나로 합쳐진 것을 확인할 수 있습니다.
            #  BallTree 방식 (더 빠르지만 메모리 많이 사용)
            concat, subway_cols = feat_add.distance_analysis_balltree(
                concat, subway_feature, df_coor, subway_coor, target='subway'
            )
            concat, bus_cols = feat_add.distance_analysis_balltree(concat, bus_feature, df_coor, bus_coor, target='bus')
            logger.info(f'For concat.Train data shape : {dt_train.shape}, Test data shape : {dt_test.shape}\n{dt_train.head(1)}\n{dt_test.head(1)}')
            #df.drop(df.columns[0], axis=1, inplace=True)
            concat.to_csv(path_feat_add)
    else:
        logger.info('>>>>feat add 존재. Loading...')
        concat = pd.read_csv(path_feat_add)
        subway_cols = [col for col in concat.columns if 'subway' in col.lower()]
        bus_cols = [col for col in concat.columns if 'bus' in col.lower()]

    transport_cols = subway_cols + bus_cols
    ##### Column cleaning
    unnamed_cols = [col for col in concat.columns if 'Unnamed' in col]
    if unnamed_cols:
        logger.info(f"Removing unnamed columns: {unnamed_cols}")
        concat = concat.drop(columns=unnamed_cols)
    logger.info(f'Total names: {len(concat.columns)} \nConcat data new column names : {transport_cols}')
    ########################################################################################################################################
    ## Feature Add 2. Clustering
    cols = ['아파트명','전용면적','층','건축년도','k-건설사(시공사)','주차대수','강남여부','신축여부','k-주거전용면적' ] + transport_cols
    
    clustering = Clustering(config)
    path_feat_add_cluster_dist = os.path.join(prep_path, 'df_feat_add_cluster_dist.csv')
    if not os.path.exists(path_feat_add_cluster_dist):
        logger.info('>>>>Dist-based Clustering 시작...')
        concat = clustering.dbscan_clustering(concat, features = ['좌표X', '좌표Y'], target='dist')
        concat.to_csv(path_feat_add_cluster_dist)
    else:
        logger.info('>>>>Dist-based Clustering 존재. Loading...')
        concat = pd.read_csv(path_feat_add_cluster_dist)
    logger.info(f'Clustering 결과 : {concat.head(3)}')
    #####
    
    path_feat_add_cluster_dist_transport = os.path.join(prep_path, 'df_feat_add_cluster_dist_transport.csv')
    if not os.path.exists(path_feat_add_cluster_dist_transport):
        logger.info('>>>>Dist/Transport-based Clustering 시작...')
        concat = clustering.dbscan_clustering(concat, features = ['좌표X', '좌표Y']+transport_cols, target='dist_transport')
        concat.to_csv(path_feat_add_cluster_dist_transport)
    else:
        logger.info('>>>>Dist/Transport-based Clustering 존재. Loading...')
        concat = pd.read_csv(path_feat_add_cluster_dist_transport)

    #clustering_analysis = ClusteringAnalysis(config)
    logger.info(f'Clustering 대상 컬럼 : {cols}')
    path_feat_add_cluster = os.path.join(prep_path, 'df_feat_add_cluster.csv')
    
    if not os.path.exists(path_feat_add_cluster):
        logger.info('>>>>Clustering 시작...')
        concat = clustering.dbscan_clustering(concat, features = cols, target='select'  )
        # clustering_analysis.find_optimal_dbscan_params(concat, features = cols,
        #                             min_samples_range = range(2, 10),
        #                             n_neighbors = 5) 
        # concat = clustering_analysis.apply_dbscan_with_saved_params(concat, features =  cols)
        concat.to_csv(path_feat_add_cluster)
    else:
        logger.info('>>>>Clustering 존재. Loading...')
        concat = pd.read_csv(path_feat_add_cluster)

    
    logger.info(f'Clustering 결과 : {concat.head(3)}')
    ########################################################################################################################################
    ##  Split Train/Test
    dt_train, dt_test, continuous_columns_v2, categorical_columns_v2 = feat_eng.split_train_test(concat)
    X_test = dt_test.drop(['target'], axis=1)
    # dt_test['target'] = y_val
    # config['X_train'] = X_train
    # config['X_val'] = X_val
    # config['y_train'] = y_train
    # config['y_val'] = y_val
    # config['model'] = model
    ########################################################################################################################################
    ##  Model Training
    model = Model(config)
    
    model_name = 'xgboost'
    split_type = 'k_fold'
    out_model_path = os.path.join(out_path, f'saved_model_{model_name}_{split_type}.pkl')
    X_train, y_train = feat_eng._prep_x_y_split_target(dt_train, flag_val=False)

    if not os.path.exists(out_model_path):
        logger.info('>>>>Model Training 시작...')
        model, _ = model.model_train(X_train, y_train, model_name, split_type)#model.model_train(X_train, X_val, y_train, y_val, model_name, split_type)
        with open(out_model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        logger.info('>>>>Model Training 존재. Loading...')
        with open(out_model_path, 'rb') as f:
            model = pickle.load(f)
    # dt_train.fillna('missing', inplace=True) 
    # prep_data = {'X_train': X_train,
    #             # 'X_val': X_val,
    #             'y_train': y_train,
    #             # 'y_val': y_val,
    #             'continuous_columns': continuous_columns_v2,
    #             'categorical_columns': categorical_columns_v2
    # }
    # out_path_data = model_instance.save_data(prep_data)
    # # loaded_data = load_data_pkl(out_path_data)
    # # print(loaded_data)
    #feat_eng.select_var(model, X_val, y_val, pred, label_encoders, categorical_columns_v2)

    # # model_name = 'XGB'
    # # type = 'k_fold'
    # # 
    ########################################################################################################################################
    logger.info('>>>>Test dataset에 대한 inference 시작...')
    real_test_pred = model.predict(X_test)
    # #real_test_pred          # 예측값들이 출력됨을 확인할 수 있습니다.

    # # 앞서 예측한 예측값들을 저장합니다.
    preds_df = pd.DataFrame(real_test_pred.astype(int), columns=["target"])

    out_pred_path = get_unique_filename(os.path.join(out_path,f'output_{model_name}_{split_type}.csv'))
    preds_df.to_csv(out_pred_path, index=False)
    logger.info(f'Inference 결과 저장 완료 : {out_pred_path}')
    logger.info(f'{preds_df.head(3)}')
    model.inference(dt_test)
    #visualizer = Visualizer(config)
if __name__ == '__main__':
    freeze_support()
    main()
