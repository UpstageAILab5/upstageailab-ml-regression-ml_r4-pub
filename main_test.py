
from pathlib import Path
import os
# import pygwalker as pyg
# import dabl
import pandas as pd

from src.logger import Logger
from src.preprocessing import DataPrep
from src.eda import EDA
from src.feature import FeatureEngineer, FeatureAdditional
from src.clustering import Clustering, ClusteringAnalysis
from src.train import Model
from src.visualization import Visualizer
# from src.utils import setup_matplotlib_korean
import pickle
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
from src.utils import get_unique_filename
import platform

def setup_font():
    # 운영체제 확인 후 폰트 설정
    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')  # Windows
    elif platform.system() == 'Darwin':          # Mac
        plt.rc('font', family='AppleGothic')
    else:
        plt.rc('font', family='NanumGothic')    # Linux
    # 마이너스 기호 깨짐 방지
    plt.rc('axes', unicode_minus=False)


def main():
    # import matplotlib.pyplot as plt
    # import matplotlib.font_manager as fm
    logger_instance = Logger()
    logger = logger_instance.logger
    setup_font()
    # setup_matplotlib_korean(logger)
    ########################################################################################################################################
    # 테스트
    # plt.figure(figsize=(3, 1))
    # plt.text(0.5, 0.5, '한글 테스트', ha='center', va='center')
    # plt.axis('off')
    # plt.show()
    if platform.system() == 'Windows':
        base_path = Path(r'D:\dev\upstageailab5-ml-regression-ml_r4')
    elif platform.system() == 'Darwin':          # Mac
        base_path = Path('/data/ephemeral/home/dev/upstageailab5-ml-regression-ml_r4')
    else:
        base_path = Path('/data/ephemeral/home/dev/upstageailab5-ml-regression-ml_r4')    # Linux
    # 마이너스 기호 깨짐 방지
    
    # setup_matplotlib_korean(logger)
    #
    out_path = os.path.join(base_path,'output')
    config ={   
            'out_path':out_path,
            'base_path':base_path,
            'subway_feature': os.path.join(base_path, 'data','subway_feature.csv'),
            'bus_feature': os.path.join(base_path, 'data','bus_feature.csv'),
            'logger': logger,
            'random_seed': 2024,
            'target': 'target',
            'thr_ratio_outlier': 0.01,
            'thr_ratio_null': 0.9,
            'wandb': {
                'project': 'project-regression_house_price',     # 필수: wandb 프로젝트명
                'entity': 'joon',          # 필수: wandb 사용자/조직명
            # 'name': 'run-name',            # 선택: 실험 실행 이름 (지정하지 않으면 자동 생성)
                'group': 'group-ml4',    # 선택: 실험 그룹명
                #'tags': ['tag1', 'tag2'],      # 선택: 실험 태그
                #'notes': 'experiment notes'     # 선택: 실험 노트
        }}

    # walker = pyg.walk(
    #     df,
    #     spec="./chart_meta_0.json",    # this json file will save your chart state, you need to click save button in ui mannual when you finish a chart, 'autosave' will be supported in the future.
    #     kernel_computation=True,          # set `kernel_computation=True`, pygwalker will use duckdb as computing engine, it support you explore bigger dataset(<=100GB).
    # )
    ########################################################################################################################################
    ### Data Prep
    data_prep = DataPrep(config)
    eda = EDA(config)
    feat_eng = FeatureEngineer(config)
    clustering = Clustering(config)
    clustering_analysis = ClusteringAnalysis(config)
    model = Model(config)
    visualizer = Visualizer(config)
    ########################################################################################################################################
    ### EDA
    prep_path = os.path.join(base_path, 'data', 'processed')
    path_baseline = os.path.join(prep_path, 'df_baseline_prep.csv')
    path_feat = os.path.join(prep_path, 'df_feat.csv')

    path_auto = os.path.join(prep_path, 'df_auto_prep.csv')
    path_feat_add = os.path.join(prep_path, 'df_feat_add.csv')

    if not os.path.exists(path_auto):
        logger.info('>>>>auto eda 존재하지 않음. 생성 시작...')
        df = data_prep.data_prep()
        #df_raw = data_prep._load_data_concat_train_test()
        df_auto = eda.automated_eda(df)
        df_auto.to_csv(path_auto)
        df.to_csv(path_baseline)
    else:
        logger.info('>>>>auto eda 존재. Loading...')
        df_auto = pd.read_csv(path_auto)
        df = pd.read_csv(path_baseline)
    ########################################################################################################################################
    ## Feature Engineering
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
        logger.info('>>>>feat add 존재하지 않음. 생성 시작...')
        feat_add = FeatureAdditional(config)
        df_coor = {'x': '좌표X', 'y': '좌표Y'}
        subway_coor = {'x': '위도', 'y': '경도'}
        bus_coor = {'x': 'X좌표', 'y': 'Y좌표'}
        subway_feature = pd.read_csv(config.get('subway_feature'))
        bus_feature = pd.read_csv(config.get('bus_feature'))

        logger.info(f'For concat.Train data shape : {dt_train.shape}, Test data shape : {dt_test.shape}\n{dt_train.head(1)}\n{dt_test.head(1)}')
        concat = pd.concat([dt_train, dt_test], axis=0).reset_index(drop=True)

        dt_train['is_test'] = 0
        dt_test['is_test'] = 1
        logger.info('is_test column added to train and test data.\nConcat train and test data.')
        concat = pd.concat([dt_train, dt_test])     # 하나의 데이터로 만들어줍니다.
        concat['is_test'].value_counts()      # train과 test data가 하나로 합쳐진 것을 확인할 수 있습니다.
            # 1. Numba + 병렬처리 조합 (메모리 효율적)
        # df, cols = feat_add.distance_analysis_optimized(
        #     df, subway_feature, df_coor, subway_coor, radius=500, target='subway'
        # )
        # 2. BallTree 방식 (더 빠르지만 메모리 많이 사용)
        concat, subway_cols = feat_add.distance_analysis_balltree(
            concat, subway_feature, df_coor, subway_coor, radius=500, target='subway'
        )
        #concat, subway_cols = feat_add.distance_analysis_parallel(concat, subway_feature, df_coor, subway_coor, 500, 'subway')
        concat, bus_cols = feat_add.distance_analysis_balltree(concat, bus_feature, df_coor, bus_coor, 500, 'bus')
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
    ########################################################################################################################################
    cols = ['아파트명','전용면적','층','건축년도','k-건설사(시공사)','주차대수','강남여부','신축여부','k-주거전용면적' ] + transport_cols
    ## Feature Add 2. Clustering
    logger.info(f'Clustering 대상 컬럼 : {cols}')
    path_feat_add_cluster = os.path.join(prep_path, 'df_feat_add_cluster.csv')
    if not os.path.exists(path_feat_add_cluster):
        logger.info('>>>>Clustering 시작...')
        clustering_analysis.find_optimal_dbscan_params(concat, features = cols,
                                    min_samples_range = range(2, 10),
                                    n_neighbors = 5) 
        concat = clustering_analysis.apply_dbscan_with_saved_params(concat, features =  cols)
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
if __name__ == '__main__':
    freeze_support()
    main()
