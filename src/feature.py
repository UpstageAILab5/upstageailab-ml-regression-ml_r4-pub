from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
# import eli5
# from eli5.sklearn import PermutationImportance
import shap
from pyproj import Transformer

from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors
import warnings
from tqdm import tqdm

from src.utils import Utils

get_unique_filename = Utils.get_unique_filename
warnings.filterwarnings('ignore')

class FeatureAdditional:
    def __init__(self, config):
        self.config = config
        self.logger_instance = config.get('logger')
        self.logger_instance.setup_logger(log_file='feature')
        self.logger = self.logger_instance.logger

    def distance_analysis_balltree(self, building_df, target_feature, building_coor, target_coor, target):
        """
        BallTree를 사용한 거리 분석
        
        Parameters:
            building_df: 건물 데이터프레임 (기준 데이터)
            target_feature: 대상(지하철/버스) 데이터프레임
            building_coor: 건물 좌표 컬럼명 {'x': 'x컬럼명', 'y': 'y컬럼명'}
            target_coor: 대상 좌표 컬럼명 {'x': '경도컬럼명', 'y': '위도컬럼명'}
            target: 대상 유형 ('subway' 또는 'bus')
        """
        self.logger.info(f"### {target} 거리 분석 시작 (BallTree)")
        
        # 데이터 소스별 좌표계 정의
        SOURCE_CRS = {
            'building': 'EPSG:4326',  # 건물: WGS84 경위도
            'bus': 'EPSG:4326',       # 버스정류장: WGS84 경위도
            'subway': 'EPSG:4326'     # 지하철역: WGS84 경위도 (수정)
        }
        
        def validate_coordinates(x, y):
            """서울 지역 좌표 검증 (경도, 위도)"""
            if not (126.5 <= x <= 127.5 and 37.0 <= y <= 38.0):
                self.logger.debug(f"서울 영역 벗어남: (경도: {x}, 위도: {y})")
                return False
            return True
        
        # 좌표 범위 확인
        self.logger.info("\n=== 입력 좌표 범위 ===")
        self.logger.info("건물 좌표:")
        self.logger.info(f"경도: {building_df[building_coor['x']].min():.6f} ~ {building_df[building_coor['x']].max():.6f}")
        self.logger.info(f"위도: {building_df[building_coor['y']].min():.6f} ~ {building_df[building_coor['y']].max():.6f}")
        
        self.logger.info(f"\n{target} 좌표:")
        self.logger.info(f"경도: {target_feature[target_coor['x']].min():.6f} ~ {target_feature[target_coor['x']].max():.6f}")
        self.logger.info(f"위도: {target_feature[target_coor['y']].min():.6f} ~ {target_feature[target_coor['y']].max():.6f}")
        
        # 좌표 검증
        building_transformed = []
        for _, row in tqdm(building_df.iterrows(), desc="건물 좌표 검증"):
            x, y = row[building_coor['x']], row[building_coor['y']]
            if validate_coordinates(x, y):
                building_transformed.append((x, y))
        
        target_transformed = []
        for _, row in tqdm(target_feature.iterrows(), desc=f"{target} 좌표 검증"):
            # 경도, 위도 순서로 저장
            lon, lat = row[target_coor['x']], row[target_coor['y']]
            if validate_coordinates(lon, lat):
                target_transformed.append((lon, lat))
        
        # 변환된 좌표 개수 확인
        self.logger.info("\n=== 유효 좌표 개수 ===")
        self.logger.info(f"건물: {len(building_transformed)} / {len(building_df)}")
        self.logger.info(f"{target}: {len(target_transformed)} / {len(target_feature)}")
        
        if not target_transformed:
            self.logger.error(f"유효한 {target} 좌표가 없습니다!")
            return None
        
        # numpy 배열로 변환
        building_transformed = np.array(building_transformed)
        target_transformed = np.array(target_transformed)
        
        # BallTree 구성 (라디안 변환)
        target_radians = np.radians(target_transformed)
        building_radians = np.radians(building_transformed)
        
        tree = BallTree(target_radians, metric='haversine')
        
        # 반경별 계산
        radius_ranges = {
            'station_area': 200,
            'direct_influence': 500,
            'indirect_influence': 1500
        }
        
        created_columns = []  # 생성된 컬럼명 저장
        
        # 각 반경별 계산
        for zone_name, radius in radius_ranges.items():
            # 반경 변환 (미터 -> 라디안)
            radius_rad = radius / 6371000.0
            
            self.logger.info(f"\n{zone_name} 계산 (반경: {radius}m)")
            
            # 반경 내 이웃 찾기
            counts = tree.query_radius(
                building_radians,
                r=radius_rad,
                count_only=True
            )
            
            # 결과 검증
            count_stats = pd.Series(counts).describe()
            self.logger.info(f"카운트 통계:\n{count_stats}")
            
            # 컬럼 추가
            count_col = f'{target}_{zone_name}_count'
            building_df[count_col] = counts
            created_columns.append(count_col)
        
        # 최단 거리 계산
        distances, _ = tree.query(building_radians, k=1)
        distances = distances * 6371000.0  # 라디안 -> 미터
        
        # 거리 검증
        self.logger.info("\n=== 거리 통계 ===")
        dist_stats = pd.Series(distances.flatten()).describe()
        self.logger.info(f"{dist_stats}")
        
        # 최단 거리 컬럼 추가
        short_col = f'{target}_shortest_distance'
        building_df[short_col] = distances.flatten()
        created_columns.append(short_col)
        
        # 역세권 구분
        zone_col = f'{target}_zone_type'
        building_df[zone_col] = 0
        
        # 역세권 구분 기준 적용
        building_df.loc[building_df[short_col] <= 200, zone_col] = 1  # 역세권
        building_df.loc[(building_df[short_col] > 200) & (building_df[short_col] <= 500), zone_col] = 2  # 직접영향권
        building_df.loc[(building_df[short_col] > 500) & (building_df[short_col] <= 1500), zone_col] = 3  # 간접영향권
        
        created_columns.append(zone_col)
        
        # 최종 결과 검증
        self.logger.info("\n=== 최종 결과 ===")
        for col in created_columns:
            value_counts = building_df[col].value_counts().sort_index()
            self.logger.info(f"\n{col}:\n{value_counts}")
        
        return building_df, created_columns

class FeatureEngineer():
    def __init__(self, config):
        self.config = config
        self.logger_instance = config.get('logger')
        self.logger_instance.setup_logger(log_file='eda')
        self.logger = self.logger_instance.logger
        self.logger.info('#### Init Feature Engineering... ')
        self.out_path = config.get('out_path')
        self.random_seed = config.get('random_seed', 2024)
    ### Feature engineering
    def feature_engineering(self, concat_select, flag_add=False):
        self.logger.info('#### Feature Engineering 시작')
        concat_select = self._prep_feat(concat_select)
        dt_train, dt_test, continuous_columns_v2, categorical_columns_v2 = self.split_train_test(concat_select)
        dt_train, dt_test, label_encoders = self.encode_label(dt_train, dt_test, categorical_columns_v2)
        if flag_add:
            self.logger.info('#### Feature Engineering 완료 - 거리 분석 포함 예정; x, y val 생성 보류.')
            return {'dt_train': dt_train, 'dt_test': dt_test, 'label_encoders': label_encoders, 'continuous_columns_v2': continuous_columns_v2, 'categorical_columns_v2': categorical_columns_v2}
        else:
            X_train, X_val, y_train, y_val = self._prep_x_y_split_target(dt_train)
            self.logger.info('#### Feature Engineering 완료 - 거리 분석 제외; x, y val 생성')
            return {'X_train': X_train, 'X_val': X_val, 'y_train': y_train, 'y_val': y_val, 'dt_test': dt_test, 'label_encoders': label_encoders, 'continuous_columns_v2': continuous_columns_v2, 'categorical_columns_v2': categorical_columns_v2}
    
    def _prep_feat(self, concat_select):
        self.logger.info('### Preparing Features...')
        self.logger.info('시군구, 년월 등 분할할 수 있는 변수들은 세부사항 고려를 용이하게 하기 위해 모두 분할해 주겠습니다.')
        concat_select['구'] = concat_select['시군구'].map(lambda x : x.split()[1])
        concat_select['동'] = concat_select['시군구'].map(lambda x : x.split()[2])
        del concat_select['시군구']

        concat_select['계약년'] = concat_select['계약년월'].astype('str').map(lambda x : x[:4])
        concat_select['계약월'] = concat_select['계약년월'].astype('str').map(lambda x : x[4:])
        del concat_select['계약년월']
        self.logger.info('시군구 -> 구, 동 / 계약년월 -> 계약년, 계약월 변수 분할 생성 완료')
        self.logger.info(concat_select.columns)
        self.logger.info('## 강남 지역: 강서구, 영등포구, 동작구, 서초구, 강남구, 송파구, 강동구 외 구 분리 시작')
        all = list(concat_select['구'].unique())
        gangnam = ['강서구', '영등포구', '동작구', '서초구', '강남구', '송파구', '강동구']
        gangbuk = [x for x in all if x not in gangnam]

        assert len(all) == len(gangnam) + len(gangbuk)       # 알맞게 분리되었는지 체크합니다.

        # 강남의 여부를 체크합니다.
        is_gangnam = []
        for x in concat_select['구'].tolist() :
            if x in gangnam :
                is_gangnam.append(1)
            else :
                is_gangnam.append(0)

        # 파생변수를 하나 만릅니다.
        concat_select['강남여부'] = is_gangnam
        self.logger.info(concat_select.columns)
        # 건축년도 분포는 아래와 같습니다. 특히 2005년이 Q3에 해당합니다.
        # 2009년 이후에 지어진 건물은 10%정도 되는 것을 확인할 수 있습니다.
        concat_select['건축년도'].describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9])
        self.logger.info('따라서 2009년 이후에 지어졌으면 비교적 신축이라고 판단하고, 신축 여부 변수를 제작해보도록 하겠습니다.')
        concat_select['신축여부'] = concat_select['건축년도'].apply(lambda x: 1 if x >= 2009 else 0)
        concat_select.head(1)       # 최종 데이터셋은 아래와 같습니다.
        self.logger.info(f'최종 데이터셋 크기: {concat_select.shape}')
        return concat_select

    def split_train_test(self, concat_select):
        # 이제 다시 train과 test dataset을 분할해줍니다. 위에서 제작해 놓았던 is_test 칼럼을 이용합니다.
        self.logger.info('#### is_test 칼럼을 이용해 train과 test dataset을 분할합니다.')
        dt_train = concat_select.query('is_test==0')
        dt_test = concat_select.query('is_test==1')

        # 이제 is_test 칼럼은 drop해줍니다.
        dt_train.drop(['is_test'], axis = 1, inplace=True)
        dt_test.drop(['is_test'], axis = 1, inplace=True)
        print(dt_train.shape, dt_test.shape)
        dt_test.head(1)
        # dt_test의 target은 일단 0으로 임의로 채워주도록 하겠습니다.
        dt_test['target'] = 0

        # 파생변수 제작으로 추가된 변수들이 존재하기에, 다시한번 연속형과 범주형 칼럼을 분리해주겠습니다.
        continuous_columns_v2 = []
        categorical_columns_v2 = []

        for column in dt_train.columns:
            if pd.api.types.is_numeric_dtype(dt_train[column]):
                continuous_columns_v2.append(column)
            else:
                categorical_columns_v2.append(column)

        self.logger.info(f"연속형 변수: {continuous_columns_v2}")
        self.logger.info(f"범주형 변수: {categorical_columns_v2}")
        return dt_train, dt_test, continuous_columns_v2, categorical_columns_v2

    def encode_label(self, dt_train, dt_test, categorical_columns_v2):
        self.logger.info('#### 범주형 변수들을 대상으로 레이블인코딩을 진행해 주겠습니다.')
        # 각 변수에 대한 LabelEncoder를 저장할 딕셔너리
        label_encoders = {}

        # Implement Label Encoding
        for col in tqdm( categorical_columns_v2 ):
            lbl = LabelEncoder()
     
            # Label-Encoding을 fit
            lbl.fit( dt_train[col].astype(str) )
            dt_train[col] = lbl.transform(dt_train[col].astype(str))
            label_encoders[col] = lbl           # 나중에 후처리를 위해 레이블인코더를 저장해주겠습니다.

            # Test 데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가해줍니다.
            dt_test[col] = dt_test[col].astype(str)
            for label in np.unique(dt_test[col]):
                if label not in lbl.classes_: # unseen label 데이터인 경우
                    lbl.classes_ = np.append(lbl.classes_, label) # 미처리 시 ValueError발생하니 주의하세요!
            dt_test[col] = lbl.transform(dt_test[col].astype(str))

            dt_train.head(1)        # 레이블인코딩이 된 모습입니다.

            assert dt_train.shape[1] == dt_test.shape[1]          # train/test dataset의 shape이 같은지 확인해주겠습니다.
        return dt_train, dt_test, label_encoders

    def _prep_x_y_split_target(self, dt_train, flag_val=True):
        self.logger.info('#### Target과 독립변수들을 분리해줍니다.')
        y_train = dt_train['target']
        X_train = dt_train.drop(['target'], axis=1)
        if flag_val:
                # Hold out split을 사용해 학습 데이터와 검증 데이터를 8:2 비율로 나누겠습니다.
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=self.random_seed)
            return X_train, X_val, y_train, y_val
        else:
            return X_train, y_train

class Clustering:
    def __init__(self, config):
        self.config = config
        self.time_delay = config.get('time_delay', 3)
        self.out_path = config.get('out_path')
        self.logger_instance = config.get('logger')
        #self.logger_instance.setup_logger(log_file='feature')
        self.logger = self.logger_instance.logger
        self.random_seed = config.get('random_seed', 2024)

    # def auto_clustering(self, df):

    #     # 클러스터링 자동화 PyCaret 설정
    #     clf = setup(data=df, session_id=self.random_seed, normalize=True, 
    #                 ignore_features=['feature3'],  # 범주형 제외
    #                 imputation_type='simple')  # 결측치 대체 (기본값은 평균/중앙값 대체)

    #     #evaluate_model(kmeans)
    #     # 다양한 클러스터링 모델 비교 및 최적 모델 추천
    #     best_model = compare_models()
    #     return best_model

    def _knn_for_dbscan_elbow(self, X_scaled, n_neighbors):
        # 데이터 준비 (예시: X_scaled는 스케일링된 수치 데이터)
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors_fit = neighbors.fit(X_scaled)
        distances, indices = neighbors_fit.kneighbors(X_scaled)

        # k-거리 플롯
        distances = np.sort(distances[:, 4], axis=0)  # k=4일 경우 사용
        plt.plot(distances)
        plt.title('k-distance Graph')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        plt.show(block=False)
        title='k-distance Graph_elbow'
        plt.savefig(os.path.join(self.out_path, title +'.png'), dpi=300, bbox_inches='tight')
        plt.pause(self.time_delay)  # 5초 동안 그래프 표시
        plt.close()

    def dbscan_clustering(self, df, features, target, batch_size=100000):
        """
        대용량 데이터를 위한 배치 처리 DBSCAN 클러스터링
        
        Parameters:
            df (DataFrame): 입력 데이터프레임
            features (list): 클러스터링에 사용할 특성 컬럼명 리스트
            target (str): 결과 컬럼명에 추가할 접미사
            batch_size (int): 배치 크기 (기본값: 100000)
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import DBSCAN
        import numpy as np
        from tqdm import tqdm
        
        # 특성 데이터 준비
        X = df[features].copy()
        n_samples = len(X)
        
        # 데이터 스케일링
        self.logger.info("데이터 스케일링 중...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 메모리 효율을 위해 배치 처리
        self.logger.info("DBSCAN 클러스터링 수행 중...")
        cluster_labels = np.zeros(n_samples, dtype=np.int32) - 1  # 기본값 -1 (노이즈)
        
        # 배치 크기 조정
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # DBSCAN 파라미터 최적화
        eps = 0.5  # 거리 임계값
        min_samples = 5  # 최소 이웃 수
        
        try:
            for i in tqdm(range(n_batches)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                batch_data = X_scaled[start_idx:end_idx]
                
                # 배치별 DBSCAN
                dbscan = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    n_jobs=-1,  # 모든 CPU 코어 사용
                    metric='euclidean'
                )
                
                batch_labels = dbscan.fit_predict(batch_data)
                
                # 배치 결과 저장
                cluster_labels[start_idx:end_idx] = batch_labels
                
                # 메모리 정리
                del batch_data
                del dbscan
                import gc
                gc.collect()
        
        except Exception as e:
            self.logger.error(f"클러스터링 중 오류 발생: {str(e)}")
            raise e
        
        # 결과를 새 컬럼으로 추가
        cluster_col_name = f'cluster_{target}'
        df[cluster_col_name] = cluster_labels
        
        # 클러스터링 결과 분석
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # 클러스터별 카운트
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
        # 로깅
        self.logger.info(f"\n=== 클러스터링 결과 ({target}) ===")
        self.logger.info(f"총 클러스터 개수: {n_clusters}")
        self.logger.info(f"노이즈 포인트 개수: {n_noise:,}")
        self.logger.info("\n클러스터별 포인트 수:")
        
        for label, count in cluster_counts.items():
            label_type = "노이즈" if label == -1 else f"클러스터 {label}"
            percentage = (count / len(df)) * 100
            self.logger.info(f"{label_type}: {count:,}개 ({percentage:.2f}%)")
        
        # 클러스터별 카운트를 새로운 컬럼으로 추가
        count_col_name = f'cluster_{target}_count'
        df[count_col_name] = df[cluster_col_name].map(cluster_counts)
        
        return df

    def kmeans_clustering(self, df, features, target, n_clusters=5, batch_size=1000):
        """
        Mini-batch K-means를 사용한 빠른 클러스터링
        
        Parameters:
            df: 데이터프레임
            features: 클러스터링할 특성 리트
            target: 결과 컬럼명 접미사
            n_clusters: 클러스터 개수 (기본값: 5)
            batch_size: 미니배치 크기 (기본값: 1000)
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import MiniBatchKMeans
        import numpy as np
        from time import time
        
        self.logger.info("클러스터링 시작...")
        start_time = time()
        
        # 데이터 준비
        X = df[features].copy()
        
        # 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Mini-batch K-means 클러스터링
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            n_init='auto',
            random_state=42,
            max_iter=100
        )
        
        # 클러스터 레이블 예측
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # 결과 컬럼 추가
        cluster_col_name = f'cluster_{target}'
        df[cluster_col_name] = cluster_labels
        
        # 클러스터별 카운트
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
        # 클러스터 중심과의 거리 계산
        distances = np.min(kmeans.transform(X_scaled), axis=1)
        distance_col_name = f'cluster_{target}_distance'
        df[distance_col_name] = distances
        
        # 각 클러스터의 크기를 새로운 컬럼으로 추가
        count_col_name = f'cluster_{target}_count'
        df[count_col_name] = df[cluster_col_name].map(cluster_counts)
        
        # 실행 시간 계산
        execution_time = time() - start_time
        
        # 결과 로깅
        self.logger.info(f"\n=== 클러스터링 결과 ({target}) ===")
        self.logger.info(f"실행 시간: {execution_time:.2f}초")
        self.logger.info(f"총 클러스터 개수: {n_clusters}")
        self.logger.info("\n클러스터별 포인트 수:")
        
        for label, count in cluster_counts.items():
            percentage = (count / len(df)) * 100
            self.logger.info(f"클러스터 {label}: {count:,}개 ({percentage:.2f}%)")
        
        # 클러스터 통계
        cluster_stats = df.groupby(cluster_col_name)[features].agg(['mean', 'std']).round(2)
        self.logger.info("\n클러스터별 통계:")
        self.logger.info(f"\n{cluster_stats}")
        
        return df
    

    
class XAI:
    def __init__(self, config):
        self.time_delay = config.get('time_delay', 3)
        self.config = config
        self.model = config.get('model')
        self.X_train = config.get('X_train')
        self.X_val = config.get('X_val')
        self.y_train = config.get('y_train')
        self.y_val = config.get('y_val')
        self.random_state = config.get('random_seed')
        self.logger_instance = config.get('logger')
        self.logger_instance.setup_logger(log_file='feature')
        self.logger = self.logger_instance.logger

    def shap_summary(self, model, X_train, X_val, y_val, y_train):
  
                # Explain the model predictions using SHAP
        explainer = shap.Explainer(model, X_train)  # Create SHAP explainer
        shap_values = explainer(X_val)  # Calculate SHAP values for the test set

        # Force plot
        plt.figure(figsize=(12, 6))
        shap.initjs()
        force_plot = shap.force_plot(
            explainer.expected_value, 
            shap_values[0].values, 
            X_val.iloc[0],
            show=False
        )
        plt.savefig(get_unique_filename(os.path.join(self.config['out_path'], 'shap_force_plot.png')), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bar Plot for Feature Importance
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig(get_unique_filename(os.path.join(self.config['out_path'], 'shap_feature_importance.png')), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("SHAP plots saved successfully")
    # Squared_error를 계산하는 함수를 정의하겠습니다.

    # def calculate_se(self, target, pred):
    #     squared_errors = (target - pred) ** 2
    #     return squared_errors

    # def select_var_importance(self, model, X_val, y_val, pred, label_encoders, categorical_columns_v2):
    #     # Permutation importance 방법을 변수 선택에 이용해보겠습니다.
    #     perm = PermutationImportance(model,        # 위에서 학습된 모델을 이용하겠습니다.
    #                                 scoring = 'neg_mean_squared_error',        # 평가 지표로는 회귀문제이기에 negative rmse를 사용합니다. (neg_mean_squared_error : 음의 평균 제곱 오차)
    #                                 random_state = self.random_seed,
    #                                 n_iter=3).fit(X_val, y_val)
    #     eli5.show_weights(perm, feature_names = X_val.columns.tolist())    # valid data에 대해 적시킵니다.

    #     # Validation dataset에 target과 pred 값을 채워주도록 하겠습니다.
    #     X_val['target'] = y_val
    #     X_val['pred'] = pred

    #     # RMSE 계산
    #     squared_errors = self.calculate_se(X_val['target'], X_val['pred'])
    #     X_val['error'] = squared_errors

    #     # Error가 큰 순서대로 sorting 해 보겠습니다.
    #     X_val_sort = X_val.sort_values(by='error', ascending=False)       # 내림차순 sorting

    #     X_val_sort.head()

    #     X_val_sort_top100 = X_val.sort_values(by='error', ascending=False).head(100)        # 예측을 잘 하지못한 top 100개의 data
    #     X_val_sort_tail100 = X_val.sort_values(by='error', ascending=False).tail(100)       # 예측을 잘한 top 100개의 data

    #     # 해석을 위해 레이블인코딩 된 변수를 복원해줍니다.
    #     error_top100 = X_val_sort_top100.copy()
    #     for column in categorical_columns_v2 :     # 앞서 레이블 인코딩에서 정의했던 categorical_columns_v2 범주형 변수 리스트를 사용합니다.
    #         error_top100[column] = label_encoders[column].inverse_transform(X_val_sort_top100[column])

    #     best_top100 = X_val_sort_tail100.copy()
    #     for column in categorical_columns_v2 :     # 앞서 레이블 인코딩에서 정의했던 categorical_columns_v2 범주형 변수 리스트를 사용합니다.
    #         best_top100[column] = label_encoders[column].inverse_transform(X_val_sort_tail100[column])

    #     print(error_top100.head(1))
    #     print(best_top100.head(1))

    #     sns.boxplot(data = error_top100, x='target')
    #     title_worst = 'The worst top100 prediction의 target 분포'
    #     plt.title(title_worst)
    #     plt.show()
    #     plt.savefig(os.path.join(self.out_path, title_worst +'.png'), dpi=300, bbox_inches='tight')

    #     sns.boxplot(data = best_top100, x='target', color='orange')
    #     title_best = 'The best top100 prediction의 target 분포'
    #     plt.title(title_best)
    #     plt.show()
    #     plt.savefig(os.path.join(self.out_path, title_best +'.png'), dpi=300, bbox_inches='tight')

    #     sns.histplot(data = error_top100, x='전용면적', alpha=0.5)
    #     sns.histplot(data = best_top100, x='전용면적', color='orange', alpha=0.5)
    #     title_hist = '전용면적 분포 비교'
    #     plt.title(title_hist)
    #     plt.savefig(os.path.join(self.out_path, title_hist +'.png'), dpi=300, bbox_inches='tight')
    #     plt.show()
