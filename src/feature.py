import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, VarianceThreshold, RFECV
from sklearn.feature_selection import SelectFromModel, RFE

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.ensemble import RandomForestRegressor

from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import Ridge

import math
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.neighbors import BallTree
from scipy import stats  

class FeatureEngineer():
    def __init__(self):
        print('#### Init Feature Engineering... ')
    @staticmethod
    def prep_feat(concat_select, year = 2009,  col_add=''):
        # 시군구, 년월 등 분할할 수 있는 변수들은 세부사항 고려를 용이하게 하기 위해 모두 분할해 주겠습니다.
                # '계약년' 열이 존재하지 않는 경우에만 추가
        try:
            if '구' not in concat_select.columns:
                concat_select['구'] = concat_select['시군구'].map(lambda x : x.split()[1])
            else:
                print('##### 구 열이 이미 존재합니다.')
            if '동' not in concat_select.columns:
                concat_select['동'] = concat_select['시군구'].map(lambda x : x.split()[2])
            else:
                print('##### 동 열이 이미 존재합니다.')
        except:
            print('##### 시군구 열이 없습니다.')
        try:    
            # concat_select['계약년'] = concat_select['계약년월'].astype('str').str[:4]
            concat_select['계약년'] = concat_select['계약년월'].astype('str').apply(lambda x: x[:4])
            concat_select['계약월'] = concat_select['계약년월'].astype('str').apply(lambda x: x[4:])
        
        #concat_select['계약년월'] = concat_select['계약년월'].astype('str').apply(lambda x: x[:4] + x[4:])
        except:
            print('##### 계약년월 열이 없습니다.')
        try:
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
        except:
            print('##### 강남여부 열이 없습니다.')
        
        try:
            # 건축년도 분포는 아래와 같습니다. 특히 2005년이 Q3에 해당합니다.
            # 2009년 이후에 지어진 건물은 10%정도 되는 것을 확인할 수 있습니다.
            concat_select['건축년도'].describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9])

            # 따라서 2009년 이후에 지어졌으면 비교적 신축이라고 판단하고, 신축 여부 변수를 제작해보도록 하겠습니다.
            concat_select['신축여부'] = concat_select['건축년도'].apply(lambda x: 1 if x >= int(year) else 0)
        except:
            print('##### 신축여부 열이 없습니다.')
        try:
            if col_add == 'address':
                concat_select['시군구+���지'] = concat_select['시구'].astype(str) + concat_select['번지'].astype(str)
        except:
            print('No address column')
        #concat_select.head(1)       # 최종 데이터셋은 아래와 같습니다.
        del concat_select['계약년월']
        del concat_select['시군구']

        return concat_select
    @staticmethod
    def prep_null_coord(df, col_coord=['좌표X', '좌표Y']):
        df['주소'] = df['시군구'] + ' ' + df['도로명'] #+ ' ' + df['아파트명']
        df_with_nulls = df[df[col_coord].isnull().any(axis=1)]  
        df_without_nulls = df[~df[col_coord].isnull().any(axis=1)]
        print(f'총 {df_with_nulls.shape[0]}개의 아파트 주소가 좌표 없음\n{df_without_nulls.shape[0]}개의 아파트 주소가 좌표 있음')

        print(f'null : {df_with_nulls.isnull().sum()}')
        print(f'null: {df_with_nulls.head(1)}')
        print(f'normal: {df_without_nulls.isnull().sum()}')
        print(f'normal: {df_without_nulls.head(1)}')
        return df_with_nulls, df_without_nulls
    @staticmethod
    def _get_lat_lon(address):
        
        """
        주소를 입력받아 해당 위치의 경도와 위도를 반환합니다.

        load_dotenv('/data/ephemeral/home/RealEstatePricePredictor/.env'): 
        .env 파일을 현재 프로젝트 폴더 경로에서 불러니다.
        .env 파일에서 KAKAO_API_KEY를 환경 변수로 불러와 api_key에 저장합니다.
        get_lat_lon 함수에서 이 api_key를 사용하여 카카오 API에 요청을 보냅니다.
        """
        path_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        print(path_env)
        load_dotenv(path_env)
        api_key = os.getenv("KAKAO_API_KEY")

        url = "https://dapi.kakao.com/v2/local/search/address.json"
        headers = {"Authorization": f"KakaoAK {api_key}"}
        params = {"query": address}
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            result = response.json()
            if result['documents']:
                address_info = result['documents'][0]['address']
                try:
                    latitude = float(address_info['y'])
                    longitude = float(address_info['x'])
                    return latitude, longitude
                except:
                    print(f'주소: {address} 좌표 변환 오류. address_info: {address_info}')
                    return None, None
            else:
                print(">>>>>>>>>해당 주소에 대한 결과가 없습니다.")
                return None, None
        else:
            print(f"Error: {response.status_code}")
            return None, None
    @staticmethod
    def get_coord_from_address(df, col_address='주소', prep_path=None):

        count = 0
        grouped = df.groupby(col_address)   
        print(f'총 {len(grouped)}개의 아파트 그룹 {col_address}가 좌표 없음')
        for i, (address, group) in tqdm(enumerate(grouped), total=len(grouped)):
            print(f'\n{i+1}번째 아파트 그룹: {address}, 그룹 내 데이터 수: {len(group)}')
            # 주소로부터 위도와 경도를 가져옴
            lat, long = FeatureEngineer._get_lat_lon(address)
            # 위도와 경도가 유효한 경우에만 할당
            if lat is not None and long is not None:
                df.loc[group.index, ['좌표X', '좌표Y']] = long, lat
                print(f'X long: {long}, Y lat: {lat}')
            else:
                print(">>>>>>>유효한 좌표를 찾을 수 없습니다.")
                count+=1
            print(df.loc[group.index][['주소', '좌표X', '좌표Y']].head(1))
            if i % 1000 == 0:
                df.to_csv(os.path.join(prep_path, f'backup_df_prep_null_coord_{i}.csv'), index=True)

        print(f'총 {len(grouped)}개의 아파트 그룹 중 {count}개의 아파트 주소가 좌 변환 오류')
        return df
    # 위경도를 이용해 두 지점간의 거리를 구하는 함수를 생성합니다.
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        radius = 6371.0

        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = radius * c
        return distance
    @staticmethod
    def distance_apt(df_source,  dict_target,  feature_source=['좌표X', '좌표Y'], feature_target=['대장_좌표X', '대장_좌표Y'], target='대장아파트'):
            # 대장 아파트의 위경도 데이터프레임을 구성합니다.
        if target == '대장아파트':  
            df_target = pd.DataFrame([{"구": k, feature_target[0]: v[1], feature_target[1]: v[0]} for k, v in dict_target.items()])
            # 데이터프레임간 결합을 합니다.
        else:
            df_target = dict_target
        target_col = f'{target}_거리'

        df_source = pd.merge(df_source, df_target, how="inner", on="구")
            # 아까 작한 haversine_distance 함수를 이용해 대장아파트와의 거리를 계산하고, 새롭게 컬럼을 구성합니다.
        df_source[target_col] = df_source.apply(lambda row: FeatureEngineer.haversine_distance(row[feature_source[1]], row[feature_source[0]], row[feature_target[1]], row[feature_target[0]]), axis=1)
        return df_source, [target_col]
    @staticmethod
    def distance_analysis(building_df, target_feature, building_coor, target_coor, target):
        """
        Haversine 거리를 사용한 거리 분석
        """
        # 반경별 계산
        distance_1 = 200
        distance_2 = 500
        distance_3 = 1500
        radius_ranges = {
            '0-1': distance_1,
            '1-2': distance_2,
            '2-3': distance_3
        }
        
        # 좌표 검증 및 변환
        building_transformed = []
        for _, row in tqdm(building_df.iterrows(), desc="건물 좌표 검증"):
            x, y = row[building_coor['x']], row[building_coor['y']]
            building_transformed.append((x, y))
        
        target_transformed = []
        for _, row in tqdm(target_feature.iterrows(), desc=f"{target} 좌표 검증"):
            lon, lat = row[target_coor['x']], row[target_coor['y']]
            target_transformed.append((lon, lat))
        
        # 거 계산
        created_columns = []
        
        for zone_name, radius in radius_ranges.items():
            count_col = f'{target}_{zone_name}_count'
            short_col = f'{target}_{zone_name}_shortest_distance'
            zone_col = f'{target}_{zone_name}_zone_type'
            
            building_df[count_col] = 0
            building_df[short_col] = np.inf
            building_df[zone_col] = 0
            
            for i, (bx, by) in enumerate(building_transformed):
                for tx, ty in target_transformed:
                    distance = FeatureEngineer.haversine_distance(by, bx, ty, tx)
                    
                    # Count
                    if distance <= radius:
                        building_df.at[i, count_col] += 1
                    
                    # Shortest Distance
                    if distance < building_df.at[i, short_col]:
                        building_df.at[i, short_col] = distance
                    
                    # Zone Type
                    if distance <= radius:
                        building_df.at[i, zone_col] = 1
            
            created_columns.extend([count_col, short_col, zone_col])
        
        # 최종 결과 검증
        print("\n=== 최종 결과 ===")
        for col in created_columns:
            value_counts = building_df[col].value_counts().sort_index()
            print(f"\n{col}:\n{value_counts}")
        
        return building_df, created_columns
    @staticmethod
    def distance_analysis_old(building_df, target_feature, building_coor, target_coor, target):
        """
        BallTree를 사용한 거리 분석
        
        Parameters:
            building_df: 건물 데이터프레임 (기준 데이터)
            target_feature: 대상(지하철/버스) 데이터프레임
            building_coor: 건물 좌표 컬럼명 {'x': 'x컬럼명', 'y': 'y컬럼명'}
            target_coor: 대상 좌표 컬럼명 {'x': '경도컬럼명', 'y': '위도컬럼명'}
            target: 대상 유형 ('subway' 또는 'bus')
        """
        # 데이터 소스별 좌표계 정의
        SOURCE_CRS = {
            'building': 'EPSG:4326',  # 건물: WGS84 경위도
            'bus': 'EPSG:4326',       # 버스정류장: WGS84 경위도
            'subway': 'EPSG:4326',     # 지하철역: WGS84 경위도 (수정)
            'gangnam_apt': 'EPSG:4326' # 강남아파트: 
        }
        
        def validate_coordinates(x, y):
            """서울 지역 좌표 검증 (경도, 위도)"""
            if not (126.5 <= x <= 127.5 and 37.0 <= y <= 38.0):
                print(f"서울 영역 벗어남: (경도: {x}, 위도: {y})")
                return False
            return True
        
        # 좌표 범위 확인
        print("\n=== 입력 좌표 범위 ===")
        print("건물 좌표:")
        print(f"경도: {building_df[building_coor['x']].min():.6f} ~ {building_df[building_coor['x']].max():.6f}")
        print(f"위도: {building_df[building_coor['y']].min():.6f} ~ {building_df[building_coor['y']].max():.6f}")
        
        print(f"\n{target} 좌표:")
        print(f"경도: {target_feature[target_coor['x']].min():.6f} ~ {target_feature[target_coor['x']].max():.6f}")
        print(f"위도: {target_feature[target_coor['y']].min():.6f} ~ {target_feature[target_coor['y']].max():.6f}")
        
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
        print("\n=== 유효 좌표 개수 ===")
        print(f"건물: {len(building_transformed)} / {len(building_df)}")
        print(f"{target}: {len(target_transformed)} / {len(target_feature)}")
        
        if not target_transformed:
            print(f"유효한 {target} 좌표가 없습니다!")
            return None
        # numpy 배열로 변환
        building_transformed = np.array(building_transformed)
        target_transformed = np.array(target_transformed)
        
        # BallTree 구성 (라디안 변환)
        target_radians = np.radians(target_transformed)
        building_radians = np.radians(building_transformed)
        tree = BallTree(target_radians, metric='haversine')
        # 반경별 계산
        distance_1 = 200
        distance_2 = 500
        distance_3 = 1500
        radius_ranges = {
            'station_area': distance_1,
            'direct_influence': distance_2,
            'indirect_influence': distance_3
        }
        
        created_columns = []  # 생성된 컬럼명 저장
        
        if target != 'gangnam_apt':
            # 각 반경별 계산
            for zone_name, radius in radius_ranges.items():
                # 경 환 (미터 -> 라디안)
                radius_rad = radius / 6371000.0
                print(f"\n{zone_name} 계산 (반경: {radius}m)")
                # 반경 내 이웃 찾기
                counts = tree.query_radius(
                    building_radians,
                    r=radius_rad,
                    count_only=True
                )
                
                # 결과 검증
                count_stats = pd.Series(counts).describe()
                print(f"카운트 통계:\n{count_stats}")

                # 컬럼 추가
                count_col = f'{target}_{zone_name}_count'
                building_df[count_col] = counts
                created_columns.append(count_col)
        
        # 최단 거리 계산
        distances, _ = tree.query(building_radians, k=1)
        distances = distances * 6371000.0  # 라디안 -> 미터
        
        # 거리 검증
        print("\n=== 거리 통계 ===")
        dist_stats = pd.Series(distances.flatten()).describe()
        print(f"{dist_stats}")
        
        # 최단 거리 컬럼 추가
        short_col = f'{target}_shortest_distance'
        building_df[short_col] = distances.flatten()
        created_columns.append(short_col)
        
        # 역세권 구분
        zone_col = f'{target}_zone_type'
        building_df[zone_col] = 0
        
        # 역세권 구분 기준 적용
        building_df.loc[building_df[short_col] <= distance_1, zone_col] = 1  # 역세권
        building_df.loc[(building_df[short_col] > distance_1) & (building_df[short_col] <= distance_2), zone_col] = 2  # 직접영향권
        building_df.loc[(building_df[short_col] > distance_2) & (building_df[short_col] <= distance_3), zone_col] = 3  # 간접영향권
        
        created_columns.append(zone_col)
        
        # 최종 결과 검증
        print("\n=== 최종 결과 ===")
        for col in created_columns:
            value_counts = building_df[col].value_counts().sort_index()
            print(f"\n{col}:\n{value_counts}")
        
        return building_df, created_columns
class FeatureSelect:
    def __init__(self):
        pass
    @staticmethod
    def filter_method(X_train, X_test, continuous_columns, categorical_columns):
        #print("Original DataFrame:\n", df.shape)
        #print("\nTransformed DataFrame:\n", df_transformed.shape)
        # FeatureSelect.calculate_vif(X_train, continuous_columns)
        # FeatureSelect.cramers_v_all(X_train, categorical_columns)
        cols_var = FeatureSelect._variance_threshold(X_train)

        #.to_csv(os.path.join(prep_path, 'df_combined_distance_feature_after_null_fill_reduced_var.csv'))
        cols_corr = FeatureSelect._corr_threshold(X_train) 

        # reduced_train_corr = reduced_train[cols_corr]
        # reduced_test_corr = reduced_test[cols_corr]

        # print(cols_corr, reduced_train_corr.shape)
        return cols_var, cols_corr
    @staticmethod
    def _variance_threshold(df, threshold = 0.1): 
        selector = VarianceThreshold(threshold=threshold)
        reduced_df = selector.fit_transform(df)

        print("Original DataFrame:\n", df.shape)
        print("Var based Reduced DataFrame:\n", pd.DataFrame(reduced_df).shape)
        selected_features = df.columns[selector.get_support()] 
        unselected_features = df.columns[~selector.get_support()]
        print(f'\n####### Selected Columns: {len(selected_features)}\n{selected_features}\n')
        print(f'########### Unselected Columns: {len(unselected_features)}\n{unselected_features}\n')
        return selected_features
    @staticmethod
    def _corr_threshold(df, threshold = 0.8):
        correlation_matrix = df.corr()
        # 상관계수 행렬을 절대값 기준으로 상관계수 기준 설정
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        # 상관계수가 threshold 이상인 열 이름 선택
        selected_features = [column for column in upper_triangle.columns if any(upper_triangle[column].abs() <= threshold)]
        # 선택된 변수 제거
        unselected_features = [column for column in df.columns if column not in selected_features]
        print(f'\n####### Selected Columns: {len(selected_features)}\n{selected_features}\n')
        print(f'########### Unselected Columns: {len(unselected_features)}\n{unselected_features}\n')
        return selected_features
    
    # 1. VIF 계산 (numerical columns)
    @staticmethod
    def calculate_vif(dataframe, numerical_columns, vif_threshold):
        print('\n##### VIF Analysis\n')
        numerical_columns = [col for col in numerical_columns if col in dataframe.columns]
        X = dataframe[numerical_columns].copy()

        if X.isnull().values.any():
            print("데이터에 결측값이 있습니다. 결측값을 처리하세요.")
        if np.isinf(X.values).any():
            print("데이터에 무한대 값이 있습니다. 무한대 값을 처리하세요.")
        
        X = X.assign(intercept=1)  # 상수항 추가
        vif_df = pd.DataFrame()
        vif_df['Feature'] = X.columns
        vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print(f'VIF: {vif_df}')
        high_vif_columns = vif_df[vif_df['VIF'] > vif_threshold]['Feature'].tolist()
        print(f'High VIF Columns: {high_vif_columns}')
        return high_vif_columns

    # 2. Cramer's V 계산 (categorical columns)
    @staticmethod
    def cramers_v(x, y):
        """
        두 범주형 변수 간의 연관성을 측정하는 Cramér's V 계산
        """
        try:
            # Series로 변환 및 결측치 처리
            x = pd.Series(x).fillna('missing')
            y = pd.Series(y).fillna('missing')
            
            print(f"Data types after conversion - x: {x.dtype}, y: {y.dtype}")
            print(f"Unique values - x: {len(x.unique())}, y: {len(y.unique())}")
            
            # 범주형으로 변환
            x = x.astype('category')
            y = y.astype('category')
            
            # 교차표 생성
            confusion_matrix = pd.crosstab(x, y)
            print(f"Contingency table shape: {confusion_matrix.shape}")
            
            if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
                print("Warning: Not enough categories for Cramer's V calculation")
                return 0
            
            # chi-square 검정
            chi2 = stats.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            
            # Cramér's V 계산
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            
            if min((kcorr-1), (rcorr-1)) <= 0:
                print("Warning: Invalid dimensions for Cramer's V calculation")
                return 0
            
            return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
            
        except Exception as e:
            print(f"Error calculating Cramer's V: {e}")
            print(f"Error type: {type(e).__name__}")
            return 0

    @staticmethod
    def cramers_v_all(df, categorical_columns, threshold):
        """
        모든 범주형 변수 쌍에 대해 Cramér's V 계산
        """
        cramers_v_pairs = {}
        features_to_drop = set()
        
        # 범주형 변수만 선택
        df_cat = df[categorical_columns].copy()
        
        for i in range(len(categorical_columns)):
            for j in range(i + 1, len(categorical_columns)):
                col1 = categorical_columns[i]
                col2 = categorical_columns[j]
                
                try:
                    print(f"\nProcessing columns: {col1} and {col2}")
                    
                    # Series로 명시적 변환
                    x = df_cat[col1].squeeze()
                    y = df_cat[col2].squeeze()
                    
                    print(f"Column types: {x.dtype}, {y.dtype}")
                    print(f"Sample values - {col1}: {x.head().tolist()}, {col2}: {y.head().tolist()}")
                    
                    # 데이터가 2차원인 경우 1차원으로 변��
                    if isinstance(x, pd.DataFrame):
                        x = x.iloc[:, 0]
                    if isinstance(y, pd.DataFrame):
                        y = y.iloc[:, 0]
                    
                    cramer_value = FeatureSelect.cramers_v(x, y)
                    cramers_v_pairs[(col1, col2)] = cramer_value
                    
                    print(f"Cramer's V value for {col1} and {col2}: {cramer_value}")
                    
                    if cramer_value > threshold:
                        features_to_drop.add(col2)
                        print(f"Adding {col2} to features to drop (Cramer's V: {cramer_value})")
                        
                except Exception as e:
                    print(f"Error processing columns {col1} and {col2}: {e}")
                    print(f"Error type: {type(e).__name__}")
                    cramers_v_pairs[(col1, col2)] = 0
        
        return cramers_v_pairs, list(features_to_drop)
    @staticmethod
    def select_features_by_kbest(X_sampled, y_sampled, original_column_names, score_func, k=20):
        k = min(k, X_sampled.shape[1])  # 특성 수의 절반 또는 20개
        print(f'k: {k}')
        
        selector = SelectKBest(score_func=score_func, k=k)
        X_filtered = selector.fit_transform(X_sampled, y_sampled)
        selected_features = original_column_names[selector.get_support()]
        X_train_sampled = pd.DataFrame(X_filtered, columns=selected_features)
        #X_test_sampled = pd.DataFrame(X_filtered, columns=original_column_names[selector.get_support()])
        print('#### Filtered columns by SelectKBest', selected_features)
        return X_train_sampled, selected_features#, X_test_sampled
    @staticmethod
    def compare_selected_features(selected_rfe, selected_sfs, list_features, original_column_names):
        set_rfe = set(selected_rfe)
        set_sfs = set(selected_sfs)
        # 공통 집합 계산
        common_features = set_rfe.intersection(set_sfs)
        union_features = set_rfe.union(set_sfs)

        # 결과 출력
        print(f"{list_features[0]}로 선택된 피처: \n{selected_rfe}\n")
        print(f"{list_features[1]}로 선택된 피처: \n{selected_sfs}\n")
        print("통 피처:\n", common_features)
        print("합집합 피처:\n", union_features)
        rest_features = list(set(original_column_names) - set(union_features))
        print(f'Rest features: \n{rest_features}\n')

        return list(common_features), list(union_features), list(rest_features)
    @staticmethod
    def wrapper_method(X_train, y_train, clf, fig_path):
        print(f'Starting Wrapper Feature Selection...\n')
        
        # SFS
        selected_features_sfs = FeatureSelect._optimized_sfs(clf, X_train, y_train, fig_path)
        # RFE
        selected_features_rfe = FeatureSelect._optimized_rfe(clf, X_train, y_train, fig_path)
        
        return selected_features_rfe, selected_features_sfs
    @staticmethod
    def _optimized_rfe(clf, X_train, y_train, fig_path):
    # 더 큰 step 사용
        step = max(1, X_train.shape[1] // 10)  # 특성 수의 10%씩 제거
        
        rfecv = RFECV(
            estimator=clf,
            step=step,
            scoring='neg_mean_absolute_error',
            cv=3,  # cv 수 줄임
            n_jobs=-1  # 병렬 처리
        )
        
        rfecv.fit(X_train, y_train)
        mean_test_scores = rfecv.cv_results_['mean_test_score']
        
        # 그래프 스타일 설정
        plt.style.use('seaborn')
        plt.figure(figsize=(12, 6))
        
        # 메인 플롯
        plt.plot(
            range(1, len(mean_test_scores) + 1), 
            mean_test_scores,
            marker='o',
            markersize=6,
            linewidth=2,
            color='#2E86C1',
            label='Cross Validation Score'
        )
        
        # 최적 포인트 강조
        optimal_num_features = np.argmax(mean_test_scores) + 1
        plt.plot(
            optimal_num_features, 
            np.max(mean_test_scores),
            'r*',
            markersize=15,
            label=f'Optimal Features: {optimal_num_features}'
        )
        
        # 그리드 추가
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 레이블과 제목
        plt.xlabel('Number of Features Selected', fontsize=12, fontweight='bold')
        plt.ylabel('Cross Validation Score (MAE)', fontsize=12, fontweight='bold')
        plt.title('Recursive Feature Elimination Cross Validation (RFECV)\nPerformance Analysis', 
                fontsize=14, fontweight='bold', pad=20)
        
        # 범례 추가
        plt.legend(loc='lower right', fontsize=10)
        
        # 여백 조정
        plt.tight_layout()
        
        # 그래프 저장 및 표시
        plt.savefig(
            os.path.join(fig_path, 'rfecv_performance.png'), 
            dpi=300, 
            bbox_inches='tight'
        )
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
        # 선택된 특성 반환
        selected_cols = X_train.columns[rfecv.support_].tolist()
        print(f'RFECV: Selected {len(selected_cols)} features: {selected_cols}')
        return selected_cols

    @staticmethod
    def _optimized_sfs(clf, X_train, y_train, fig_path, k_features='best'):
        print('Sequential Feature Selection (SFS)')
        sfs = SFS(
            clf,
            k_features=k_features,
            forward=True,
            floating=False,
            scoring='neg_mean_absolute_error',
            cv=3,
            n_jobs=-1
        )
        print(f'SFS Configuration: {sfs}')
        
        # 모델 학습
        sfs = sfs.fit(X_train, y_train)
        
        # 메트릭 데이터 추출
        metrics_dict = sfs.get_metric_dict()
        
        # 그래프 스타일 설정
        plt.style.use('seaborn')
        plt.figure(figsize=(12, 6))
        
        # 데이터 준비
        x_values = []
        y_means = []
        y_stds = []
        
        for k in sorted(metrics_dict.keys()):
            x_values.append(k)
            y_means.append(metrics_dict[k]['avg_score'])
            y_stds.append(metrics_dict[k]['std_dev'])
        
        # 메인 라인 플롯
        plt.plot(x_values, y_means, 
                marker='o', 
                markersize=8,
                linewidth=2,
                color='#2E86C1',
                label='Average Score')
        
        # 표준편차 영역 추가
        plt.fill_between(x_values,
                        [m - s for m, s in zip(y_means, y_stds)],
                        [m + s for m, s in zip(y_means, y_stds)],
                        alpha=0.2,
                        color='#2E86C1',
                        label='Standard Deviation')
        
        # 최적 포인트 강조
        best_k = max(metrics_dict.keys(), key=lambda k: metrics_dict[k]['avg_score'])
        best_score = metrics_dict[best_k]['avg_score']
        plt.plot(best_k, best_score,
                'r*',
                markersize=15,
                label=f'Optimal Features: {best_k}')
        
        # 그리드 및 레이블 설정
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Number of Features', fontsize=12, fontweight='bold')
        plt.ylabel('Cross Validation Score (MAE)', fontsize=12, fontweight='bold')
        plt.title('Sequential Feature Selection (SFS)\nPerformance Analysis',
                fontsize=14, fontweight='bold', pad=20)
        
        # 범례 설정
        plt.legend(loc='best', fontsize=10)
        
        # 여백 조정
        plt.tight_layout()
        
        # 그래프 저장 및 표시
        plt.savefig(
            os.path.join(fig_path, 'sfs_performance.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
        # 선택된 특성 처리
        selected_features_sfs = X_train.columns[list(sfs.k_feature_idx_)].tolist()
        print(f'Selected Features Score: {sfs.k_score_:.4f}')
        print(f'SFS: Selected {len(selected_features_sfs)} features: {selected_features_sfs}')
        
        return selected_features_sfs
   
def main_prep_null_coordinate(df_raw, prep_path):
   
    cols_address=['주소','시군구', '도로명','아파트명', '좌표X', '좌표Y']
    df_na_coord, df_coord = FeatureEngineer.prep_null_coord(df_raw, ['좌표X', '좌표Y'])
    df_na_coord = FeatureEngineer.get_coord_from_address(df_na_coord, col_address='주소', prep_path=prep_path)
    
    df_combined = pd.concat([df_coord, df_na_coord]).sort_index()
    print(df_combined.shape)
    df_combined.to_csv(os.path.join(prep_path, 'df_raw_null_prep_coord.csv'), index=True)
    return df_combined
def main():
    # 지역구별 대장 아파트들을 입력합니다.
    lead_house = {
        "강서구" : (37.56520754904415, 126.82349451366355),
        "관악구" : (37.47800896704934, 126.94178722423047),
        "강남구" : (37.530594054209146, 127.0262701317293),
        "강동구" : (37.557175745977375, 127.16359581113558),
        "광진구" : (37.543083184171, 127.0998363490422),
        "구로구" : (37.51045944660659, 126.88687199829572),
        "금천구" : (37.459818907487936, 126.89741481874103),
        "노원구" : (37.63952738902813, 127.07234254197617),
        "도봉구" : (37.65775043994647, 127.04345013224447),
        "동대문구" : (37.57760781415707, 127.05375628992316),
        "동작구" : (37.509881249641495, 126.9618159122961),
        "마포구" : (37.54341664563958, 126.93601641235335),
        "서대문구" : (37.55808950436837, 126.9559315685538),
        "서초구" : (37.50625410912666, 126.99846468032919),
        "성동구" : (37.53870643389788, 127.04496220606433),
        "성북구" : (37.61158435092128, 127.02699796439015),
        "송파구" : (37.512817775046074, 127.08340371063358),
        "양천구" : (37.526754982736556, 126.86618704123521),
        "영등포구" : (37.52071403351804, 126.93668907644046),
        "용산구" : (37.521223570097305, 126.97345317787784),
        "은평구" : (37.60181702377437, 126.9362806808709),
        "종로구" : (37.56856915384472, 126.96687674967252),
        "중구" : (37.5544678205846, 126.9634879236162),
        "중랑구" : (37.58171824083332, 127.08183326205129),
        "강북구" : (37.61186335979484, 127.02822407466175)
    }
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, 'data')
    prep_path = os.path.join(data_path, 'preprocessed')

    # df = pd.read_csv(os.path.join(prep_path, 'df_raw_null_prep_coord.csv'), index_col=0)
    # ### 1. 좌표 null 값 외부 데이터로 대체
    # df_combined = main_prep_null_coordinate(df_raw=df, prep_path=prep_path)
    # ##############
    # 2. 남은 결측치, 도로명 주소 기준으로 최빈값 imputation
    feature_source = ['좌표X', '좌표Y']
    # df_combined = _fill_missing_values(df, target_cols=feature_source, group_cols=['도로명주소', '시군구', '도로명', '아파트명'], is_categorical=False)
    # print(f'처리 후 결측치:\n{df_combined[feature_source].isnull().sum()}')
    # ###
    ### 3. 건물 좌표 기준 대상 좌표 거리 분석: 강남 아파트
    feature_target = ['대장_좌표X', '대장_좌표Y']
    # df_combined[["시", "구", "동"]] = df_combined["시군구"].str.split(" ", expand=True)
    # print(df_combined.columns)
    # df_target = pd.DataFrame([{"구": k, feature_target[0]: v[1], feature_target[1]: v[0]} for k, v in lead_house.items()])
    # # Custom 함수 사용
    # # df_combined, cols_distance = distance_analysis(building_df=df_combined, target_feature=df_target, building_coor={'x': '좌표X', 'y': '좌표Y'}, target_coor={'x': '대장_좌표X', 'y': '대장_좌표Y'}, target='gangnam_apt')
    # # Stages.ai 게시판 함수 사용 
    # df_combined, col_apt = distance_apt(df_source=df_combined, dict_target=lead_house, feature_source=feature_source, feature_target=feature_target)
    # #(둘 중 하나만 사용해도 무방. 검증 필요)

    df_combined = pd.read_csv(os.path.join(prep_path, 'df_combined_distance_feature_after_null_fill.csv'), index_col=0)
    #############
    ### 4. 건물 좌표 준 대상 좌표 거리 분석: 지하철, 버스
    df_coor = {'x': '좌표X', 'y': '좌표Y'}
    subway_coor = {'x': '경도', 'y':'위도' }
    bus_coor = {'x': 'X좌표', 'y': 'Y좌표'}
    # 버스 지하철 데이터 로드
    path_subway_feature = os.path.join(data_path, 'subway_feature.csv')
    path_bus_feature = os.path.join(data_path, 'bus_feature.csv')
    
    subway_feature = pd.read_csv(path_subway_feature)
    bus_feature = pd.read_csv(path_bus_feature, index_col=0)
    
    df_combined, col_subway  = FeatureEngineer.distance_analysis_old(
            df_combined, subway_feature, df_coor, subway_coor, target='subway'
        )
    df_combined, col_bus = FeatureEngineer.distance_analysis_old(df_combined, bus_feature, df_coor, bus_coor, target='bus')

    #df_combined, col_subway = distance_apt(df_combined, subway_feature, list(df_coor.values()),list(subway_coor.values()),target='subway')
    #df_combined, col_bus = distance_apt(df_combined, bus_feature, list(df_coor.values()),list(bus_coor.values()),target='bus')
    #df.drop(df.columns[0], axis=1, inplace=True)
    #concat.to_csv(path_feat_add)
    total_cols = col_subway + col_bus# +col_apt
    # feat_cols = cols_distance + transport_cols
    print(total_cols)
    df_combined.to_csv(os.path.join(prep_path, 'df_combined_distance_feature_after_null_fill_complete.csv'), index=True)
if __name__ == "__main__":
    main()