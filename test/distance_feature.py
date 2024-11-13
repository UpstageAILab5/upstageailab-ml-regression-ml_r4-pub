import   os
import requests
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import BallTree

from typing import List, Dict

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

def _get_lat_lon(address):
	
    """
    주소를 입력받아 해당 위치의 경도와 위도를 반환합니다.

    load_dotenv('/data/ephemeral/home/RealEstatePricePredictor/.env'): 
    .env 파일을 현재 프로젝트 폴더 경로에서 불러옵니다.
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

def get_coord_from_address(df, col_address='주소', prep_path=None):

    count = 0
    grouped = df.groupby(col_address)   
    print(f'총 {len(grouped)}개의 아파트 그룹 {col_address}가 좌표 없음')
    for i, (address, group) in tqdm(enumerate(grouped), total=len(grouped)):
        print(f'\n{i+1}번째 아파트 그룹: {address}, 그룹 내 데이터 수: {len(group)}')
        # 주소로부터 위도와 경도를 가져옴
        lat, long = _get_lat_lon(address)
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

    print(f'총 {len(grouped)}개의 아파트 그룹 중 {count}개의 아파트 주소가 좌표 변환 오류')
    return df
# 위경도를 이용해 두 지점간의 거리를 구하는 함수를 생성합니다.
import math
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

def distance_apt(df_source,  dict_target,  feature_source=['좌표X', '좌표Y'], feature_target=['대장_좌표X', '대장_좌표Y'], target='대장아파트'):
        # 대장 아파트의 위경도 데이터프레임을 구성합니다.
    if target == '대장아파트':  
        df_target = pd.DataFrame([{"구": k, feature_target[0]: v[1], feature_target[1]: v[0]} for k, v in dict_target.items()])
        # 데이터프레임간 결합을 합니다.
    else:
        df_target = dict_target
    target_col = f'{target}_거리'

    df_source = pd.merge(df_source, df_target, how="inner", on="구")
        # 아까 제작한 haversine_distance 함수를 이용해 대장아파트와의 거리를 계산하고, 새롭게 컬럼을 구성합니다.
    df_source[target_col] = df_source.apply(lambda row: haversine_distance(row[feature_source[1]], row[feature_source[0]], row[feature_target[1]], row[feature_target[0]]), axis=1)
    return df_source, [target_col]
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
    
    # 거리 계산
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
                distance = haversine_distance(by, bx, ty, tx)
                
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
            # 반경 변환 (미터 -> 라디안)
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



def main_prep_null_coordinate(df_raw, prep_path):
   
    cols_address=['주소','시군구', '도로명','아파트명', '좌표X', '좌표Y']
    df_na_coord, df_coord = prep_null_coord(df_raw, ['좌표X', '좌표Y'])
    df_na_coord = get_coord_from_address(df_na_coord, col_address='주소', prep_path=prep_path)
    
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
    ### 4. 건물 좌표 ���준 대상 좌표 거리 분석: 지하철, 버스
    df_coor = {'x': '좌표X', 'y': '좌표Y'}
    subway_coor = {'x': '경도', 'y':'위도' }
    bus_coor = {'x': 'X좌표', 'y': 'Y좌표'}
    # 버스 지하철 데이터 로드
    path_subway_feature = os.path.join(data_path, 'subway_feature.csv')
    path_bus_feature = os.path.join(data_path, 'bus_feature.csv')
    
    subway_feature = pd.read_csv(path_subway_feature)
    bus_feature = pd.read_csv(path_bus_feature, index_col=0)
    
    df_combined, col_subway  = distance_analysis_old(
            df_combined, subway_feature, df_coor, subway_coor, target='subway'
        )
    df_combined, col_bus = distance_analysis_old(df_combined, bus_feature, df_coor, bus_coor, target='bus')

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
