from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
# import eli5
# from eli5.sklearn import PermutationImportance
from tqdm import tqdm
from geopy.distance import geodesic
import shap
from pyproj import Transformer
from src.utils import get_unique_filename
from src.utils import get_unique_filename
from scipy.spatial import cKDTree
from numba import jit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        self.logger = config.get('logger')

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
class FeatureAdditional:
    def __init__(self, config):
        self.config = config
        self.logger = config.get('logger')
    def distance_analysis(self, df, subway_feature, df_coor, subway_coor, radius, target):
        """
        최적화된 거리 분석 수행
        """
        self.logger.info(f"### {target} 거리 분석 시작")
        
        # 좌표계 변환기 설정
        transformer = Transformer.from_crs(
            "EPSG:5181",
            "EPSG:4326",
            always_xy=True
        )
        
        # 모든 좌표를 한 번에 변환 (벡터화)
        subway_points = subway_feature[[subway_coor['x'], subway_coor['y']]].values
        building_points = df[[df_coor['x'], df_coor['y']]].values
        
        # 벡터화된 좌표 변환
        subway_transformed = np.array([transformer.transform(x, y) for x, y in subway_points])
        building_transformed = np.array([transformer.transform(x, y) for x, y in building_points])
        
        # KD-Tree 구성
        tree = cKDTree(subway_transformed)
        
        # 새로운 컬럼명들
        count_col = f'{target}_near_{radius}m_count'
        avg_col = f'{target}_avg_distance'
        short_col = f'{target}_shortest_distance'
        
        # 거리 계산 (KD-Tree 사용)
        distances, _ = tree.query(building_transformed, k=len(subway_transformed))
        
        # 결과 계산 (벡터화)
        df[count_col] = np.sum(distances <= radius, axis=1)
        df[short_col] = np.min(distances, axis=1)
        
        # 평균 거리 계산
        mask = distances <= radius
        avg_distances = np.where(
            np.any(mask, axis=1),
            np.sum(distances * mask, axis=1) / np.sum(mask, axis=1),
            radius
        )
        df[avg_col] = avg_distances
        
        self.logger.info(f"거리 분석 완료: {count_col}, {avg_col}, {short_col} 컬럼 생성")
        
        # 통계 출력
        for col in tqdm([count_col, avg_col, short_col]):
            stats = df[col].describe()
            self.logger.info(f"\n{col} 통계:\n{stats}")
        
        return df, [count_col, avg_col, short_col]
    
    def parallel_distance_calc(self, chunk, subway_feature, df_coor, subway_coor, radius, target):
        """단일 청크에 대한 거리 계산"""
        transformer = Transformer.from_crs(
            "EPSG:5181",
            "EPSG:4326",
            always_xy=True
        )
        
        # 지하철 좌표 변환
        subway_coords = []
        for _, row in subway_feature.iterrows():
            x, y = transformer.transform(
                row[subway_coor['x']], 
                row[subway_coor['y']]
            )
            subway_coords.append((y, x))
        
        results = []
        for idx, row in chunk.iterrows():
            # 건물 좌표 변환
            x1, y1 = transformer.transform(
                row[df_coor['x']], 
                row[df_coor['y']]
            )
            
            # 거리 계산
            distances = [geodesic((y1, x1), coord).meters for coord in subway_coords]
            
            # 결과 저장
            results.append({
                'idx': idx,
                'count': sum(1 for d in distances if d <= radius),
                'avg': np.mean([d for d in distances if d <= radius]) if any(d <= radius for d in distances) else radius,
                'shortest': min(distances)
            })
            
        return results

    def distance_analysis_parallel(self, df, subway_feature, df_coor, subway_coor, radius, target):
        """병렬 처리를 사용한 거리 분석"""
        self.logger.info(f"### {target} 거리 분석 시작 (병렬 처리)")
        
        # 컬럼명 정의
        count_col = f'{target}_near_{radius}m_count'
        avg_col = f'{target}_avg_distance'
        short_col = f'{target}_shortest_distance'
        
        # CPU 코어 수 확인 및 청크 분할
        n_cores = multiprocessing.cpu_count()
        chunks = np.array_split(df, n_cores)
        
        # 병렬 처리
        all_results = []
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = [
                executor.submit(
                    self.parallel_distance_calc, 
                    chunk, 
                    subway_feature, 
                    df_coor, 
                    subway_coor, 
                    radius, 
                    target
                ) for chunk in chunks
            ]
            
            # tqdm으로 진행상황 표시
            for future in tqdm(
                as_completed(futures), 
                total=len(futures),
                desc=f"{target} 거리 계산",
                ncols=100
            ):
                results = future.result()
                all_results.extend(results)
        
        # 결과를 데이터프레임에 적용
        self.logger.info("결과 데이터프레임 업데이트 중...")
        for result in tqdm(all_results, desc="결과 업데이트"):
            idx = result['idx']
            df.loc[idx, count_col] = result['count']
            df.loc[idx, avg_col] = result['avg']
            df.loc[idx, short_col] = result['shortest']
        
        self.logger.info(f"거리 분석 완료: {count_col}, {avg_col}, {short_col} 컬럼 생성")
        return df, [count_col, avg_col, short_col]
    
#     주요 최적화 포인트:
# 1. 벡터화 연산
# 반복문 대신 NumPy 벡터화 연산 사용
# 좌표 변환도 한 번에 처리
# 2. KD-Tree 사용
# 공간 인덱싱으로 거리 계산 최적화
# O(n²) → O(n log n) 복잡도로 감소

# average_distance: 집에서 가까운 정류장까지의 평균 거리
# shortest_distance: 가장 가까운 정류장까지의 거리
# num_nearby_stations: 특정 거리(예: 500m, 1km 등) 이내에 있는 정류장
class FeatureEngineer():
    def __init__(self, config):
        self.config = config
        self.logger = config.get('logger')
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
