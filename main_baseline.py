# visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf', # ttf 파일이 저장되어 있는 경로
    name='NanumBarunGothic')                        # 이 폰트의 원하는 이름 설정
fm.fontManager.ttflist.insert(0, fe)              # Matplotlib에 폰트 추가
plt.rcParams.update({'font.size': 10, 'font.family': 'NanumBarunGothic'}) # 폰트 설정
plt.rc('font', family='NanumBarunGothic')
import seaborn as sns
import xgboost as xgb
# utils
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import warnings;warnings.filterwarnings('ignore')

# Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import os
import eli5
from eli5.sklearn import PermutationImportance
# 필요한 데이터를 load 하겠습니다. 경로는 환경에 맞게 지정해주면 됩니다.
# from sklearn.feature_selection import RFE
# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from typing import Dict, List
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error
random_seed = 2024

class DataPrep():
    def __init__(self, config):
        self.out_path = config.get('out_path')
        print('#### Init Data Prep.. ')

    def load_feat_data(self, path):
        dt = pd.read_csv(path)
        print('Train data shape : ', dt.shape, 'Test data shape : ', dt.shape)
        # Train과 Test data를 살펴보겠습니다.
        print(dt.head(1))
        return dt

    def load_data(self, base_path):
        data_path = os.path.join(base_path, 'data')

        train_path = os.path.join(data_path, 'train.csv')
        test_path  = os.path.join(data_path, 'test.csv')
        dt = pd.read_csv(train_path)
        dt_test = pd.read_csv(test_path)

        # Train data와 Test data shape은 아래와 같습니다.
        print('Train data shape : ', dt.shape, 'Test data shape : ', dt_test.shape)
        # Train과 Test data를 살펴보겠습니다.
        print(dt.head(1))
        print(dt_test.head(1))      # 부동산 실거래가(=Target) column이 제외된 모습입니다.

        # Data Prep
        # train/test 구분을 위한 칼럼을 하나 만들어 줍니다.
        dt['is_test'] = 0
        dt_test['is_test'] = 1
        concat = pd.concat([dt, dt_test])     # 하나의 데이터로 만들어줍니다.
        concat['is_test'].value_counts()      # train과 test data가 하나로 합쳐진 것을 확인할 수 있습니다.
        # 칼럼 이름을 쉽게 바꿔주겠습니다. 다른 칼럼도 사용에 따라 바꿔주셔도 됩니다!
        concat = concat.rename(columns={'전용면적(㎡)':'전용면적'})
        return concat
    ##### Null / Outlier

    def prep_null(self, concat):
        # 실제로 결측치라고 표시는 안되어있지만 아무 의미도 갖지 않는 element들이 아래와 같이 존재합니다.
        # 아래 3가지의 경우 모두 아무 의미도 갖지 않는 element가 포함되어 있습니다.
        print(concat['등기신청일자'].value_counts())
        print(concat['거래유형'].value_counts())
        print(concat['중개사소재지'].value_counts())

        # 위 처럼 아무 의미도 갖지 않는 칼럼은 결측치와 같은 역할을 하므로, np.nan으로 채워 결측치로 인식되도록 합니다.
        concat['등기신청일자'] = concat['등기신청일자'].replace(' ', np.nan)
        concat['거래유형'] = concat['거래유형'].replace('-', np.nan)
        concat['중개사소재지'] = concat['중개사소재지'].replace('-', np.nan)

        # EDA에 앞서 결측치를 확인해보겠습니다.
        concat.isnull().sum()

        # 변수별 결측치의 비율을 plot으로 그려보면 아래와 같습니다.
        fig = plt.figure(figsize=(13, 2))
        missing = concat.isnull().sum() / concat.shape[0]
        missing = missing[missing > 0]
        missing.sort_values(inplace=True)
        missing.plot.bar(color='orange')
        title='변수별 결측치 비율'
        plt.title(title)
        plt.savefig(os.path.join(self.out_path, title +'.png'), dpi=300, bbox_inches='tight')

        plt.show()
        self.out_path
        


        # Null값이 100만개 이상인 칼럼은 삭제해보도록 하겠습니다.
        print('* 결측치가 100만개 이하인 변수들 :', list(concat.columns[concat.isnull().sum() <= 1000000]))     # 남겨질 변수들은 아래와 같습니다.
        print('* 결측치가 100만개 이상인 변수들 :', list(concat.columns[concat.isnull().sum() >= 1000000]))

        # 위에서 결측치가 100만개 이하인 변수들만 골라 새로운 concat_select 객체로 저장해줍니다.
        selected = list(concat.columns[concat.isnull().sum() <= 1000000])
        concat_select = concat[selected]
        concat_select.isnull().sum()     # 결측치가 100만개 초과인 칼럼이 제거된 모습은 아래와 같습니다.
        # target변수는 test dataset 개수만큼(9272) 결측치가 존재함을 확인할 수 있습니다.
        return concat_select


    def prep_inter(self, concat_select):

        # Interpolation
        # 연속형 변수는 선형보간을 해주고, 범주형변수는 알수없기에 “unknown”이라고 임의로 보간해 주겠습니다.
        concat_select.info()
        # 본번, 부번의 경우 float로 되어있지만 범주형 변수의 의미를 가지므로 object(string) 형태로 바꾸어주고 아래 작업을 진행하겠습니다.
        concat_select['본번'] = concat_select['본번'].astype('str')
        concat_select['부번'] = concat_select['부번'].astype('str')
        # 먼저, 연속형 변수와 범주형 변수를 위 info에 따라 분리해주겠습니다.
        continuous_columns = []
        categorical_columns = []

        for column in concat_select.columns:
            if pd.api.types.is_numeric_dtype(concat_select[column]):
                continuous_columns.append(column)
            else:
                categorical_columns.append(column)

        print("연속형 변수:", continuous_columns)
        print("범주형 변수:", categorical_columns)

        # 범주형 변수에 대한 보간
        concat_select[categorical_columns] = concat_select[categorical_columns].fillna('NULL')

        # 연속형 변수에 대한 보간 (선형 보간)
        concat_select[continuous_columns] = concat_select[continuous_columns].interpolate(method='linear', axis=0)
        concat_select.isnull().sum()         # 결측치가 보간된 모습을 확인해봅니다.

        # 이상치 제거 이전의 shape은 아래와 같습니다.
        print(concat_select.shape)
        # 대표적인 연속형 변수인 “전용 면적” 변수 관련한 분포를 먼저 살펴보도록 하겠습니다.
        # fig = plt.figure(figsize=(7, 3))
        # try:
        #     sns.boxplot(data = concat_select, x = '전용면적(m)', color='lightgreen')
        # except:
        #     sns.boxplot(data = concat_select, x = '전용면적', color='lightgreen')

        # title = '전용면적 분포'
        # plt.title(title)
        # plt.xlabel('Area')
        # plt.show()
        # plt.savefig(os.path.join(self.out_path, title +'.png'), dpi=300, bbox_inches='tight')
        return concat_select

    # 이상치 제거 방법에는 IQR을 이용하겠습니다.
    @staticmethod
    def remove_outliers_iqr(dt, column_name):
        df = dt.query('is_test == 0')       # train data 내에 있는 이상치만 제거하도록 하겠습니다.
        df_test = dt.query('is_test == 1')

        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

        result = pd.concat([df, df_test])   # test data와 다시 합쳐주겠습니다.
        return result

class FeatureEngineer():
    def __init__(self, config):
        print('#### Init Feature Engineering... ')
        self.out_path = config.get('out_path')
    ### Feature engineering
    def prep_feat(self, concat_select):
        # 시군구, 년월 등 분할할 수 있는 변수들은 세부사항 고려를 용이하게 하기 위해 모두 분할해 주겠습니다.
        concat_select['구'] = concat_select['시군구'].map(lambda x : x.split()[1])
        concat_select['동'] = concat_select['시군구'].map(lambda x : x.split()[2])
        del concat_select['시군구']

        concat_select['계약년'] = concat_select['계약년월'].astype('str').map(lambda x : x[:4])
        concat_select['계약월'] = concat_select['계약년월'].astype('str').map(lambda x : x[4:])
        del concat_select['계약년월']
        print(concat_select.columns)
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
        print(concat_select.columns)
        # 건축년도 분포는 아래와 같습니다. 특히 2005년이 Q3에 해당합니다.
        # 2009년 이후에 지어진 건물은 10%정도 되는 것을 확인할 수 있습니다.
        concat_select['건축년도'].describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9])
        # 따라서 2009년 이후에 지어졌으면 비교적 신축이라고 판단하고, 신축 여부 변수를 제작해보도록 하겠습니다.
        concat_select['신축여부'] = concat_select['건축년도'].apply(lambda x: 1 if x >= 2009 else 0)
        concat_select.head(1)       # 최종 데이터셋은 아래와 같습니다.
        print(concat_select.shape)
        return concat_select

    
    # def sum_distances_from_a_to_b(a, a_coor, b, b_coor, target):
    #     x_a = a_coor.get('x')
    #     y_a = a_coor.get('y')
    #     x_b = b_coor.get('x')
    #     y_b = b_coor.get('y')

    #     a_coords = a[[x_a, y_a]].values
    #     b_coords = b[[x_b, y_b]].values

    #     # KDTree를 사용하여 거리 계산
    #     tree = cKDTree(b_coords)
    #     distance_sums = tree.query(a_coords, k=len(b_coords))[0].sum(axis=1)

    #     a[f'distance_sum_{target}'] = distance_sums
    #     return a
    @staticmethod
    def sum_distances_from_a_to_b(a, a_coor, b, b_coor, target, batch_size=1000):
        """
        Parameters:
            a (pd.DataFrame): DataFrame containing x, y coordinates in columns 'x' and 'y'.
            b (pd.DataFrame): DataFrame containing x, y coordinates in columns 'x' and 'y'.
            batch_size (int): Number of rows to process in each batch.

        Returns:
            pd.DataFrame: Updated DataFrame 'a' with a new column 'distance_sum_b' representing
                        the sum of distances from each (x, y) coordinate in 'a' to all (x, y) coordinates in 'b'.
        """
        x_a = a_coor.get('x')
        y_a = a_coor.get('y')
        x_b = b_coor.get('x')
        y_b = b_coor.get('y')

        # Extract coordinate arrays
        b_coords = b[[x_b, y_b]].values
        distance_sums = []

        # Process in batches
        for start in tqdm(range(0, len(a), batch_size)):
            end = min(start + batch_size, len(a))
            batch_a = a.iloc[start:end]
            a_coords = batch_a[[x_a, y_a]].values

            # Calculate distances for the current batch
            distances = np.sqrt(
                np.sum((a_coords[:, np.newaxis, :] - b_coords[np.newaxis, :, :]) ** 2, axis=2)
            )

            # Sum distances for each point in the batch
            batch_distance_sums = distances.sum(axis=1)
            distance_sums.extend(batch_distance_sums)

        # Add new column to DataFrame 'a'
        a[f'distance_sum_{target}'] = distance_sums

        return a
    # def sum_distances_from_a_to_b(a, a_coor, b, b_coor, target):
    #     """
    #     Parameters:
    #         a (pd.DataFrame): DataFrame containing x, y coordinates in columns 'x' and 'y'.
    #         b (pd.DataFrame): DataFrame containing x, y coordinates in columns 'x' and 'y'.

    #     Returns:
    #         pd.DataFrame: Updated DataFrame 'a' with a new column 'distance_sum_b' representing
    #                     the sum of distances from each (x, y) coordinate in 'a' to all (x, y) coordinates in 'b'.
    #     """
    #     x_a = a_coor.get('x')
    #     y_a = a_coor.get('y')
    #     x_b = b_coor.get('x')
    #     y_b = b_coor.get('y')

    #     # Extract coordinate arrays
    #     a_coords = a[[x_a, y_a]].values
    #     b_coords = b[[x_b, y_b]].values

    #     # Calculate pairwise distances
    #     distances = np.sqrt(
    #         np.sum((a_coords[:, np.newaxis, :] - b_coords[np.newaxis, :, :]) ** 2, axis=2)
    #     )

    #     # Sum distances for each point in 'a'
    #     distance_sums = distances.sum(axis=1)

    #     # Add new column to DataFrame 'a'
    #     a[f'distance_sum_{target}'] = distance_sums

        return a
    def split_train_test(self, concat_select):
        # 이제 다시 train과 test dataset을 분할해줍니다. 위에서 제작해 놓았던 is_test 칼럼을 이용합니다.
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

        print("연속형 변수:", continuous_columns_v2)
        print("범주형 변수:", categorical_columns_v2)
        return dt_train, dt_test, continuous_columns_v2, categorical_columns_v2

    def encode_label(self, dt_train, dt_test, continuous_columns_v2, categorical_columns_v2):
        # 아래에서 범주형 변수들을 대상으로 레이블인코딩을 진행해 주겠습니다.
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
        return dt_train, label_encoders

    def split_dataset(self, dt_train):
        # Target과 독립변수들을 분리해줍니다.
        y_train = dt_train['target']
        X_train = dt_train.drop(['target'], axis=1)

        # Hold out split을 사용해 학습 데이터와 검증 데이터를 8:2 비율로 나누겠습니다.
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed)
        return X_train, X_val, y_train, y_val
    # Squared_error를 계산하는 함수를 정의하겠습니다.
    def calculate_se(self, target, pred):
        squared_errors = (target - pred) ** 2
        return squared_errors

    def select_var(self, model, X_val, y_val, pred, label_encoders, categorical_columns_v2):
        # Permutation importance 방법을 변수 선택에 이용해보겠습니다.
        perm = PermutationImportance(model,        # 위에서 학습된 모델을 이용하겠습니다.
                                    scoring = 'neg_mean_squared_error',        # 평가 지표로는 회귀문제이기에 negative rmse를 사용합니다. (neg_mean_squared_error : 음의 평균 제곱 오차)
                                    random_state = random_seed,
                                    n_iter=3).fit(X_val, y_val)
        eli5.show_weights(perm, feature_names = X_val.columns.tolist())    # valid data에 대해 적합시킵니다.

        # Validation dataset에 target과 pred 값을 채워주도록 하겠습니다.
        X_val['target'] = y_val
        X_val['pred'] = pred

        # RMSE 계산
        squared_errors = self.calculate_se(X_val['target'], X_val['pred'])
        X_val['error'] = squared_errors

        # Error가 큰 순서대로 sorting 해 보겠습니다.
        X_val_sort = X_val.sort_values(by='error', ascending=False)       # 내림차순 sorting

        X_val_sort.head()

        X_val_sort_top100 = X_val.sort_values(by='error', ascending=False).head(100)        # 예측을 잘 하지못한 top 100개의 data
        X_val_sort_tail100 = X_val.sort_values(by='error', ascending=False).tail(100)       # 예측을 잘한 top 100개의 data

        # 해석을 위해 레이블인코딩 된 변수를 복원해줍니다.
        error_top100 = X_val_sort_top100.copy()
        for column in categorical_columns_v2 :     # 앞서 레이블 인코딩에서 정의했던 categorical_columns_v2 범주형 변수 리스트를 사용합니다.
            error_top100[column] = label_encoders[column].inverse_transform(X_val_sort_top100[column])

        best_top100 = X_val_sort_tail100.copy()
        for column in categorical_columns_v2 :     # 앞서 레이블 인코딩에서 정의했던 categorical_columns_v2 범주형 변수 리스트를 사용합니다.
            best_top100[column] = label_encoders[column].inverse_transform(X_val_sort_tail100[column])

        print(error_top100.head(1))
        print(best_top100.head(1))

        sns.boxplot(data = error_top100, x='target')
        title_worst = 'The worst top100 prediction의 target 분포'
        plt.title(title_worst)
        plt.show()
        plt.savefig(os.path.join(self.out_path, title_worst +'.png'), dpi=300, bbox_inches='tight')

        sns.boxplot(data = best_top100, x='target', color='orange')
        title_best = 'The best top100 prediction의 target 분포'
        plt.title(title_best)
        plt.show()
        plt.savefig(os.path.join(self.out_path, title_best +'.png'), dpi=300, bbox_inches='tight')

        sns.histplot(data = error_top100, x='전용면적', alpha=0.5)
        sns.histplot(data = best_top100, x='전용면적', color='orange', alpha=0.5)
        title_hist = '전용면적 분포 비교'
        plt.title(title_hist)
        plt.savefig(os.path.join(self.out_path, title_hist +'.png'), dpi=300, bbox_inches='tight')
        plt.show()


class Model():
    def __init__(self, config):
        self.out_path = config.get('out_path')
        print('#### Init Model Train... ')
    ### Model training
    def model_train(self, X_train, X_val, y_train, y_val, type='default'):
    # RandomForestRegressor를 이용해 회귀 모델을 적합시키겠습니다.
        #model = RandomForestRegressor(n_estimators=5, criterion='squared_error', random_state=1, n_jobs=-1)
        sweep_configs = {
            "xgboost": {
                'method': 'bayes',
                'metric': {'name': 'rmse', 'goal': 'minimize'},
                'parameters': {
                    'eta': 0.3,
                    'max_depth':10,
                    'subsample': 0.6239,
                    'colsample_bytree': 0.5305,
                    'gamma': 4.717,
                    'reg_lambda': 5.081, 
                    'alpha': 0.4902,
                    'n_estimators': 551
                }
            },
            "random_forest": {
            'method': 'bayes',
            'metric': {'name': 'rmse', 'goal': 'minimize'},
            'parameters': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'log2'
            }
        },
            }
        
        config = sweep_configs['xgboost']['parameters'] #['random_forest']['parameters']#
        model_name = 'XGB'
        model = xgb.XGBRegressor(
                reg_alpha=config.get('alpha'),   
                colsample_bytree=config.get('colsample_bytree'),
                eta=config.get('eta'),
                gamma=config.get('gamma'),
                max_depth=config.get('max_depth'),
                n_estimators=config.get('n_estimators'),
                reg_lambda=config.get('reg_lambda'),
                subsample=config.get('subsample'), 
            )
        # model = RandomForestRegressor(
        #         n_estimators=config.get('n_estimators'),
        #         max_depth=config.get('max_depth'),
        #         min_samples_split=config.get('min_samples_split'),
        #         min_samples_leaf=config.get('min_samples_leaf'),
        #         max_features=config.get('max_features')
        #     )
        

        # smote = SMOTE(random_state=42)
        # X_train, y_train = smote.fit_resample(X_train, y_train)
        # Recursive Feature Elimination 예시
        # rfe = RFE(estimator=model, n_features_to_select=10)
        # X_rfe = rfe.fit_transform(X_train, y)

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if isinstance(y_train, pd.Series):
            y_val = y_val.values
    

        from sklearn.metrics import make_scorer, mean_squared_error
        from sklearn.model_selection import cross_val_score, KFold
       
        if type == 'k_fold':
            # Ensure X and y are vertically stacked properly
            X = np.vstack((X_train, X_val))
            y = np.hstack((y_train, y_val))  # Use hstack since y is a 1D array

            kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

            # Correct the make_scorer usage
            rmse_scorer = make_scorer(mean_squared_error, squared=False, greater_is_better=False)

            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring=rmse_scorer)
            model.fit(X_train, y_train)
            # Calculate the RMSE
            pred = model.predict(X_val)
            rmse = -np.mean(cv_scores)  # Negative sign because scoring returns negative values
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_val)

            # 회귀 관련 metric을 통해 train/valid의 모델 적합 결과를 관찰합니다.
            print(f'RMSE test: {np.sqrt(metrics.mean_squared_error(y_val, pred))}')

            # 위 feature importance를 시각화해봅니다.
            importances = pd.Series(model.feature_importances_, index=list(X_train.columns))
            importances = importances.sort_values(ascending=False)
            title_feat = "Feature Importances"
            plt.figure(figsize=(10,8))
            plt.title(title_feat)
            sns.barplot(x=importances, y=importances.index)
            plt.show()
            
            plt.savefig(os.path.join(self.out_path, title_feat +'.png'), dpi=300, bbox_inches='tight')

        # 학습된 모델을 저장합니다. Pickle 라이브러리를 이용하겠습니다.
        out_path = os.path.join(self.out_path,f'saved_model_{model_name}_{type}.pkl' )
        with open(out_path, 'wb') as f:
            pickle.dump(model, f)
        return model, pred

    def k_fold_train(self, dt_train, k=5):
        y = dt_train['target']
        X = dt_train.drop(['target'], axis=1)
        kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
        # 또는 StratifiedKFold (타겟 불균형시)
        # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        

        
        # models=[]
        # for train_index, val_index in kf.split(X, y):
        #     X_train, X_val = X[train_index], X[val_index]
        #     y_train, y_val = y[train_index], y[val_index]
    
        #     model, pred = self.model_train(X_train, X_val, y_train, y_val, self.out_path)
        #     models.append({'model':model,'pred':pred})
 
        return 

    def inference(self, dt_test):
        print('inference start.')
        dt_test.head(2)      # test dataset에 대한 inference를 진행해보겠습니다.
        # 저장된 모델을 불러옵니다.
        out_model_path = os.path.join(self.out_path, 'saved_model.pkl')
        with open(out_model_path, 'rb') as f:
            model = pickle.load(f)

        X_test = dt_test.drop(['target'], axis=1)

        # Test dataset에 대한 inference를 진행합니다.
        real_test_pred = model.predict(X_test)
        #real_test_pred          # 예측값들이 출력됨을 확인할 수 있습니다.

        # 앞서 예측한 예측값들을 저장합니다.
        preds_df = pd.DataFrame(real_test_pred.astype(int), columns=["target"])
        preds_df.to_csv(os.path.join(self.out_path,'output.csv'), index=False)
        return preds_df

    def load_data_pkl(self, data_path):
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        except:
            print('err')
            data = None
        return data

    def save_data(self, prep_data):
        out_path = os.path.join(self.out_path,'prep_data.pkl')
        try:
            with open(out_path, 'wb') as f:
                    pickle.dump(prep_data, f)
            print('Dataset Saved to ', out_path)
        except:
            print('error.')
        return out_path


class EDA:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = config.get('logger')
    def analyze_missing_vals(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_stats = pd.DataFrame({
            'missing_count': df.isnull().sum(),
            'missing_ratio': df.isnull().sum() / len(df) * 100
        })
        return missing_stats.sort_values('missing_ratio', ascending=False)
    def analyze_dist(self, df: pd.DataFrame, numerical_cols: List[str]) -> None:
        """수치형 변수의 분포를 시각화하고 기초 통계량을 출력"""
        if not numerical_cols:
            self.logger.warning("수치형 변수가 없습니다!")
            return
        
        # 데이터 전처리: 문자열을 숫자로 변환
        df_numeric = df[numerical_cols].copy()
        for col in numerical_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df_numeric[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 기초 통계량 출력
        self.logger.info("\n=== 기초 통계량 ===")
        self.logger.info(f"\n{df_numeric.describe()}")
        
        # 그래프 생성
        n_cols = len(numerical_cols)
        fig, axes = plt.subplots(n_cols, 2, figsize=(15, 5*n_cols))
        
        # 변수가 1개인 경우 axes 처리
        if n_cols == 1:
            axes = axes.reshape(1, 2)
        
        for idx, col in enumerate(numerical_cols):
            data = df_numeric[col].dropna()  # 결측치 제외
            
            # 히스토그램 + KDE
            sns.histplot(data=data, kde=True, ax=axes[idx, 0])
            axes[idx, 0].set_title(f'{col} - 히스토그램 (결측치 제외)')
            
            # 박스플롯
            sns.boxplot(y=data, ax=axes[idx, 1])
            axes[idx, 1].set_title(f'{col} - 박스플롯 (결측치 제외)')
            
            # 이상치 및 결측치 정보 출력
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)].shape[0]
            missing_count = df_numeric[col].isnull().sum()
            
            self.logger.info(f"\n{col}:")
            self.logger.info(f"- 결측치: {missing_count}개 ({missing_count/len(df)*100:.2f}%)")
            self.logger.info(f"- 이상치: {outlier_count}개 ({outlier_count/len(data)*100:.2f}%)")
        
        plt.tight_layout()
        plt.show()
        
        # 상관관계 분석 (변수가 2개 이상인 경우)
        if n_cols > 1:
            plt.figure(figsize=(10, 8))
            correlation = df_numeric.corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
            plt.title('변수 간 상관관계')
            plt.tight_layout()
            plt.show()

    def analyze_corr(self, df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        plt.figure(figsize=(10,8))
        sns.heatmap(df[numerical_cols].corr(),
                    annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.show()


    def plot_histograms(self):
        """수치형 변수에 대한 히스토그램을 생성합니다."""
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numeric_cols].hist(figsize=(15, 10), bins=20)
        plt.tight_layout()
        plt.show()

def main():
    base_path = '/data/ephemeral/home/dev/upstageailab5-ml-regression-ml_r4'
    out_path = os.path.join(base_path,'output')
    config ={'out_path':out_path}
    data_prep = DataPrep(config)
    feat_eng = FeatureEngineer(config)
    model_instance = Model(config)
    df = data_prep.load_data(base_path)
    df = data_prep.prep_null(df)
    df = data_prep.prep_inter(df)
    # 위 방법으로 전용 면적에 대한 이상치를 제거해보겠습니다.
    cols = ['계약년', '전용면적', '강남여부', '구', '건축년도', '좌표X', '좌표Y', '동']
    # df = DataPrep.remove_outliers_iqr(df, '전용면적')
    # # 이상치 제거 후의 shape은 아래와 같습니다. 약 10만개의 데이터가 제거된 모습을 확인할 수 있습니다.
    # print(df.shape)
    # df['is_test'].value_counts()     # 또한, train data만 제거되었습니다.

    # #### Feat eng
    # df = feat_eng.prep_feat(df)
    # df_coor = {'x': '좌표X', 'y': '좌표Y'}

    # df_subway = data_prep.load_feat_data(os.path.join(base_path, 'data','subway_feature.csv'))
    # df_bus = data_prep.load_feat_data(os.path.joing(base_path, 'data','bus_feature.csv'))
    # subway_coor = {'x': '위도', 'y': '경도'}
    # bus_coor = {'x': 'X좌표', 'y': 'Y좌표'}

    # # df = feat_eng.sum_distances_from_a_to_b(df, df_coor , df_subway, subway_coor, 'subway')
    # # df = feat_eng.sum_distances_from_a_to_b(df, df_coor , df_bus, bus_coor, 'bus')

    path_feat = os.path.join(out_path, 'data_feature_bus_subway.csv')
    # df.to_csv(path_feat,  index=False)

    df = pd.read_csv(path_feat)#, index_col=False)
    df.head()

### split
    dt_train, dt_test, continuous_columns_v2, categorical_columns_v2 = feat_eng.split_train_test(df)
    dt_train, label_encoders = feat_eng.encode_label(dt_train, dt_test, continuous_columns_v2, categorical_columns_v2)
    X_train, X_val, y_train, y_val = feat_eng.split_dataset(dt_train)

    prep_data = {'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val,
                'continuous_columns': continuous_columns_v2,
                'categorical_columns': categorical_columns_v2
    }
    out_path_data = model_instance.save_data(prep_data)
    # loaded_data = load_data_pkl(out_path_data)
    # print(loaded_data)
    # model, pred = model_instance.model_train(X_train, X_val, y_train, y_val)
    models =model_instance.k_fold_train(dt_train)

    # feat_eng.select_var(model, X_val, y_val, pred, label_encoders, categorical_columns_v2)
    # model_instance.inference(dt_test)

if __name__ == '__main__':
    main()