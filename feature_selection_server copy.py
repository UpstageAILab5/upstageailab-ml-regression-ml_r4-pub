
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
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

# 데이터 로드
def feature_selection_sfs(clf, X_train, y_train, path, k_features='best'):
    # Sequential Feature Selection (forward or backward)
    print('Sequential Feature Selection (SFS)')
    sfs = SFS(clf,
            k_features=k_features,  # 선택할 피처 수
            forward=True,  # True for forward selection, False for backward
            floating=False,
            scoring='neg_mean_absolute_error',  # 평가 지표
            cv=5)
    print(f'SFS: {sfs}')
    sfs = sfs.fit(X_train, y_train)
    print('Selected features:', sfs.k_feature_idx_)
    print('CV Score:', sfs.k_score_)
    # SFS 과정 시각화
    fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
    title='Sequential Feature Selection (SFS) Performance'
    plt.title(title)

    plt.grid()
    
    plt.show()
    plt.savefig(os.path.join(path, title +'.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return sfs.k_feature_idx_

def variance_threshold(df): 
    selector = VarianceThreshold(threshold=0.1)
    reduced_df = selector.fit_transform(df)

    print("Original DataFrame:\n", df.shape)
    print("Reduced DataFrame:\n", pd.DataFrame(reduced_df).shape)
    return pd.DataFrame(reduced_df)
def feature_selection_rfe(clf, X_train, y_train, path):
    # RFE 모델 생성
    #feature_names = iris.feature_names
    # 모델 및 RFE 수행
    rfecv = RFECV(estimator=clf, step=1, scoring='neg_mean_absolute_error', cv=5)
    rfecv.fit(X_train, y_train)

        # RFECV 시각화
    plt.figure(figsize=(8, 4))
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Cross Validation Score (Accuracy)')
    title='RFECV Performance'
    plt.title(title)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    plt.savefig(os.path.join(path, title +'.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return rfecv.support_

# One-Hot Encoding


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from tqdm import tqdm

def auto_label_encode(df, cols_exclude, encoder_type='onehot', batch_size=3000):
    """
    문자열 또는 카테고리형 변수를 자동으로 Label Encoding하는 함수.
    인자로 받은 DataFrame을 변환한 새로운 DataFrame을 반환합니다.

    Parameters:
    df (pd.DataFrame): 입력 DataFrame
    cols_exclude (list): 인코딩에서 제외할 열 목록
    encoder_type (str): 인코더 유형 ('onehot' 또는 'label')
    batch_size (int): 배치 크기

    Returns:
    pd.DataFrame: 변환된 DataFrame
    """
    # 제외할 열을 제외한 나머지 열 선택
    cols_rest = df.columns.difference(cols_exclude)
    df_transformed = df.loc[:, cols_rest].copy()
    
    print(f'cols_exclude: {cols_exclude}')
    if encoder_type == 'onehot':
        encoder = OneHotEncoder(sparse_output=False)
    elif encoder_type == 'label':
        encoder = LabelEncoder()

    encoded_batches = []

    for column in tqdm(df_transformed.columns):
        # 문자열 또는 카테고리형 열을 찾음
        if df_transformed[column].dtype == 'object' or df_transformed[column].dtype.name == 'category':
            # Null 값 처리
            df_transformed[column] = df_transformed[column].astype(str).fillna('Missing')
            
            # 배치 처리
            for start in tqdm(range(0, len(df_transformed), batch_size)):
                end = min(start + batch_size, len(df_transformed))
                batch = df_transformed.iloc[start:end]
                
                if encoder_type == 'onehot':
                    encoded = encoder.fit_transform(batch[[column]])  # 2D로 변환
                    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
                elif encoder_type == 'label':
                    encoded = encoder.fit_transform(batch[column])
                    encoded_df = pd.DataFrame(encoded, columns=[column])
                
                encoded_batches.append(encoded_df)
    
    # 모든 배치를 결합
    df_encoded = pd.concat(encoded_batches, axis=0).reset_index(drop=True)
    
    # 인코딩된 데이터프레임과 제외된 열을 결합
    df_result = pd.concat([df[cols_exclude].reset_index(drop=True), df_encoded], axis=1).sort_index(axis=1)
    
    print(df.shape, df_result.shape)
    return df_result
def auto_label_encode_parallel(df, cols_exclude, encoder_type='onehot', batch_size=10000, n_jobs=-1):
    """
    병렬 처리를 사용하여 문자열 또는 카테고리형 변수를 자동으로 Label Encoding하는 함수.
    """
    cols_rest = df.columns.difference(cols_exclude)
    df_transformed = df.loc[:, cols_rest].copy()
    
    print(f'cols_exclude: {cols_exclude}')
    if encoder_type == 'onehot':
        encoder = OneHotEncoder(sparse_output=False)
    elif encoder_type == 'label':
        encoder = LabelEncoder()

    encoded_batches = Parallel(n_jobs=n_jobs)(
        delayed(parallel_encode)(df_transformed, encoder, column, batch_size)
        for column in tqdm(df_transformed.columns)
        if df_transformed[column].dtype == 'object' or df_transformed[column].dtype.name == 'category'
    )
    
    df_encoded = pd.concat(encoded_batches, axis=1).reset_index(drop=True)
    df_result = pd.concat([df[cols_exclude].reset_index(drop=True), df_encoded], axis=1).sort_index(axis=1)
    
    print(df.shape, df_result.shape)
    return df_result
class Utils:
    @staticmethod
    def concat_train_test(dt, dt_test):
        Utils.remove_unnamed_columns(dt)
        Utils.remove_unnamed_columns(dt_test)
        dt['is_test'] = 0
        dt_test['is_test'] = 1
        dt_test['target'] = 0
        concat = pd.concat([dt, dt_test], axis=0).reset_index(drop=True)
        print(concat['is_test'].value_counts())
        return concat
    @staticmethod
    def unconcat_train_test(concat):
        Utils.remove_unnamed_columns(concat)
        dt = concat.query('is_test==0')
        # y_train = dt['target']
        dt.drop(columns=['is_test'], inplace=True)
        dt_test = concat.query('is_test==1')
        dt_test.drop(columns=['target', 'is_test'], inplace=True)
        return dt, dt_test
    @staticmethod
    def clean_df(df):
        df = Utils.remove_unnamed_columns(df)
        duplicated_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicated_cols:
            print("중복된 컬럼:", duplicated_cols)
            # 방법 1: 중복 컬럼 제거 (첫 번째 컬럼 유지)
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
        return df
    @staticmethod
    def remove_unnamed_columns(df):
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            print(f"Removing unnamed columns: {unnamed_cols}")  
            df = df.drop(columns=unnamed_cols)
        return df
    @staticmethod
    def categorical_numeric(df):
        # 파생변수 제작으로 추가된 변수들이 존재하기에, 다시한번 연속형과 범주형 칼럼을 분리해주겠습니다.
        continuous_columns_v2 = []
        categorical_columns_v2 = []

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                continuous_columns_v2.append(column)
            else:
                categorical_columns_v2.append(column)
        return continuous_columns_v2, categorical_columns_v2
    @staticmethod
    def chk_train_test_data(df):
        dt = df.query('is_test==0')
        dt_test = df.query('is_test==1')
        print(f'train data shape: {dt.shape}, test data shape: {dt_test.shape}')
        print(dt['target'].info())
        print(dt_test['target'].info())
        return
def optimize_dataframe(df):
    """
    데이터프레임의 데이터 타입을 최적화하여 메모리 사용량을 줄입니다.
    """
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='unsigned' if df[col].dtype == 'int64' else 'float')
    return df
from joblib import Parallel, delayed

def parallel_encode(df, encoder, column, batch_size):
    encoded_batches = []
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch = df.iloc[start:end]
        encoded = encoder.fit_transform(batch[[column]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
        encoded_batches.append(encoded_df)
    return pd.concat(encoded_batches, axis=0)

import numpy as np
def encode_label(dt_train, dt_test, categorical_columns_v2):
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
def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'data')
    prep_path = os.path.join(data_path, 'preprocessed')
    fig_path = os.path.join(base_path, 'output', 'figures')
    #df = pd.read_csv(os.path.join(prep_path, 'df_raw_null_prep_coord.csv'))
    df = pd.read_csv(os.path.join(prep_path, 'df_combined_distance_feature_after_null_fill.csv'), index_col=0)
    Utils.chk_train_test_data(df)
## Encode categorical variables
    cols_exclude = ['target', 'is_test']
    
    df_train, df_test = Utils.unconcat_train_test(df)
    y_train = df_train['target']
    X_train = df_train.drop(columns=['target'])
    X_test = df_test
    continuous_features, categorical_features = Utils.categorical_numeric(X_train)
    X_train, X_test, encode_labels = encode_label(X_train, X_test, categorical_features)

    #df_transformed = auto_label_encode_parallel(df, cols_exclude, 'onehot')#auto_label_encode(df, cols_exclude, 'onehot')
    print("Original DataFrame:\n", df.shape)
    #print("\nTransformed DataFrame:\n", df_transformed.shape)
    reduced_df = variance_threshold(X_train)
    reduced_df.to_csv(os.path.join(prep_path, 'df_combined_distance_feature_after_null_fill_reduced.csv'))

    X_train['target'] = y_train

    concat_encoded = Utils.concat_train_test(X_train, X_test)

    concat_encoded.to_csv(os.path.join(prep_path, 'df_combined_distance_feature_after_null_fill_encoded.csv'))

    # y_train = X_train['target']
    # X_train = X_train.drop(columns=['target'])
    # print(X_train.columns)
## Select Features
    rf = RandomForestRegressor(random_state=42)
    #feature_selection_sfs(rf, X_train, y_train, fig_path)
    feature_selection_rfe(rf, X_train, y_train, fig_path)
    
    

if __name__ == '__main__':
    main()


