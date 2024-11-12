
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

# 데이터 로드
def feature_selection_sfs(clf, X_train, y_train, k_features='best'):
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
    plt.title('Sequential Feature Selection (SFS) Performance')
    plt.grid()
    plt.show()
    return sfs.k_feature_idx_

def variance_threshold(df): 
    selector = VarianceThreshold(threshold=0.1)
    reduced_df = selector.fit_transform(df)

    print("Original DataFrame:\n", df)
    print("Reduced DataFrame:\n", pd.DataFrame(reduced_df))

def feature_selection_rfe(clf, X_train, y_train):
    # RFE 모델 생성
    #feature_names = iris.feature_names
    # 모델 및 RFE 수행
    rfecv = RFECV(estimator=clf, step=1, scoring='neg_mean_absolute_error', cv=5)
    rfecv.fit(X_train, y_train)

        # RFECV 시각화
    plt.figure(figsize=(8, 4))
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Cross Validation Score (Accuracy)')
    plt.title('RFECV Performance')
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    return rfecv.support_
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def auto_label_encode(df, cols_exclude):
    """
    문자열 또는 카테고리형 변수를 자동으로 Label Encoding하는 함수.
    인자로 받은 DataFrame을 변환한 새로운 DataFrame을 반환합니다.

    Parameters:
    df (pd.DataFrame): 입력 DataFrame

    Returns:
    pd.DataFrame: 변환된 DataFrame
    """
    df_rest = df.copy()[cols_exclude]
    df_transformed = df.copy().drop(columns=cols_exclude, inplace=False)

  
    print(f'cols_exclude: {cols_exclude}')
    le = LabelEncoder()

    for column in tqdm(df_transformed.columns):
        # 문자열 또는 카테고리형 열을 찾음
        if df_transformed[column].dtype == 'object' or df_transformed[column].dtype.name == 'category':
            # Null 값 처리
            df_transformed[column] = df_transformed[column].astype(str).fillna('Missing')
            # Label Encoding 적용
            df_transformed[column] = le.fit_transform(df_transformed[column])
    print(df.shape, df_transformed.shape)
    return pd.concat([df_rest, df_transformed], axis=1).sort_index(axis=1)
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
    def chk_train_test_data(df):
        dt = df.query('is_test==0')
        dt_test = df.query('is_test==1')
        print(f'train data shape: {dt.shape}, test data shape: {dt_test.shape}')
        print(dt['target'].info())
        print(dt_test['target'].info())
        return
def main():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'preprocessed')
    df = pd.read_csv(os.path.join(path, 'df_raw_null_prep_coord.csv'))
    Utils.chk_train_test_data(df)
## Encode categorical variables
    cols_exclude = ['target', 'is_test']
    df_transformed = auto_label_encode(df, cols_exclude)
    print("Original DataFrame:\n", df.shape)
    print("\nTransformed DataFrame:\n", df_transformed.shape)

    X_train, x_test = Utils.unconcat_train_test(df_transformed)
    y_train = X_train['target']
    X_train = X_train.drop(columns=['target'])
    print(X_train.columns)
## Select Features
    rf = RandomForestRegressor(random_state=42)
    feature_selection_sfs(rf, X_train, y_train)
    feature_selection_rfe(rf, X_train, y_train)

if __name__ == '__main__':
    main()


