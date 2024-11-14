import os
import pandas as pd

def chk_index_duplicated(df):
    print(df.index.duplicated().sum())
    return df.index.duplicated().sum()

class Utils:
    @staticmethod
    def clean_column_names(df):
        # 하이픈을 언더바로 변경
        df.columns = df.columns.str.replace('-', '_', regex=False)
        # 알파벳, 숫자, 언더바를 제외한 특수 기호 제거
        df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)
        return df
    @staticmethod
    def prepare_test_data(X_test, model):
        """
        테스트 데이터를 예측을 위해 준비
        """
        # target 컬럼 제거
        if 'target' in X_test.columns:
            X_test = X_test.drop(['target'], axis=1)
        
        # 학습에 사용된 컬럼 확인
        train_columns = model.feature_names_in_
        
        # 누락된 컬럼 체크
        missing_cols = set(train_columns) - set(X_test.columns)
        if missing_cols:
            raise ValueError(f"테스트 데이터에 다음 컬럼이 없습니다: {missing_cols}")
        
        # 불필요한 컬럼 체크
        extra_cols = set(X_test.columns) - set(train_columns)
        if extra_cols:
            print(f"다음 컬럼은 예측에 사용되지 않습니다: {extra_cols}")
        
        # 학습에 사용된 컬럼만 선택하고 순서 맞추기
        X_test = X_test[train_columns]
        
        return X_test
    @staticmethod
    def get_unique_filename(filepath: str) -> str:
        """
        파일이 이미 존재할 경우 파일명_1, 파일명_2 등으로 변경
        
        Parameters:
        -----------
        filepath : str
            원본 파일 경로
        
        Returns:
        --------
        str : 유니크한 파일 경로
        """
        if not os.path.exists(filepath):
            return filepath
        
        # 파일 경로와 확장자 분리
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        
        # 새로운 파일명 생성
        counter = 1
        while True:
            new_filename = f"{name}_{counter}{ext}"
            new_filepath = os.path.join(directory, new_filename)
            
            if not os.path.exists(new_filepath):
                return new_filepath
            counter += 1
    @staticmethod
    def concat_train_test(dt_origin, dt_test_origin):
        dt = dt_origin.copy().reset_index(drop=True)
        dt_test = dt_test_origin.copy().reset_index(drop=True)
        # Add is_test and target columns
        dt['is_test'] = 0
        dt_test['target'] = 0  # Ensure target column consistency
        dt_test['is_test'] = 1

        # Concatenate and reset index
        combined = pd.concat([dt, dt_test], axis=0).reset_index(drop=True)
        # Display counts of is_test values for verification
        print(combined['is_test'].value_counts())
        return combined
    @staticmethod
    def unconcat_train_test(concat):
        Utils.remove_unnamed_columns(concat)
        if 'is_test' not in concat.columns:
            raise ValueError("'is_test' 열이 데이터프레임에 없습니다.")
        # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
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
        print(f'train data shape: {dt.shape},{dt.columns}\ntest data shape: {dt_test.shape},{dt_test.columns}')
        print(dt['target'].info())
        print(dt_test['target'].info())
