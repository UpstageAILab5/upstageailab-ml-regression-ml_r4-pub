import os
import pandas as pd

def chk_index_duplicated(df):
    print(df.index.duplicated().sum())
    return df.index.duplicated().sum()

class Utils:
    @staticmethod
    def clean_feature_names(df):
        """
        데이터프레임의 컬럼 이름에서 특수 문자를 제거하거나 대체
        """
        # 컬럼 이름 매핑 사전 생성
        column_mapping = {
            col: col.replace('+', '_plus_')
            .replace('-', '_minus_')
            .replace('[', '_')
            .replace(']', '_')
            .replace(' ', '_')
            .replace('/', '_')
            .replace('(', '_')
            .replace(')', '_')
            .replace(',', '_')
            .strip('_') # 끝에 있는 언더스코어 제거
            for col in df.columns
        }
        
        # 컬럼 이름 변경
        df_cleaned = df.rename(columns=column_mapping)
        
        return df_cleaned, column_mapping
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
    def unconcat_train_test(concat_select):
        concat_select = Utils.remove_unnamed_columns(concat_select)
                # 이제 다시 train과 test dataset을 분할해줍니다. 위에서 제작해 놓았던 is_test 칼럼을 이용합니다.
        dt_train = concat_select.query('is_test==0')
        dt_test = concat_select.query('is_test==1')

        # 이제 is_test 칼럼은 drop해줍니다.
        dt_train.drop(['is_test'], axis = 1, inplace=True)
        dt_test.drop(['is_test'], axis = 1, inplace=True)
        dt_test['target'] = 0
        print(dt_train.shape, dt_test.shape)
        return dt_train, dt_test
    # def unconcat_train_test(concat):
    #     """
    #     결합된 데이터프레임을 train과 test로 분리하는 함수
        
    #     Args:
    #         concat (pd.DataFrame): 'is_test' 열을 포함한 결합된 데이터프레임
            
    #     Returns:
    #         tuple: (train_df, test_df) - 분리된 train과 test 데이터프레임
    #     """
    #     # Unnamed 열 제거 및 데이터프레임 복사
    #     concat = Utils.remove_unnamed_columns(concat)
        
    #     # 'is_test' 열 존재 여부 확인
    #     if 'is_test' not in concat.columns:
    #         raise ValueError("'is_test' 열이 데이터프레임에 없습니다.")
        
    #     # 데이터프레임 복사
    #     concat_copy = concat.copy()
        
    #     # train 데이터 추출 및 처리
    #     train_df = concat_copy[~concat_copy['is_test']].copy()
    #     train_df.drop(columns=['is_test'], inplace=True)
        
    #     # test 데이터 추출 및 처리
    #     test_df = concat_copy[concat_copy['is_test']].copy()
    #     test_df.drop(columns=['target', 'is_test'], inplace=True)
        
        return train_df, test_df
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
