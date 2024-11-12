import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import Utils
from typing import List, Dict
get_unique_filename = Utils.get_unique_filename

class DataPrep():
    def __init__(self, config):
        self.config = config
        self.logger_instance = config.get('logger')
        self.logger_instance.setup_logger(log_file='preprocessing')
        self.logger = self.logger_instance.logger
        self.base_path = config.get('path').get('base')
        self.data_path = config.get('path').get('data')
        self.out_path = config.get('path').get('out')
        self.prep_path = config.get('path').get('prep')
        self.time_delay = config.get('time_delay', 3)
        self.train_path = os.path.join(self.data_path, 'train.csv')
        self.test_path  = os.path.join(self.data_path, 'test.csv')
        self.target = config.get('target')
        self.thr_ratio_outlier = config.get('thr_ratio_outlier')
        self.thr_null = config.get('thr_null')
        self.thr_detect_categorical = float(config.get('thr_detect_categorical'))
        self.logger.info(f'thr_detect_categorical: {self.thr_detect_categorical}')
        self.thr_ratio_null = config.get('thr_ratio_null')
        self.df_eda = pd.DataFrame()    
        # 경로 확인
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f'Train 파일을 찾을 수 없습니다: {self.train_path}')
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f'Test 파일을 찾을 수 없습니다: {self.test_path}')
        
        self.logger.info('#### Init Data Prep.. ')
    
    def data_prep(self, remove_cols: List[str] = None):
        ####
        self.logger.info('#### Data Prep starts...\n Load Data.')
        concat = self._load_data_concat_train_test()
        self.logger.info(f'Data prep 시작 전 데이터 shape: {concat.shape}')
        concat.to_csv(os.path.join(self.prep_path, 'df_raw.csv'))
        before_profile = DataPrep.get_data_profile(concat, stage="before")
        before_profile.to_csv(os.path.join(self.out_path, 'report_profile_raw.csv'))
        ######################################################################
        #concat = self._null_allocation(concat, remove_cols)
        concat = concat.drop(columns=remove_cols)
        self.logger.info(f'{remove_cols} 칼럼 제거 -> {concat.shape}\n{concat.columns}')
        concat = self._remove_null(concat)
        self.logger.info(f'결측치 처리 후 데이터 shape: {concat.shape}')
        ####
        concat = self._interpolation(concat)
        self.logger.info(f'보간 후 데이터 shape: {concat.shape}')
        # 대표적인 연속형 변수인 “전용 면적” 변수 관련한 분포를 먼저 살펴보도록 하겠습니다.
        fig = plt.figure(figsize=(7, 3))
        sns.boxplot(data = concat, x = '전용면적', color='lightgreen')
        title = '전용면적 분포'
        plt.title(title)
        plt.xlabel('Area')
        plt.show(block=False)
        plt.savefig(get_unique_filename(os.path.join(self.prep_path, title +'.png')), dpi=300, bbox_inches='tight')
        plt.pause(self.time_delay)  # 5초 동안 그래프 표시
        plt.close()
        
        #concat = self.remove_outliers_iqr(concat, '전용면적')
        
        
        # after_profile = DataPrep.get_data_profile(concat, stage="after")
        # # 전후 비교 테이블 생성
        # comparison = pd.concat([before_profile, after_profile], 
        #                     keys=['Before', 'After'], 
        #                     axis=1)
        # # 변화량 계산
        # changes = pd.DataFrame()
        # for col in comparison.index:
        #     if col in before_profile.index and col in after_profile.index:
        #         numeric_cols = ['missing_ratio', 'unique_ratio']
        #         if pd.api.types.is_numeric_dtype(concat[col]):
        #             numeric_cols.extend(['mean', 'std', 'outlier_ratio'])
                
        #         for metric in numeric_cols:
        #             if metric in before_profile.loc[col] and metric in after_profile.loc[col]:
        #                 before_val = before_profile.loc[col, metric]
        #                 after_val = after_profile.loc[col, metric]
        #                 changes.loc[col, f'{metric}_change'] = after_val - before_val
        
        # # 결과 저장 및 출력
        # comparison_path = os.path.join(self.out_path, 'data_prep_comparison_baseline.csv')
        # changes_path = os.path.join(self.out_path, 'data_prep_changes_baseline.csv')
        # comparison.to_csv(comparison_path)
        # changes.to_csv(changes_path)
        
        # # 결과 출력
        # self.logger.info('\n=== Data Preparation Comparison ===')
        # self.logger.info('\nBefore vs After Statistics:')
        # self.logger.info(f'\n{comparison}')
        # self.logger.info('\nKey Changes:')
        # self.logger.info(f'\n{changes}')
        
        self.logger.info('#### Data Prep ends...')
        return concat

    @staticmethod
    def get_data_profile(df: pd.DataFrame, stage: str = "before") -> pd.DataFrame:
        """데이터프레임의 주요 특성을 분석하여 프로파일 생성"""
        profile = {}
        
        for col in df.columns:
            stats = {
                'type': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'missing_ratio': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_ratio': (df[col].nunique() / len(df)) * 100
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # 수치형 변수 통계
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                
                stats.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'outlier_count': len(outliers),
                    'outlier_ratio': (len(outliers) / len(df)) * 100
                })
            else:
                # 범주형 변수 통계
                stats.update({
                    'most_common': df[col].mode().iloc[0] if not df[col].empty else None,
                    'most_common_ratio': (df[col].value_counts().iloc[0] / len(df)) * 100 if not df[col].empty else 0
                })
                
            profile[col] = stats
        
        return pd.DataFrame.from_dict(profile, orient='index')
    
    def _load_data_concat_train_test(self):
        try:
            # 데이터 로드
            dt = pd.read_csv(self.train_path)
            dt_test = pd.read_csv(self.test_path)
            # 데이터 합치기
            # concat = pd.concat([dt, dt_test], axis=0).reset_index(drop=True)
            
            # self.logger.info(f'데이터 로드 완료: {concat.shape}')
            # Train data와 Test data shape은 아래와 같습니다.
            self.logger.info(f'train/test 구분을 위한 칼럼을 하나 만들어 줍니다. Train data shape : {dt.shape}, Test data shape : {dt_test.shape}\n')
            # Train과 Test data를 살펴보겠습니다.
    
            dt['is_test'] = 0
            dt_test['is_test'] = 1
            self.logger.info('is_test column added to train and test data.\nConcat train and test data.')
            concat = pd.concat([dt, dt_test])     # 하나의 데이터로 만들어줍니다.
            concat['is_test'].value_counts()      # train과 test data가 하나로 합쳐진 것을 확인할 수 있습니다.
            self.logger.info('칼럼 이름을 쉽게 바꿔주겠습니다. 다른 칼럼도 사용에 따라 바꿔주셔도 됩니다!')
            concat = concat.rename(columns={'전용면적(㎡)':'전용면적'})
            self.logger.info(f'Concat data shape : {concat.shape}\n')
            self.logger.info(f'Column name converted from 전용면적(㎡) to 전용면적 : {concat.columns}')
            self.logger.info('#### End Data Prep.. ')
            return concat
            
        except Exception as e:
            self.logger.error(f'데이터 로드 중 오류 발생: {str(e)}')
            return None
        
    ##### Null / Outlier
    def _null_allocation(self, concat, remove_cols: List[str] = None):
        # 실제로 결측치라고 표시는 안되어있지만 아무 의미도 갖지 않는 element들이 아래와 같이 존재합니다.
        # 아래 3가지의 경우 모두 아무 의미도 갖지 않는 element가 포함되어 있습니다.
        self.logger.info('### Null Prep. 아무 의미도 갖지 않는 칼럼은 결측치와 같은 역할을 하므로, np.nan으로 채워 결측치로 인식되도록 합니다.')
        if remove_cols:
            for col in remove_cols:
                self.logger.info(f'{col} column nan 처리')
                concat[col] = np.nan#concat[col].replace(' ', np.nan)   
        return concat
    
    def _null_check(self, concat):
        # EDA에 앞서 결측치를 확인해보겠습니다.
        self.logger.info('### Null Check.. ')
        self.logger.info('결측치 확인')
        self.logger.info(concat.isnull().sum())
        # 변수별 결측치의 비율을 plot으로 그려보면 아래 같습니다.
        fig = plt.figure(figsize=(13, 2))
        missing = concat.isnull().sum() / concat.shape[0]
        missing = missing[missing > 0]
        missing.sort_values(inplace=True)
        missing.plot.bar(color='orange')
        title='변수별 결측치 비율'
        plt.title(title)
        plt.savefig(get_unique_filename(os.path.join(self.out_path, title +'.png')), dpi=300, bbox_inches='tight')
        
        plt.show(block=False)
        plt.pause(self.time_delay)  # 5초 동안 그래프 표시
        plt.close()
        
    def _remove_null(self, concat):
        """결측치가 특정 비율 이상인 컬럼 제거"""
        # 각 컬럼별 결측치 비율 계산 (전체 행 수 대비)
        ratio_null = concat.isnull().sum() / len(concat)
        #count_null = concat.isnull().sum()
        # threshold 기준으로 컬럼 필터링
        columns_to_keep = ratio_null[ratio_null <= self.thr_ratio_null].index
        columns_to_drop = ratio_null[ratio_null > self.thr_ratio_null].index
        # 로깅
        self.logger.info(f'* 결측치 비율이 {self.thr_ratio_null} 이하인 변수들: {list(columns_to_keep)}')
        self.logger.info(f'* 결측치 비율이 {self.thr_ratio_null} 초과인 변수들: {list(columns_to_drop)}')
        
        # 선택된 컬럼만 반환
        return concat[columns_to_keep]
    
    def _fill_missing_values(self, 
                            df: pd.DataFrame, 
                            target_cols: List[str],
                            group_cols: List[str] = ['도로명주소', '시군구', '도로명', '아파트명'],
                            is_categorical: bool = True) -> pd.DataFrame:
        """결측치 채우기 함수"""
        df_filled = df.copy()
        
        # 결측치 현황 출력
        null_before = df_filled[target_cols].isnull().sum()
        print(f"처리 전 결측치:\n{null_before}\n")
        
        # 상세 단위부터 큰 단위까지 순차적으로 결측치 채우기
        for i in range(len(group_cols), 0, -1):
            current_groups = group_cols[:i]
            
            # 그룹화 컬럼이 데이터에 있는지 확인
            valid_groups = [col for col in current_groups if col in df_filled.columns]
            
            if not valid_groups:
                continue
                
            for col in target_cols:
                # 결측치가 있는 경우에만 처리
                if df_filled[col].isnull().any():
                    if is_categorical:
                        # 범주형: 최빈값으로 채우기
                        fill_values = df_filled.groupby(valid_groups)[col].transform(
                            lambda x: x.mode().iloc[0] if not x.mode().empty else x
                        )
                    else:
                        # 수치형: 평균값으로 채우기
                        fill_values = df_filled.groupby(valid_groups)[col].transform('mean')
                    
                    # NaN이 아닌 값만 업데이트
                    mask = df_filled[col].isnull() & fill_values.notna()
                    df_filled.loc[mask, col] = fill_values[mask]
                    
                    filled_count = null_before[col] - df_filled[col].isnull().sum()
                    if filled_count > 0:
                        print(f"{col}: {valid_groups}기준으로 {filled_count}개 채움")
        
        # 남은 결측치 처리
        for col in target_cols:
            if df_filled[col].isnull().any():
                if is_categorical:
                    fill_value = df_filled[col].mode().iloc[0] if not df_filled[col].mode().empty else 'Unknown'
                    method = "최빈값"
                else:
                    fill_value = df_filled[col].mean()
                    method = "평균값"
                    
                missing_count = df_filled[col].isnull().sum()
                df_filled[col] = df_filled[col].fillna(fill_value)
                print(f"{col}: 전체 {method}({fill_value})으로 {missing_count}개 채움")
        
        # 결과 확인
        null_after = df_filled[target_cols].isnull().sum()
        print(f"\n처리 후 결측치:\n{null_after}")
        print(f"\n채워진 결측치 수:\n{null_before - null_after}")
        
        return df_filled
    def _detect_column_types(self, 
                            df: pd.DataFrame,
                            unique_ratio_threshold: float = 1e-7,
                            exclude_cols: List[str] = None) -> Dict[str, List[str]]:
        """
        데이터프레임의 컬럼들을 categorical/numerical로 분류
        
        Args:
            df: 데이터프레임
            unique_ratio_threshold: unique 값 비율 임계값 (default: 0.05)
            exclude_cols: 제외할 컬럼 리스트
        
        Returns:
            Dict[str, List[str]]: {'categorical': [...], 'numerical': [...]}
        """
        if exclude_cols is None:
            exclude_cols = set()
        else:
            exclude_cols = set(exclude_cols)
        
        columns_to_analyze = set(df.columns) - exclude_cols
        
        categorical_cols = []
        numerical_cols = []
        
        total_rows = len(df)
        
        for col in columns_to_analyze:
            # 데이터 타입 먼저 확인
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            
            # object나 category 타입은 바로 categorical로 분류
            if not is_numeric or df[col].dtype in ['object', 'category']:
                categorical_cols.append(col)
                continue
                
            # 수치형 데이터만 unique 비율 계산
            unique_ratio = df[col].nunique() / total_rows
            
            if unique_ratio < unique_ratio_threshold:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        print(f"Categorical columns ({len(categorical_cols)}): {sorted(categorical_cols)}")
        print(f"Numerical columns ({len(numerical_cols)}): {sorted(numerical_cols)}")
        
        return {
            'categorical': sorted(categorical_cols),
            'numerical': sorted(numerical_cols)
        }
    def _interpolation(self, concat: pd.DataFrame) -> pd.DataFrame:
        """결측치 보간"""
        concat_select = concat.copy()
        
        # 컬럼 타입 감지
        self.logger.info('컬럼 타입 감지')
        column_types = self._detect_column_types(concat_select, unique_ratio_threshold = self.thr_detect_categorical,
                                                exclude_cols = ['is_target', 'is_test'])
        categorical_columns = column_types['categorical']
        numerical_columns = column_types['numerical']
        self.logger.info(f"연속형 변수: {numerical_columns}")
        self.logger.info(f"범주형 변수: {categorical_columns}")
        
        group_cols = ['시군구', '도로명', '아파트명']
        self.logger.info('범주형 변수 결측치 처리')
        self.logger.info(concat_select[categorical_columns].isnull().sum())
        # 범주형 변수 결측치 처리
        if categorical_columns:
            filled_cat = self._fill_missing_values(
                concat_select, 
                target_cols=categorical_columns, 
                group_cols=group_cols,
                is_categorical=True
            )
            # 각 컬럼별로 값 복사
            for col in categorical_columns:
                concat_select[col] = filled_cat[col]
        self.logger.info('범주형 변수 결측치 처리 완료')
        self.logger.info(concat_select[categorical_columns].isnull().sum())

        self.logger.info('수치형 변수 결측치 처리')
        self.logger.info(concat_select[numerical_columns].isnull().sum())
        # 수치형 변수 결측치 처리
        if numerical_columns:
            filled_num = self._fill_missing_values(
                concat_select, 
                target_cols=numerical_columns, 
                group_cols=group_cols,
                is_categorical=False
            )
            # 각 컬럼별로 값 복사
            for col in numerical_columns:
                concat_select[col] = filled_num[col]
        self.logger.info('수치형 변수 결측치 처리 완료')
        self.logger.info(concat_select[numerical_columns].isnull().sum())
        return concat_select
#     def _interpolation(self, concat):
#         # Interpolation
#         self.logger.info('#### Interpolation starts...\n연속형 변수는 선형보간을 해주고, 범주형변수는 알수없기에 “unknown”이라고 임의로 보간해 주겠습니다.')
#         self.logger.info('본번, 부번의 경우 float로 되어있지만 범주형 변수의 의미를 가지므로 object(string) 형태로 바꾸어주고 아래 작업을 진행하겠습니다.')
#         concat_select = concat.copy()
        
#         concat_select['본번'] = concat_select['본번'].astype('str')
#         concat_select['부번'] = concat_select['부번'].astype('str')
#         # 먼저, 연속형 변수와 범주형 변수를 위 info에 따라 분리해주겠습니다.
#         continuous_columns = []
#         categorical_columns = []
#         for column in concat_select.columns:
#             if pd.api.types.is_numeric_dtype(concat_select[column]):
#                 continuous_columns.append(column)
#             else:
#                 categorical_columns.append(column)
#         self.logger.info(f"연속형 변수: {continuous_columns}")
#         self.logger.info(f"범주형 변수: {categorical_columns}")
#         # 범주 변수에 대한 보간
#         #concat_select[categorical_columns] = concat_select[categorical_columns].fillna('NULL')
#         # 연속형 변수에 대한 보간 (선형 보간)
#    #     coord_cols=['좌표X', '좌표Y']
        
#         concat_select[categorical_columns] = self._fill_missing_values(concat_select, categorical_columns
#                                                                        , group_cols=group_cols, is_categorical=True)
#         concat_select[continuous_columns] = self._fill_missing_values(concat_select, continuous_columns
#                                                                        , group_cols=group_cols, is_categorical=False)
#         #concat_select[continuous_columns] = concat_select[continuous_columns].interpolate(method='linear', axis=0)
#         self.logger.info('결측치가 보간된 모습을 확인해봅니다.')
#         self.logger.info(concat_select.isnull().sum())

#         return concat_select

    # 이상치 제거 방법에는 IQR을 이용하겠습니다.
    
    def remove_outliers_iqr(self,dt, column_name):
        df = dt.query('is_test == 0')       # train data 내에 있는 이상치만 제거하도록 하겠습니다.
        df_test = dt.query('is_test == 1')

        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

        result = pd.concat([df, df_test])   # test data와 다시 합쳐주겠습니다.
        self.logger.info(f'{column_name} 이상치 제거 후 데이터 shape: {dt.shape}')
        return result