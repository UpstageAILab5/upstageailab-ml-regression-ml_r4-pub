import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import get_unique_filename

class DataPrep():
    def __init__(self, config):
        self.time_delay = config.get('time_delay', 3)
        self.base_path = config.get('base_path')
        self.data_path = os.path.join(self.base_path, 'data')
        self.train_path = os.path.join(self.data_path, 'train.csv')
        self.test_path  = os.path.join(self.data_path, 'test.csv')
        self.out_path = config.get('out_path')
        self.target = config.get('target')
        self.config = config
        self.logger = config.get('logger')
        self.thr_ratio_outlier = config.get('thr_ratio_outlier')
        self.thr_ratio_null = config.get('thr_ratio_null')        
        # 경로 확인
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f'Train 파일을 찾을 수 없습니다: {self.train_path}')
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f'Test 파일을 찾을 수 없습니다: {self.test_path}')
        
        self.logger.info('#### Init Data Prep.. ')
    
    def data_prep(self):
        self.logger.info('#### Data Prep starts...')
        concat = self._load_data_concat_train_test()
        before_profile = DataPrep.get_data_profile(concat, stage="before")

        concat = self._null_prep(concat)
        self._null_check(concat)
        concat = self._remove_null(concat)
        self._null_check(concat)
        concat = self._interpolation(concat)
        after_profile = DataPrep.get_data_profile(concat, stage="after")
        # 전후 비교 테이블 생성
        comparison = pd.concat([before_profile, after_profile], 
                            keys=['Before', 'After'], 
                            axis=1)
        # 변화량 계산
        changes = pd.DataFrame()
        for col in comparison.index:
            if col in before_profile.index and col in after_profile.index:
                numeric_cols = ['missing_ratio', 'unique_ratio']
                if pd.api.types.is_numeric_dtype(concat[col]):
                    numeric_cols.extend(['mean', 'std', 'outlier_ratio'])
                
                for metric in numeric_cols:
                    if metric in before_profile.loc[col] and metric in after_profile.loc[col]:
                        before_val = before_profile.loc[col, metric]
                        after_val = after_profile.loc[col, metric]
                        changes.loc[col, f'{metric}_change'] = after_val - before_val
        
        # 결과 저장 및 출력
        comparison_path = os.path.join(self.out_path, 'data_prep_comparison_baseline.csv')
        changes_path = os.path.join(self.out_path, 'data_prep_changes_baseline.csv')
        comparison.to_csv(comparison_path)
        changes.to_csv(changes_path)
        
        # 결과 출력
        self.logger.info('\n=== Data Preparation Comparison ===')
        self.logger.info('\nBefore vs After Statistics:')
        self.logger.info(f'\n{comparison}')
        self.logger.info('\nKey Changes:')
        self.logger.info(f'\n{changes}')
        
        self.logger.info('#### Data Prep ends...')
        return concat

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
            concat = pd.concat([dt, dt_test], axis=0).reset_index(drop=True)
            
            self.logger.info(f'데이터 로드 완료: {concat.shape}')
            # Train data와 Test data shape은 아래와 같습니다.
            self.logger.info(f'Train data shape : {dt.shape}, Test data shape : {dt_test.shape}\n{dt.head(1)}\n{dt_test.head(1)}')
            # Train과 Test data를 살펴보겠습니다.
            # train/test 구분을 위한 칼럼을 하나 만들어 줍니다.
            dt['is_test'] = 0
            dt_test['is_test'] = 1
            self.logger.info('is_test column added to train and test data.\nConcat train and test data.')
            concat = pd.concat([dt, dt_test])     # 하나의 데이터로 만들어줍니다.
            concat['is_test'].value_counts()      # train과 test data가 하나로 합쳐진 것을 확인할 수 있습니다.
            # 칼럼 이름을 쉽게 바꿔주겠습니다. 다른 칼럼도 사용에 따라 바꿔주셔도 됩니다!
            concat = concat.rename(columns={'전용면적(㎡)':'전용면적'})
            self.logger.info(f'Concat data shape : {concat.shape}\n{concat.head(1)}')
            self.logger.info(f'Column name converted from 전용면적(㎡) to 전용면적 : {concat.columns}')
            self.logger.info('#### End Data Prep.. ')
            return concat
            
        except Exception as e:
            self.logger.error(f'데이터 로드 중 오류 발생: {str(e)}')
            return None
        
    ##### Null / Outlier
    def _null_prep(self, concat):
        # 실제로 결측치라고 표시는 안되어있지만 아무 의미도 갖지 않는 element들이 아래와 같이 존재합니다.
        # 아래 3가지의 경우 모두 아무 의미도 갖지 않는 element가 포함되어 있습니다.
        self.logger.info('### Null Prep.. ')
        self.logger.info(f'아래 3가지의 경우 모두 아무 의미도 갖지 않는 element가 포함되어 있습니다.\n{concat["등기신청일자"].value_counts()}')
        self.logger.info(f'{concat["거래유형"].value_counts()}')
        self.logger.info(f'{concat["중개사소재지"].value_counts()}')

        self.logger.info('위 처럼 아무 의미도 갖지 않는 칼럼은 결측치와 같은 역할을 하므로, np.nan으로 채워 결측치로 인식되도록 합니다.')
        concat['등기신청일자'] = concat['등기신청일자'].replace(' ', np.nan)
        concat['거래유형'] = concat['거래유형'].replace('-', np.nan)
        concat['중개사소재지'] = concat['중개사소재지'].replace('-', np.nan)
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
        
        # threshold 기준으로 컬럼 필터링
        columns_to_keep = ratio_null[ratio_null <= self.thr_ratio_null].index
        columns_to_drop = ratio_null[ratio_null > self.thr_ratio_null].index
        
        # 로깅
        self.logger.info(f'* 결측치 비율이 {self.thr_ratio_null*100}% 이하인 변수들: {list(columns_to_keep)}')
        self.logger.info(f'* 결측치 비율이 {self.thr_ratio_null*100}% 초과인 변수들: {list(columns_to_drop)}')
        
        # 선택된 컬럼만 반환
        return concat[columns_to_keep]

    def _interpolation(self, concat_select):
        # Interpolation
        self.logger.info('#### Interpolation starts...\n연속형 변수는 선형보간을 해주고, 범주형변수는 알수없기에 “unknown”이라고 임의로 보간해 주겠습니다.')
        self.logger.info('본번, 부번의 경우 float로 되어있지만 범주형 변수의 의미를 가지므로 object(string) 형태로 바꾸어주고 아래 작업을 진행하겠습니다.')
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
        self.logger.info(f"연속형 변수: {continuous_columns}")
        self.logger.info(f"범주형 변수: {categorical_columns}")
        # 범주 변수에 대한 보간
        concat_select[categorical_columns] = concat_select[categorical_columns].fillna('NULL')
        # 연속형 변수에 대한 보간 (선형 보간)
        concat_select[continuous_columns] = concat_select[continuous_columns].interpolate(method='linear', axis=0)
        self.logger.info('결측치가 보간된 모습을 확인해봅니다.')
        self.logger.info(concat_select.isnull().sum())

        # 이상치 제거 이전의 shape은 아래와 같습니다.
        self.logger.info(concat_select.shape)
        # 대표적인 연속형 변수인 “전용 면적” 변수 관련한 분포를 먼저 살펴보도록 하겠습니다.
        fig = plt.figure(figsize=(7, 3))
        try:
            sns.boxplot(data = concat_select, x = '전용면적(m)', color='lightgreen')
        except:
            sns.boxplot(data = concat_select, x = '전용면적', color='lightgreen')

        title = '전용면적 분포'
        plt.title(title)
        plt.xlabel('Area')
        plt.show(block=False)
        plt.pause(self.time_delay)  # 5초 동안 그래프 표시
        plt.close()
        plt.savefig(get_unique_filename(os.path.join(self.out_path, title +'.png')), dpi=300, bbox_inches='tight')
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