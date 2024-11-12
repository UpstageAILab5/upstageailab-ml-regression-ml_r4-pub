import os
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import wandb

# Importing XGBoost and LightGBM
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict

####

def load_config(path):
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config
def parse_comma_separated_list(param: str) -> List[str]:
    """
    쉼표로 구분된 문자열을 리스트로 변환합니다.
    """
    if not param:
        return []
    return [item.strip() for item in param.split(',')]

def encode_label(X: pd.DataFrame, X_test: pd.DataFrame, categorical_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    범주형 변수에 대해 라벨 인코딩을 수행합니다.
    """
    # 범주형 변수 리스트를 쉼표로 구분된 문자열에서 리스트로 변환
    if isinstance(categorical_features, str):
        categorical_features = parse_comma_separated_list(categorical_features)
    
    # 실제 데이터프레임에 존재하는 컬럼만 선택
    categorical_features = [col for col in categorical_features if col in X.columns]
    
    encoders = {}
    for feature in categorical_features:
        if feature in X.columns:  # 컬럼이 존재하는지 확인
            le = LabelEncoder()
            # 결측치가 있다면 먼저 처리
            X[feature] = X[feature].fillna('Unknown')
            if X_test is not None:
                X_test[feature] = X_test[feature].fillna('Unknown')
            
            # 학습 데이터와 테스트 데이터를 합쳐서 인코딩
            combined_data = pd.concat([X[feature], X_test[feature] if X_test is not None else pd.Series()])
            le.fit(combined_data)
            
            # 변환된 값을 정수형으로 변환
            X[feature] = le.transform(X[feature]).astype(int)
            if X_test is not None:
                X_test[feature] = le.transform(X_test[feature]).astype(int)
            
            encoders[feature] = le
    
    # 모든 범주형 변수가 정수형으로 변환되었는지 확인
    for feature in categorical_features:
        if not np.issubdtype(X[feature].dtype, np.integer):
            print(f"Warning: {feature} is not integer type after encoding: {X[feature].dtype}")
    
    return X, X_test, encoders

def remove_unnamed_columns(df):
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        print(f"Removing unnamed columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)
    return df
def unconcat_train_test(concat):
    remove_unnamed_columns(concat)
    
    dt = concat.query('is_test==0')
    # y_train = dt['target']
    dt.drop(columns=['is_test'], inplace=True)
    dt_test = concat.query('is_test==1')
    columns_to_drop = [col for col in ['target', 'is_test'] if col in dt_test.columns]
    dt_test.drop(columns=columns_to_drop, inplace=True)
    return dt, dt_test

def concat_train_test(dt, dt_test):
    remove_unnamed_columns(dt)
    remove_unnamed_columns(dt_test)
    dt['is_test'] = 0
    dt_test['is_test'] = 1
    dt_test['target'] = 0
    concat = pd.concat([dt, dt_test], axis=0).reset_index(drop=True)
    print(concat['is_test'].value_counts())

    return concat

from typing import List, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm

class DataScaler:
    def __init__(self, 
              
                 categorical_cols: List[str] = None,
                 continuous_cols: List[str] = None,
                 exclude_cols: List[str] = None):
        """
        데이터 스케일링을 위한 클래스
        
        Args:
            target_col: 타겟 컬럼명 (스케일링에서 제외)
            categorical_cols: 범주형 변수 리스트
            continuous_cols: 연속형 변수 리스트
            exclude_cols: 스케일링에서 제외할 컬럼 리스트
        """
     
        self.categorical_cols = set(categorical_cols) if categorical_cols else set()
        self.continuous_cols = set(continuous_cols) if continuous_cols else set()
        self.exclude_cols = set(exclude_cols) if exclude_cols else set()
        self.scalers = {}
        
    def _get_appropriate_scaler(self, col_name: str, data: pd.Series) -> object:
        """
        컬럼 특성에 맞는 스케일러 반환
        
        Args:
            col_name: 컬럼명
            data: 스케일링할 데이터
            
        Returns:
            스케일러 객체
        """
        # 범주형 변수는 StandardScaler 사용
        if col_name in self.categorical_cols:
            return StandardScaler()
        
        # 연속형 변수는 왜도에 따라 스케일러 결정
        if abs(data.skew()) > 1:
            # 심한 왜도는 RobustScaler + PowerTransformer
            return Pipeline([
                ('robust', RobustScaler()),
                ('power', PowerTransformer(method='yeo-johnson'))
            ])
        
        # 나머지는 RobustScaler
        return RobustScaler()
        
    def scale_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        특성별 적절한 스케일링 적용
        
        Args:
            df: 스케일링할 데이터프레임
            is_train: 학습 데이터 여부
            
        Returns:
            스케일링된 데이터프레임
        """
        scaled_df = df.copy()
        
        # 스케일링할 컬럼 결정
        if self.continuous_cols or self.categorical_cols:
            # 지정된 컬럼이 있는 경우
            scale_cols = self.continuous_cols | self.categorical_cols
        else:
            # 지정된 컬럼이 없는 경우 수치형 컬럼 자동 선택
            scale_cols = set(df.select_dtypes(include=['int64', 'float64']).columns)
        
        # 제외할 컬럼 처리
        exclude_set = self.exclude_cols.copy()
        exclude_set.add('is_test')  # 항상 제외
        
        scale_cols = scale_cols - exclude_set
        
        # 실제 존재하는 컬럼만 선택
        actual_scale_cols = scale_cols & set(df.columns)
        
        # 누락된 컬럼 확인
        missing_cols = scale_cols - actual_scale_cols
        if missing_cols:
            print(f"Warning: 다음 컬럼들이 데이터에 없습니다: {sorted(missing_cols)}")
        
        print(f"스케일링 적용 컬럼: {sorted(actual_scale_cols)}")
        
        # 스케일링 수행
        for col in tqdm(actual_scale_cols, desc='Scaling columns'):
            if is_train:
                # 학습 데이터: 스케일러 생성 및 적용
                scaler = self._get_appropriate_scaler(col, scaled_df[col])
                self.scalers[col] = scaler
                scaled_values = scaler.fit_transform(scaled_df[[col]]).ravel()
                scaled_df[col] = scaled_values
            else:
                # 테스트 데이터: 기존 스케일러 적용
                if col in self.scalers:
                    scaled_values = self.scalers[col].transform(scaled_df[[col]]).ravel()
                    scaled_df[col] = scaled_values
                else:
                    print(f"Warning: {col} 럼의 스케���러가 없습니다.")
        
        return scaled_df
    

def custom_scaling_df(df, numerical_features, categorical_features=None):
    """Apply custom scaling to the DataFrame."""
    existing_numerical_cols = [
        col for col in numerical_features if col in df.columns
    ]
    print(f'existing_numerical_cols: {existing_numerical_cols}')
    
    df_train, df_test = unconcat_train_test(df) # train, test data 분리
    # # categorical, numerical 기준은 baseline 으로 임시 작성. 변경 가능.
    # categorical_features =['전용면적', '계약일', '층', '건축년도', 'k-전체동수', 'k-전체세대수', 'k-연면적', 'k-주거전용면적', 'k-관리비부과면적', 'k-전용면적별세대현황(60㎡이하)', 'k-전용면적별세대현황(60㎡~85㎡이하)', 'k-85㎡~135㎡이하', '건축면적', '주차대수', '좌표X', '좌표Y', 'target', '강남여부', '신축여부']
    # numerical_features = ['번지', '본번', '부번', '아파트명', '도로명', 'k-단지분류(아파트,주상복합등등)', 'k-전화번호', 'k-팩스번호', 'k-세대타입(분양형태)', 'k-관리방식', 'k-복도유형', 'k-난방방식', 'k-건설사(시공사)', 'k-시행사', 'k-사용검사일-사용승인일', 'k-수정일자', '고용보험관리번호', '경비비관리형태', '세대전기계약방법', '청소비관리형태', '기타/의무/임대/임의=1/2/3/4', '단지승인일', '사용허가여부', '관리비 업로드', '단지신청일', '구', '동', '계약년', '계약월']
    # add_features = ['구', '동', '계약년', '계약월']
    # numerical_features = list(set(numerical_features)-set(add_features))

    data_scaler = DataScaler(
                                categorical_cols=categorical_features,
                                continuous_cols=numerical_features,
                                exclude_cols=['target']) # 아직 
    if not df_train[existing_numerical_cols].empty:
        df_train[existing_numerical_cols] = data_scaler.scale_features(df_train[existing_numerical_cols], is_train=True)
    if not df_test[existing_numerical_cols].empty:
        df_test[existing_numerical_cols] = data_scaler.scale_features(df_test[existing_numerical_cols], is_train=False)

    # 1. Numerical Features 들의 경우, outlier removal 없이 이상치에 강건하게 Robust Scaling 을 먼저 적용합니다.
    # 학습 데이터 전처리

    # 2. Categorical Features 들의 경우, Feature engineering 후 encoding 까지 완료된 이후에 표준화 스케일링을 별도 적용합니다
    print(df_train.isnull().sum(), df_test.isnull().sum())
    print(df_train.shape, df_test.shape)
    return df_train, df_test


def load_data(paths, additional_feature_paths=None, base_path=None):
    """Load datasets and concatenate additional features if provided."""
    dataframes = [pd.read_csv(os.path.join(base_path, path)) for path in paths]
    df = pd.concat(dataframes, axis=1)
    
    if additional_feature_paths:
        additional_features = [
            pd.read_csv(os.path.join(base_path, path)) for path in additional_feature_paths
        ]
        df_additional = pd.concat(additional_features, axis=1)
        df = pd.concat([df, df_additional], axis=1)
    df = remove_unnamed_columns(df)

    return df


def handle_nulls(df, strategy='mean', exclude_columns=None):
    """Handle null values in the DataFrame, excluding specified columns."""
    if exclude_columns is None:
        exclude_columns = []

    # 실제 존재하는 컬럼만 제외
    actual_exclude_columns = [col for col in exclude_columns if col in df.columns]

    # 제외할 컬럼을 분리
    df_excluded = df[actual_exclude_columns]
    df_to_impute = df.drop(columns=actual_exclude_columns)

    if strategy == 'drop':
        df_imputed = df_to_impute.dropna()
    elif strategy == 'null':
        df_imputed = df_to_impute
    else:
        imputer = SimpleImputer(strategy=strategy)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df_to_impute), columns=df_to_impute.columns
        )

    # 제외한 컬럼을 다시 합치기
    df_result = pd.concat([df_imputed, df_excluded], axis=1)
    
    return df_result


def handle_outliers(df, method='zscore', threshold=3, exclude_columns='target'):
    """Handle outliers in the DataFrame, excluding specified columns."""
    if exclude_columns is None:
        exclude_columns = []
    actual_exclude_columns = [col for col in exclude_columns if col in df.columns]
    # 제외할 컬럼을 분리
    df_excluded = df[actual_exclude_columns]
    df_to_process = df.drop(columns=actual_exclude_columns)

    if method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df_to_process._get_numeric_data()))
        mask = (z_scores < threshold).all(axis=1)
        df_processed = df_to_process[mask]
    elif method == 'iqr':
        Q1 = df_to_process.quantile(0.25)
        Q3 = df_to_process.quantile(0.75)
        IQR = Q3 - Q1
        mask = (df_to_process >= Q1 - 1.5 * IQR) & (df_to_process <= Q3 + 1.5 * IQR)
        df_processed = df_to_process[mask]
    elif method == 'null':
        df_processed = df_to_process
    else:
        df_processed = df_to_process

    # 제외한 컬럼을 다시 합치기
    df_result = pd.concat([df_processed, df_excluded], axis=1)
    
    return df_result


def custom_scaling(df, method='standard', categorical_features=None, numerical_features=None):
    """Apply custom scaling to the DataFrame."""
    if method == 'custom':
        # Custom scaling example: log transformation
        
        df_train, df_test = custom_scaling_df(df, numerical_features, categorical_features)
        return df_train, df_test
    else:
        df_train, df_test = unconcat_train_test(df)
        return df_train, df_test
    


def feature_engineering(df, method='none'):
    """Apply feature engineering to the DataFrame."""
    from sklearn.preprocessing import PolynomialFeatures
    if method == 'polynomial':
        poly = PolynomialFeatures(degree=2, include_bias=False)
        df_fe = pd.DataFrame(
            poly.fit_transform(df),
            columns=poly.get_feature_names_out(df.columns)
        )
        return df_fe
    elif method == 'interaction':
        poly = PolynomialFeatures(
            degree=2, include_bias=False, interaction_only=True
        )
        df_fe = pd.DataFrame(
            poly.fit_transform(df),
            columns=poly.get_feature_names_out(df.columns)
        )
        return df_fe
    else:
        return df


def feature_selection_correlation(X, y, threshold=0.5):
    """Select features based on correlation with the target."""
    df = pd.concat([X, y], axis=1)
    corr = df.corr()
    target_corr = corr[y.name].abs().sort_values(ascending=False)
    selected_features = target_corr[
        target_corr > threshold
    ].index.drop(y.name, errors='ignore')
    return X[selected_features]

def encode_categorical_features(X: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
    """
    범주형 변수를 라벨 인코딩하여 정수형으로 변환합니다.
    """
    for feature in categorical_features:
        if feature in X.columns:
            le = LabelEncoder()
            # 결측치가 있다면 'Unknown'으로 채우기
            X[feature] = X[feature].fillna('Unknown')
            # 라벨 인코딩 적용
            X[feature] = le.fit_transform(X[feature])
    return X

def load_and_preprocess_data(config):
    """
    데이터 로드 및 전처리를 수행합니다.
    """
    # 데이터 로드
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prep_path = os.path.join(base_path, 'data', 'preprocessed')
    data_paths = config.data_paths
    additional_feature_paths = parse_comma_separated_list(config.additional_feature_paths)
    
    df = load_data(data_paths, additional_feature_paths, prep_path)
    
    # 타겟 변수 분리
    target = config.target_variable
    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in dataset.")
    
    y = df[target]
    X = df.drop(columns=[target])
    
    # 결측치 처리
    X = handle_nulls(X, strategy=config.null_strategy, exclude_columns=[target])
    y = y.loc[X.index]
    
    # 범주형 변수 인코딩
    categorical_features = parse_comma_separated_list(config.categorical_features)
    X = encode_categorical_features(X, categorical_features)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.dataset_split_ratio, random_state=config.random_state
    )
    
    return X_train, X_test, y_train, y_test

def main():
    """Run the data pipeline and model training."""
    wandb.init()
    config = wandb.config
    
    # 데이터 로드 및 전처리
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    
    # 모델 초기화
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=100, random_state=config.random_state
        ),
        'xgboost': xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.1, random_state=config.random_state,
            enable_categorical=True
        ),
        'lightgbm': lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, random_state=config.random_state,
            force_col_wise=True, min_data_in_leaf=20, min_data_in_bin=3, verbose=-1
        )
    }
    
    model = models.get(config.model_type)
    if not model:
        raise ValueError(f"모델 타입 '{config.model_type}'이(가) 지원되지 않습니다.")
    
    # 모델 학습 및 평가
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    wandb.log({'rmse': rmse})
    print(f"Test RMSE: {rmse}")

    # 예측 결과 저장
    y_pred_df = pd.DataFrame(y_pred, columns=[config.target_variable])
    output_filename = f'y_pred_{config.model_type}_{wandb.run.id}.csv'
    y_pred_df.to_csv(os.path.join(base_path, 'output', output_filename), index=False)
    print(f"예측 결과 저장 완료: {output_filename}")


if __name__ == '__main__':
    # Read configuration from YAML file
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(base_path)
    with open(os.path.join(base_path, 'test','config_sweep.yaml'), 'r',  encoding='utf-8') as file:
        yaml_config = yaml.safe_load(file)
    print(f'selected_features: {yaml_config["selected_features"]}')
    print(f'additional_feature_paths: {yaml_config["additional_feature_paths"]}')

    categorical_features = ['k-건설사(시공사)', 'k-관리방식', 'k-난방방식', 'k-단지분류(아파트,주상복합등등)', 'k-복도유형', 'k-사용검사일-사용승인일', 'k-세대타입(분양형태)', 'k-수정일자', 'k-시행사', '경비비관리형태', '관리비 업로드', '기타/의무/임대/임의=1/2/3/4', '단지승인일', '단지신청일', '도로명', '번지', '사용허가여부', '세대전기계약방법',  '아파트명', '청소비관리형태']
    numerical_features = ['k-85㎡~135㎡이하', 'k-관리비부과면적', 'k-연면적', 'k-전용면적별세대현황(60㎡~85㎡이하)', 'k-전용면적별세대현황(60㎡이하)', 'k-전체동수', 'k-전체세대수', 'k-주거전용면적', '건축년도', '건축면적', '계약일', '본번', '부번', '전용면적', '좌표X', '좌표Y', '주차대수', '층']
    selected_features = ['전용면적', '강남여부', '신축여부', '구', '동', '건축년도', '건축면적', '좌표X', '좌표Y', '주차대수', '층', '계약년', 'subway_shortest_distance', 'bus_shortest_distance', '대장아파트_거리']
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'rmse', 'goal': 'minimize'},
        'parameters': {
            'data_paths': {'values': yaml_config['data_paths']},
            'additional_feature_paths': {
                'values': yaml_config['additional_feature_paths']
            },
            'target_variable': {'values': yaml_config['target_variables']},
            # 'null_strategy': {'values': yaml_config['null_strategies']},
            # 'outlier_method': {'values': yaml_config['outlier_methods']},
            # 'outlier_threshold': {'values': yaml_config['outlier_thresholds']},
            'scaling_method': {'values': yaml_config['scaling_methods']},
            # 'feature_engineering_method': {
            #     'values': yaml_config['feature_engineering_methods']
            # },
            'feature_selection_method': {
                'values': yaml_config['feature_selection_methods']
            },
            'correlation_threshold': {
                'values': yaml_config['correlation_thresholds']
            },
            'dataset_split_ratio': {
                'values': yaml_config['dataset_split_ratios']
            },
            'selected_features': {'values': selected_features},
            'split_method': {'values': yaml_config['split_methods']},
            'k_folds': {'values': yaml_config['k_folds']},
            'model_type': {'values': yaml_config['model_types']},
            'categorical_features': {'values': categorical_features},
            'numerical_features': {'values': numerical_features},
            'random_state': {'values': yaml_config['random_states']}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='regression_optimization')
    wandb.agent(sweep_id, function=main, count=200)
