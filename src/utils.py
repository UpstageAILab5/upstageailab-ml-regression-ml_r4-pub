import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import platform
from pathlib import Path
from typing import Dict, Any, List
import yaml
from pathlib import Path
class FileCache:
    def __init__(self, logger=None):
        self.logger = logger

    def load_or_create(self, file_path, create_func, index_col=0, *args, **kwargs):
        """파일이 존재하면 로드하고, 없으면 생성하여 저장"""
        if os.path.exists(file_path):
            if self.logger:
                self.logger.info(f'>>>> 결과 존재. Loading from {file_path}...')
            return pd.read_csv(file_path, index_col=index_col)
        
        if self.logger:
            self.logger.info(f'>>>> 결과 존재하지 않음. 생성 시작...')
        result = create_func(*args, **kwargs)
        result.to_csv(file_path, index=True if index_col is not None else False)
        return result
class Utils:
    def __init__(self, logger):
        self.current_platform = platform.system()
        self.logger = logger
        self.time_delay = 3
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
    def detect_column_types(
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
    def get_latest_file_with_string(directory, search_string):
        """특정 문자열을 포함하는 파일 중 가장 최신 파일의 경로를 반환합니다."""
        latest_file = None
        latest_time = 0

        for root, _, files in os.walk(directory):
            for file in files:
                if search_string in file:
                    file_path = os.path.join(root, file)
                    file_time = os.path.getmtime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file_path

        return latest_file
    @staticmethod
    def list_files(directory, ext='.csv'):
        ext_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(ext):
                    ext_files.append(os.path.join(root, file))
        return ext_files
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
    def save_file(self, data, filepath):
        """파일 확장자를 확인한 후 데이터를 저장합니다."""
        filename = self.add_timestamp_to_filename(filename)
        ext = os.path.splitext(filepath)[1]
        full_path = self.create_path_relative_to_script(filepath)
        self.ensure_directory_exists(os.path.dirname(full_path))
        print('Saved to ', full_path)
        if ext == '.txt':
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
        elif ext == '.json':
            import json
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        elif ext == '.csv':
            if isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            data.to_csv(filepath,encoding='utf-8-sig' )
        else:
            raise ValueError(f"지원하지 않는 파일 확장자입니다: {ext}")
    @staticmethod
    def load_file(self, filepath):
        """파일 확장자를 확인한 후 파일을 불러옵니다."""
        ext = os.path.splitext(filepath)[1]
        full_path = self.create_path_relative_to_script(filepath)
        if ext == '.txt':
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.json':
            import json
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif ext =='.csv':
            return pd.read_csv(filepath, encoding='utf-8-sig') 
        else:
            raise ValueError(f"지원하지 않는 파일 확장자입니다: {ext}")
    @staticmethod
    def print_data_info(data):
        """데이터의 타입, 크기, 내용 출력합니다."""
        print(f"타입: {type(data)}")
        if hasattr(data, '__len__'):
            print(f"크기: {len(data)}")
        print(f"내용: {data}")

    @staticmethod
    def timeit(func):
        """함수의 실행 시간을 측정하는 데코레이터입니다."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"'{func.__name__}' 함수 실행 시간: {end_time - start_time:.6f}초")
            return result
        return wrapper
    
    @staticmethod
    def find_elements_with_substring(lst, substring):
        indices = [index for index, element in enumerate(lst) if substring in element]
        elements = [lst[index] for index in indices]
        return indices, elements
    @staticmethod
    def load_nested_yaml(yaml_path: str) -> Dict[str, Any]:
        """
        계층적 구조의 YAML 파일을 딕셔너리로 로드
        
        Parameters:
            yaml_path: YAML 파일 경로
        Returns:
            dict: 중첩된 딕셔너리
        """
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"YAML 파일 로드 중 오류 발생: {str(e)}")
            return {}

    # 중첩된 설정값 접근을 위한 헬퍼 함수
    @staticmethod
    def get_nested_value(dict_obj: Dict[str, Any], key_path: str, default=None) -> Any:
        """
        계층적 딕셔너리에서 점(.) 표기법으로 값 접근
        
        Parameters:
            dict_obj: 중첩된 딕셔너리
            key_path: 점으로 구분된 키 경로 (예: "coordinates.subway.x")
            default: 기본값
        """
        keys = key_path.split('.')
        value = dict_obj
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    @staticmethod
    def clean_df(df):
        df = Utils.remove_unnamed_columns(df)
        duplicated_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicated_cols:
            print("중복된 컬럼:", duplicated_cols)
            # 방법 1: 중복 컬럼 제거 (첫 번째 컬럼 유지)
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
        #col_to_drop=[col for col in df.columns if 'target' in col]
        #print('col_to_drop:',col_to_drop)
        #df.drop(columns=col_to_drop, inplace=True)
        return df

    @staticmethod
    def remove_unnamed_columns(df):
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            print(f"Removing unnamed columns: {unnamed_cols}")
            df = df.drop(columns=unnamed_cols)
        return df
    def setup_font_and_path_platform(self):
        # 운영체제 확인 후 폰트 설정
        if platform.system() == 'Windows':
            plt.rc('font', family='Malgun Gothic')  # Windows
        elif platform.system() == 'Darwin':          # Mac
            plt.rc('font', family='AppleGothic')
        else:
            plt.rc('font', family='NanumGothic')    # Linux
        # 마이너스 기호 깨짐 방지
        self.logger.info(f'{self.current_platform} platform. Font: {plt.rcParams["font.family"]}')
        plt.rc('axes', unicode_minus=False)
        # 테스트
    #     plt.figure(figsize=(3, 1))
    #     plt.text(0.5, 0.5, '처음과 같이 이제와 항상 영원히', ha='center', va='center')
    #     plt.axis('off')
    #     # 현재 figure 저장
    #     fig = plt.gcf()
    #     # 그래프 표시 (non-blocking)
    #     plt.show(block=False)
    #     #time.sleep(self.time_delay)
    #     plt.pause(self.time_delay)
    #     plt.close()
    #    # delay = time_delay if time_delay is not None else self.time_delay
    #     # figure 종료
    #     plt.close('all')  # 모든 figure 종료
    #     # 메모리 정리
    #     plt.clf()
        self.current_platform=platform.system()

class PathManager:
    def __init__(self, base_path):
        # 문자열이나 Path 객체를 받아서 Path 객체로 변환
        self.base_path = Path(base_path).resolve()
        self.os_type = platform.system()
        self.paths = {}
        
        # 기본 경로 설정
        base_paths = {
            'logs': 'logs',
            'config': 'config',
            'output': 'output',
            'data': 'data'
        }
        
        # 추가 하위 경로 설정
        sub_paths = {
            'models': ['output', 'models'],
            'report': ['output', 'report'],
            'processed': ['data', 'preprocessed']
        }
        
        # 기본 경로 생성
        self.paths = self.add_paths(base_paths)
        
        # 하위 경로 생성
        for name, [parent, subdir] in sub_paths.items():
            path = self.create_subdir(parent, subdir)
            if path:
                self.paths[f'{name}_path'] = path
        
        # config 파일 로드 및 경로 업데이트
        config_file = self.get_path('config') / 'config.yaml'
        if config_file.exists():
            config = Utils.load_nested_yaml(str(config_file))
            # 모든 경로를 문자열로 변환하여 저장
            path_dict = {k: str(v).replace('\\', '/') for k, v in self.paths.items()}
            config.update(path_dict)
            self.config = config
        else:
            self.config = {}
    
    def _normalize_path(self, path_str):
        """경로 문자열을 운영체제에 맞게 정규화"""
        return str(Path(path_str.replace('\\', '/')))
    
    def add_paths(self, paths_config):
        """경로 추가 및 생성"""
        result = {}
        for name, rel_path in paths_config.items():
            # 경로 정규화 및 결합
            full_path = (self.base_path / self._normalize_path(rel_path)).resolve()
            # 디렉토리 생성
            full_path.mkdir(parents=True, exist_ok=True)
            # 경로 저장 (항상 forward slash 사용)
            result[name] = full_path
        return result
    
    def get_path(self, name, as_str=False):
        """
        경로 가져오기
        
        Parameters:
            name: 경로 이름
            as_str: True면 문자열 반환, False면 Path 객체 반환
        """
        path = self.paths.get(name)
        if path and as_str:
            return str(path)
        return path
    
    def create_subdir(self, parent, name):
        """하위 디렉토리 생성"""
        parent_path = self.get_path(parent)
        if parent_path:
            # 경로 정규화
            normalized_name = self._normalize_path(name)
            new_path = (parent_path / normalized_name).resolve()
            new_path.mkdir(parents=True, exist_ok=True)
            return new_path
        return None
    
    def get_all_paths(self, as_str=False):
        """모든 경로 조회"""
        if as_str:
            return {k: str(v) for k, v in self.paths.items()}
        return self.paths
