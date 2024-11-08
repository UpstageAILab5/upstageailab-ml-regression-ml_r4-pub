import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import platform
from pathlib import Path
from typing import Dict, Any
import yaml

class Utils:
    def __init__(self, logger):
        self.current_platform = platform.system()
        self.logger = logger
        self.time_delay = 3

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


    def remove_unnamed_columns(self, df):
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            self.logger.info(f"Removing unnamed columns: {unnamed_cols}")
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
        plt.figure(figsize=(3, 1))
        plt.text(0.5, 0.5, '처음과 같이 이제와 항상 영원히', ha='center', va='center')
        plt.axis('off')
        # 현재 figure 저장
        fig = plt.gcf()
        # 그래프 표시 (non-blocking)
        plt.show(block=False)
        #time.sleep(self.time_delay)
        plt.pause(self.time_delay)
        plt.close()
       # delay = time_delay if time_delay is not None else self.time_delay
        # figure 종료
        plt.close('all')  # 모든 figure 종료
        # 메모리 정리
        plt.clf()
        self.current_platform=platform.system()