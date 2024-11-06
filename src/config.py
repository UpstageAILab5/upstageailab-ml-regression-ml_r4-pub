
import os
from src.logger import Logger
import pprint

class Config:
    def __init__(self, root_path):
        logger = Logger()
        #root_path ='/data/ephemeral/home'
        #root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        print('#'*10 +'Project House price Prediction' +'#'*10)
        print(f'- Root path: {root_path}\n')
        data_path = os.path.join(root_path,'data')
        output_path = os.path.join(root_path, 'output')
        for path in [data_path, output_path]:
            os.makedirs(path, exist_ok=True)
        csv_files = Config.list_files(directory=data_path, ext='csv')

        self.config = {
            'logger': logger.setup_logger(),
            'target': 'target',
            'data_path': data_path,
            'output_path': output_path,
            'data': {
                'csv_files': csv_files,
                'categorical_features': [],
                'numerical_features': []
            },
            'prep': {
                'missing_thr': 0.0005,
                'unique_thr': 0.0005,
                'outlier_thr': 0.0005
            }
            
        }
        pprint.pprint(self.config, indent=4)
    @staticmethod
    def list_files(directory, ext):
        found_files = []
        ext = ext.lower()
        print(f'Listing files in {directory} with extension {ext}')
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(f'.{ext}'):
                    print(f'Found file: {file}')
                    found_files.append(os.path.join(root, file))
        print(f'Found{len(found_files)} files')
        return found_files
    