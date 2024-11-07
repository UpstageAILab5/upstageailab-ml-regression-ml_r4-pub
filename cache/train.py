from typing import Dict, Type
import yaml
import wandb
from src.models.base_model import BaseModel
from src.preprocessing import DataPreprocessor
from src.features import FeatureEngineer

class ModelTrainer:
    """모델 학습 관리 클래스"""
    
    def __init__(self, config_path: str, model_class: Type[BaseModel]):
        self.config = self._load_config(config_path)
        self.model_class = model_class
        self.preprocessor = DataPreprocessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def prepare_data(self):
        """데이터 준비"""
        # 데이터 전처리 및 피처 엔지니어링
        data = self.preprocessor.load_data()
        data = self.preprocessor.preprocess(data)
        data = self.feature_engineer.create_features(data)
        return self.preprocessor.split_data(data)
        
    def train_model(self):
        """모델 학습 실행"""
        # wandb 초기화
        wandb.init(project=self.config.get('wandb_project', 'real-estate-price'),
                  config=self.config)
                  
        # 데이터 준비
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data()
        
        # 모델 학습
        model = self.model_class(self.config)
        model.train(X_train, y_train, eval_set=(X_val, y_val))
        
        # 테스트 평가
        test_metrics = model.evaluate(X_test, y_test)
        model.log_metrics(test_metrics, prefix='test')
        
        # 모델 저장
        model.save(self.config['model_path'])
        
        return model
        
    def hyperparameter_search(self):
        """하이퍼파라미터 탐색"""
        sweep_config = self.config.get('sweep_config', {})
        if not sweep_config:
            raise ValueError("Sweep 설정이 없습니다.")
            
        sweep_id = wandb.sweep(sweep_config, 
                             project=self.config.get('wandb_project'))
        wandb.agent(sweep_id, self.train_model, count=sweep_config.get('count', 10))
