from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import wandb
from src.utils.metrics import Metrics

class BaseModel(ABC):
    """모델 추상 클래스"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.metrics = Metrics()
        
    @abstractmethod
    def _create_model(self) -> None:
        """모델 객체 생성"""
        pass
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, 
              eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> 'BaseModel':
        """모델 학습"""
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측 수행"""
        pass
        
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """모델 평가"""
        predictions = self.predict(X)
        return {
            'rmse': self.metrics.rmse(y, predictions),
            'mae': self.metrics.mae(y, predictions),
            'r2': self.metrics.r2_score(y, predictions)
        }
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = '') -> None:
        """지표 로깅"""
        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
        wandb.log(metrics)
    
    def save(self, path: str) -> None:
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        self._save_model(path)
        self.logger.info(f"모델 저장 완료: {path}")
    
    def load(self, path: str) -> None:
        """모델 로드"""
        self.model = self._load_model(path)
        self.logger.info(f"모델 로드 완료: {path}")
    
    @abstractmethod
    def _save_model(self, path: str) -> None:
        """모델 저장 구현"""
        pass
    
    @abstractmethod
    def _load_model(self, path: str) -> None:
        """모델 로드 구현"""
        pass
