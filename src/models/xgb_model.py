import xgboost as xgb
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost 모델 클래스"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_params = self._get_model_params()
        self.train_params = self._get_train_params()
        self._create_model()
        
    def _get_model_params(self) -> Dict:
        """모델 파라미터 설정"""
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 100
        }
        return {**default_params, **self.config.get('model_params', {})}
        
    def _get_train_params(self) -> Dict:
        """학습 파라미터 설정"""
        default_params = {
            'early_stopping_rounds': 10,
            'verbose': False
        }
        return {**default_params, **self.config.get('train_params', {})}
        
    def _create_model(self) -> None:
        """XGBoost 모델 생성"""
        self.model = None  # 학습 시점에 생성
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
              eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> 'XGBoostModel':
        """모델 학습"""
        dtrain = xgb.DMatrix(X, label=y)
        evals = [(dtrain, 'train')]
        
        if eval_set is not None:
            X_val, y_val = eval_set
            deval = xgb.DMatrix(X_val, label=y_val)
            evals.append((deval, 'eval'))
        
        self.model = xgb.train(
            params=self.model_params,
            dtrain=dtrain,
            evals=evals,
            **self.train_params
        )
        
        # 학습 결과 로깅
        train_metrics = self.evaluate(X, y)
        self.log_metrics(train_metrics, prefix='train')
        
        if eval_set is not None:
            eval_metrics = self.evaluate(X_val, y_val)
            self.log_metrics(eval_metrics, prefix='eval')
            
        # Feature importance 로깅
        self._log_feature_importance()
        
        return self
        
    def _log_feature_importance(self) -> None:
        """Feature importance 로깅"""
        importance = self.model.get_score(importance_type='gain')
        wandb.log({"feature_importance": wandb.plot.bar(
            wandb.Table(data=[[k, v] for k, v in importance.items()],
                       columns=["feature", "importance"])
        )})
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
        
    def _save_model(self, path: str) -> None:
        """모델 저장"""
        self.model.save_model(path)
        
    def _load_model(self, path: str) -> None:
        """모델 로드"""
        model = xgb.Booster()
        model.load_model(path)
        return model