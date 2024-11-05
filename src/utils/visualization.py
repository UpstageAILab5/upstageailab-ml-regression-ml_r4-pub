import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, auc
import wandb

class Visualizer:
    """모델 학습 및 평가 결과 시각화 클래스"""
    
    def __init__(self, save_dir: str = 'visualizations'):
        """
        Parameters
        ----------
        save_dir : str
            시각화 결과물 저장 디렉토리
        """
        self.save_dir = save_dir
        self._setup_style()
        
    def _setup_style(self):
        """시각화 스타일 설정"""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_training_history(self,
                            history: Dict[str, List[float]],
                            metrics: List[str] = ['loss', 'rmse'],
                            save_path: Optional[str] = None) -> None:
        """학습 히스토리 시각화
        
        Parameters
        ----------
        history : Dict[str, List[float]]
            학습 히스토리 (e.g., {'loss': [...], 'val_loss': [...], ...})
        metrics : List[str]
            시각화할 메트릭 리스트
        save_path : Optional[str]
            저장 경로
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
            
        for ax, metric in zip(axes, metrics):
            train_metric = history[metric]
            val_metric = history.get(f'val_{metric}', None)
            
            epochs = range(1, len(train_metric) + 1)
            ax.plot(epochs, train_metric, 'b-', label=f'Train {metric}')
            if val_metric:
                ax.plot(epochs, val_metric, 'r-', label=f'Validation {metric}')
                
            ax.set_title(f'{metric.capitalize()} vs. Epochs')
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            wandb.log({f"training_history": wandb.Image(plt)})
            
        plt.close()
        
    def plot_confusion_matrix(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            labels: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> None:
        """혼동 행렬 시각화
        
        Parameters
        ----------
        y_true : np.ndarray
            실제 레이블
        y_pred : np.ndarray
            예측 레이블
        labels : Optional[List[str]]
            클래스 레이블
        save_path : Optional[str]
            저장 경로
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_path:
            plt.savefig(save_path)
            wandb.log({f"confusion_matrix": wandb.Image(plt)})
            
        plt.close()
        
    def plot_feature_importance(self,
                              importance: Dict[str, float],
                              title: str = 'Feature Importance',
                              save_path: Optional[str] = None) -> None:
        """특성 중요도 시각화
        
        Parameters
        ----------
        importance : Dict[str, float]
            특성 중요도 딕셔너리
        title : str
            그래프 제목
        save_path : Optional[str]
            저장 경로
        """
        plt.figure(figsize=(12, 6))
        
        # 중요도 기준 정렬
        sorted_idx = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features, values = zip(*sorted_idx)
        
        sns.barplot(x=list(values), y=list(features))
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        
        if save_path:
            plt.savefig(save_path)
            wandb.log({f"feature_importance": wandb.Image(plt)})
            
        plt.close()
        
    def plot_regression_results(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              save_path: Optional[str] = None) -> None:
        """회귀 결과 시각화
        
        Parameters
        ----------
        y_true : np.ndarray
            실제값
        y_pred : np.ndarray
            예측값
        save_path : Optional[str]
            저장 경로
        """
        plt.figure(figsize=(10, 6))
        
        # 산점도
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # 이상적인 예측선
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.title('Predicted vs Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        if save_path:
            plt.savefig(save_path)
            wandb.log({f"regression_results": wandb.Image(plt)})
            
        plt.close()
        
    def plot_residuals(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      save_path: Optional[str] = None) -> None:
        """잔차 분석 시각화
        
        Parameters
        ----------
        y_true : np.ndarray
            실제값
        y_pred : np.ndarray
            예측값
        save_path : Optional[str]
            저장 경로
        """
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 잔차 vs 예측값
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        
        # 잔차 분포
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_title('Residuals Distribution')
        ax2.set_xlabel('Residuals')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            wandb.log({f"residuals_analysis": wandb.Image(plt)})
            
        plt.close()
        
    def plot_learning_curves(self,
                           train_sizes: np.ndarray,
                           train_scores: np.ndarray,
                           val_scores: np.ndarray,
                           save_path: Optional[str] = None) -> None:
        """학습 곡선 시각화
        
        Parameters
        ----------
        train_sizes : np.ndarray
            훈련 데이터 크기
        train_scores : np.ndarray
            훈련 점수
        val_scores : np.ndarray
            검증 점수
        save_path : Optional[str]
            저장 경로
        """
        plt.figure(figsize=(10, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, val_mean, label='Cross-validation score')
        
        plt.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.1)
        
        plt.title('Learning Curves')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            wandb.log({f"learning_curves": wandb.Image(plt)})
            
        plt.close()
