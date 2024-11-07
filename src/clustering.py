from pycaret.clustering import setup, create_model, evaluate_model,plot_model, assign_model

# 데이터프레임 예제 생성
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from typing import Tuple, List
import os
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from numba import jit
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class Clustering:
    def __init__(self, config):
        self.config = config
        self.time_delay = config.get('time_delay', 3)
        self.out_path = config.get('out_path')
        self.logger = config.get('logger')
        self.random_seed = config.get('random_seed', 2024)

    # def auto_clustering(self, df):

    #     # 클러스터링 자동화 PyCaret 설정
    #     clf = setup(data=df, session_id=self.random_seed, normalize=True, 
    #                 ignore_features=['feature3'],  # 범주형 제외
    #                 imputation_type='simple')  # 결측치 대체 (기본값은 평균/중앙값 대체)

    #     #evaluate_model(kmeans)
    #     # 다양한 클러스터링 모델 비교 및 최적 모델 추천
    #     best_model = compare_models()
    #     return best_model

    def _knn_for_dbscan_elbow(self, X_scaled, n_neighbors):
        # 데이터 준비 (예시: X_scaled는 스케일링된 수치 데이터)
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors_fit = neighbors.fit(X_scaled)
        distances, indices = neighbors_fit.kneighbors(X_scaled)

        # k-거리 플롯
        distances = np.sort(distances[:, 4], axis=0)  # k=4일 경우 사용
        plt.plot(distances)
        plt.title('k-distance Graph')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        plt.show(block=False)
        title='k-distance Graph_elbow'
        plt.savefig(os.path.join(self.out_path, title +'.png'), dpi=300, bbox_inches='tight')
        plt.pause(self.time_delay)  # 5초 동안 그래프 표시
        plt.close()

    def dbscan_clustering(self, df):
        # DBSCAN 모델 생성
        dbscan_model = create_model('dbscan')

        # 모델 평가 시각화 (DBSCAN은 군집 수가 가변적임)
        plot_model(dbscan_model, plot='cluster')  # 클러스터 시각화 (2D)

        # DBSCAN 모델 할당된 데이터 보기
        clustered_df_dbscan = assign_model(dbscan_model)
        print(clustered_df_dbscan.head())

        # 계층적 클러스터링 모델 생성
        hierarchical_model = create_model('hclust')

        # 모델 시각화 (덴드로그램)
        plot_model(hierarchical_model, plot='dendrogram')  # 덴드로그램 시각화

        # 계층적 클러스터링 모델 할당된 데이터 보기
        clustered_df_hclust = assign_model(hierarchical_model)
        print(clustered_df_hclust.head())

        df['feature1'] = df['feature1'].str.replace(',', '')
        df['feature1'] = pd.to_numeric(df['feature1'], errors='coerce')


    def kmeans_clustering(self, df):
    # PyCaret 클러스터링 설정
        # session_id: 시드값으로, 재현 가능한 결과를 위해 설정
        clf = setup(data=df, session_id=self.random_seed, normalize=True)  # normalize=True 옵션은 피처 스케일링을 자동으로 수행
        # K-Means 모델 생성
        kmeans = create_model('kmeans', num_clusters=3)  # num_clusters 설정은 선택사항

        # 모델 결과 시각화 (다양한 방법 사용 가능)
        plot_model(kmeans, plot='elbow')  # Elbow plot으로 클러스터 수 결정
        plot_model(kmeans, plot='silhouette')  # Silhouette plot 시각화

        # 클러스터링 할당된 데이터 보기
        clustered_df = assign_model(kmeans)
        print(clustered_df.head())

class ClusteringAnalysis:
    def __init__(self, config: dict):
        self.config = config
        self.time_delay = config.get('time_delay', 3)
        self.logger = config.get('logger')
        self.out_path = config.get('out_path', 'output')
        self.n_jobs = min(multiprocessing.cpu_count(), 8)  # 병렬 처리 코어 수 제한
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_distances_numba(X: np.ndarray) -> np.ndarray:
        """Numba로 최적화된 거리 계산"""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                distances[i, j] = distances[j, i] = dist
        return distances

    def find_optimal_dbscan_params(self, df: pd.DataFrame, 
                                 features: List[str],
                                 min_samples_range: range = range(2, 10),
                                 n_neighbors: int = 5) -> Tuple[float, int]:
        """최적화된 DBSCAN 파라미터 탐색"""
        # 데이터 전처리 (벡터화)
        X = df[features].copy()
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 병렬 처리로 eps 값 찾기
        optimal_eps = self._find_optimal_eps_parallel(X_scaled, n_neighbors)
        
        # 병렬 처리로 min_samples 값 찾기
        optimal_min_samples = self._find_optimal_min_samples_parallel(
            X_scaled, optimal_eps, min_samples_range)
        
        self.logger.info(f"최적 파라미터 - eps: {optimal_eps:.3f}, min_samples: {optimal_min_samples}")
        
        # 결과 시각화 및 저장
        self._visualize_clustering(X_scaled, optimal_eps, optimal_min_samples)
        self._save_dbscan_params(optimal_eps, optimal_min_samples)
        
        return optimal_eps, optimal_min_samples
    
    def _find_optimal_eps_parallel(self, X_scaled: np.ndarray, n_neighbors: int) -> float:
        """병렬 처리를 사용한 최적 eps 값 탐색"""
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        neighbors_fit = neighbors.fit(X_scaled)
        distances, _ = neighbors_fit.kneighbors(X_scaled)
        
        # 거리 계산 벡터화
        distances = np.sort(distances[:, n_neighbors-1])
        
        # Elbow point 찾기
        knee_locator = KneeLocator(
            range(len(distances)), 
            distances,
            curve='convex', 
            direction='increasing'
        )
        
        return distances[knee_locator.elbow]
    
    @staticmethod
    def _parallel_dbscan(params: Tuple) -> Tuple[int, int]:
        """DBSCAN 클러스터링을 수행하는 정적 메서드"""
        min_samples, X, eps = params
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        return n_clusters, n_noise
    
    def _find_optimal_min_samples(self, X_scaled: np.ndarray, 
                                eps: float, 
                                min_samples_range: range) -> int:
        """순차적 처리를 사용한 최적 min_samples 값 탐색"""
        n_clusters_list = []
        n_noise_list = []
        
        # tqdm 추가
        for min_samples in tqdm(min_samples_range, 
                              desc="Finding optimal min_samples",
                              position=0):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=self.n_jobs)
            labels = dbscan.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            n_clusters_list.append(n_clusters)
            n_noise_list.append(n_noise)
        
        optimal_idx = np.argmax(np.diff(n_clusters_list))
        return min_samples_range[optimal_idx]
    
    def apply_dbscan_with_saved_params(self, df: pd.DataFrame, 
                                     features: List[str]) -> pd.DataFrame:
        """최적화된 DBSCAN 적용"""
        eps, min_samples = self.load_dbscan_params()
        
        # 데이터 전처리 진행률 표시
        self.logger.info("데이터 전처리 시작...")
        X = df[features].apply(pd.to_numeric, errors='coerce')
        
        # 배치 처리로 대규모 데이터 처리
        batch_size = 10000
        results = []
        
        for batch_start in tqdm(range(0, len(X), batch_size),
                              desc="Processing batches",
                              position=0):
            batch_end = min(batch_start + batch_size, len(X))
            batch = X.iloc[batch_start:batch_end]
            
            # 스케일링
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(batch)
            
            # DBSCAN 실행
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=self.n_jobs)
            labels = dbscan.fit_predict(X_scaled)
            results.extend(labels)
        
        df['cluster_id'] = results
        
        # 결과 요약
        n_clusters = len(set(results)) - (1 if -1 in results else 0)
        n_noise = results.count(-1)
        
        self.logger.info(f"클러스터 수: {n_clusters}")
        self.logger.info(f"노이즈 포인트 수: {n_noise}")
        
        return df

    def _batch_process(self, X: np.ndarray, batch_size: int = 1000):
        """대용량 데이터를 위한 배치 처리"""
        for i in range(0, len(X), batch_size):
            yield X[i:i + batch_size]

    def _batch_process(self, X: np.ndarray, batch_size: int = 1000):
        """대용량 데이터를 위한 배치 처리"""
        total_batches = len(X) // batch_size + (1 if len(X) % batch_size else 0)
        
        for i in tqdm(range(0, len(X), batch_size),
                     desc="Processing data batches",
                     total=total_batches,
                     position=0):
            yield X[i:i + batch_size]
        
        # 2D로 시각화 (처음 두 특성 사용)
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                            c=labels, cmap='viridis')
        title='DBSCAN Clustering Results'
        plt.title(f'{title}\n'
                 f'eps={eps:.3f}, min_samples={min_samples}')
        plt.colorbar(scatter)
        plt.show(block=False)
        
        plt.title(title)
        plt.savefig(os.path.join(self.out_path, title +'.png'), dpi=300, bbox_inches='tight')
        
        plt.pause(self.time_delay)  # 5초 동안 그래프 표시
        plt.close()
        
        # 클러스터링 결과 요약
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        self.logger.info(f"Number of clusters: {n_clusters}")
        self.logger.info(f"Number of noise points: {n_noise}")
    
    def _save_dbscan_params(self, eps: float, min_samples: int) -> None:
        """DBSCAN 최적 파라미터 저장"""
        params = {'eps': eps, 'min_samples': min_samples}
        with open(os.path.join(self.out_path, 'dbscan_params.json'), 'w') as f:
            json.dump(params, f)
        self.logger.info("DBSCAN parameters saved successfully.")
    
    def load_dbscan_params(self) -> Tuple[float, int]:
        """저장된 DBSCAN 파라미터 불러오기"""
        with open(os.path.join(self.out_path, 'dbscan_params.json'), 'r') as f:
            params = json.load(f)
        self.logger.info("DBSCAN parameters loaded successfully.")
        return params['eps'], params['min_samples']