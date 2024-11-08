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
        self.n_jobs = min(multiprocessing.cpu_count(), 8)

    def find_optimal_dbscan_params(self, df: pd.DataFrame, 
                                 features: List[str],
                                 min_samples_range: range = range(2, 10),
                                 n_neighbors: int = 5) -> Tuple[float, int]:
        """최적화된 DBSCAN 파라미터 탐색"""
        self.logger.info("DBSCAN 파라미터 최적화 시작...")
        
        # 데이터 전처리
        self.logger.info("데이터 전처리 중...")
        X = df[features].copy()
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # 스케일링
        self.logger.info("데이터 스케일링 중...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # eps 값 찾기
        self.logger.info("최적 eps 값 탐색 중...")
        optimal_eps = self._find_optimal_eps(X_scaled, n_neighbors)
        
        # min_samples 값 찾기
        self.logger.info("최적 min_samples 값 탐색 중...")
        optimal_min_samples = self._find_optimal_min_samples(
            X_scaled, optimal_eps, min_samples_range)
        
        self.logger.info(f"최적 파라미터 - eps: {optimal_eps:.3f}, min_samples: {optimal_min_samples}")
        
        # 결과 저장
        self._save_dbscan_params(optimal_eps, optimal_min_samples)
        
        # 결과 시각화
        self._visualize_clustering(X_scaled, optimal_eps, optimal_min_samples)
        
        return optimal_eps, optimal_min_samples

    def _find_optimal_eps(self, X_scaled: np.ndarray, n_neighbors: int) -> float:
        """최적 eps 값 탐색"""
        self.logger.info("KNN fit 진행 중...")
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        neighbors_fit = neighbors.fit(X_scaled)
        
        self.logger.info("거리 계산 중...")
        distances, _ = neighbors_fit.kneighbors(X_scaled)
        distances = np.sort(distances[:, n_neighbors-1])
        
        self.logger.info("Elbow point 탐색 중...")
        knee_locator = KneeLocator(
            range(len(distances)), 
            distances,
            curve='convex', 
            direction='increasing'
        )
        
        return distances[knee_locator.elbow]

    def _parallel_dbscan(self, X_scaled: np.ndarray, min_samples: int, eps: float) -> dict:
        """DBSCAN을 실행하는 독립적인 함수"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=self.n_jobs)
        labels = dbscan.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        return {
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }

    def _find_optimal_min_samples(self, X_scaled: np.ndarray, 
                                eps: float, 
                                min_samples_range: range) -> int:
        """병렬 처리로 최적 min_samples 값 탐색"""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for min_samples in min_samples_range:
                future = executor.submit(
                    self._parallel_dbscan,
                    X_scaled,
                    min_samples,
                    eps
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), 
                             total=len(futures),
                             desc="Finding optimal min_samples"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"병렬 처리 중 오류 발생: {str(e)}")
        
        # 결과 정렬 및 최적값 찾기
        results.sort(key=lambda x: x['min_samples'])
        n_clusters_list = [r['n_clusters'] for r in results]
        optimal_idx = np.argmax(np.diff(n_clusters_list))
        
        return min_samples_range[optimal_idx]

    def _visualize_clustering(self, X_scaled: np.ndarray, 
                            eps: float, 
                            min_samples: int) -> None:
        """클러스터링 결과 시각화"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=self.n_jobs)
        labels = dbscan.fit_predict(X_scaled)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                            c=labels, cmap='viridis')
        plt.title(f'DBSCAN Clustering Results\n'
                 f'eps={eps:.3f}, min_samples={min_samples}')
        plt.colorbar(scatter)
        
        # 저장 및 표시
        plt.savefig(os.path.join(self.out_path, 'dbscan_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(self.time_delay)
        plt.close()

    def apply_dbscan_with_saved_params(self, df: pd.DataFrame, 
                                     features: List[str]) -> pd.DataFrame:
        """저장된 파라미터로 DBSCAN 적용"""
        eps, min_samples = self.load_dbscan_params()
        
        self.logger.info("데이터 전처리 시작...")
        X = df[features].apply(pd.to_numeric, errors='coerce')
        
        # 배치 처리
        batch_size = 10000
        results = []
        
        for batch_start in tqdm(range(0, len(X), batch_size),
                              desc="Processing DBSCAN batches"):
            batch_end = min(batch_start + batch_size, len(X))
            batch = X.iloc[batch_start:batch_end]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(batch)
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=self.n_jobs)
            labels = dbscan.fit_predict(X_scaled)
            results.extend(labels)
        
        df['cluster_id'] = results
        
        n_clusters = len(set(results)) - (1 if -1 in results else 0)
        n_noise = results.count(-1)
        
        self.logger.info(f"클러스터 수: {n_clusters}")
        self.logger.info(f"노이즈 포인트 수: {n_noise}")
        
        return df

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