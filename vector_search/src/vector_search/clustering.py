import numpy as np
from typing import List
import logging

class Cluster:
    def __init__(self, centroid: np.ndarray):
        self.centroid = centroid
        self.indices: List[int] = []
        
    def add_point(self, index: int):
        self.indices.append(index)
        
    def clear_points(self):
        self.indices = []

class ClusterManager:
    def __init__(self, n_clusters: int, batch_size: int):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
    def initialize_clusters(self, vectors: np.ndarray) -> List[Cluster]:
        """Initialize cluster centroids using k-means++ initialization."""
        idx = np.random.randint(len(vectors))
        centroids = [vectors[idx]]
        
        for _ in range(self.n_clusters - 1):
            distances = np.full(len(vectors), np.inf)
            for centroid in centroids:
                centroid_distances = np.sum((vectors - centroid) ** 2, axis=1)
                distances = np.minimum(distances, centroid_distances)
            
            probabilities = distances / distances.sum()
            idx = np.random.choice(len(vectors), p=probabilities)
            centroids.append(vectors[idx])
        
        return [Cluster(centroid) for centroid in centroids]
    
    def assign_points_to_clusters(self, vectors: np.ndarray, clusters: List[Cluster], start_idx: int, end_idx: int) -> None:
        batch = vectors[start_idx:end_idx]
        
        distances = np.zeros((len(batch), len(clusters)))
        for i, cluster in enumerate(clusters):
            distances[:, i] = np.sum((batch - cluster.centroid) ** 2, axis=1)
        
        nearest_clusters = np.argmin(distances, axis=1)
        for i, cluster_idx in enumerate(nearest_clusters):
            clusters[cluster_idx].add_point(start_idx + i)
    
    def update_centroids(self, vectors: np.ndarray, clusters: List[Cluster]) -> float:
        max_shift = 0.0
        
        for cluster in clusters:
            if not cluster.indices:
                continue
                
            new_centroid = np.mean(vectors[cluster.indices], axis=0)
            shift = np.sum((new_centroid - cluster.centroid) ** 2)
            max_shift = max(max_shift, shift)
            cluster.centroid = new_centroid
            
        return max_shift
    
    def build_clusters(self, vectors: np.ndarray) -> List[Cluster]:
        self.logger.info("Building clusters...")
        
        clusters = self.initialize_clusters(vectors)
        
        for iteration in range(100):
            for cluster in clusters:
                cluster.clear_points()
            
            for start_idx in range(0, len(vectors), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(vectors))
                self.assign_points_to_clusters(vectors, clusters, start_idx, end_idx)
            
            max_shift = self.update_centroids(vectors, clusters)
            if max_shift < 1e-4:
                break
                
        self.logger.info(f"Built {len(clusters)} clusters in {iteration + 1} iterations")
        return clusters