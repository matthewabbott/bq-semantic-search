import numpy as np
from typing import List, Tuple
from .clustering import Cluster

class SearchStrategy:
    def __init__(self, use_squared_distance: bool = False):
        self.use_squared_distance = use_squared_distance

    def search(self, query: np.ndarray, vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class ExactSearch(SearchStrategy):
    def search(self, query: np.ndarray, vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        query_norm = np.sum(query ** 2, axis=1, keepdims=True)
        vector_norm = np.sum(vectors ** 2, axis=1)
        dot_product = np.dot(query, vectors.T)
        
        distances = query_norm + vector_norm - 2 * dot_product
        if not self.use_squared_distance:
            distances = np.sqrt(np.maximum(distances, 0))
        
        k = min(k, len(vectors))
        indices = np.argsort(distances, axis=1)[:, :k]
        distances = np.take_along_axis(distances, indices, axis=1)
        
        return distances, indices

class ApproximateSearch(SearchStrategy):
    def __init__(self, clusters: List[Cluster], use_squared_distance: bool = False):
        super().__init__(use_squared_distance)
        self.clusters = clusters
        
    def search(self, query: np.ndarray, vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        # For cluster assignment, always use squared distances as it's more efficient
        cluster_distances = []
        for cluster in self.clusters:
            dist = np.sum((query - cluster.centroid) ** 2, axis=1)
            cluster_distances.append(dist)
        
        cluster_distances = np.stack(cluster_distances, axis=1)
        nearest_clusters = np.argsort(cluster_distances, axis=1)[:, :3]
        
        all_distances = []
        all_indices = []
        
        for query_idx in range(len(query)):
            search_indices = []
            for cluster_idx in nearest_clusters[query_idx]:
                search_indices.extend(self.clusters[cluster_idx].indices)
            
            search_indices = np.array(search_indices)
            search_vectors = vectors[search_indices]
            
            q = query[query_idx:query_idx+1]
            query_norm = np.sum(q ** 2)
            vector_norm = np.sum(search_vectors ** 2, axis=1)
            dot_product = np.dot(q, search_vectors.T)[0]
            
            distances = query_norm + vector_norm - 2 * dot_product
            if not self.use_squared_distance:
                distances = np.sqrt(np.maximum(distances, 0))
            
            k_actual = min(k, len(search_indices))
            nearest = np.argsort(distances)[:k_actual]
            
            all_distances.append(distances[nearest])
            all_indices.append(search_indices[nearest])
        
        return np.array(all_distances), np.array(all_indices)