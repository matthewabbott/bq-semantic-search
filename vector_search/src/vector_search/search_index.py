import numpy as np
from typing import Tuple
import logging
from .clustering import Cluster, ClusterManager
from .search_strategies import SearchStrategy, ExactSearch, ApproximateSearch

class SearchIndex:
    def __init__(self, dimension: int, n_clusters: int = 100, batch_size: int = 1000, 
                 rebuild_threshold: float = 0.3, use_squared_distance: bool = False):
        self.dimension = dimension
        self.cluster_manager = ClusterManager(n_clusters, batch_size)
        self.rebuild_threshold = rebuild_threshold
        self.use_squared_distance = use_squared_distance
        self.vectors = None
        self.clusters = []
        self.last_clustered_size = 0
        self.logger = logging.getLogger(__name__)
        
    def add(self, vectors: np.ndarray) -> None:
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")
        
        vectors = vectors.astype(np.float32)
            
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            
        current_size = len(self.vectors)
        should_rebuild = (
            current_size >= self.cluster_manager.n_clusters and
            (self.last_clustered_size == 0 or
            abs(current_size - self.last_clustered_size) / self.last_clustered_size >= self.rebuild_threshold)
        )
            
        if should_rebuild:
            self.clusters = self.cluster_manager.build_clusters(self.vectors)
            self.last_clustered_size = current_size
            
        self.logger.info(f"Index now contains {current_size} vectors")
        
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.vectors is None:
            raise ValueError("Index is empty")
            
        if query.shape[1] != self.dimension:
            raise ValueError(f"Expected query vectors of dimension {self.dimension}, got {query.shape[1]}")
            
        query = query.astype(np.float32)
        
        strategy: SearchStrategy
        if not self.clusters:
            strategy = ExactSearch(use_squared_distance=self.use_squared_distance)
        else:
            strategy = ApproximateSearch(self.clusters, use_squared_distance=self.use_squared_distance)
            
        return strategy.search(query, self.vectors, k)
    
    def __len__(self) -> int:
        return 0 if self.vectors is None else len(self.vectors)
        
    @property
    def is_empty(self) -> bool:
        return self.vectors is None or len(self.vectors) == 0
        
    @property
    def memory_usage(self) -> int:
        vector_memory = 0 if self.vectors is None else self.vectors.nbytes
        cluster_memory = sum(
            c.centroid.nbytes + len(c.indices) * 4
            for c in self.clusters
        )
        return vector_memory + cluster_memory