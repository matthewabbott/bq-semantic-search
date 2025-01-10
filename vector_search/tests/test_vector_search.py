import numpy as np
import logging
import pytest
from vector_search import SearchIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_vectors(num_vectors: int, dimension: int) -> np.ndarray:
    """Some random test vectors."""
    return np.random.randn(num_vectors, dimension).astype(np.float32)

def test_basic_search_workflow():
    """
    Test both exact and approximate search with squared distances.
    """
    # Test parameters
    num_vectors = 1000
    dimension = 128
    n_clusters = 10  # Smaller number for quick testing
    
    logger.info(f"Generating {num_vectors} test vectors of dimension {dimension}")
    vectors = generate_test_vectors(num_vectors, dimension)
    
    # Initialize index
    logger.info("Initializing search index")
    index = SearchIndex(
        dimension=dimension,
        n_clusters=n_clusters,
        batch_size=100,
        rebuild_threshold=0.3,
        use_squared_distance=True
    )
    
    # Add vectors
    logger.info("Adding vectors to index")
    index.add(vectors)
    
    # Generate a test query
    logger.info("Generating test query")
    query = generate_test_vectors(1, dimension)
    
    # Test exact search (before clusters are built)
    logger.info("Testing exact search")
    distances, indices = index.search(query, k=5)
    logger.info(f"Exact search results - squared distances: {distances[0]}, indices: {indices[0]}")
    
    # Verify exact search results
    assert len(distances[0]) <= 5, "Exact search should return at most k results"
    assert len(indices[0]) <= 5, "Exact search should return at most k results"
    assert all(idx < len(vectors) for idx in indices[0]), "Exact search indices should be valid"
    assert all(d >= 0 for d in distances[0]), "Squared distances should be non-negative"
    
    # Verify distances are squared (should be larger than Euclidean distances)
    query_vec = query[0]
    for idx, dist in zip(indices[0], distances[0]):
        vec = vectors[idx]
        euclidean_dist = np.sqrt(np.sum((query_vec - vec) ** 2))
        assert dist >= euclidean_dist ** 2 * 0.95, "Distances should be squared"
    
    # Add more vectors to trigger cluster building
    logger.info("Adding more vectors to trigger clustering")
    more_vectors = generate_test_vectors(100, dimension)
    index.add(more_vectors)
    
    # Test approximate search (after clusters are built)
    logger.info("Testing approximate search")
    distances, indices = index.search(query, k=5)
    logger.info(f"Approximate search results - squared distances: {distances[0]}, indices: {indices[0]}")
    
    # Verify approximate search results
    assert len(distances[0]) <= 5, "Approximate search should return at most k results"
    assert len(indices[0]) <= 5, "Approximate search should return at most k results"
    assert all(idx < len(vectors) + len(more_vectors) for idx in indices[0]), "Approximate search indices should be valid"
    assert all(d >= 0 for d in distances[0]), "Squared distances should be non-negative"

def test_empty_index():
    """Check behavior with empty index."""
    index = SearchIndex(dimension=10)
    query = generate_test_vectors(1, 10)
    
    with pytest.raises(ValueError, match="Index is empty"):
        index.search(query)

def test_dimension_mismatch():
    """Check error handling for dimension mismatch."""
    index = SearchIndex(dimension=10)
    wrong_dim_vectors = generate_test_vectors(100, 20)
    
    with pytest.raises(ValueError, match="Expected vectors of dimension 10"):
        index.add(wrong_dim_vectors)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--log-cli-level=INFO"])