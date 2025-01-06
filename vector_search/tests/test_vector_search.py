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
    Workflow of exact and approximate search.
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
        rebuild_threshold=0.3
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
    logger.info(f"Exact search results - distances: {distances[0]}, indices: {indices[0]}")
    
    # Verify exact search results
    assert len(distances[0]) <= 5, "Exact search should return at most k results"
    assert len(indices[0]) <= 5, "Exact search should return at most k results"
    assert all(idx < len(vectors) for idx in indices[0]), "Exact search indices should be valid"
    
    # Add more vectors to trigger cluster building
    logger.info("Adding more vectors to trigger clustering")
    more_vectors = generate_test_vectors(100, dimension)
    index.add(more_vectors)
    
    # Test approximate search (after clusters are built)
    logger.info("Testing approximate search")
    distances, indices = index.search(query, k=5)
    logger.info(f"Approximate search results - distances: {distances[0]}, indices: {indices[0]}")
    
    # Verify approximate search results
    assert len(distances[0]) <= 5, "Approximate search should return at most k results"
    assert len(indices[0]) <= 5, "Approximate search should return at most k results"
    assert all(idx < len(vectors) + len(more_vectors) for idx in indices[0]), "Approximate search indices should be valid"

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
    # Note: python -m pytest tests/ -v --log-cli-level=INFO
    pytest.main([__file__, "-v"])
    
"""
from vector_search directory:

- Run all tests
pytest tests/

- Run with verbose output
pytest tests/ -v

- Run a specific test file
pytest tests/test_vector_search.py

- Run a specific test function
pytest tests/test_vector_search.py::test_basic_search_workflow -v
"""