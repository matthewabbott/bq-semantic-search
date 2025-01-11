from sentence_transformers import SentenceTransformer
from vector_search import SearchIndex
import numpy as np
from pathlib import Path
import pickle
import logging
from typing import List, Dict, Tuple, Optional

from .post_manager import PostManager
from .tag_index import TagIndex

logger = logging.getLogger(__name__)

class SemanticSearch:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.post_manager = PostManager()
        self.tag_index = TagIndex()
        logger.info(f"Initialized BQ search with model: {model_name}")

    def load_data(self, data_dir: Path) -> None:
        """
        Load all necessary data from the data directory
        
        Args:
            data_dir: Path to the data directory containing necessary files
        """
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at {data_dir}")
        
        embeddings_path = data_dir / 'embeddings.npy'
        index_path = data_dir / 'search_index.pkl'
        json_path = data_dir / 'banished_quest.json'
        
        # Load posts
        self.post_manager.load_posts(json_path)
        
        # Build tag index
        self.tag_index.build_index(self.post_manager.get_posts())
        
        # Load embeddings and create search index
        self._load_or_create_index(embeddings_path, index_path)
        
        logger.info(f"Successfully loaded all data from {data_dir}")

    def _load_or_create_index(self, embeddings_path: Path, index_path: Path) -> None:
        """Handle loading/creating of search index"""
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found at {embeddings_path}. "
                "Please run compute_embeddings.py first."
            )
        
        embeddings = np.load(embeddings_path)
        if len(embeddings) != len(self.post_manager.get_posts()):
            raise ValueError(
                f"Mismatch between number of embeddings ({len(embeddings)}) "
                f"and posts ({len(self.post_manager.get_posts())})"
            )
        
        self.index = self._initialize_index(embeddings, index_path)

    def _initialize_index(self, embeddings: np.ndarray, index_path: Path) -> SearchIndex:
        """Initialize search index"""
        try:
            if index_path.exists():
                logger.info("Loading existing search index...")
                with open(index_path, 'rb') as f:
                    return pickle.load(f)
            
            logger.info("Creating new BQ search index...")
            index = SearchIndex(dimension=embeddings.shape[1], use_squared_distance=False)
            index.add(embeddings)
            
            with open(index_path, 'wb') as f:
                pickle.dump(index, f)
            
            return index
            
        except Exception as e:
            logger.error(f"Error initializing BQ search index: {str(e)}")
            raise

    def search(self, query: str, k: int = 5, include_tags: Optional[List[str]] = None,
              exclude_tags: Optional[List[str]] = None) -> List[Dict]:
        """Perform semantic search with optional tag filtering"""
        if not self.index:
            raise ValueError("Index not initialized. Please load data first.")

        logger.info(f"Starting search with query: '{query}', k={k}")
        if include_tags:
            logger.info(f"Include tags: {include_tags}")
        if exclude_tags:
            logger.info(f"Exclude tags: {exclude_tags}")

        # First filter by tags
        if include_tags or exclude_tags:
            all_indices = list(range(len(self.post_manager.get_posts())))
            logger.info(f"Total posts available: {len(all_indices)}")
            
            filtered_indices = self.tag_index.filter_posts(
                all_indices, 
                include_tags, 
                exclude_tags
            )
            
            logger.info(f"Posts after tag filtering: {len(filtered_indices)}")
            
            if not filtered_indices:
                logger.info("No posts match the tag criteria, returning empty results")
                return []
            
            # Convert filtered indices to set for faster lookup
            filtered_set = set(filtered_indices)
            
            # Do semantic search
            logger.info("Encoding search query...")
            query_vector = self.model.encode([query])
            
            logger.info(f"Performing semantic search with expanded k={len(filtered_indices) + k}")
            distances, indices = self.index.search(query_vector, len(filtered_indices) + k)
            
            # Filter and format results
            filtered_results = []
            filtered_distances = []
            
            # Convert NumPy arrays to Python lists for easier handling
            indices_list = indices[0].tolist()
            distances_list = distances[0].tolist()
            
            logger.info("Filtering semantic results by tag criteria...")
            for idx, distance in zip(indices_list, distances_list):
                if idx in filtered_set:
                    filtered_results.append(idx)
                    filtered_distances.append(distance)
                    if len(filtered_results) >= k:
                        break
            
            if not filtered_results:
                logger.info("No results passed both semantic search and tag filters")
                return []
            
            logger.info(f"Final number of results after all filtering: {len(filtered_results)}")
            return self._format_results(filtered_results, np.array(filtered_distances))
        else:
            logger.info("No tag filtering requested, performing basic semantic search")
            query_vector = self.model.encode([query])
            distances, indices = self.index.search(query_vector, k)
            return self._format_results(indices[0].tolist(), distances[0])

    def _format_results(self, indices: List[int], distances: np.ndarray) -> List[Dict]:
        """Format search results into response structure"""
        logger.info("Formatting search results")
        
        if not indices:
            logger.info("No indices provided, returning empty results")
            return []
            
        if isinstance(distances, np.ndarray):
            logger.debug("Converting distances from numpy array to list")
            distances = distances.tolist()
        
        max_distance = max(distances) + 1
        results = []
        
        for idx, distance in zip(indices, distances):
            post = self.post_manager.get_post_by_index(idx)
            similarity = 1 - (distance / max_distance)
            
            results.append({
                'text': post['text'][:100] + "...",  # Log just the start of the text
                'similarity': float(similarity),
                'id': post['id'],
                'thread_id': post['thread_id'],
                'timestamp': post['timestamp'],
                'author': post['author'],
                'metadata': post['metadata'],
                'archive_url': f"https://steelbea.me/banished/archive/{post['thread_id']}/#p{post['id']}"
            })
        
        logger.info(f"Formatted {len(results)} results")
        logger.debug(f"Result IDs: {[r['id'] for r in results]}")
        logger.debug(f"Similarities: {[r['similarity'] for r in results]}")
        
        return results

    def get_available_tags(self) -> List[str]:
        """Get list of all available tags"""
        return self.tag_index.get_available_tags()