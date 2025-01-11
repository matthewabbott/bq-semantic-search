from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pickle
from typing import List, Dict
import logging
from pathlib import Path
from vector_search import SearchIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchQuery(BaseModel):
    query: str
    num_results: int = Field(default=5, ge=1, le=50)

class SearchResult(BaseModel):
    text: str
    similarity: float
    id: int
    thread_id: int
    timestamp: int
    author: Dict
    metadata: Dict

class SemanticSearch:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.posts = []
        logger.info(f"Initialized with model: {model_name}")
    
    def check_data_files(self, json_path: Path, embeddings_path: Path) -> None:
        """Verify that all required data files exist."""
        if not json_path.exists():
            raise FileNotFoundError(
                f"Posts data file not found at {json_path}. "
                "Please ensure banished_quest.json is in the data directory."
            )
        
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found at {embeddings_path}. "
                "Please run compute_embeddings.py first."
            )
    
    def load_or_create_index(self, embeddings: np.ndarray, index_path: Path) -> SearchIndex:
        """Load serialized index if it exists, otherwise create and serialize new index."""
        try:
            if index_path.exists():
                logger.info("Found serialized index. Loading...")
                with open(index_path, 'rb') as f:
                    return pickle.load(f)
            
            logger.info("No serialized index found. Creating new index...")
            dimension = embeddings.shape[1]
            logger.info(f"Initializing SearchIndex with dimension {dimension}")
            index = SearchIndex(dimension=dimension, use_squared_distance=False)
            
            logger.info("Adding embeddings to index...")
            index.add(embeddings)
            logger.info("Search index created successfully")
            
            # Serialize the index
            logger.info("Serializing index for future use...")
            with open(index_path, 'wb') as f:
                pickle.dump(index, f)
            logger.info(f"Index serialized to {index_path}")
            
            return index
            
        except Exception as e:
            logger.error(f"Error handling index: {str(e)}")
            raise
    
    def load_data(self, filepath: str):
        """Load and process data from the JSON file."""
        try:
            json_path = Path(filepath)
            data_dir = json_path.parent
            embeddings_path = data_dir / 'embeddings.npy'
            index_path = data_dir / 'search_index.pkl'
            
            # Check JSON and embeddings exist
            self.check_data_files(json_path, embeddings_path)
            
            logger.info("Loading posts from JSON...")
            with open(json_path, 'r', encoding='utf-8') as f:
                self.posts = json.load(f)
            logger.info(f"Loaded {len(self.posts)} posts")
            
            logger.info("Loading pre-computed embeddings...")
            embeddings = np.load(embeddings_path)
            
            if len(embeddings) != len(self.posts):
                raise ValueError(
                    f"Mismatch between number of embeddings ({len(embeddings)}) "
                    f"and posts ({len(self.posts)})"
                )
            
            # Load or create the search index
            self.index = self.load_or_create_index(embeddings, index_path)
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search for similar posts."""
        if not self.index:
            raise ValueError("Index not initialized. Please load data first.")

        # Encode query and search
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(query_vector, k)
        
        # Convert distances to similarities
        max_distance = np.max(distances[0]) + 1
        similarities = 1 - (distances[0] / max_distance)

        # Format results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx < 0:  # Handle invalid indices
                continue
            post = self.posts[idx]
            results.append(SearchResult(
                text=post['text'],
                similarity=float(similarity),
                id=post['id'],
                thread_id=post['thread_id'],
                timestamp=post['timestamp'],
                author=post['author'],
                metadata=post['metadata']
            ))
        
        return results

# Initialize FastAPI app
app = FastAPI(
    title="Banished Quest Semantic Search",
    description="Search through Banished Quest content using semantic similarity",
    version="1.0.0",
    root_path="/bq-search"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search engine
search_engine = SemanticSearch()

@app.on_event("startup")
async def startup_event():
    """Load data on startup."""
    try:
        data_path = Path("data/banished_quest.json")
        search_engine.load_data(str(data_path))
    except Exception as e:
        logger.error(f"Failed to load data at startup: {str(e)}")
        raise

@app.post("/search", response_model=List[SearchResult])
async def search(query: SearchQuery):
    """
    Search endpoint that returns semantically similar texts.
    """
    try:
        results = search_engine.search(query.query, query.num_results)
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)