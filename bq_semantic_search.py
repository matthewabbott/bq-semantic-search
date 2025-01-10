from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import numpy as np
import json
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
        
    def load_data(self, filepath: str):
            """Load and process data from the JSON file."""
            try:
                logger.info("Loading posts from JSON...")
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.posts = json.load(f)
                logger.info(f"Loaded {len(self.posts)} posts")
                
                # Load pre-computed embeddings
                embeddings_path = Path('data/embeddings.npy')
                if not embeddings_path.exists():
                    raise FileNotFoundError(
                        "Embeddings file not found. Please run compute_embeddings.py first."
                    )
                
                logger.info("Loading pre-computed embeddings...")
                embeddings = np.load(str(embeddings_path))
                
                if len(embeddings) != len(self.posts):
                    raise ValueError(
                        f"Mismatch between number of embeddings ({len(embeddings)}) "
                        f"and posts ({len(self.posts)})"
                    )
                
                # Create search index
                logger.info("Creating search index...")
                dimension = embeddings.shape[1]
                logger.info(f"Initializing SearchIndex with dimension {dimension}...")
                self.index = SearchIndex(dimension=dimension)
                logger.info("Adding embeddings to index...")
                self.index.add(embeddings)
                logger.info("Search index created successfully")
                
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
        # Note: [vector_search] returns actual distances, not square distances
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
    version="1.0.0"
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
    data_path = Path("data/banished_quest.json")
    if data_path.exists():
        search_engine.load_data(str(data_path))
    else:
        logger.warning("No data file found at startup")

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