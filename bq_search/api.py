# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from pathlib import Path

from .bq_semantic_search import SemanticSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchQuery(BaseModel):
    query: str
    num_results: int = Field(default=5, ge=1, le=50)
    include_tags: Optional[List[str]] = None
    exclude_tags: Optional[List[str]] = None

class SearchResponse(BaseModel):
    text: str
    similarity: float
    id: int
    thread_id: int
    timestamp: int
    author: dict
    metadata: dict
    archive_url: str

app = FastAPI(
    title="Banished Quest Semantic Search",
    description="Search through Banished Quest content using semantic similarity",
    version="1.0.0",
    root_path="/bq-search"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_engine = SemanticSearch()

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    try:
        # Get the project root (from bq-semantic-search/bq-search/api.py)
        project_root = Path(__file__).parent.parent
        data_dir = project_root / 'data'
        
        logger.info(f"Loading data from {data_dir}")
        search_engine.load_data(data_dir)
        logger.info("BQ Search engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to load BQ data at startup: {str(e)}")
        raise

@app.post("/search", response_model=List[SearchResponse])
async def search(query: SearchQuery):
    """Search endpoint with tag filtering support"""
    try:
        results = search_engine.search(
            query.query,
            query.num_results,
            query.include_tags,
            query.exclude_tags
        )
        return results
    except Exception as e:
        logger.error(f"BQ Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tags", response_model=List[str])
async def get_tags():
    """Get all available tags"""
    try:
        return search_engine.get_available_tags()
    except Exception as e:
        logger.error(f"Error fetching BQ tags: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)