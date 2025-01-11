# BQ Semantic Search 

Back-end for semantic search of posts from Banished Quest (https://suptg.thisisnotatrueending.com/archive.html?tags=Banished%20Quest)

## Technical Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** - Fast and modern web framework for building APIs
- **[Sentence Transformers](https://www.sbert.net/)** - Semantic similarity transformation using all-MiniLM-L6-v2 model
- **[PyTorch](https://pytorch.org/)** - Machine learning backend (CPU-only version)
- **[Uvicorn](https://www.uvicorn.org/)** - ASGI server for running the FastAPI application
- **[vector_search]** - Custom kNN search implementation optimized for small VPS deployments

## Project Structure

```
bq-semantic-search/
├── compute_embeddings.py                # Script to generate embedding vectors from post text
│
├── bq_search/                           # Main package containing search service
│   ├── __init__.py                     # Package initialization and version info
│   ├── api.py                          # FastAPI routes and endpoint definitions
│   ├── bq_semantic_search.py           # Core semantic search implementation using sentence transformers
│   ├── post_manager.py                 # Handles loading and accessing post data from JSON
│   └── tag_index.py                    # Maintains inverted index for efficient tag filtering
├── data/
│   ├── banished_quest.json             # Raw post data in JSON format
│   ├── embeddings.npy                  # NumPy array of computed text embeddings
│   └── search_index.pkl                # Serialized vector search index for fast lookups
├── vector_search/                      # Vector search implementation (for my little VPS)
│   └── src/vector_search/
│       ├── __init__.py                 # Vector search package initialization
│       ├── clustering.py               # Optional clustering for large-scale search optimization
│       ├── search_index.py             # Core vector similarity search implementation
│       └── search_strategies.py        # Search algorithm implementations
│   └── tests/
│       ├── __init__.py                 # Test package initialization
│       └── test_vector_search.py       # Unit tests for vector search functionality
│
├── bq-search.html                      # Web component (deployed at /var/www/html/)
└── bq-search.service                   # Systemd service file (deployed at /etc/systemd/system/)
```

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
# Install PyTorch CPU version first
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the local vector_search package
pip install -e vector_search/

# Install other dependencies
pip install --no-cache-dir fastapi uvicorn sentence-transformers
```

3. Prepare data:
```bash
mkdir -p data
# Place banished_quest.json in the data directory
python compute_embeddings.py  # Generate embeddings and index
```

4. Configure service:
```bash
# Copy service file
sudo cp bq-search.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bq-search
sudo systemctl start bq-search
```

The service will run on `http://127.0.0.1:8000`

## API Endpoints

### Search Posts

`POST /search`

Search for posts with optional tag filtering.

Request body:
```json
{
  "query": "your search query",
  "num_results": 5,
  "include_tags": ["tag1", "tag2"],  // optional
  "exclude_tags": ["tag3"]           // optional
}
```

Response:
```json
[
  {
    "text": "post content",
    "similarity": 0.8532,
    "id": 123456,
    "thread_id": 789,
    "timestamp": 1392517470,
    "author": {
      "name": "author_name",
      "trip_code": "!trip.code"
    },
    "metadata": {
      "tags": ["tag1", "tag2"],
      "has_file": true,
      "file_name": "image.jpg",
      "inbound_links": 4,
      "outbound_links": 2
    },
    "archive_url": "https://steelbea.me/banished/archive/789/#p123456"
  }
]
```

### Get Available Tags

`GET /tags`

Get list of available tags for filtering.

Response:
```json
["tag1", "tag2", "tag3", "tag4", "tag5"]
```

## Development

To run the server in development mode:
```bash
uvicorn bq_search.api:app --reload --host 127.0.0.1 --port 8000
```