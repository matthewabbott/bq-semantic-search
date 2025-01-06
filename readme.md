
# BQ Semantic Search 

Back-end for searching posts from Banished Quest (https://suptg.thisisnotatrueending.com/archive.html?tags=Banished%20Quest)

## Technical Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** (0.115.6) - Crutch for making a quick REST app
- **[Sentence Transformers](https://www.sbert.net/)** (3.3.1) - Semantic similarity transformation magic (all-MiniLM-L6-v2 model for sentence embeddings)
- **[PyTorch](https://pytorch.org/)** (2.5.1) - Machine learning magic back-end (CPU-only version).
- **[Uvicorn](https://www.uvicorn.org/)** (0.34.0) - The server functionality that actually runs the REST app and handles web requests.
- **[vector_search]** - custom implementation of kNN search to use with all-MiniLM-L6-v2 on my itty bitty VPS

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
pip install --no-cache-dir fastapi uvicorn faiss-cpu sentence-transformers
```

3. Prepare your data:
```bash
mkdir -p data
# Currently this is just hardcoded to read from banished_quest.json the data directory
```

4. Run the server:
```bash
python app.py
```

The server will start at `http://localhost:8000`

## API Usage

### Search Endpoint

`POST /search`

Request body:
```json
{
  "query": "your search query",
  "num_results": 5
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
    }
  }
]
```
