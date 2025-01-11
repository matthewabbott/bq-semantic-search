bq-semantic-search/
├── compute_embeddings.py
├── bq_search/
│   ├── __init__.py
│   ├── api.py
│   ├── bq_semantic_search.py
│   ├── post_manager.py
│   └── tag_index.py
├── data/
│   ├── banished_quest.json
│   ├── embeddings.npy
│   └── search_index.pkl
├── vector_search/
│	│   └── src/vector_search/
│	│		├── __init__.py
│	│   	├── clustering.py
│	│   	├── search_index.py
│	│   	└── search_strategies.py
│	└── tests/
│		├── __init__.py
│   	└── test_vector_search.py
│ 
├── bq-search.html (at /var/www/html/bq-search.html)
└── bq-search.service (at /etc/systemd/system/bq-search.service)