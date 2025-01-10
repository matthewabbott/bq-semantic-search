from sentence_transformers import SentenceTransformer
import numpy as np
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_embeddings():
    logger.info("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    logger.info("Loading posts from JSON...")
    with open('data/banished_quest.json', 'r', encoding='utf-8') as f:
        posts = json.load(f)
    
    logger.info(f"Loaded {len(posts)} posts")
    texts = [post['text'] for post in posts]
    
    logger.info("Computing embeddings...")
    batch_size = 8
    total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size else 0)
    
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True
    )
    
    logger.info("Saving embeddings...")
    np.save('data/embeddings.npy', embeddings)
    logger.info("Complete. Embeddings saved to data/embeddings.npy")

if __name__ == "__main__":
    compute_embeddings()