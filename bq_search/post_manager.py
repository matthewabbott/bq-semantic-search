# post_manager.py
from pathlib import Path
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class PostManager:
    """Manages loading and access to Banished Quest post data"""
    def __init__(self):
        self.posts: List[Dict] = []
    
    def load_posts(self, filepath: Path) -> None:
        """Load posts from JSON file"""
        logger.info(f"Loading BQ posts from {filepath}...")
        if not filepath.exists():
            raise FileNotFoundError(f"Posts data file not found at {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            self.posts = json.load(f)
        logger.info(f"Loaded {len(self.posts)} BQ posts")
    
    def get_posts(self) -> List[Dict]:
        """Get all posts"""
        return self.posts
    
    def get_post_by_index(self, idx: int) -> Dict:
        """Get single post by index"""
        return self.posts[idx]
    
    def get_post_count(self) -> int:
        """Get total number of posts"""
        return len(self.posts)