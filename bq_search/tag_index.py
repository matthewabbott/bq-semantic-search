# tag_index.py
from typing import Dict, Set, List, Optional
import logging

logger = logging.getLogger(__name__)

class TagIndex:
    """Manages an inverted index for Banished Quest post tags"""
    def __init__(self):
        self.tag_to_posts: Dict[str, Set[int]] = {}
        self.unique_tags: Set[str] = set()
    
    def build_index(self, posts: List[Dict]) -> None:
        """Build inverted index from posts"""
        logger.info("Building BQ tag index...")
        self.tag_to_posts.clear()
        self.unique_tags.clear()
        
        for idx, post in enumerate(posts):
            post_tags = post['metadata'].get('tags', [])
            self.unique_tags.update(post_tags)
            
            for tag in post_tags:
                if tag not in self.tag_to_posts:
                    self.tag_to_posts[tag] = set()
                self.tag_to_posts[tag].add(idx)
                
        logger.info(f"Found {len(self.unique_tags)} unique BQ tags")
        logger.debug(f"Tag distribution: {self._get_tag_distribution()}")

    def filter_posts(self, candidate_indices: List[int], 
                    include_tags: Optional[List[str]] = None,
                    exclude_tags: Optional[List[str]] = None) -> List[int]:
        """Filter post indices based on tag criteria"""
        logger.info(f"Starting tag filtering with {len(candidate_indices)} candidates")
        
        if not include_tags and not exclude_tags:
            logger.info("No tag filters specified, returning all candidates")
            return candidate_indices
            
        candidates = set(candidate_indices)
        
        if include_tags:
            logger.info(f"Filtering for posts with tags: {include_tags}")
            tag_sets = [self.tag_to_posts.get(tag, set()) for tag in include_tags]
            
            if not all(tag_sets):
                missing_tags = [tag for tag, s in zip(include_tags, tag_sets) if not s]
                logger.warning(f"Some include tags not found in index: {missing_tags}")
                return []
                
            matching_posts = set.intersection(*tag_sets)
            original_count = len(candidates)
            candidates &= matching_posts
            logger.info(f"After include filtering: {len(candidates)} posts (from {original_count})")
        
        if exclude_tags:
            logger.info(f"Filtering out posts with tags: {exclude_tags}")
            excluded_posts = set.union(*[self.tag_to_posts.get(tag, set()) for tag in exclude_tags])
            original_count = len(candidates)
            candidates -= excluded_posts
            logger.info(f"After exclude filtering: {len(candidates)} posts (from {original_count})")
        
        result = sorted(list(candidates))
        logger.info(f"Final tag-filtered result count: {len(result)}")
        return result

    def get_available_tags(self) -> List[str]:
        """Get sorted list of all available tags"""
        return sorted(list(self.unique_tags))
    
    def _get_tag_distribution(self) -> Dict[str, int]:
        """Get distribution of tag usage"""
        return {tag: len(posts) for tag, posts in self.tag_to_posts.items()}