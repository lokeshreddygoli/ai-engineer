"""
Semantic cache built from first principles.

Core insight: A cache miss is expensive (search + clustering analysis).
A cache hit on a "close enough" query is valuable.

The tunable parameter: SIMILARITY_THRESHOLD
- Controls what counts as a cache hit
- Too high (0.95+): almost never hits, cache is useless
- Too low (0.70-): hits on unrelated queries, wrong results
- Sweet spot (0.80-0.90): balances false positives vs cache utility

This threshold reveals the system's behavior:
- At 0.95: cache is conservative, only exact semantic matches
- At 0.85: cache is aggressive, catches paraphrases
- At 0.75: cache is very aggressive, catches related queries

The choice of threshold is a business decision: 
- High threshold = correctness priority (medical, legal domains)
- Low threshold = speed priority (exploratory search)

For this system, we use 0.82 as a reasonable middle ground.

Cache structure:
- List of (query_embedding, query_text, result, cluster_distribution)
- On lookup: find closest embedding above threshold
- Cluster distribution helps: if query is about cluster 3, only check
  cache entries also about cluster 3 (faster for large caches)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.82):
        """
        Initialize semantic cache.
        
        Args:
            similarity_threshold: Minimum similarity to consider a cache hit
                                 Range [0, 1]. Higher = stricter matching.
        """
        self.similarity_threshold = similarity_threshold
        self.cache_entries = []  # List of cache entries
        self.hit_count = 0
        self.miss_count = 0
        
    def lookup(self, query_embedding: np.ndarray, 
               cluster_distribution: Dict[int, float] = None) -> Optional[Tuple[str, float, Dict]]:
        """
        Look up query in cache.
        
        Args:
            query_embedding: Query vector
            cluster_distribution: Soft cluster assignment for query
            
        Returns:
            Tuple of (matched_query_text, similarity_score, result_dict) or None
        """
        if not self.cache_entries:
            return None
        
        # Normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        best_match = None
        best_similarity = -1
        
        for entry in self.cache_entries:
            cached_embedding = entry['embedding']
            cached_norm = cached_embedding / (np.linalg.norm(cached_embedding) + 1e-8)
            
            # Cosine similarity
            similarity = float(np.dot(query_norm, cached_norm))
            
            # Check if above threshold
            if similarity >= self.similarity_threshold:
                # Cluster filter: if both have cluster info, prefer same clusters
                if cluster_distribution and entry.get('cluster_distribution'):
                    # Check overlap in dominant clusters
                    query_clusters = set(cluster_distribution.keys())
                    cached_clusters = set(entry['cluster_distribution'].keys())
                    
                    # If no overlap, penalize similarity
                    if not query_clusters.intersection(cached_clusters):
                        similarity *= 0.9  # 10% penalty for different clusters
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry
        
        if best_match is not None:
            self.hit_count += 1
            return (
                best_match['query_text'],
                best_similarity,
                best_match['result']
            )
        
        self.miss_count += 1
        return None
    
    def store(self, query_text: str, query_embedding: np.ndarray, 
              result: Dict, cluster_distribution: Dict[int, float] = None):
        """
        Store query result in cache.
        
        Args:
            query_text: Original query text
            query_embedding: Query embedding vector
            result: Result to cache (dict with search results)
            cluster_distribution: Soft cluster assignment for query
        """
        entry = {
            'query_text': query_text,
            'embedding': query_embedding,
            'result': result,
            'cluster_distribution': cluster_distribution or {}
        }
        self.cache_entries.append(entry)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        
        return {
            'total_entries': len(self.cache_entries),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': round(hit_rate, 3),
            'similarity_threshold': self.similarity_threshold
        }
    
    def clear(self):
        """Clear cache and reset stats."""
        self.cache_entries = []
        self.hit_count = 0
        self.miss_count = 0
    
    def set_threshold(self, threshold: float):
        """
        Adjust similarity threshold.
        
        This is the tunable parameter that reveals system behavior.
        Changing it mid-session shows how cache hit rate responds.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be in [0, 1]")
        self.similarity_threshold = threshold
    
    def save(self, path: str):
        """Persist cache to disk."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Convert embeddings to lists for JSON serialization
        serializable_entries = []
        for entry in self.cache_entries:
            serializable_entries.append({
                'query_text': entry['query_text'],
                'embedding': entry['embedding'].tolist(),
                'result': entry['result'],
                'cluster_distribution': entry['cluster_distribution']
            })
        
        with open(f"{path}/cache.json", "w") as f:
            json.dump({
                'entries': serializable_entries,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'similarity_threshold': self.similarity_threshold
            }, f)
    
    def load(self, path: str):
        """Load cache from disk."""
        with open(f"{path}/cache.json", "r") as f:
            data = json.load(f)
        
        self.cache_entries = []
        for entry in data['entries']:
            self.cache_entries.append({
                'query_text': entry['query_text'],
                'embedding': np.array(entry['embedding']),
                'result': entry['result'],
                'cluster_distribution': entry['cluster_distribution']
            })
        
        self.hit_count = data['hit_count']
        self.miss_count = data['miss_count']
        self.similarity_threshold = data['similarity_threshold']
