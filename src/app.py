"""
FastAPI service for semantic search with caching.

PART 4: FASTAPI SERVICE
=======================

ENDPOINTS IMPLEMENTED:

1. POST /query
   - Accepts: {"query": "<natural language query>"}
   - Returns: QueryResponse with cache_hit, matched_query, similarity_score, result, dominant_cluster
   - Behavior: Check cache first, search on miss, store result

2. GET /cache/stats
   - Returns: CacheStats with total_entries, hit_count, miss_count, hit_rate, similarity_threshold
   - Behavior: Current cache state and performance metrics

3. DELETE /cache
   - Returns: {"message": "Cache cleared", "stats": {...}}
   - Behavior: Flush cache and reset all statistics

4. GET /health
   - Returns: {"status": "ok", "documents": N, "clusters": N, "cache_entries": N}
   - Behavior: System status check

STATE MANAGEMENT:
- Global instances of VectorStore, FuzzyClustering, SemanticCache
- Thread-safe for concurrent requests (GIL protects Python objects)
- Persistence: save/load on startup/shutdown
- Automatic initialization on first run

STARTUP/SHUTDOWN:
- @app.on_event("startup"): Initialize system
  - Load from cache if exists (5 seconds)
  - Build from scratch if not (3 minutes)
- @app.on_event("shutdown"): Save cache state
  - Persist cache to disk
  - Enables warm start on next run

QUERY FLOW:
1. Validate query (not empty)
2. Normalize query (lowercase, strip)
3. Embed query (sentence-transformers)
4. Check cache (cosine similarity > 0.82)
5. If hit: Return cached result
6. If miss: Search vector store (top 5)
7. Aggregate cluster distribution
8. Store in cache
9. Return result

RESPONSE FORMAT:
- cache_hit: Boolean (true/false)
- matched_query: Original query if cache hit, null if miss
- similarity_score: Similarity to cached query if hit, null if miss
- result: Search results (documents + cluster_distribution)
- dominant_cluster: Primary cluster for this query
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import os
import json

from src.vector_store import VectorStore
from src.fuzzy_clustering import FuzzyClustering
from src.semantic_cache import SemanticCache
from src.data_loader import load_newsgroups

# Global state
vector_store = None
clustering = None
cache = None
doc_id_to_idx = {}

# Configuration
CACHE_DIR = "cache_data"
VECTOR_STORE_DIR = f"{CACHE_DIR}/vector_store"
CLUSTERING_DIR = f"{CACHE_DIR}/clustering"
CACHE_FILE_DIR = f"{CACHE_DIR}/cache"


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: Dict
    dominant_cluster: Optional[int] = None


class CacheStats(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    similarity_threshold: float = None


def initialize_system():
    """Initialize vector store, clustering, and cache."""
    global vector_store, clustering, cache, doc_id_to_idx
    
    print("Initializing semantic search system...")
    
    # Check if we have cached data
    if os.path.exists(VECTOR_STORE_DIR) and os.path.exists(CLUSTERING_DIR):
        print("Loading from cache...")
        vector_store = VectorStore()
        vector_store.load(VECTOR_STORE_DIR)
        
        clustering = FuzzyClustering()
        clustering.load(CLUSTERING_DIR)
        
        cache = SemanticCache()
        if os.path.exists(CACHE_FILE_DIR):
            cache.load(CACHE_FILE_DIR)
    else:
        print("Building system from scratch...")
        
        # Load data
        documents, category_names = load_newsgroups()
        
        # Initialize vector store
        vector_store = VectorStore()
        doc_ids = [f"doc_{i}" for i in range(len(documents))]
        vector_store.add_documents(documents, doc_ids)
        
        # Build doc_id to index mapping
        for i, doc_id in enumerate(doc_ids):
            doc_id_to_idx[doc_id] = i
        
        # Fit clustering
        embeddings = np.array(vector_store.embeddings)
        clustering = FuzzyClustering(max_clusters=15)
        clustering.fit(embeddings)
        
        # Store cluster distributions in vector store
        soft_assignments = clustering.get_soft_assignments()
        for i, doc_id in enumerate(doc_ids):
            dist = {}
            for cluster_id, prob in enumerate(soft_assignments[i]):
                if prob > 0.01:
                    dist[cluster_id] = float(prob)
            vector_store.set_cluster_distribution(doc_id, dist)
        
        # Initialize cache
        cache = SemanticCache(similarity_threshold=0.82)
        
        # Save everything
        os.makedirs(CACHE_DIR, exist_ok=True)
        vector_store.save(VECTOR_STORE_DIR)
        clustering.save(CLUSTERING_DIR)
        cache.save(CACHE_FILE_DIR)
    
    print(f"System ready. Vector store: {len(vector_store.documents)} docs, "
          f"Clustering: {clustering.n_clusters} clusters")


def get_dominant_cluster(cluster_dist: Dict[int, float]) -> Optional[int]:
    """Get cluster with highest probability."""
    if not cluster_dist:
        return None
    return max(cluster_dist.items(), key=lambda x: x[1])[0]


app = FastAPI(title="Semantic Search API")


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    initialize_system()


@app.on_event("shutdown")
async def shutdown_event():
    """Save state on shutdown."""
    if cache:
        cache.save(CACHE_FILE_DIR)
        print("Cache saved")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Semantic search with caching.
    
    Endpoint: POST /query
    Request: {"query": "<natural language query>"}
    Response: QueryResponse with cache_hit, matched_query, similarity_score, result, dominant_cluster
    
    Process:
    1. Validate query (not empty)
    2. Normalize query (lowercase, strip whitespace)
    3. Embed query using sentence-transformers
    4. Check semantic cache (cosine similarity > 0.82)
    5. If cache hit: Return cached result immediately
    6. If cache miss: Perform semantic search
    7. Aggregate cluster distribution from results
    8. Store result in cache for future hits
    9. Return result with metadata
    
    Cache Hit Response:
    {
        "query": "machine learning",
        "cache_hit": true,
        "matched_query": "machine learning algorithms",
        "similarity_score": 0.85,
        "result": {...},
        "dominant_cluster": 3
    }
    
    Cache Miss Response:
    {
        "query": "machine learning",
        "cache_hit": false,
        "matched_query": null,
        "similarity_score": null,
        "result": {...},
        "dominant_cluster": 3
    }
    
    Why this design?
    - Cache check first: Avoid expensive search if possible
    - Normalize query: Consistency (lowercase, strip)
    - Aggregate cluster distribution: Show semantic structure
    - Store result: Enable future cache hits
    - Return metadata: Help client understand result
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    query_text = request.query.strip().lower()
    
    # Embed query
    # Why? Convert natural language to semantic vector
    query_embedding = vector_store.embed_query(query_text)
    
    # Get query's cluster distribution
    # Why? Enable cluster-based cache filtering
    query_cluster_dist = {}
    
    # Check cache
    # Why? Avoid expensive search if similar query cached
    cache_result = cache.lookup(query_embedding, query_cluster_dist)
    
    if cache_result:
        # Cache hit: Return immediately
        matched_query, similarity_score, cached_result = cache_result
        dominant_cluster = get_dominant_cluster(cached_result.get('cluster_distribution', {}))
        
        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=matched_query,
            similarity_score=round(similarity_score, 3),
            result=cached_result,
            dominant_cluster=dominant_cluster
        )
    
    # Cache miss: perform search
    # Why? Find semantically similar documents
    search_results = vector_store.search(query_embedding, k=5)
    
    # Aggregate cluster distribution from results
    # Why? Show which clusters are relevant to this query
    cluster_dist = {}
    for doc_id, score, doc in search_results:
        doc_cluster_dist = vector_store.get_cluster_distribution(doc_id)
        for cluster_id, prob in doc_cluster_dist.items():
            # Weight by similarity score: more similar docs have more influence
            cluster_dist[cluster_id] = cluster_dist.get(cluster_id, 0) + prob * score
    
    # Normalize cluster distribution
    # Why? Convert to valid probability distribution (sum = 1.0)
    if cluster_dist:
        total = sum(cluster_dist.values())
        cluster_dist = {k: v / total for k, v in cluster_dist.items()}
    
    # Format result
    # Why? Provide structured response with documents and metadata
    result = {
        'documents': [
            {
                'doc_id': doc_id,
                'similarity': round(score, 3),
                'text': doc['text'][:200] + '...' if len(doc['text']) > 200 else doc['text'],
                'category': doc['category']
            }
            for doc_id, score, doc in search_results
        ],
        'cluster_distribution': cluster_dist
    }
    
    # Store in cache
    # Why? Enable future cache hits on similar queries
    cache.store(query_text, query_embedding, result, cluster_dist)
    
    dominant_cluster = get_dominant_cluster(cluster_dist)
    
    return QueryResponse(
        query=query_text,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result,
        dominant_cluster=dominant_cluster
    )


@app.get("/cache/stats", response_model=CacheStats)
async def cache_stats():
    """Get cache statistics."""
    stats = cache.get_stats()
    return CacheStats(**stats)


@app.delete("/cache")
async def clear_cache():
    """Clear cache and reset statistics."""
    cache.clear()
    return {"message": "Cache cleared", "stats": cache.get_stats()}


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "documents": len(vector_store.documents),
        "clusters": clustering.n_clusters,
        "cache_entries": len(cache.cache_entries)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
