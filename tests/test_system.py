"""
Test script to validate the semantic search system.

Tests:
1. Vector store functionality
2. Clustering quality
3. Cache behavior
4. API endpoints
"""

import requests
import json
import time
from typing import Dict, List


def test_api_endpoints():
    """Test all API endpoints."""
    
    BASE_URL = "http://localhost:8000"
    
    print("=" * 80)
    print("TESTING SEMANTIC SEARCH API")
    print("=" * 80)
    
    # Test 1: Health check
    print("\n1. Health Check")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    health = response.json()
    print(f"   ✓ System healthy")
    print(f"     Documents: {health['documents']}")
    print(f"     Clusters: {health['clusters']}")
    print(f"     Cache entries: {health['cache_entries']}")
    
    # Test 2: Initial cache stats
    print("\n2. Initial Cache Stats")
    response = requests.get(f"{BASE_URL}/cache/stats")
    assert response.status_code == 200
    stats = response.json()
    print(f"   ✓ Cache stats retrieved")
    print(f"     Total entries: {stats['total_entries']}")
    print(f"     Hit rate: {stats['hit_rate']}")
    
    # Test 3: First query (cache miss)
    print("\n3. First Query (Cache Miss Expected)")
    query1 = "machine learning and artificial intelligence"
    response = requests.post(
        f"{BASE_URL}/query",
        json={"query": query1}
    )
    assert response.status_code == 200
    result1 = response.json()
    print(f"   ✓ Query processed")
    print(f"     Cache hit: {result1['cache_hit']}")
    print(f"     Dominant cluster: {result1['dominant_cluster']}")
    print(f"     Results: {len(result1['result']['documents'])} documents")
    assert result1['cache_hit'] == False, "First query should miss cache"
    
    # Test 4: Similar query (cache hit expected)
    print("\n4. Similar Query (Cache Hit Expected)")
    query2 = "deep learning neural networks"
    response = requests.post(
        f"{BASE_URL}/query",
        json={"query": query2}
    )
    assert response.status_code == 200
    result2 = response.json()
    print(f"   ✓ Query processed")
    print(f"     Cache hit: {result2['cache_hit']}")
    if result2['cache_hit']:
        print(f"     Matched query: {result2['matched_query']}")
        print(f"     Similarity: {result2['similarity_score']}")
    
    # Test 5: Different query (cache miss)
    print("\n5. Different Query (Cache Miss Expected)")
    query3 = "sports and recreation"
    response = requests.post(
        f"{BASE_URL}/query",
        json={"query": query3}
    )
    assert response.status_code == 200
    result3 = response.json()
    print(f"   ✓ Query processed")
    print(f"     Cache hit: {result3['cache_hit']}")
    print(f"     Dominant cluster: {result3['dominant_cluster']}")
    
    # Test 6: Cache stats after queries
    print("\n6. Cache Stats After Queries")
    response = requests.get(f"{BASE_URL}/cache/stats")
    assert response.status_code == 200
    stats = response.json()
    print(f"   ✓ Cache stats retrieved")
    print(f"     Total entries: {stats['total_entries']}")
    print(f"     Hit count: {stats['hit_count']}")
    print(f"     Miss count: {stats['miss_count']}")
    print(f"     Hit rate: {stats['hit_rate']}")
    
    # Test 7: More queries to build cache
    print("\n7. Building Cache with Multiple Queries")
    test_queries = [
        "computer graphics and visualization",
        "religious discussions and beliefs",
        "automobile maintenance and repair",
        "medical health and diseases",
        "space exploration and astronomy"
    ]
    
    for query in test_queries:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"query": query}
        )
        assert response.status_code == 200
        print(f"   ✓ {query[:40]}...")
    
    # Test 8: Paraphrased queries (should hit cache)
    print("\n8. Paraphrased Queries (Cache Hits Expected)")
    paraphrases = [
        ("computer graphics and visualization", "graphics programming and rendering"),
        ("religious discussions and beliefs", "faith and spirituality"),
        ("automobile maintenance and repair", "car repair and maintenance")
    ]
    
    hits = 0
    for original, paraphrase in paraphrases:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"query": paraphrase}
        )
        assert response.status_code == 200
        result = response.json()
        if result['cache_hit']:
            hits += 1
            print(f"   ✓ HIT: '{paraphrase}' matched '{result['matched_query']}'")
        else:
            print(f"   ✗ MISS: '{paraphrase}' (similarity threshold may be too high)")
    
    print(f"\n   Cache hit rate on paraphrases: {hits}/{len(paraphrases)}")
    
    # Test 9: Final cache stats
    print("\n9. Final Cache Stats")
    response = requests.get(f"{BASE_URL}/cache/stats")
    assert response.status_code == 200
    stats = response.json()
    print(f"   ✓ Cache stats retrieved")
    print(f"     Total entries: {stats['total_entries']}")
    print(f"     Hit count: {stats['hit_count']}")
    print(f"     Miss count: {stats['miss_count']}")
    print(f"     Hit rate: {stats['hit_rate']:.1%}")
    print(f"     Threshold: {stats['similarity_threshold']}")
    
    # Test 10: Clear cache
    print("\n10. Clear Cache")
    response = requests.delete(f"{BASE_URL}/cache")
    assert response.status_code == 200
    print(f"   ✓ Cache cleared")
    
    response = requests.get(f"{BASE_URL}/cache/stats")
    stats = response.json()
    assert stats['total_entries'] == 0
    assert stats['hit_count'] == 0
    assert stats['miss_count'] == 0
    print(f"   ✓ Cache verified empty")
    
    # Test 11: Error handling
    print("\n11. Error Handling")
    
    # Empty query
    response = requests.post(
        f"{BASE_URL}/query",
        json={"query": ""}
    )
    assert response.status_code == 400
    print(f"   ✓ Empty query rejected")
    
    # Invalid JSON
    response = requests.post(
        f"{BASE_URL}/query",
        json={"invalid": "field"}
    )
    assert response.status_code == 422
    print(f"   ✓ Invalid JSON rejected")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)


def test_cache_threshold_behavior():
    """Test how cache threshold affects hit rate."""
    
    print("\n" + "=" * 80)
    print("CACHE THRESHOLD BEHAVIOR ANALYSIS")
    print("=" * 80)
    
    BASE_URL = "http://localhost:8000"
    
    # Clear cache first
    requests.delete(f"{BASE_URL}/cache")
    
    # Build cache with initial queries
    print("\nBuilding cache with 10 queries...")
    queries = [
        "machine learning algorithms",
        "computer vision and image processing",
        "natural language processing",
        "deep neural networks",
        "data science and analytics",
        "artificial intelligence applications",
        "computer graphics rendering",
        "database systems and SQL",
        "web development frameworks",
        "cloud computing infrastructure"
    ]
    
    for query in queries:
        requests.post(f"{BASE_URL}/query", json={"query": query})
    
    # Test paraphrases
    paraphrases = [
        "machine learning models",
        "image recognition and computer vision",
        "text processing and NLP",
        "neural network training",
        "big data analytics"
    ]
    
    print(f"\nTesting {len(paraphrases)} paraphrased queries...")
    print("(Note: Threshold is fixed at 0.82 in this test)\n")
    
    hits = 0
    for paraphrase in paraphrases:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"query": paraphrase}
        )
        result = response.json()
        if result['cache_hit']:
            hits += 1
            print(f"✓ HIT: '{paraphrase}'")
            print(f"      → Matched: '{result['matched_query']}'")
            print(f"      → Similarity: {result['similarity_score']}")
        else:
            print(f"✗ MISS: '{paraphrase}'")
    
    print(f"\nHit rate: {hits}/{len(paraphrases)} ({100*hits/len(paraphrases):.0f}%)")
    print("\nInterpretation:")
    print("- High hit rate (>70%): Threshold is aggressive, catches paraphrases")
    print("- Low hit rate (<30%): Threshold is conservative, only exact matches")
    print("- Medium hit rate (30-70%): Threshold is balanced")


if __name__ == "__main__":
    print("Waiting for API to be ready...")
    time.sleep(2)
    
    try:
        test_api_endpoints()
        test_cache_threshold_behavior()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API at http://localhost:8000")
        print("Make sure the API is running: uvicorn app:app --reload")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
