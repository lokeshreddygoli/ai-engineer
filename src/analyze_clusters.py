"""
Analyze and visualize clustering results.

This script demonstrates:
1. Cluster quality and semantic coherence
2. Boundary cases (documents with ambiguous cluster membership)
3. Model uncertainty (entropy analysis)
4. Cluster-to-category mapping
"""

import numpy as np
from src.vector_store import VectorStore
from src.fuzzy_clustering import FuzzyClustering
from src.data_loader import load_newsgroups
import os


def analyze_clusters():
    """Run comprehensive cluster analysis."""
    
    print("=" * 80)
    print("SEMANTIC CLUSTERING ANALYSIS - 20 Newsgroups Dataset")
    print("=" * 80)
    
    # Load data
    documents, category_names = load_newsgroups()
    
    # Initialize vector store
    vector_store = VectorStore()
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    vector_store.add_documents(documents, doc_ids)
    
    # Fit clustering
    embeddings = np.array(vector_store.embeddings)
    clustering = FuzzyClustering(max_clusters=15)
    n_clusters = clustering.fit(embeddings)
    
    print(f"\n✓ Optimal clusters determined: {n_clusters}")
    print(f"  (Using BIC criterion to balance fit vs complexity)")
    
    # Store cluster distributions
    soft_assignments = clustering.get_soft_assignments()
    for i, doc_id in enumerate(doc_ids):
        dist = {}
        for cluster_id, prob in enumerate(soft_assignments[i]):
            if prob > 0.01:
                dist[cluster_id] = float(prob)
        vector_store.set_cluster_distribution(doc_id, dist)
    
    # Analysis 1: Cluster composition
    print("\n" + "=" * 80)
    print("CLUSTER COMPOSITION & SEMANTIC COHERENCE")
    print("=" * 80)
    
    for cluster_id in range(n_clusters):
        members = clustering.get_cluster_members(cluster_id, threshold=0.2)
        
        if not members:
            continue
        
        print(f"\n--- Cluster {cluster_id} ({len(members)} strong members) ---")
        
        # Get top categories in this cluster
        category_counts = {}
        for doc_idx, prob in members[:50]:  # Top 50 members
            category = documents[doc_idx]['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Top categories: {', '.join([f'{cat} ({count})' for cat, count in top_categories])}")
        
        # Show sample documents
        print(f"Sample documents (top 3 by probability):")
        for doc_idx, prob in members[:3]:
            text = documents[doc_idx]['text'][:100].replace('\n', ' ')
            category = documents[doc_idx]['category']
            print(f"  [{prob:.2f}] {category}: {text}...")
    
    # Analysis 2: Boundary cases (high uncertainty)
    print("\n" + "=" * 80)
    print("BOUNDARY CASES - DOCUMENTS WITH AMBIGUOUS MEMBERSHIP")
    print("=" * 80)
    print("\nThese documents belong to multiple clusters with similar probabilities.")
    print("They represent the 'fuzzy' nature of semantic categories.\n")
    
    # Find documents with high entropy (uncertain cluster membership)
    entropies = []
    for i in range(len(documents)):
        probs = soft_assignments[i]
        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append((i, entropy))
    
    # Sort by entropy (highest first)
    entropies.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 5 most ambiguous documents:\n")
    for rank, (doc_idx, entropy) in enumerate(entropies[:5], 1):
        doc = documents[doc_idx]
        probs = soft_assignments[doc_idx]
        
        # Get top 3 clusters
        top_clusters = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"{rank}. Entropy: {entropy:.2f}")
        print(f"   Category: {doc['category']}")
        print(f"   Cluster distribution: {', '.join([f'C{c}:{p:.2f}' for c, p in top_clusters])}")
        print(f"   Text: {doc['text'][:120]}...")
        print()
    
    # Analysis 3: Cluster entropy (model uncertainty per cluster)
    print("=" * 80)
    print("CLUSTER UNCERTAINTY - WHERE IS THE MODEL MOST UNCERTAIN?")
    print("=" * 80)
    print("\nHigher entropy = more ambiguous cluster boundaries\n")
    
    cluster_entropies = []
    for cluster_id in range(n_clusters):
        entropy = clustering.get_cluster_entropy(cluster_id)
        cluster_entropies.append((cluster_id, entropy))
    
    cluster_entropies.sort(key=lambda x: x[1], reverse=True)
    
    print("Cluster uncertainty ranking:")
    for cluster_id, entropy in cluster_entropies:
        members = clustering.get_cluster_members(cluster_id, threshold=0.1)
        print(f"  Cluster {cluster_id}: entropy={entropy:.2f}, members={len(members)}")
    
    # Analysis 4: Soft vs hard assignments
    print("\n" + "=" * 80)
    print("SOFT VS HARD ASSIGNMENTS - WHY FUZZY CLUSTERING MATTERS")
    print("=" * 80)
    
    # Show a document that would be misclassified with hard assignment
    print("\nExample: Document with split membership\n")
    
    # Find a document with ~50/50 split between two clusters
    for i in range(len(documents)):
        probs = soft_assignments[i]
        top_2 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:2]
        
        if top_2[0][1] > 0.3 and top_2[1][1] > 0.2 and top_2[0][1] + top_2[1][1] < 0.8:
            doc = documents[i]
            print(f"Document: {doc['text'][:150]}...")
            print(f"Original category: {doc['category']}")
            print(f"\nSoft assignment (fuzzy):")
            for cluster_id, prob in top_2:
                print(f"  Cluster {cluster_id}: {prob:.1%}")
            print(f"\nHard assignment (traditional): Cluster {top_2[0][0]}")
            print(f"  → Loses {top_2[1][1]:.1%} of information!")
            break
    
    # Analysis 5: Category-to-cluster mapping
    print("\n" + "=" * 80)
    print("CATEGORY-TO-CLUSTER MAPPING")
    print("=" * 80)
    print("\nHow do original 20 categories map to discovered clusters?\n")
    
    category_to_clusters = {}
    for i, doc in enumerate(documents):
        category = doc['category']
        if category not in category_to_clusters:
            category_to_clusters[category] = {}
        
        probs = soft_assignments[i]
        for cluster_id, prob in enumerate(probs):
            if prob > 0.1:
                if cluster_id not in category_to_clusters[category]:
                    category_to_clusters[category][cluster_id] = 0
                category_to_clusters[category][cluster_id] += 1
    
    # Show a few categories
    for category in sorted(category_to_clusters.keys())[:5]:
        clusters = category_to_clusters[category]
        top_clusters = sorted(clusters.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"{category}:")
        print(f"  Spans clusters: {', '.join([f'C{c}({n})' for c, n in top_clusters])}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print(f"""
1. Discovered {n_clusters} semantic clusters (vs 20 original categories)
   - Suggests real semantic structure is simpler than category labels
   
2. Soft assignments reveal document ambiguity
   - Many documents belong to multiple clusters
   - Hard clustering would lose this nuance
   
3. Boundary cases are most interesting
   - High-entropy documents show where categories blur
   - These are often the most semantically rich
   
4. Cluster entropy varies significantly
   - Some clusters have clear boundaries
   - Others are inherently fuzzy (expected for news data)
   
5. Cache efficiency implication
   - Cluster membership helps filter cache lookups
   - Related queries likely share cluster distribution
   - Reduces search space for large caches
""")


if __name__ == "__main__":
    analyze_clusters()
