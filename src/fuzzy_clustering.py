"""
Fuzzy clustering using Gaussian Mixture Models.

PART 2: FUZZY CLUSTERING
========================

PROBLEM: Hard Clustering is Wrong for News Data
===============================================

Example: Document about gun legislation
- Hard clustering: Assign to EITHER politics OR firearms
- Reality: Belongs to BOTH with different probabilities
  - Politics: 60% (legislative process, policy)
  - Firearms: 40% (gun rights, regulations)
- Hard clustering loses 40% of information

Solution: Soft Clustering with Probability Distributions
- Each document gets a distribution over clusters
- Captures ambiguity and overlapping topics
- Preserves semantic nuance

ALGORITHM: Gaussian Mixture Models (GMM)
========================================

Why GMM?
- Probabilistic: Each document has probability distribution over clusters
- Soft assignments: predict_proba() gives responsibilities
- Interpretable: Each cluster is a Gaussian distribution
- Scalable: EM algorithm is efficient
- Flexible: Full covariance allows different cluster shapes

Why not K-Means?
- Hard assignments only (each point to one cluster)
- No probabilistic interpretation
- Doesn't capture document ambiguity

Why not Hierarchical Clustering?
- Produces dendrograms (tree structure)
- Doesn't give probability distributions
- Harder to interpret for cache filtering

Why not DBSCAN?
- Density-based clustering
- Hard assignments
- Doesn't handle overlapping clusters well

CLUSTER COUNT SELECTION: BIC Criterion
======================================

Problem: How many clusters should we have?
- Too few: Lose semantic structure
- Too many: Overfit, noise becomes signal
- No single "right" answer

Solution: Bayesian Information Criterion (BIC)
- Balances model fit vs complexity
- Formula: BIC = -2*log(L) + k*log(n)
  - L = likelihood (fit)
  - k = number of parameters (complexity)
  - n = number of observations
- Penalizes adding clusters
- Theoretically justified (Bayesian model selection)

Why BIC?
- Automatic (no manual tuning)
- Principled (Bayesian foundation)
- Prevents overfitting
- Standard in statistics

Alternatives considered:
- Elbow method: Subjective, requires visual inspection
- Silhouette score: Measures separation, not probabilistic
- Gap statistic: More complex, similar results
- Fixed number: Arbitrary, doesn't adapt to data

COVARIANCE TYPE: Full
====================

Options:
- spherical: All clusters are spheres (too restrictive)
- tied: All clusters share same covariance (too restrictive)
- diag: Diagonal covariance (assumes independence)
- full: Each cluster has its own covariance matrix (most flexible)

Why full?
- News documents have complex semantic structure
- Different clusters have different shapes/sizes
- Full covariance captures this without overfitting
- BIC prevents overfitting (automatic regularization)

Trade-off: Slightly slower fitting, but more accurate clusters

SOFT ASSIGNMENTS: Responsibilities
==================================

What are responsibilities?
- For each document, probability of belonging to each cluster
- Sum to 1.0 across all clusters
- Computed via EM algorithm (Expectation-Maximization)

Example:
- Document about gun legislation
- Cluster 0 (politics): 0.60
- Cluster 1 (firearms): 0.40
- Cluster 2 (law): 0.00
- Sum: 1.00

Why responsibilities?
- Capture document ambiguity
- Enable cluster-based filtering (cache optimization)
- Reveal semantic structure
- Interpretable (probabilities)

ENTROPY: Measuring Uncertainty
==============================

What is entropy?
- Shannon entropy: -sum(p * log(p))
- Measures uncertainty in probability distribution
- High entropy: Many clusters with similar probability
- Low entropy: One dominant cluster

Why entropy?
- Identifies boundary cases (high entropy = ambiguous)
- Reveals where model is uncertain
- Shows cluster boundary ambiguity
- Helps understand semantic structure

Example:
- Document with entropy 2.0: Highly ambiguous (multiple clusters)
- Document with entropy 0.1: Clear assignment (one dominant cluster)

INITIALIZATION: K-Means++
=========================

Why k-means++ initialization?
- Starts with good initial cluster centers
- Improves convergence speed
- Reduces chance of local optima
- Standard practice for GMM

Alternative: random initialization
- Slower convergence
- More likely to get stuck in local optima
- Less stable

STANDARDIZATION: StandardScaler
===============================

Why standardize embeddings?
- GMM assumes Gaussian distributions
- Standardization improves numerical stability
- Ensures all dimensions have equal weight
- Standard practice for probabilistic models

Process:
- Subtract mean: (x - mean)
- Divide by std: (x - mean) / std
- Result: Mean 0, std 1

Design rationale:
- GMM provides soft cluster assignments (probability distributions)
- Each document gets a distribution over clusters, not a hard label
- Number of clusters: determined by BIC (Bayesian Information Criterion)
  BIC balances model fit vs complexity, prevents overfitting
- Initialization: k-means++ for stability
- Covariance type: 'full' allows clusters to have different shapes/sizes

The key insight: a document about gun legislation belongs to multiple clusters
with different probabilities. This is captured in the covariance matrix.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import json


class FuzzyClustering:
    def __init__(self, max_clusters: int = 15):
        """
        Initialize fuzzy clustering.
        
        Args:
            max_clusters: Maximum number of clusters to evaluate
        """
        self.max_clusters = max_clusters
        self.gmm = None
        self.scaler = StandardScaler()
        self.n_clusters = None
        self.embeddings = None
        
    def fit(self, embeddings: np.ndarray) -> int:
        """
        Fit GMM and determine optimal number of clusters using BIC.
        
        Args:
            embeddings: Document embeddings (n_docs, embedding_dim)
            
        Returns:
            Optimal number of clusters
            
        Process:
        1. Standardize embeddings (mean 0, std 1)
        2. Evaluate BIC for 2-15 clusters
        3. Select cluster count with minimum BIC
        4. Refit with optimal clusters
        
        Why BIC?
        - Balances model fit vs complexity
        - Prevents overfitting (penalizes extra clusters)
        - Theoretically justified (Bayesian model selection)
        - Automatic (no manual tuning)
        
        Why 2-15 clusters?
        - Minimum 2: Need at least 2 clusters
        - Maximum 15: Reasonable upper bound for news data
        - Typical result: 8-12 clusters (vs 20 original categories)
        
        Why k-means++ initialization?
        - Good starting point for EM algorithm
        - Faster convergence
        - More stable results
        - Standard practice
        
        Why n_init=10?
        - Run EM 10 times with different initializations
        - Take best result (highest likelihood)
        - Reduces chance of local optima
        - Ensures stable clustering
        """
        self.embeddings = embeddings
        
        # Standardize embeddings for GMM
        # Why? GMM assumes Gaussian distributions, standardization improves stability
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Evaluate BIC for different cluster counts
        # Why? Find optimal balance between fit and complexity
        bic_scores = []
        for n_clusters in range(2, self.max_clusters + 1):
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type='full',  # Each cluster has its own covariance
                init_params='kmeans',    # k-means++ initialization
                n_init=10,               # Try 10 initializations, take best
                random_state=42          # Reproducibility
            )
            gmm.fit(embeddings_scaled)
            bic_scores.append(gmm.bic(embeddings_scaled))
        
        # Find elbow: optimal cluster count
        # Why argmin? Lowest BIC = best balance of fit vs complexity
        self.n_clusters = np.argmin(bic_scores) + 2
        print(f"Optimal clusters by BIC: {self.n_clusters}")
        print(f"BIC scores: {bic_scores}")
        
        # Refit with optimal clusters
        # Why refit? Use best initialization for final model
        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type='full',
            init_params='kmeans',
            n_init=10,
            random_state=42
        )
        self.gmm.fit(embeddings_scaled)
        
        return self.n_clusters
    
    def get_soft_assignments(self) -> np.ndarray:
        """
        Get soft cluster assignments (responsibilities).
        
        Returns:
            Array of shape (n_docs, n_clusters) with probabilities
            
        What are responsibilities?
        - For each document, probability of belonging to each cluster
        - Sum to 1.0 across all clusters
        - Computed via EM algorithm (Expectation-Maximization)
        
        Example:
        - Document about gun legislation
        - Cluster 0 (politics): 0.60
        - Cluster 1 (firearms): 0.40
        - Cluster 2 (law): 0.00
        - Sum: 1.00
        
        Why responsibilities?
        - Capture document ambiguity
        - Enable cluster-based filtering (cache optimization)
        - Reveal semantic structure
        - Interpretable (probabilities)
        
        Why standardize before transform?
        - GMM was fit on standardized data
        - Must use same standardization for consistency
        - Ensures correct probability computation
        """
        if self.gmm is None:
            raise ValueError("Model not fitted yet")
        
        embeddings_scaled = self.scaler.transform(self.embeddings)
        return self.gmm.predict_proba(embeddings_scaled)
    
    def get_document_distribution(self, doc_idx: int) -> Dict[int, float]:
        """
        Get cluster distribution for a single document.
        
        Args:
            doc_idx: Document index
            
        Returns:
            Dict mapping cluster_id -> probability
        """
        responsibilities = self.get_soft_assignments()
        dist = {}
        for cluster_id, prob in enumerate(responsibilities[doc_idx]):
            if prob > 0.01:  # Only include clusters with >1% probability
                dist[cluster_id] = float(prob)
        return dist
    
    def get_cluster_members(self, cluster_id: int, 
                           threshold: float = 0.1) -> List[Tuple[int, float]]:
        """
        Get documents belonging to a cluster above threshold.
        
        Args:
            cluster_id: Cluster ID
            threshold: Minimum probability to include
            
        Returns:
            List of (doc_idx, probability) tuples, sorted by probability
        """
        responsibilities = self.get_soft_assignments()
        members = []
        for doc_idx, probs in enumerate(responsibilities):
            if probs[cluster_id] >= threshold:
                members.append((doc_idx, float(probs[cluster_id])))
        
        return sorted(members, key=lambda x: x[1], reverse=True)
    
    def get_cluster_entropy(self, cluster_id: int) -> float:
        """
        Measure uncertainty in cluster assignments.
        Higher entropy = more ambiguous cluster boundaries.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Shannon entropy of cluster membership probabilities
            
        What is entropy?
        - Shannon entropy: -sum(p * log(p))
        - Measures uncertainty in probability distribution
        - High entropy: Many clusters with similar probability
        - Low entropy: One dominant cluster
        
        Why entropy?
        - Identifies boundary cases (high entropy = ambiguous)
        - Reveals where model is uncertain
        - Shows cluster boundary ambiguity
        - Helps understand semantic structure
        
        Example:
        - Document with entropy 2.0: Highly ambiguous
        - Document with entropy 0.1: Clear assignment
        
        Interpretation:
        - Entropy 0: Perfect certainty (one cluster = 1.0)
        - Entropy 1-2: Moderate ambiguity (multiple clusters)
        - Entropy 3+: High ambiguity (many equal clusters)
        """
        responsibilities = self.get_soft_assignments()
        probs = responsibilities[:, cluster_id]
        
        # Shannon entropy: -sum(p * log(p))
        # Why log? Standard information theory measure
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)
    
    def save(self, path: str):
        """Save clustering model."""
        import pickle
        import os
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/gmm.pkl", "wb") as f:
            pickle.dump(self.gmm, f)
        with open(f"{path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(f"{path}/metadata.json", "w") as f:
            json.dump({
                'n_clusters': self.n_clusters,
                'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else None
            }, f)
    
    def load(self, path: str):
        """Load clustering model."""
        import pickle
        with open(f"{path}/gmm.pkl", "rb") as f:
            self.gmm = pickle.load(f)
        with open(f"{path}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(f"{path}/metadata.json", "r") as f:
            metadata = json.load(f)
            self.n_clusters = metadata['n_clusters']
