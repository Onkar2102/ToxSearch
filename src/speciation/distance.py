"""
distance.py

Ensemble distance computation functions for speciation.
Combines genotype (prompt embedding) and phenotype (response scores) distances.

Implements: D_ensemble = wg * D_cos(g1, g2) + wp * D_pheno(p1, p2)
where wg + wp = 1, recommended: wg = 0.7, wp = 0.3
"""

import numpy as np
from typing import Union, Optional, List

from .phenotype_distance import phenotype_distance


def semantic_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    """
    Compute cosine distance between two L2-normalized embeddings.
    
    Cosine distance is defined as: d(u,v) = 1 - cos(u,v) = 1 - u·v
    (for normalized vectors, cosine similarity = dot product)
    
    Properties:
    - Range: [0, 2] where 0 = identical, 2 = maximally different
    - 0 = same meaning, 1 = orthogonal, 2 = opposite meaning
    - Used as the primary distance metric for clustering
    
    This is the core distance function used throughout speciation:
    - Leader-Follower clustering (theta_sim threshold)
    - Island merging (theta_merge threshold)
    - Migration topology (k-nearest neighbors)
    - Silhouette computation (intra/inter-species distances)
    
    Args:
        e1: First embedding vector (L2-normalized, shape: (embedding_dim,))
        e2: Second embedding vector (L2-normalized, shape: (embedding_dim,))
    
    Returns:
        Cosine distance as float in range [0, 2]
    """
    # Verify L2-normalization
    norm_e1 = np.linalg.norm(e1)
    norm_e2 = np.linalg.norm(e2)
    if not (np.isclose(norm_e1, 1.0) and np.isclose(norm_e2, 1.0)):
        raise ValueError(f"Embeddings must be L2-normalized. Got norms: {norm_e1}, {norm_e2}")
    
    # For normalized vectors: cosine_similarity = dot product
    cosine_similarity = np.dot(e1, e2)
    # Clip to [-1, 1] to handle floating-point errors
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    # Convert similarity to distance
    return float(1.0 - cosine_similarity)


def semantic_distances_batch(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    Compute distances from a query embedding to multiple embeddings (vectorized).
    
    This is an optimized batch version of semantic_distance() that computes
    distances to multiple targets in a single matrix operation.
    Used in Leader-Follower clustering to find nearest leader efficiently.
    
    Args:
        query_embedding: Query embedding vector (L2-normalized, shape: (embedding_dim,))
        embeddings: Target embeddings (L2-normalized, shape: (num_targets, embedding_dim))
                    or single vector (embedding_dim,) which will be reshaped
    
    Returns:
        1D numpy array of distances (shape: (num_targets,)) in range [0, 2]
    """
    # Ensure 2D array for matrix multiplication
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    
    # Vectorized dot product: (num_targets, dim) @ (dim,) = (num_targets,)
    cosine_similarities = embeddings @ query_embedding
    # Clip to handle floating-point errors
    cosine_similarities = np.clip(cosine_similarities, -1.0, 1.0)
    # Convert to distances
    return 1.0 - cosine_similarities


def ensemble_distance(
    e1: np.ndarray,
    e2: np.ndarray,
    p1: Optional[np.ndarray] = None,
    p2: Optional[np.ndarray] = None,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3
) -> float:
    """
    Compute ensemble distance combining genotype (prompt embedding) and phenotype (response scores).
    
    Formula: D_ensemble = wg * D_cos(g1, g2) + wp * D_pheno(p1, p2)
    
    Where:
    - D_cos: Cosine distance between prompt embeddings (range [0, 2])
    - D_pheno: Normalized Euclidean distance between phenotype vectors (range [0, 1])
    - wg, wp: Weights such that wg + wp = 1
    
    The phenotype distance is normalized to [0, 1], so we scale it to [0, 2] to match
    the cosine distance range, or we can keep it as [0, 1] and scale cosine distance.
    
    For consistency with existing thresholds (theta_sim, theta_merge), we normalize
    both components to [0, 1] range:
    - D_cos_normalized = D_cos / 2.0  (cosine distance [0, 2] -> [0, 1])
    - D_pheno is already [0, 1]
    
    Then: D_ensemble = wg * D_cos_normalized + wp * D_pheno
    
    Args:
        e1: First prompt embedding (L2-normalized, shape: (embedding_dim,))
        e2: Second prompt embedding (L2-normalized, shape: (embedding_dim,))
        p1: First phenotype vector (8D response scores, shape: (8,)) or None
        p2: Second phenotype vector (8D response scores, shape: (8,)) or None
        w_genotype: Weight for genotype (prompt embedding) distance (default: 0.7)
        w_phenotype: Weight for phenotype (response scores) distance (default: 0.3)
    
    Returns:
        Ensemble distance in range [0, 1]
    """
    # Validate weights
    if abs(w_genotype + w_phenotype - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got w_genotype={w_genotype}, w_phenotype={w_phenotype}")
    
    # Compute genotype distance (cosine distance, range [0, 2])
    d_genotype = semantic_distance(e1, e2)
    
    # Normalize genotype distance to [0, 1]
    d_genotype_norm = d_genotype / 2.0
    
    # Compute phenotype distance (normalized Euclidean, range [0, 1])
    if p1 is not None and p2 is not None:
        d_phenotype = phenotype_distance(p1, p2)
    else:
        d_phenotype = 0.0
    
    # Ensemble distance
    d_ensemble = w_genotype * d_genotype_norm + w_phenotype * d_phenotype
    
    return float(d_ensemble)


def ensemble_distances_batch(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    query_phenotype: Optional[np.ndarray] = None,
    phenotypes: Optional[Union[np.ndarray, List[Optional[np.ndarray]]]] = None,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3
) -> np.ndarray:
    """
    Compute ensemble distances from a query to multiple targets (vectorized).
    
    Args:
        query_embedding: Query prompt embedding (L2-normalized, shape: (embedding_dim,))
        embeddings: Target embeddings (L2-normalized, shape: (num_targets, embedding_dim))
        query_phenotype: Query phenotype vector (8D, shape: (8,)) or None
        phenotypes: Target phenotypes (shape: (num_targets, 8)) or None
        w_genotype: Weight for genotype distance (default: 0.7)
        w_phenotype: Weight for phenotype distance (default: 0.3)
    
    Returns:
        1D numpy array of ensemble distances (shape: (num_targets,)) in range [0, 1]
    """
    # Validate weights
    if abs(w_genotype + w_phenotype - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got w_genotype={w_genotype}, w_phenotype={w_phenotype}")
    
    # Compute genotype distances (range [0, 2])
    d_genotype = semantic_distances_batch(query_embedding, embeddings)
    
    # Normalize to [0, 1]
    d_genotype_norm = d_genotype / 2.0
    
    # Compute phenotype distances if available
    if query_phenotype is not None and phenotypes is not None:
        # Check if we have valid phenotypes (not all None)
        valid_phenotypes = []
        valid_indices = []
        for i, p in enumerate(phenotypes):
            if p is not None:
                valid_phenotypes.append(p)
                valid_indices.append(i)
        
        if valid_phenotypes and len(valid_phenotypes) == len(phenotypes):
            # All phenotypes are valid, use batch computation
            from .phenotype_distance import phenotype_distances_batch
            phenotypes_array = np.array(valid_phenotypes)
            d_phenotype = phenotype_distances_batch(query_phenotype, phenotypes_array)
            # Ensemble distances
            d_ensemble = w_genotype * d_genotype_norm + w_phenotype * d_phenotype
            return d_ensemble
        elif valid_phenotypes:
            # Some phenotypes are valid, compute mixed
            from .phenotype_distance import phenotype_distances_batch
            phenotypes_array = np.array(valid_phenotypes)
            d_phenotype_valid = phenotype_distances_batch(query_phenotype, phenotypes_array)
            # Build full array
            d_phenotype = np.full(len(phenotypes), 0.0)
            for idx, orig_idx in enumerate(valid_indices):
                d_phenotype[orig_idx] = d_phenotype_valid[idx]
            # For invalid indices, phenotype distance is 0 (fallback to genotype-only)
            # Ensemble distances
            d_ensemble = w_genotype * d_genotype_norm + w_phenotype * d_phenotype
            return d_ensemble
    
    # If phenotypes unavailable, fall back to genotype-only distances
    return d_genotype_norm
