"""
distance.py

Semantic distance computation functions for speciation.
All distance-related calculations are centralized here.
"""

import numpy as np
from typing import Union


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
