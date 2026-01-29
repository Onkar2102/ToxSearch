"""
phenotype_distance.py

Phenotype distance computation functions for speciation.
Computes distance between phenotype vectors (response scores).
"""

import numpy as np
from typing import Optional

from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()


# Standard order of phenotype scores (8 dimensions)
PHENOTYPE_SCORE_ORDER = [
    'toxicity',
    'threat',
    'profanity',
    'sexually_explicit',
    'identity_attack',
    'flirtation',
    'insult',
    'severe_toxicity'
]



def extract_phenotype_vector(genome: dict, logger=None) -> Optional[np.ndarray]:
    """
    Extract phenotype vector (8D response scores) from genome.
    
    The phenotype vector consists of 8 toxicity-related scores from
    moderation_result.google.scores in a fixed order.
    
    Args:
        genome: Genome dictionary containing moderation_result
        
    Returns:
        8D numpy array of phenotype scores, or None if scores not available
    """

    if logger is None:
        logger = get_logger("PhenotypeDistance")

    if not genome or "moderation_result" not in genome:
        return None
    
    moderation_result = genome.get("moderation_result")
    if not moderation_result or "google" not in moderation_result:
        return None
    
    google_scores = moderation_result.get("google", {})
    if not isinstance(google_scores, dict) or "scores" not in google_scores:
        return None
    
    scores = google_scores.get("scores", {})
    if not isinstance(scores, dict):
        return None
    
    # Extract scores in fixed order, defaulting to 0.0 if missing
    phenotype = np.array([
        float(scores.get(score_name, 0.0))
        for score_name in PHENOTYPE_SCORE_ORDER
    ], dtype=np.float32)
    
    # Validate range
    if not np.all((phenotype >= 0.0) & (phenotype <= 1.0)):
        invalid_indices = np.where((phenotype < 0.0) | (phenotype > 1.0))[0]
        logger.warning(f"Phenotype scores out of [0,1] range: indices {invalid_indices}")
        # Clip to valid range
        phenotype = np.clip(phenotype, 0.0, 1.0)
    
    return phenotype


def phenotype_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute normalized Euclidean distance between two phenotype vectors.
    
    Normalized to range [0, 1] by dividing by maximum possible distance.
    Maximum distance occurs when one vector is all zeros and the other is all ones:
    max_distance = sqrt(8 * 1²) = √8 ≈ 2.828
    
    Formula:
        D_pheno = ||p1 - p2||_2 / √8
    
    Args:
        p1: First phenotype vector (8D, shape: (8,))
        p2: Second phenotype vector (8D, shape: (8,))
    
    Returns:
        Normalized phenotype distance in range [0, 1]
    """
    if p1 is None or p2 is None:
        return 1.0  # Maximum distance if phenotype unavailable
    
    # Ensure numpy arrays
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    
    # Euclidean distance
    diff = p1 - p2
    euclidean_dist = np.linalg.norm(diff)
    
    # Normalize to [0, 1] range
    # Maximum possible distance: sqrt(8 * 1²) = √8 ≈ 2.828
    # For vectors in [0,1]^D, max Euclidean distance = √D
    max_distance = np.sqrt(len(p1))
    normalized_dist = min(euclidean_dist / max_distance, 1.0)
    
    return float(normalized_dist)


def phenotype_distances_batch(
    query_phenotype: np.ndarray,
    phenotypes: np.ndarray
) -> np.ndarray:
    """
    Compute phenotype distances from a query phenotype to multiple phenotypes (vectorized).
    
    Args:
        query_phenotype: Query phenotype vector (8D, shape: (8,))
        phenotypes: Target phenotypes (shape: (num_targets, 8))
                    or single vector (8,) which will be reshaped
    
    Returns:
        1D numpy array of normalized distances (shape: (num_targets,)) in range [0, 1]
    """
    if query_phenotype is None:
        # If query has no phenotype, return maximum distance for all
        if phenotypes.ndim == 1:
            return np.array([1.0])
        return np.ones(len(phenotypes))
    
    # Ensure 2D array
    if phenotypes.ndim == 1:
        phenotypes = phenotypes.reshape(1, -1)
    
    # Vectorized Euclidean distance computation
    # ||query - target||_2 for each target
    diff = phenotypes - query_phenotype
    euclidean_dists = np.linalg.norm(diff, axis=1)
    
    # Normalize to [0, 1]
    # Maximum possible distance: sqrt(D * 1²) = √D
    # For vectors in [0,1]^D, max Euclidean distance = √D
    max_distance = np.sqrt(query_phenotype.shape[0])
    normalized_dists = np.clip(euclidean_dists / max_distance, 0.0, 1.0)
    
    return normalized_dists


__all__ = [
    "extract_phenotype_vector",
    "phenotype_distance",
    "phenotype_distances_batch",
    "PHENOTYPE_SCORE_ORDER"
]
