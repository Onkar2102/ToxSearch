"""
cluster_quality.py

Post-hoc cluster quality metrics for RQ2: Cluster Quality analysis.
Calculates Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.

These metrics measure how well the speciation (clustering) is performing:
- Silhouette Score: Measures how similar genomes are to their own species vs other species
- Davies-Bouldin Index: Ratio of within-species distances to between-species distances (lower is better)
- Calinski-Harabasz Index: Ratio of between-species variance to within-species variance (higher is better)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from utils import get_custom_logging, get_system_utils

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _ = get_system_utils()


def calculate_silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
    logger=None
) -> float:
    """
    Calculate Silhouette Score for cluster quality.
    
    Silhouette Score measures how similar an object is to its own cluster compared to other clusters.
    Score range: [-1, 1] where:
    - 1 = sample is well matched to its cluster
    - 0 = sample is on or very close to the decision boundary between clusters
    - -1 = sample might have been assigned to the wrong cluster
    
    For speciation quality:
    - High score (>0.5): Species are well-separated and internally cohesive
    - Low score (<0.25): Species overlap or have poor internal cohesion
    
    Args:
        embeddings: 2D array of embeddings (N, D)
        labels: 1D array of species IDs (N,)
        logger: Optional logger instance
        
    Returns:
        Silhouette score (float) or 0.0 if calculation fails
    """
    _logger = logger or get_logger("ClusterQuality")
    
    try:
        from sklearn.metrics import silhouette_score
        
        # Need at least 2 samples per cluster and at least 2 clusters
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            _logger.warning("Need at least 2 clusters for silhouette score")
            return 0.0
        
        # Filter out clusters with only 1 sample
        valid_mask = np.zeros(len(labels), dtype=bool)
        for label in unique_labels:
            count = np.sum(labels == label)
            if count >= 2:
                valid_mask |= (labels == label)
        
        if np.sum(valid_mask) < 4:  # Need at least 4 samples total
            _logger.warning("Not enough valid samples for silhouette score")
            return 0.0
        
        filtered_embeddings = embeddings[valid_mask]
        filtered_labels = labels[valid_mask]
        
        # Recalculate unique labels after filtering
        unique_filtered = np.unique(filtered_labels)
        if len(unique_filtered) < 2:
            _logger.warning("Not enough clusters after filtering for silhouette score")
            return 0.0
        
        score = silhouette_score(filtered_embeddings, filtered_labels, metric='cosine')
        return float(score)
        
    except ImportError:
        _logger.warning("sklearn not available for silhouette score calculation")
        return 0.0
    except Exception as e:
        _logger.warning(f"Failed to calculate silhouette score: {e}")
        return 0.0


def calculate_davies_bouldin_index(
    embeddings: np.ndarray,
    labels: np.ndarray,
    logger=None
) -> float:
    """
    Calculate Davies-Bouldin Index for cluster quality.
    
    The Davies-Bouldin Index measures the average similarity between clusters,
    where similarity compares the size of clusters to distances between cluster centers.
    
    Score: >= 0 where lower values indicate better clustering.
    
    For speciation quality:
    - Low score (<1.0): Species are compact and well-separated
    - High score (>2.0): Species overlap significantly
    
    Args:
        embeddings: 2D array of embeddings (N, D)
        labels: 1D array of species IDs (N,)
        logger: Optional logger instance
        
    Returns:
        Davies-Bouldin Index (float) or -1.0 if calculation fails
    """
    _logger = logger or get_logger("ClusterQuality")
    
    try:
        from sklearn.metrics import davies_bouldin_score
        
        # Need at least 2 clusters
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            _logger.warning("Need at least 2 clusters for Davies-Bouldin index")
            return -1.0
        
        score = davies_bouldin_score(embeddings, labels)
        return float(score)
        
    except ImportError:
        _logger.warning("sklearn not available for Davies-Bouldin calculation")
        return -1.0
    except Exception as e:
        _logger.warning(f"Failed to calculate Davies-Bouldin index: {e}")
        return -1.0


def calculate_calinski_harabasz_index(
    embeddings: np.ndarray,
    labels: np.ndarray,
    logger=None
) -> float:
    """
    Calculate Calinski-Harabasz Index (Variance Ratio Criterion) for cluster quality.
    
    The Calinski-Harabasz Index is the ratio of between-cluster dispersion to
    within-cluster dispersion. Higher values indicate better-defined clusters.
    
    Score: >= 0 where higher values indicate better clustering.
    
    For speciation quality:
    - High score: Species are well-separated with high inter-species variance
    - Low score: Species are mixed or have high intra-species variance
    
    Args:
        embeddings: 2D array of embeddings (N, D)
        labels: 1D array of species IDs (N,)
        logger: Optional logger instance
        
    Returns:
        Calinski-Harabasz Index (float) or -1.0 if calculation fails
    """
    _logger = logger or get_logger("ClusterQuality")
    
    try:
        from sklearn.metrics import calinski_harabasz_score
        
        # Need at least 2 clusters
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            _logger.warning("Need at least 2 clusters for Calinski-Harabasz index")
            return -1.0
        
        score = calinski_harabasz_score(embeddings, labels)
        return float(score)
        
    except ImportError:
        _logger.warning("sklearn not available for Calinski-Harabasz calculation")
        return -1.0
    except Exception as e:
        _logger.warning(f"Failed to calculate Calinski-Harabasz index: {e}")
        return -1.0


def calculate_cluster_quality_metrics(
    outputs_path: Optional[str] = None,
    logger=None
) -> Dict[str, Any]:
    """
    Calculate all cluster quality metrics from the current population.
    
    Reads elites.json and reserves.json to extract embeddings and species assignments,
    then calculates quality metrics.
    
    Args:
        outputs_path: Path to outputs directory (defaults to get_outputs_path())
        logger: Optional logger instance
        
    Returns:
        Dictionary with cluster quality metrics:
        - silhouette_score: Silhouette Score [-1, 1], higher is better
        - davies_bouldin_index: Davies-Bouldin Index >= 0, lower is better
        - calinski_harabasz_index: Calinski-Harabasz Index >= 0, higher is better
        - num_samples: Number of samples used
        - num_clusters: Number of clusters/species
    """
    _logger = logger or get_logger("ClusterQuality")
    
    metrics = {
        "silhouette_score": 0.0,
        "davies_bouldin_index": -1.0,
        "calinski_harabasz_index": -1.0,
        "num_samples": 0,
        "num_clusters": 0
    }
    
    try:
        if outputs_path is None:
            outputs_path = str(get_outputs_path())
        
        outputs_dir = Path(outputs_path)
        elites_path = outputs_dir / "elites.json"
        reserves_path = outputs_dir / "reserves.json"
        
        # Load genomes
        all_genomes = []
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                all_genomes.extend(json.load(f))
        if reserves_path.exists():
            with open(reserves_path, 'r', encoding='utf-8') as f:
                all_genomes.extend(json.load(f))
        
        if not all_genomes:
            _logger.warning("No genomes found for cluster quality calculation")
            return metrics
        
        # Extract embeddings and labels
        embeddings_list = []
        labels_list = []
        
        for genome in all_genomes:
            embedding = genome.get("prompt_embedding")
            species_id = genome.get("species_id")
            
            if embedding is not None and species_id is not None:
                embeddings_list.append(embedding)
                labels_list.append(species_id)
        
        if len(embeddings_list) < 4:
            _logger.warning(f"Not enough genomes with embeddings ({len(embeddings_list)}) for cluster quality")
            return metrics
        
        embeddings = np.array(embeddings_list)
        labels = np.array(labels_list)
        
        metrics["num_samples"] = len(labels)
        metrics["num_clusters"] = len(np.unique(labels))
        
        # Calculate metrics
        metrics["silhouette_score"] = round(calculate_silhouette_score(embeddings, labels, _logger), 4)
        metrics["davies_bouldin_index"] = round(calculate_davies_bouldin_index(embeddings, labels, _logger), 4)
        metrics["calinski_harabasz_index"] = round(calculate_calinski_harabasz_index(embeddings, labels, _logger), 4)
        
        _logger.info(
            f"Cluster quality metrics: silhouette={metrics['silhouette_score']:.4f}, "
            f"davies_bouldin={metrics['davies_bouldin_index']:.4f}, "
            f"calinski_harabasz={metrics['calinski_harabasz_index']:.4f}, "
            f"samples={metrics['num_samples']}, clusters={metrics['num_clusters']}"
        )
        
        return metrics
        
    except Exception as e:
        _logger.error(f"Failed to calculate cluster quality metrics: {e}", exc_info=True)
        return metrics


def save_cluster_quality_to_tracker(
    outputs_path: Optional[str] = None,
    logger=None
) -> bool:
    """
    Calculate and save cluster quality metrics to EvolutionTracker.json.
    
    This is a post-hoc metric calculation that can be run after evolution completes.
    
    Args:
        outputs_path: Path to outputs directory (defaults to get_outputs_path())
        logger: Optional logger instance
        
    Returns:
        True if successful, False otherwise
    """
    _logger = logger or get_logger("ClusterQuality")
    
    try:
        if outputs_path is None:
            outputs_path = str(get_outputs_path())
        
        outputs_dir = Path(outputs_path)
        tracker_path = outputs_dir / "EvolutionTracker.json"
        
        if not tracker_path.exists():
            _logger.warning("EvolutionTracker.json not found")
            return False
        
        # Calculate metrics
        metrics = calculate_cluster_quality_metrics(outputs_path, _logger)
        
        # Load and update tracker
        with open(tracker_path, 'r', encoding='utf-8') as f:
            tracker = json.load(f)
        
        tracker["cluster_quality"] = metrics
        
        with open(tracker_path, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, indent=2, ensure_ascii=False)
        
        _logger.info("Saved cluster quality metrics to EvolutionTracker.json")
        return True
        
    except Exception as e:
        _logger.error(f"Failed to save cluster quality metrics: {e}", exc_info=True)
        return False


__all__ = [
    "calculate_silhouette_score",
    "calculate_davies_bouldin_index",
    "calculate_calinski_harabasz_index",
    "calculate_cluster_quality_metrics",
    "save_cluster_quality_to_tracker"
]
