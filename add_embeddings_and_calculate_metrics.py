#!/usr/bin/env python3
"""
Script to add embeddings to genomes in elites.json and reserves.json,
then calculate cluster quality metrics.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from speciation.embeddings import get_embedding_model
from utils.cluster_quality import calculate_cluster_quality_metrics
from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()
logger = get_logger("AddEmbeddings")


def add_embeddings_to_genomes(genomes, model, batch_size=64, logger=None):
    """
    Add embeddings to genomes that don't have them.
    
    Args:
        genomes: List of genome dictionaries
        model: EmbeddingModel instance
        batch_size: Batch size for embedding computation
        logger: Optional logger
        
    Returns:
        Tuple of (updated_genomes, success_count, failure_count)
    """
    if logger is None:
        logger = get_logger("AddEmbeddings")
    
    # Filter genomes that need embeddings
    genomes_needing_embeddings = [
        (i, g) for i, g in enumerate(genomes) 
        if "prompt_embedding" not in g or g.get("prompt_embedding") is None
    ]
    
    if not genomes_needing_embeddings:
        logger.info("All genomes already have embeddings")
        return genomes, len(genomes), 0
    
    logger.info(f"Computing embeddings for {len(genomes_needing_embeddings)} genomes...")
    
    # Extract prompts
    prompts = [g.get("prompt", "") for _, g in genomes_needing_embeddings]
    
    # Compute embeddings in batch
    try:
        embeddings = model.encode(prompts, batch_size=batch_size, show_progress=True)
    except Exception as e:
        logger.error(f"Failed to compute embeddings: {e}", exc_info=True)
        raise
    
    # Add embeddings to genomes
    success_count = 0
    failure_count = 0
    
    for (idx, genome), embedding in zip(genomes_needing_embeddings, embeddings):
        try:
            if embedding is not None and len(embedding) > 0:
                # Convert numpy array to list (JSON-compatible)
                genomes[idx]["prompt_embedding"] = embedding.tolist()
                success_count += 1
            else:
                failure_count += 1
                genome_id = genome.get("id", "unknown")
                logger.warning(f"Embedding computation returned None/empty for genome {genome_id}")
        except Exception as e:
            failure_count += 1
            genome_id = genome.get("id", "unknown")
            logger.warning(f"Failed to add embedding for genome {genome_id}: {e}")
    
    logger.info(f"Embedding computation: {success_count} success, {failure_count} failures")
    return genomes, success_count, failure_count


def process_file(file_path, model, logger):
    """Process a single JSON file to add embeddings."""
    logger.info(f"Processing {file_path}...")
    
    # Load genomes
    with open(file_path, 'r', encoding='utf-8') as f:
        genomes = json.load(f)
    
    if not genomes:
        logger.warning(f"No genomes found in {file_path}")
        return 0, 0
    
    logger.info(f"Loaded {len(genomes)} genomes from {file_path}")
    
    # Add embeddings
    start_time = time.time()
    updated_genomes, success_count, failure_count = add_embeddings_to_genomes(
        genomes, model, batch_size=64, logger=logger
    )
    elapsed = time.time() - start_time
    
    logger.info(f"Computed embeddings in {elapsed:.2f}s: {success_count} success, {failure_count} failures")
    
    # Save updated genomes
    logger.info(f"Saving updated genomes to {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(updated_genomes, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Successfully saved {len(updated_genomes)} genomes with embeddings")
    return success_count, failure_count


def main():
    """Main function."""
    outputs_dir = Path("data/outputs/20260117_1152")
    
    if not outputs_dir.exists():
        logger.error(f"Outputs directory not found: {outputs_dir}")
        return 1
    
    elites_path = outputs_dir / "elites.json"
    reserves_path = outputs_dir / "reserves.json"
    
    if not elites_path.exists():
        logger.error(f"elites.json not found: {elites_path}")
        return 1
    
    if not reserves_path.exists():
        logger.error(f"reserves.json not found: {reserves_path}")
        return 1
    
    logger.info("=" * 60)
    logger.info("Adding embeddings to genomes")
    logger.info("=" * 60)
    
    # Get embedding model
    logger.info("Loading embedding model...")
    model = get_embedding_model(model_name="all-MiniLM-L6-v2")
    logger.info("Embedding model loaded")
    
    # Process elites.json
    logger.info("\n" + "-" * 60)
    logger.info("Processing elites.json")
    logger.info("-" * 60)
    elites_success, elites_fail = process_file(elites_path, model, logger)
    
    # Process reserves.json
    logger.info("\n" + "-" * 60)
    logger.info("Processing reserves.json")
    logger.info("-" * 60)
    reserves_success, reserves_fail = process_file(reserves_path, model, logger)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Embedding Summary")
    logger.info("=" * 60)
    logger.info(f"Elites: {elites_success} success, {elites_fail} failures")
    logger.info(f"Reserves: {reserves_success} success, {reserves_fail} failures")
    logger.info(f"Total: {elites_success + reserves_success} success, {elites_fail + reserves_fail} failures")
    
    # Calculate cluster quality metrics
    logger.info("\n" + "=" * 60)
    logger.info("Calculating cluster quality metrics")
    logger.info("=" * 60)
    
    start_time = time.time()
    metrics = calculate_cluster_quality_metrics(
        outputs_path=str(outputs_dir),
        temp_path=None,  # Use elites.json and reserves.json only
        logger=logger
    )
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("Cluster Quality Metrics")
    logger.info("=" * 60)
    logger.info(f"Calculation time: {elapsed:.2f} seconds")
    logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f} (range: [-1, 1], higher is better)")
    logger.info(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f} (range: [0, ∞], lower is better)")
    logger.info(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.4f} (range: [0, ∞], higher is better)")
    logger.info(f"QD Score (multiplicative): {metrics['qd_score']:.4f}")
    logger.info(f"Pareto-Optimal QD Score: {metrics['pareto_qd_score']:.4f}")
    logger.info(f"  - Pareto-optimal species: {metrics['pareto_optimal_count']}/{metrics['num_clusters']}")
    logger.info(f"  - Pareto coverage: {metrics['pareto_coverage']:.4f} ({metrics['pareto_coverage']*100:.2f}%)")
    logger.info(f"Number of samples: {metrics['num_samples']}")
    logger.info(f"Number of clusters: {metrics['num_clusters']}")
    
    # Save metrics to a file
    metrics_path = outputs_dir / "cluster_quality_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nMetrics saved to: {metrics_path}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
