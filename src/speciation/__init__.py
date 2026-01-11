"""
Speciation module for Dynamic Islands framework.

Implements Leader-Follower clustering with semantic embeddings for maintaining
diverse species (islands) that evolve independently.

Key features:
- Species limited to top 100 genomes by fitness
- Cluster 0 (reserves) holds all non-elite genomes (max 1000)
- Constant radius (theta_sim) for all species - no dynamic adjustment
- Cluster origin tracking (merge/natural) in speciation_state.json
- reserves.json stores Cluster 0 individuals (replaces legacy limbo.json)
"""

# Main module
from .SpeciationModule import SpeciationModule

# Configuration
from .config import SpeciationConfig

# Data structures
from .species import Individual, Species, SpeciesMode, IslandMode, generate_species_id, SpeciesIdGenerator

# Embeddings
from .embeddings import (
    EmbeddingModel, compute_and_save_embeddings, remove_embeddings_from_temp, get_embedding_model
)

# Distance functions
from .distance import (
    semantic_distance, semantic_distances_batch,
    cosine_similarity, normalize_embedding
)

# Clustering
from .leader_follower import (
    leader_follower_clustering, find_nearest_leader, update_species_leaders
)

# Cluster 0 (reserves)
from .reserves import (
    Cluster0, Cluster0Individual, should_enter_cluster0, CLUSTER_0_ID
)

# Modes
from .modes import (
    detect_stagnation, enter_explore_mode, enter_exploit_mode, enter_default_mode,
    update_island_mode, update_all_island_modes, get_mode_statistics
)

# Intra-island
from .intra_island import (
    select_parents_elite_focused, survivor_selection,
    select_parents_from_species, compute_breeding_budget
)

# Merging
from .merging import detect_merge_candidates, merge_islands, process_merges, should_merge

# Extinction
from .extinction import (
    should_extinct, detect_extinction_candidates, repopulate_from_cluster0,
    repopulate_from_global_best, process_extinctions, find_global_best
)



# Metrics
from .metrics import (
    GenerationMetrics, SpeciationMetricsTracker, compute_diversity_metrics,
    compute_solution_diversity, get_species_statistics, log_generation_summary
)

# Main entry point (similar to run_evolution)
from .SpeciationModule import (
    run_speciation,
    get_speciation_module,
    reset_speciation_module,
    get_speciation_statistics,
    update_evolution_tracker_with_speciation
)

__all__ = [
    # Main classes
    "SpeciationModule", "SpeciationConfig",
    "Individual", "Species", "SpeciesMode", "IslandMode", "generate_species_id", "SpeciesIdGenerator",
    
    # Embeddings
    "EmbeddingModel", "compute_and_save_embeddings", "remove_embeddings_from_temp", "get_embedding_model",
    
    # Distance
    "semantic_distance", "semantic_distances_batch",
    "cosine_similarity", "normalize_embedding",
    
    # Clustering
    "leader_follower_clustering", "find_nearest_leader", "update_species_leaders",
    
    # Cluster 0
    "Cluster0", "Cluster0Individual", "should_enter_cluster0", "CLUSTER_0_ID",
    
    # Modes
    "detect_stagnation", "enter_explore_mode", "enter_exploit_mode", "enter_default_mode",
    "update_island_mode", "update_all_island_modes", "get_mode_statistics",
    
    # Intra-island
    "select_parents_elite_focused", "survivor_selection",
    "select_parents_from_species", "compute_breeding_budget",
    
    # Merging
    "detect_merge_candidates", "merge_islands", "process_merges", "should_merge",
    
    # Extinction
    "should_extinct", "detect_extinction_candidates", "repopulate_from_cluster0",
    "repopulate_from_global_best", "process_extinctions", "find_global_best",
    
    # Metrics
    "GenerationMetrics", "SpeciationMetricsTracker", "compute_diversity_metrics",
    "compute_solution_diversity", "get_species_statistics", "log_generation_summary",
    
    # Main entry point
    "run_speciation",
    "get_speciation_module",
    "reset_speciation_module",
    "get_speciation_statistics",
    "update_evolution_tracker_with_speciation",
]
