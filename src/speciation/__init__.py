"""
Speciation module for Plan A+ Dynamic Islands framework.

Implements Leader-Follower clustering with semantic embeddings for maintaining
diverse species (islands) that evolve independently.
"""

# Main module
from .SpeciationModule import SpeciationModule

# Configuration
from .config import PlanAPlusConfig

# Data structures
from .island import Individual, Species, IslandMode, generate_species_id, SpeciesIdGenerator

# Embeddings
from .embeddings import (
    EmbeddingModel, compute_embedding, compute_embeddings_batch,
    semantic_distance, semantic_distances_batch, get_embedding_model,
    cosine_similarity, normalize_embedding
)

# Clustering
from .leader_follower import (
    leader_follower_clustering, incremental_clustering, reassign_to_species,
    find_nearest_leader, update_species_leaders
)

# Limbo
from .limbo import LimboBuffer, LimboIndividual, should_enter_limbo

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
    should_extinct, detect_extinction_candidates, repopulate_from_limbo,
    repopulate_from_global_best, process_extinctions, find_global_best
)

# Migration
from .migration import (
    compute_semantic_topology, select_migrant, perform_migration,
    process_migrations, get_topology_statistics
)

# Adaptive thresholds
from .adaptive_threshold import (
    adjust_island_radius, compute_silhouette_score, trigger_split_event,
    process_adaptive_thresholds, compute_all_silhouette_scores, get_adaptive_statistics
)

# Metrics
from .metrics import (
    GenerationMetrics, SpeciationMetricsTracker, compute_diversity_metrics,
    compute_solution_diversity, get_species_statistics, log_generation_summary
)

__all__ = [
    "SpeciationModule", "PlanAPlusConfig",
    "Individual", "Species", "IslandMode", "generate_species_id", "SpeciesIdGenerator",
    "EmbeddingModel", "compute_embedding", "compute_embeddings_batch",
    "semantic_distance", "semantic_distances_batch", "get_embedding_model",
    "cosine_similarity", "normalize_embedding",
    "leader_follower_clustering", "incremental_clustering", "reassign_to_species",
    "find_nearest_leader", "update_species_leaders",
    "LimboBuffer", "LimboIndividual", "should_enter_limbo",
    "detect_stagnation", "enter_explore_mode", "enter_exploit_mode", "enter_default_mode",
    "update_island_mode", "update_all_island_modes", "get_mode_statistics",
    "select_parents_elite_focused", "survivor_selection",
    "select_parents_from_species", "compute_breeding_budget",
    "detect_merge_candidates", "merge_islands", "process_merges", "should_merge",
    "should_extinct", "detect_extinction_candidates", "repopulate_from_limbo",
    "repopulate_from_global_best", "process_extinctions", "find_global_best",
    "compute_semantic_topology", "select_migrant", "perform_migration",
    "process_migrations", "get_topology_statistics",
    "adjust_island_radius", "compute_silhouette_score", "trigger_split_event",
    "process_adaptive_thresholds", "compute_all_silhouette_scores", "get_adaptive_statistics",
    "GenerationMetrics", "SpeciationMetricsTracker", "compute_diversity_metrics",
    "compute_solution_diversity", "get_species_statistics", "log_generation_summary",
]

