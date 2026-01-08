"""
config.py

Configuration parameters for Dynamic Islands speciation framework.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SpeciationConfig:
    """
    Configuration parameters for speciation framework.
    
    This dataclass centralizes all hyperparameters for the Dynamic Islands framework,
    allowing easy tuning and experimentation. All parameters have sensible defaults.
    
    Attributes:
        # Clustering Parameters
        theta_sim: Similarity threshold for species assignment (semantic distance).
                  Individuals within this distance of a leader become followers.
                  Range: [0, 2] where 0 = identical, 2 = maximally different.
                  Default: 0.4 (moderate similarity required)
                  Also used as the constant radius for all species.
        
        theta_merge: Merge threshold for combining similar islands.
                     Species with leader distance < theta_merge are candidates for merging.
                     Must be < theta_sim to prevent premature merging.
                     Default: 0.2 (tighter than theta_sim)
        
        # Cluster 0 Parameters
        viability_baseline: Minimum fitness required to enter cluster 0.
                           High-fitness outliers that don't fit existing species
                           are preserved here for potential speciation.
                           Range: [0, 1]
                           Default: 0.3 (moderate fitness threshold)
        
        cluster0_ttl: Time-to-live in generations for cluster 0 individuals.
                      After this many generations without speciation, individuals expire.
                      Prevents cluster 0 from growing unbounded.
                      Default: 10 generations
        
        cluster0_min_cluster_size: Minimum cluster size required for cluster 0 speciation.
                                   When cluster 0 individuals form a cohesive cluster of this size,
                                   they can create a new species.
                                   Default: 2 (minimum viable species)
        
        cluster0_max_capacity: Maximum individuals in cluster 0.
                               When exceeded, lowest-fitness individuals are removed.
                               Default: 1000 individuals
        
        removal_threshold: Fitness threshold for removing individuals from cluster 0.
                          Individuals with fitness below this are archived to under_performing.json.
                          If None, no threshold filtering is applied.
                          Default: None (set dynamically based on population)
        
        # Island Management Parameters
        max_island_capacity: Maximum individuals per species (keeps top by fitness).
                             When exceeded, lowest-fitness members are sent to cluster 0.
                             Default: 100 individuals
        
        min_island_size: Minimum island size before extinction.
                         Islands smaller than this are considered extinct.
                         Default: 2 (minimum viable population)
        
        stagnation_window: Number of generations to look back for stagnation detection.
                            Used in mode switching logic.
                            Default: 5 generations
        
        max_stagnation: Maximum generations without improvement before extinction.
                        Islands that stagnate beyond this threshold are extinguished.
                        Default: 20 generations
        
        # Mode Switching Parameters
        explore_mutation_multiplier: Mutation rate multiplier in EXPLORE mode.
                                      Higher values increase exploration diversity.
                                      Default: 2.0 (double mutation rate)
        
        exploit_mutation_multiplier: Mutation rate multiplier in EXPLOIT mode.
                                      Lower values focus on exploitation.
                                      Default: 0.5 (half mutation rate)
        
        explore_selection_pressure: Selection pressure in EXPLORE mode.
                                    Lower values allow more diversity in parent selection.
                                    Range: [0, 1]
                                    Default: 0.7 (moderate pressure)
        
        improvement_slope_threshold: Fitness slope threshold to trigger EXPLOIT mode.
                                      When fitness trend slope > this, switch to EXPLOIT.
                                      Default: 0.01 (positive improvement trend)
        
        decline_slope_threshold: Fitness slope threshold to trigger EXPLORE mode.
                                 When fitness trend slope < this, switch to EXPLORE.
                                 Default: -0.001 (slight decline)
        
        # Migration Parameters
        migration_frequency: Perform migration every N generations.
                             Migration happens periodically, not every generation.
                             Default: 5 (every 5 generations)
        
        k_neighbors: Number of nearest neighbors for migration topology.
                     Each island can migrate to its k nearest semantic neighbors.
                     Default: 3 neighbors
        
        migration_selection: Method for selecting migrants.
                            Options: "most_unique", "random", "best"
                            Default: "most_unique" (maximize diversity transfer)
        
        # Adaptive Threshold Parameters
        silhouette_threshold: Trigger split if silhouette score below this.
                              Low silhouette indicates poor cluster cohesion.
                              Range: [-1, 1], typically [0, 1]
                              Default: 0.5 (moderate cohesion)
        
        silhouette_check_frequency: Check silhouette scores every N generations.
                                    Silhouette computation is expensive, so done periodically.
                                    Default: 10 (every 10 generations)
        
        # Embedding Parameters
        embedding_model: Sentence-transformer model name for prompt embeddings.
                          Model must be compatible with sentence-transformers library.
                          Default: "all-MiniLM-L6-v2" (384-dim, fast, high quality)
        
        embedding_dim: Expected embedding dimensionality.
                        Should match the chosen model's output dimension.
                        Default: 384 (for all-MiniLM-L6-v2)
        
        embedding_batch_size: Batch size for embedding computation.
                               Larger batches are faster but use more memory.
                               Default: 64 prompts per batch
        
        # Repopulation Parameters
        repopulation_mutation_rate: Mutation rate for creating new individuals during repopulation.
                                     When islands are extinguished, new ones are created via mutation.
                                     Range: [0, 1]
                                     Default: 0.3 (moderate mutation)
        
        repopulation_size: Initial size of repopulated islands.
                           New islands created from cluster 0 or global best start with this many members.
                           Default: 20 individuals
    """
    
    # Clustering
    theta_sim: float = 0.4
    theta_merge: float = 0.2
    
    # Cluster 0
    viability_baseline: float = 0.3
    cluster0_ttl: int = 10
    cluster0_min_cluster_size: int = 2
    cluster0_max_capacity: int = 1000
    removal_threshold: Optional[float] = None
    
    # Island Management
    max_island_capacity: int = 100
    min_island_size: int = 2
    stagnation_window: int = 5
    max_stagnation: int = 20
    
    # Mode Switching
    explore_mutation_multiplier: float = 2.0
    exploit_mutation_multiplier: float = 0.5
    explore_selection_pressure: float = 0.7
    improvement_slope_threshold: float = 0.01
    decline_slope_threshold: float = -0.001
    
    # Migration
    migration_frequency: int = 5
    k_neighbors: int = 3
    migration_selection: str = "most_unique"
    
    # Adaptive Thresholds (splitting only, no radius adjustment)
    silhouette_threshold: float = 0.5
    silhouette_check_frequency: int = 10
    
    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 64
    
    # Repopulation
    repopulation_mutation_rate: float = 0.3
    repopulation_size: int = 20
    
    def __post_init__(self):
        """
        Validate configuration parameters after initialization.
        
        Ensures all parameters are within valid ranges and relationships
        between parameters are correct (e.g., theta_merge < theta_sim).
        Raises AssertionError if validation fails.
        """
        # Validate clustering thresholds
        assert 0 <= self.theta_sim <= 2, f"theta_sim must be in [0, 2]"
        assert 0 <= self.theta_merge <= 2, f"theta_merge must be in [0, 2]"
        # Merge threshold must be tighter than similarity threshold
        assert self.theta_merge < self.theta_sim, "theta_merge should be < theta_sim"
        
        # Validate cluster 0 parameters
        assert 0 <= self.viability_baseline <= 1, "viability_baseline must be in [0, 1]"
        assert self.cluster0_ttl > 0, "cluster0_ttl must be positive"
        assert self.cluster0_max_capacity > 0, "cluster0_max_capacity must be positive"
        
        # Validate island management
        assert self.max_island_capacity > 0, "max_island_capacity must be positive"
        
        # Validate removal threshold if provided
        if self.removal_threshold is not None:
            assert 0 <= self.removal_threshold <= 1, "removal_threshold must be in [0, 1]"
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary with all configuration parameters as key-value pairs.
            Useful for saving config to JSON or logging.
        """
        return {
            "theta_sim": self.theta_sim,
            "theta_merge": self.theta_merge,
            "viability_baseline": self.viability_baseline,
            "cluster0_ttl": self.cluster0_ttl,
            "cluster0_min_cluster_size": self.cluster0_min_cluster_size,
            "cluster0_max_capacity": self.cluster0_max_capacity,
            "removal_threshold": self.removal_threshold,
            "max_island_capacity": self.max_island_capacity,
            "min_island_size": self.min_island_size,
            "stagnation_window": self.stagnation_window,
            "max_stagnation": self.max_stagnation,
            "explore_mutation_multiplier": self.explore_mutation_multiplier,
            "exploit_mutation_multiplier": self.exploit_mutation_multiplier,
            "explore_selection_pressure": self.explore_selection_pressure,
            "improvement_slope_threshold": self.improvement_slope_threshold,
            "decline_slope_threshold": self.decline_slope_threshold,
            "migration_frequency": self.migration_frequency,
            "k_neighbors": self.k_neighbors,
            "migration_selection": self.migration_selection,
            "silhouette_threshold": self.silhouette_threshold,
            "silhouette_check_frequency": self.silhouette_check_frequency,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "embedding_batch_size": self.embedding_batch_size,
            "repopulation_mutation_rate": self.repopulation_mutation_rate,
            "repopulation_size": self.repopulation_size,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "SpeciationConfig":
        """
        Create SpeciationConfig instance from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters.
                        Unknown keys are ignored (filtered by dataclass fields).
        
        Returns:
            SpeciationConfig instance with values from dictionary.
        
        Example:
            >>> config = SpeciationConfig.from_dict({"theta_sim": 0.5, "cluster0_ttl": 15})
        """
        # Filter to only include valid dataclass fields
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# Default configuration instance for convenience
DEFAULT_CONFIG = SpeciationConfig()
