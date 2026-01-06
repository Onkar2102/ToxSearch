"""
config.py

Configuration parameters for Plan A+ Dynamic Islands speciation framework.
"""

from dataclasses import dataclass


@dataclass
class PlanAPlusConfig:
    """
    Configuration parameters for Plan A+ speciation framework.
    
    Attributes:
        # Clustering
        theta_sim: Similarity threshold for species assignment (distance).
        theta_merge: Merge threshold for combining similar islands.
        
        # Limbo
        viability_baseline: Minimum fitness to enter limbo.
        limbo_ttl: Time-to-live in generations for limbo individuals.
        limbo_min_cluster_size: Minimum cluster size for limbo speciation.
        
        # Island Management
        max_island_capacity: Maximum individuals per island.
        min_island_size: Minimum island size before extinction.
        stagnation_window: Generations for stagnation detection.
        max_stagnation: Generations before extinction.
        
        # Mode Switching
        explore_mutation_multiplier: Mutation rate multiplier in EXPLORE mode.
        exploit_mutation_multiplier: Mutation rate multiplier in EXPLOIT mode.
        explore_selection_pressure: Selection pressure in EXPLORE mode.
        improvement_slope_threshold: Slope to trigger EXPLOIT mode.
        decline_slope_threshold: Slope to trigger EXPLORE mode.
        
        # Migration
        migration_frequency: Migrate every N generations.
        k_neighbors: Number of neighbors for migration topology.
        migration_selection: Method for selecting migrants.
        
        # Adaptive Thresholds
        silhouette_threshold: Trigger split if silhouette below this.
        radius_shrink_factor: Factor to shrink radius when over capacity.
        silhouette_check_frequency: Check silhouette every N generations.
        
        # Embedding
        embedding_model: Sentence-transformer model name.
        embedding_dim: Embedding dimensionality.
        embedding_batch_size: Batch size for embedding computation.
        
        # Repopulation
        repopulation_mutation_rate: Mutation rate for repopulation.
        repopulation_size: Size of repopulated islands.
    """
    
    # Clustering
    theta_sim: float = 0.4
    theta_merge: float = 0.2
    
    # Limbo
    viability_baseline: float = 0.3
    limbo_ttl: int = 10
    limbo_min_cluster_size: int = 2
    
    # Island Management
    max_island_capacity: int = 50
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
    
    # Adaptive Thresholds
    silhouette_threshold: float = 0.5
    radius_shrink_factor: float = 0.9
    silhouette_check_frequency: int = 10
    
    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 64
    
    # Repopulation
    repopulation_mutation_rate: float = 0.3
    repopulation_size: int = 20
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert 0 <= self.theta_sim <= 2, f"theta_sim must be in [0, 2]"
        assert 0 <= self.theta_merge <= 2, f"theta_merge must be in [0, 2]"
        assert self.theta_merge < self.theta_sim, "theta_merge should be < theta_sim"
        assert 0 <= self.viability_baseline <= 1, "viability_baseline must be in [0, 1]"
        assert self.limbo_ttl > 0, "limbo_ttl must be positive"
        assert self.max_island_capacity > 0, "max_island_capacity must be positive"
        assert 0 < self.radius_shrink_factor < 1, "radius_shrink_factor must be in (0, 1)"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "theta_sim": self.theta_sim,
            "theta_merge": self.theta_merge,
            "viability_baseline": self.viability_baseline,
            "limbo_ttl": self.limbo_ttl,
            "limbo_min_cluster_size": self.limbo_min_cluster_size,
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
            "radius_shrink_factor": self.radius_shrink_factor,
            "silhouette_check_frequency": self.silhouette_check_frequency,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "embedding_batch_size": self.embedding_batch_size,
            "repopulation_mutation_rate": self.repopulation_mutation_rate,
            "repopulation_size": self.repopulation_size,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "PlanAPlusConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


DEFAULT_CONFIG = PlanAPlusConfig()

