"""
reserves.py

Cluster 0 (reserves buffer) management for speciation.
Holding area for individuals that don't fit existing species (ID >= 1).
Cluster 0 is a special cluster with ID 0 that holds outliers and removed individuals.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from scipy.cluster.hierarchy import linkage, fcluster

from .species import Individual, Species, generate_species_id
from .distance import semantic_distance

if TYPE_CHECKING:
    from .config import SpeciationConfig

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


# Cluster 0 ID (reserved, species IDs start from 1)
CLUSTER_0_ID = 0


@dataclass
class Cluster0Individual:
    """
    Wrapper for individuals in Cluster 0 (reserves) with TTL (Time-To-Live) tracking.
    
    Cluster 0 individuals are outliers that don't fit existing species.
    They are preserved for a limited time (TTL) to allow potential speciation
    if enough similar individuals accumulate.
    
    Attributes:
        individual: The Individual instance in Cluster 0 (reserves)
        entered_at: Generation when individual entered Cluster 0 (reserves)
        ttl: Time-to-live (remaining generations before expiration)
    """
    individual: Individual
    entered_at: int
    ttl: int = 10
    
    def __hash__(self):
        return hash(self.individual.id)
    
    def __eq__(self, other):
        return isinstance(other, Cluster0Individual) and self.individual.id == other.individual.id




class Cluster0:
    """
    Cluster 0 (reserves) for individuals that don't fit existing species.
    
    Cluster 0 (ID=0) is a special holding area for:
    1. High-fitness outliers that are semantically distant from all species leaders
    2. Individuals removed from species when they exceed max capacity (100)
    3. Individuals that may form new species if enough similar ones accumulate
    
    Key features:
    - TTL mechanism: Individuals expire after TTL generations (prevents unbounded growth)
    - Max capacity: Limited to cluster0_max_capacity (default 1000) individuals
    - Removal threshold: Individuals below removal_threshold are archived
    - Speciation detection: When Cluster 0 individuals form cohesive clusters, they create new species
    - Agglomerative clustering: Uses hierarchical clustering to find groups in Cluster 0
    
    This preserves diversity by giving novel high-fitness solutions a chance to
    form new species rather than being discarded.
    """
    
    def __init__(
        self,
        default_ttl: int = 10,
        min_cluster_size: int = 2,
        theta_sim: float = 0.4,
        max_capacity: int = 1000,
        removal_threshold: Optional[float] = None,
        logger=None
    ):
        """
        Initialize Cluster 0 (reserves).
        
        Args:
            default_ttl: Default time-to-live in generations
            min_cluster_size: Minimum cluster size for speciation
            theta_sim: Semantic distance threshold for clustering (also used as species radius)
            max_capacity: Maximum number of individuals in Cluster 0 (reserves) (default: 1000)
            removal_threshold: Fitness threshold for removal (individuals below are archived)
            logger: Optional logger instance
        """
        self.members: List[Cluster0Individual] = []
        self.default_ttl = default_ttl
        self.min_cluster_size = min_cluster_size
        self.theta_sim = theta_sim
        self.max_capacity = max_capacity
        self.removal_threshold = removal_threshold
        self.logger = logger or get_logger("Cluster0")
        self.speciation_events: List[Dict] = []  # Track speciation events from Cluster 0 (reserves)
    
    @property
    def size(self) -> int:
        """Number of individuals in Cluster 0 (reserves)."""
        return len(self.members)
    
    @property
    def individuals(self) -> List[Individual]:
        """Get all individuals (without TTL wrapper)."""
        return [lm.individual for lm in self.members]
    
    def add(self, individual: Individual, generation: int, ttl: Optional[int] = None) -> None:
        """
        Add individual to Cluster 0 (reserves).
        
        Sets the individual's species_id to 0 (Cluster 0).
        
        Args:
            individual: Individual to add
            generation: Current generation number
            ttl: Optional time-to-live (uses default if not specified)
        """
        # Avoid duplicates
        for lm in self.members:
            if lm.individual.id == individual.id:
                return
        
        # Mark individual as belonging to Cluster 0 (reserves)
        individual.species_id = CLUSTER_0_ID
        
        self.members.append(Cluster0Individual(
            individual=individual,
            entered_at=generation,
            ttl=ttl or self.default_ttl
        ))
    
    def add_batch(self, individuals: List[Individual], generation: int, ttl: Optional[int] = None) -> None:
        """Add multiple individuals to Cluster 0 (reserves)."""
        for ind in individuals:
            self.add(ind, generation, ttl)
    
    def remove(self, individual: Individual) -> bool:
        """Remove an individual from Cluster 0 (reserves)."""
        for i, lm in enumerate(self.members):
            if lm.individual.id == individual.id:
                self.members.pop(i)
                return True
        return False
    
    def remove_batch(self, individuals: List[Individual]) -> int:
        """Remove multiple individuals from Cluster 0 (reserves). Returns count removed."""
        ids = {ind.id for ind in individuals}
        original = len(self.members)
        self.members = [lm for lm in self.members if lm.individual.id not in ids]
        return original - len(self.members)
    
    def update_ttl(self, current_generation: int) -> List[Individual]:
        """
        Decrement TTL for all Cluster 0 (reserves) individuals and remove expired ones.
        
        Called each generation to manage cluster 0 size. Expired individuals
        are removed to prevent unbounded growth.
        
        Args:
            current_generation: Current generation number (for logging)
        
        Returns:
            List of expired individuals (removed from Cluster 0)
        """
        expired = []
        remaining = []
        for lm in self.members:
            lm.ttl -= 1  # Decrement TTL
            if lm.ttl <= 0:
                expired.append(lm.individual)  # Mark for removal
            else:
                remaining.append(lm)  # Keep
        self.members = remaining
        if expired:
            self.logger.info(f"Expired {len(expired)} individuals from Cluster 0 (reserves)")
        return expired
    
    def filter_by_removal_threshold(self, removal_threshold: Optional[float] = None, max_fitness: Optional[float] = None) -> List[Individual]:
        """
        Remove individuals with fitness below (removal_threshold)% of population's max fitness.
        
        For reserves (Cluster 0), we keep genomes scoring equal to or more than 
        (removal_threshold)% of population's max fitness score.
        
        These individuals should be archived to under_performing.json.
        
        Args:
            removal_threshold: Percentage threshold (0-1, e.g., 0.1 = 10%) (uses instance value if not provided)
            max_fitness: Maximum fitness in the population (required if removal_threshold is provided)
        
        Returns:
            List of removed individuals (to be archived)
        """
        threshold_pct = removal_threshold if removal_threshold is not None else self.removal_threshold
        
        if threshold_pct is None or max_fitness is None:
            return []  # No threshold or max_fitness, nothing to filter
        
        # Calculate actual threshold: (removal_threshold)% of max fitness
        threshold = threshold_pct * max_fitness
        
        removed = []
        remaining = []
        
        for lm in self.members:
            if lm.individual.fitness < threshold:
                removed.append(lm.individual)
            else:
                remaining.append(lm)
        
        self.members = remaining
        
        if removed:
            self.logger.info(f"Removed {len(removed)} individuals below {threshold_pct*100:.1f}% of max fitness ({threshold:.4f}) from Cluster 0 (reserves)")
        
        return removed
    
    def enforce_capacity(self) -> List[Individual]:
        """
        Enforce maximum capacity by removing lowest-fitness individuals.
        
        If Cluster 0 (reserves) size exceeds max_capacity, removes individuals until
        size equals max_capacity, keeping highest-fitness individuals.
        
        Removed individuals should be archived to under_performing.json.
        
        Returns:
            List of removed individuals (to be archived)
        """
        if self.size <= self.max_capacity:
            return []
        
        # Sort by fitness (highest first)
        self.members.sort(key=lambda x: x.individual.fitness, reverse=True)
        
        # Keep top max_capacity
        removed_members = self.members[self.max_capacity:]
        self.members = self.members[:self.max_capacity]
        
        removed = [lm.individual for lm in removed_members]
        
        if removed:
            self.logger.info(f"Enforced capacity: removed {len(removed)} lowest-fitness individuals from Cluster 0 (reserves)")
        
        return removed
    
    def check_speciation(self, current_generation: int) -> Optional[Species]:
        """
        Check if Cluster 0 (reserves) individuals can form a new species via clustering.
        
        This is the key mechanism for creating new species from Cluster 0. When enough
        similar high-fitness individuals accumulate, they can form a cohesive cluster
        and create a new species.
        
        Algorithm:
        1. Check if enough individuals in Cluster 0 (>= min_cluster_size)
        2. Perform agglomerative clustering on Cluster 0 individuals
        3. Find largest cohesive cluster (all pairwise distances < theta_sim)
        4. If cluster size >= min_cluster_size, create new species
        5. Remove cluster members from Cluster 0
        
        Args:
            current_generation: Current generation number
        
        Returns:
            New Species if speciation occurred, else None
        """
        # Check minimum size requirement
        if len(self.members) < self.min_cluster_size:
            return None
        
        # Filter to individuals with embeddings
        individuals = [lm.individual for lm in self.members if lm.individual.embedding is not None]
        if len(individuals) < self.min_cluster_size:
            return None
        
        # Cluster individuals
        clusters = self._agglomerative_clustering(individuals, self.theta_sim)
        
        # Check clusters from largest to smallest
        for cluster in sorted(clusters, key=len, reverse=True):
            # Verify cluster is cohesive and large enough
            if len(cluster) >= self.min_cluster_size and self._check_cohesion(cluster, self.theta_sim):
                # Create new species with highest-fitness individual as leader
                leader = max(cluster, key=lambda x: x.fitness)
                new_species = Species(
                    id=generate_species_id(),
                    leader=leader,
                    members=list(cluster),
                    radius=self.theta_sim,  # Constant radius
                    created_at=current_generation,
                    last_improvement=current_generation,
                    cluster_origin="natural",  # Formed naturally from Cluster 0
                    parent_ids=None,
                    parent_id=None
                )
                # Remove from Cluster 0 (they're now in a species)
                self.remove_batch(cluster)
                # Track speciation event
                self.speciation_events.append({
                    "generation": current_generation,
                    "species_id": new_species.id,
                    "size": len(cluster),
                    "leader_fitness": leader.fitness,
                    "origin": "cluster_0_speciation"
                })
                self.logger.info(f"Speciation event! Created species {new_species.id} from {len(cluster)} Cluster 0 (reserves) individuals")
                return new_species
        return None
    
    def _agglomerative_clustering(self, individuals: List[Individual], threshold: float) -> List[List[Individual]]:
        """
        Perform agglomerative hierarchical clustering on Cluster 0 (reserves) individuals.
        
        Uses scipy's linkage and fcluster to find groups of similar individuals.
        
        Args:
            individuals: List of individuals to cluster
            threshold: Distance threshold for clustering
        
        Returns:
            List of clusters (each cluster is a list of individuals)
        """
        if len(individuals) < 2:
            return [individuals] if individuals else []
        
        n = len(individuals)
        embeddings = np.array([ind.embedding for ind in individuals])
        # Compute pairwise distances (condensed distance matrix)
        distances = [semantic_distance(embeddings[i], embeddings[j]) for i in range(n) for j in range(i + 1, n)]
        
        if not distances:
            return [individuals]
        
        try:
            # Hierarchical clustering with average linkage
            Z = linkage(np.array(distances), method='average')
            # Form clusters at threshold distance
            labels = fcluster(Z, t=threshold, criterion='distance')
        except Exception:
            # Fallback: return all as single cluster
            return [individuals]
        
        # Group individuals by cluster label
        clusters: Dict[int, List[Individual]] = {}
        for ind, label in zip(individuals, labels):
            clusters.setdefault(label, []).append(ind)
        return list(clusters.values())
    
    def _check_cohesion(self, cluster: List[Individual], threshold: float) -> bool:
        """
        Verify that all pairwise distances in cluster are below threshold.
        
        A cohesive cluster means all members are semantically similar to each other,
        not just to the leader. This ensures the cluster is truly homogeneous.
        
        Args:
            cluster: List of individuals to check
            threshold: Maximum allowed pairwise distance
        
        Returns:
            True if cluster is cohesive (all distances < threshold), else False
        """
        if len(cluster) < 2:
            return True  # Single individual is trivially cohesive
        # Check all pairwise distances
        for i, ind1 in enumerate(cluster):
            for ind2 in cluster[i + 1:]:
                if ind1.embedding is None or ind2.embedding is None:
                    continue
                if semantic_distance(ind1.embedding, ind2.embedding) > threshold:
                    return False  # Found pair that's too far apart
        return True  # All pairs are within threshold
    
    def get_best(self, n: int = 1) -> List[Individual]:
        """Get top N individuals by fitness."""
        sorted_members = sorted(self.members, key=lambda x: x.individual.fitness, reverse=True)
        return [lm.individual for lm in sorted_members[:n]]
    
    def pop_best(self) -> Optional[Individual]:
        """Remove and return the best individual."""
        if not self.members:
            return None
        best_idx = max(range(len(self.members)), key=lambda i: self.members[i].individual.fitness)
        return self.members.pop(best_idx).individual
    
    def clear(self) -> None:
        """Clear all members from Cluster 0 (reserves)."""
        self.members = []
    
    def to_dict(self) -> Dict:
        """Serialize Cluster 0 (reserves) to dictionary for JSON storage."""
        return {
            "cluster_id": CLUSTER_0_ID,
            "size": self.size,
            "max_capacity": self.max_capacity,
            "removal_threshold": self.removal_threshold,
            "members": [
                {
                    "id": lm.individual.id,
                    "prompt": lm.individual.prompt,
                    "fitness": lm.individual.fitness,
                    "ttl": lm.ttl,
                    "entered_at": lm.entered_at
                }
                for lm in self.members
            ],
            "speciation_events": self.speciation_events[-10:]
        }




def should_enter_cluster0(
    individual: Individual,
    species: Dict[int, Species],
    theta_sim: float,
    viability_baseline: float
) -> bool:
    """
    Check if an individual should enter Cluster 0 (reserves).
    
    An individual should enter Cluster 0 if:
    1. Has an embedding (can be clustered)
    2. Has fitness > viability_baseline (high-fitness outlier)
    3. Is semantically distant from all existing species leaders (doesn't fit anywhere)
    
    Args:
        individual: Individual to check
        species: Dict of existing species (ID >= 1)
        theta_sim: Semantic distance threshold
        viability_baseline: Minimum fitness for Cluster 0
    
    Returns:
        True if individual should enter Cluster 0 (reserves), else False
    """
    # Must have embedding and sufficient fitness
    if individual.embedding is None or individual.fitness <= viability_baseline:
        return False
    # Check if fits into any existing species
    for sp in species.values():
        if sp.leader.embedding is not None:
            if semantic_distance(individual.embedding, sp.leader.embedding) < theta_sim:
                return False  # Fits into existing species, don't send to Cluster 0
    return True  # Doesn't fit anywhere and has high fitness → Cluster 0
