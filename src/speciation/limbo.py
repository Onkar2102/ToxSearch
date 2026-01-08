"""
limbo.py

Limbo buffer management for speciation.
Holding area for high-fitness outliers that don't fit existing species.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, TYPE_CHECKING
from scipy.cluster.hierarchy import linkage, fcluster

from .island import Individual, Species, generate_species_id
from .distance import semantic_distance

if TYPE_CHECKING:
    from .config import PlanAPlusConfig

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


@dataclass
class LimboIndividual:
    """
    Wrapper for individuals in limbo buffer with TTL (Time-To-Live) tracking.
    
    Limbo individuals are high-fitness outliers that don't fit existing species.
    They are preserved for a limited time (TTL) to allow potential speciation
    if enough similar individuals accumulate.
    
    Attributes:
        individual: The Individual instance in limbo
        entered_at: Generation when individual entered limbo
        ttl: Time-to-live (remaining generations before expiration)
    """
    individual: Individual
    entered_at: int
    ttl: int = 10
    
    def __hash__(self):
        return hash(self.individual.id)
    
    def __eq__(self, other):
        return isinstance(other, LimboIndividual) and self.individual.id == other.individual.id


class LimboBuffer:
    """
    Limbo buffer for high-fitness outliers that don't fit existing species.
    
    The limbo buffer serves as a "holding area" for promising individuals that:
    1. Have high fitness (above viability_baseline)
    2. Are semantically distant from all existing species leaders
    3. May form new species if enough similar individuals accumulate
    
    Key features:
    - TTL mechanism: Individuals expire after TTL generations (prevents unbounded growth)
    - Speciation detection: When limbo individuals form cohesive clusters, they create new species
    - Agglomerative clustering: Uses hierarchical clustering to find groups in limbo
    
    This preserves diversity by giving novel high-fitness solutions a chance to
    form new species rather than being discarded.
    """
    
    def __init__(self, default_ttl: int = 10, min_cluster_size: int = 2, theta_sim: float = 0.4, logger=None):
        """
        Initialize limbo buffer.
        
        Args:
            default_ttl: Default time-to-live in generations
            min_cluster_size: Minimum cluster size for speciation
            theta_sim: Semantic distance threshold for clustering
            logger: Optional logger instance
        """
        self.members: List[LimboIndividual] = []
        self.default_ttl = default_ttl
        self.min_cluster_size = min_cluster_size
        self.theta_sim = theta_sim
        self.logger = logger or get_logger("LimboBuffer")
        self.speciation_events: List[Dict] = []  # Track speciation events from limbo
    
    @property
    def size(self) -> int:
        return len(self.members)
    
    @property
    def individuals(self) -> List[Individual]:
        return [lm.individual for lm in self.members]
    
    def add(self, individual: Individual, generation: int, ttl: Optional[int] = None) -> None:
        """Add individual to limbo."""
        for lm in self.members:
            if lm.individual.id == individual.id:
                return
        self.members.append(LimboIndividual(individual=individual, entered_at=generation, ttl=ttl or self.default_ttl))
    
    def add_batch(self, individuals: List[Individual], generation: int, ttl: Optional[int] = None) -> None:
        for ind in individuals:
            self.add(ind, generation, ttl)
    
    def remove(self, individual: Individual) -> bool:
        for i, lm in enumerate(self.members):
            if lm.individual.id == individual.id:
                self.members.pop(i)
                return True
        return False
    
    def remove_batch(self, individuals: List[Individual]) -> int:
        ids = {ind.id for ind in individuals}
        original = len(self.members)
        self.members = [lm for lm in self.members if lm.individual.id not in ids]
        return original - len(self.members)
    
    def update_ttl(self, current_generation: int) -> List[Individual]:
        """
        Decrement TTL for all limbo individuals and remove expired ones.
        
        Called each generation to manage limbo buffer size. Expired individuals
        are removed to prevent unbounded growth. This ensures limbo doesn't
        accumulate indefinitely.
        
        Args:
            current_generation: Current generation number (for logging)
        
        Returns:
            List of expired individuals (removed from limbo)
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
            self.logger.info(f"Expired {len(expired)} individuals from limbo")
        return expired
    
    def check_speciation(self, current_generation: int) -> Optional[Species]:
        """
        Check if limbo individuals can form a new species via clustering.
        
        This is the key mechanism for creating new species from limbo. When enough
        similar high-fitness individuals accumulate in limbo, they can form a
        cohesive cluster and create a new species.
        
        Algorithm:
        1. Check if enough individuals in limbo (>= min_cluster_size)
        2. Perform agglomerative clustering on limbo individuals
        3. Find largest cohesive cluster (all pairwise distances < theta_sim)
        4. If cluster size >= min_cluster_size, create new species
        5. Remove cluster members from limbo
        
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
        
        # Cluster limbo individuals
        clusters = self._agglomerative_clustering(individuals, self.theta_sim)
        
        # Check clusters from largest to smallest
        for cluster in sorted(clusters, key=len, reverse=True):
            # Verify cluster is cohesive and large enough
            if len(cluster) >= self.min_cluster_size and self._check_cohesion(cluster, self.theta_sim):
                # Create new species with highest-fitness individual as leader
                leader = max(cluster, key=lambda x: x.fitness)
                new_species = Species(
                    id=generate_species_id(), leader=leader, members=list(cluster),
                    radius=self.theta_sim, created_at=current_generation, last_improvement=current_generation
                )
                # Remove from limbo (they're now in a species)
                self.remove_batch(cluster)
                # Track speciation event
                self.speciation_events.append({
                    "generation": current_generation, "species_id": new_species.id,
                    "size": len(cluster), "leader_fitness": leader.fitness
                })
                self.logger.info(f"Speciation event! Created species {new_species.id} from {len(cluster)} limbo individuals")
                return new_species
        return None
    
    def _agglomerative_clustering(self, individuals: List[Individual], threshold: float) -> List[List[Individual]]:
        """
        Perform agglomerative hierarchical clustering on limbo individuals.
        
        Uses scipy's linkage and fcluster to find groups of similar individuals.
        This is more sophisticated than simple distance thresholding and can
        discover natural clusters in limbo.
        
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
        sorted_members = sorted(self.members, key=lambda x: x.individual.fitness, reverse=True)
        return [lm.individual for lm in sorted_members[:n]]
    
    def pop_best(self) -> Optional[Individual]:
        if not self.members:
            return None
        best_idx = max(range(len(self.members)), key=lambda i: self.members[i].individual.fitness)
        return self.members.pop(best_idx).individual
    
    def clear(self) -> None:
        self.members = []
    
    def to_dict(self) -> Dict:
        return {
            "size": self.size,
            "members": [{"id": lm.individual.id, "fitness": lm.individual.fitness, "ttl": lm.ttl} for lm in self.members],
            "speciation_events": self.speciation_events[-10:]
        }


def should_enter_limbo(individual: Individual, species: Dict[int, Species], theta_sim: float, viability_baseline: float) -> bool:
    """
    Check if an individual should enter limbo buffer.
    
    An individual should enter limbo if:
    1. Has an embedding (can be clustered)
    2. Has fitness > viability_baseline (high-fitness outlier)
    3. Is semantically distant from all existing species leaders (doesn't fit anywhere)
    
    Args:
        individual: Individual to check
        species: Dict of existing species
        theta_sim: Semantic distance threshold
        viability_baseline: Minimum fitness for limbo
    
    Returns:
        True if individual should enter limbo, else False
    """
    # Must have embedding and sufficient fitness
    if individual.embedding is None or individual.fitness <= viability_baseline:
        return False
    # Check if fits into any existing species
    for sp in species.values():
        if sp.leader.embedding is not None:
            if semantic_distance(individual.embedding, sp.leader.embedding) < theta_sim:
                return False  # Fits into existing species, don't send to limbo
    return True  # Doesn't fit anywhere and has high fitness → limbo

