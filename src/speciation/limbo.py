"""
limbo.py

Limbo buffer management for Plan A+ speciation.
Holding area for high-fitness outliers that don't fit existing species.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, TYPE_CHECKING
from scipy.cluster.hierarchy import linkage, fcluster

from .island import Individual, Species, generate_species_id
from .embeddings import semantic_distance

if TYPE_CHECKING:
    from .config import PlanAPlusConfig

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


@dataclass
class LimboIndividual:
    """Wrapper for individuals in limbo with TTL tracking."""
    individual: Individual
    entered_at: int
    ttl: int = 10
    
    def __hash__(self):
        return hash(self.individual.id)
    
    def __eq__(self, other):
        return isinstance(other, LimboIndividual) and self.individual.id == other.individual.id


class LimboBuffer:
    """Limbo buffer for high-fitness outliers."""
    
    def __init__(self, default_ttl: int = 10, min_cluster_size: int = 2, theta_sim: float = 0.4, logger=None):
        self.members: List[LimboIndividual] = []
        self.default_ttl = default_ttl
        self.min_cluster_size = min_cluster_size
        self.theta_sim = theta_sim
        self.logger = logger or get_logger("LimboBuffer")
        self.speciation_events: List[Dict] = []
    
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
        """Decrement TTL and remove expired."""
        expired = []
        remaining = []
        for lm in self.members:
            lm.ttl -= 1
            if lm.ttl <= 0:
                expired.append(lm.individual)
            else:
                remaining.append(lm)
        self.members = remaining
        if expired:
            self.logger.info(f"Expired {len(expired)} individuals from limbo")
        return expired
    
    def check_speciation(self, current_generation: int) -> Optional[Species]:
        """Check if limbo individuals can form a new species."""
        if len(self.members) < self.min_cluster_size:
            return None
        
        individuals = [lm.individual for lm in self.members if lm.individual.embedding is not None]
        if len(individuals) < self.min_cluster_size:
            return None
        
        clusters = self._agglomerative_clustering(individuals, self.theta_sim)
        
        for cluster in sorted(clusters, key=len, reverse=True):
            if len(cluster) >= self.min_cluster_size and self._check_cohesion(cluster, self.theta_sim):
                leader = max(cluster, key=lambda x: x.fitness)
                new_species = Species(
                    id=generate_species_id(), leader=leader, members=list(cluster),
                    radius=self.theta_sim, created_at=current_generation, last_improvement=current_generation
                )
                self.remove_batch(cluster)
                self.speciation_events.append({
                    "generation": current_generation, "species_id": new_species.id,
                    "size": len(cluster), "leader_fitness": leader.fitness
                })
                self.logger.info(f"Speciation event! Created species {new_species.id} from {len(cluster)} limbo individuals")
                return new_species
        return None
    
    def _agglomerative_clustering(self, individuals: List[Individual], threshold: float) -> List[List[Individual]]:
        if len(individuals) < 2:
            return [individuals] if individuals else []
        
        n = len(individuals)
        embeddings = np.array([ind.embedding for ind in individuals])
        distances = [semantic_distance(embeddings[i], embeddings[j]) for i in range(n) for j in range(i + 1, n)]
        
        if not distances:
            return [individuals]
        
        try:
            Z = linkage(np.array(distances), method='average')
            labels = fcluster(Z, t=threshold, criterion='distance')
        except Exception:
            return [individuals]
        
        clusters: Dict[int, List[Individual]] = {}
        for ind, label in zip(individuals, labels):
            clusters.setdefault(label, []).append(ind)
        return list(clusters.values())
    
    def _check_cohesion(self, cluster: List[Individual], threshold: float) -> bool:
        if len(cluster) < 2:
            return True
        for i, ind1 in enumerate(cluster):
            for ind2 in cluster[i + 1:]:
                if ind1.embedding is None or ind2.embedding is None:
                    continue
                if semantic_distance(ind1.embedding, ind2.embedding) > threshold:
                    return False
        return True
    
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
    """Check if individual should enter limbo."""
    if individual.embedding is None or individual.fitness <= viability_baseline:
        return False
    for sp in species.values():
        if sp.leader.embedding is not None:
            if semantic_distance(individual.embedding, sp.leader.embedding) < theta_sim:
                return False
    return True

