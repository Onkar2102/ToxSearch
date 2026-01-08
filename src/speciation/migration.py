"""
migration.py

Inter-island migration for speciation.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

from .island import Individual, Species
from .distance import semantic_distance, semantic_distances_batch

if TYPE_CHECKING:
    from .limbo import LimboBuffer

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def compute_semantic_topology(species: Dict[int, Species], k_neighbors: int = 3) -> Dict[int, List[int]]:
    """
    Build semantic neighbor graph for migration topology.
    
    Creates a graph where each species is connected to its k nearest semantic neighbors
    (based on leader embedding distances). This topology determines which species
    can exchange migrants.
    
    Migration topology enables:
    - Genetic exchange between related species
    - Diversity transfer without breaking species boundaries
    - Exploration of semantic space between species
    
    Args:
        species: Dict of species
        k_neighbors: Number of nearest neighbors per species
    
    Returns:
        Dict mapping species_id -> list of neighbor species_ids
    """
    topology = {sid: [] for sid in species.keys()}
    species_list = list(species.items())
    
    if len(species_list) <= 1:
        return topology
    
    for sid1, sp1 in species_list:
        if sp1.leader.embedding is None:
            continue
        
        distances = []
        for sid2, sp2 in species_list:
            if sid1 != sid2 and sp2.leader.embedding is not None:
                dist = semantic_distance(sp1.leader.embedding, sp2.leader.embedding)
                distances.append((sid2, dist))
        
        distances.sort(key=lambda x: x[1])
        topology[sid1] = [sid for sid, _ in distances[:k_neighbors]]
    
    return topology


def select_migrant(members: List[Individual], target_leader_embedding: np.ndarray,
                   selection_method: str = "most_unique") -> Optional[Individual]:
    """Select a migrant from source island."""
    if not members or len(members) <= 1:
        return None
    
    candidates = [m for m in members[1:] if m.embedding is not None]
    if not candidates:
        return None
    
    if selection_method == "most_unique":
        max_dist, most_unique = -1, None
        for ind in candidates:
            dist = semantic_distance(ind.embedding, target_leader_embedding)
            if dist > max_dist:
                max_dist, most_unique = dist, ind
        return most_unique
    elif selection_method == "random":
        return random.choice(candidates)
    elif selection_method == "best":
        return max(candidates, key=lambda x: x.fitness)
    else:
        return random.choice(candidates)


def perform_migration(species: Dict[int, Species], topology: Dict[int, List[int]],
                     current_generation: int, migration_frequency: int = 5,
                     max_capacity: int = 50, selection_method: str = "most_unique",
                     limbo: Optional["LimboBuffer"] = None, logger=None) -> List[Dict]:
    """Migrate individuals between neighbor islands."""
    if logger is None:
        logger = get_logger("Migration")
    
    if current_generation % migration_frequency != 0:
        return []
    
    events = []
    
    for sid1, neighbors in topology.items():
        if not neighbors:
            continue
        
        sp1 = species.get(sid1)
        if sp1 is None or sp1.size <= 2:
            continue
        
        sid2 = random.choice(neighbors)
        sp2 = species.get(sid2)
        if sp2 is None or sp2.leader.embedding is None:
            continue
        
        migrant = select_migrant(sp1.members, sp2.leader.embedding, selection_method)
        if migrant is None:
            continue
        
        sp1.remove_member(migrant)
        sp2.add_member(migrant)
        
        event = {"generation": current_generation, "from": sid1, "to": sid2, "migrant_id": migrant.id}
        
        if sp2.size > max_capacity:
            weakest = min([m for m in sp2.members if m != sp2.leader], key=lambda x: x.fitness, default=None)
            if weakest:
                sp2.remove_member(weakest)
                event["ejected_id"] = weakest.id
                if limbo:
                    limbo.add(weakest, current_generation)
        
        events.append(event)
        logger.debug(f"Migration: {migrant.id} from {sid1} to {sid2}")
    
    if events:
        logger.info(f"Performed {len(events)} migration(s)")
    
    return events


def process_migrations(species: Dict[int, Species], current_generation: int,
                       migration_frequency: int = 5, k_neighbors: int = 3,
                       max_capacity: int = 50, selection_method: str = "most_unique",
                       limbo: Optional["LimboBuffer"] = None, logger=None) -> Tuple[Dict[int, Species], List[Dict]]:
    """Full migration processing pipeline."""
    if logger is None:
        logger = get_logger("Migration")
    
    if current_generation % migration_frequency != 0:
        return species, []
    
    topology = compute_semantic_topology(species, k_neighbors)
    events = perform_migration(species, topology, current_generation, 1, max_capacity, selection_method, limbo, logger)
    
    return species, events


def get_topology_statistics(species: Dict[int, Species], topology: Dict[int, List[int]]) -> Dict:
    """Get topology statistics."""
    if not topology:
        return {"n_islands": 0, "avg_neighbors": 0.0, "isolated": 0}
    
    counts = [len(n) for n in topology.values()]
    return {
        "n_islands": len(topology),
        "avg_neighbors": sum(counts) / len(counts) if counts else 0.0,
        "isolated": sum(1 for c in counts if c == 0)
    }

