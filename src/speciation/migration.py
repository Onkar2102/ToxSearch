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
    from .reserves import Cluster0

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
    """
    Select a migrant from a source island to send to target island.
    
    Different selection strategies balance genetic diversity with fitness:
    - "most_unique": Select individual most different from target leader
      (maximizes diversity transfer between islands)
    - "random": Random selection (baseline)
    - "best": Select highest-fitness individual (greedy migration)
    
    The leader is excluded from migration (never migrates out).
    
    Args:
        members: List of members in source island
        target_leader_embedding: Target island leader's embedding
        selection_method: Migration selection method
    
    Returns:
        Selected Individual, or None if no suitable candidates
    """
    # Skip if too few members (leader is always kept)
    if not members or len(members) <= 1:
        return None
    
    # Candidates are non-leader members with embeddings
    candidates = [m for m in members[1:] if m.embedding is not None]
    if not candidates:
        return None
    
    # Apply selection strategy
    if selection_method == "most_unique":
        # Find individual most different from target (maximize diversity)
        max_dist, most_unique = -1, None
        for ind in candidates:
            dist = semantic_distance(ind.embedding, target_leader_embedding)
            if dist > max_dist:
                max_dist, most_unique = dist, ind
        return most_unique
    elif selection_method == "random":
        # Random selection
        return random.choice(candidates)
    elif selection_method == "best":
        # High-fitness selection
        return max(candidates, key=lambda x: x.fitness)
    else:
        # Default to random
        return random.choice(candidates)


def perform_migration(
    species: Dict[int, Species],
    topology: Dict[int, List[int]],
    current_generation: int,
    migration_frequency: int = 5,
    max_capacity: int = 100,
    selection_method: str = "most_unique",
    cluster0: Optional["Cluster0"] = None,
    logger=None
) -> List[Dict]:
    """
    Execute migration events between neighboring islands.
    
    Migration improves genetic exchange between related species:
    1. For each species with neighbors (from topology)
    2. Select a random neighbor to migrate to
    3. Select a migrant using selection_method
    4. Move migrant from source to target island
    5. If target exceeds capacity (100), eject weakest member to cluster 0
    
    This implements island hopping: individuals move between islands following
    the semantic topology (similar species are neighbors).
    
    Args:
        species: Dict of all current species
        topology: Migration topology (species_id -> list of neighbor IDs)
        current_generation: Current generation number
        migration_frequency: Perform migration every N generations (check before calling)
        max_capacity: Maximum members per species after migration (default: 100)
        selection_method: Migrant selection strategy
        cluster0: Optional cluster 0 (for ejected members)
        logger: Optional logger instance
    
    Returns:
        List of migration events (for logging/metrics)
    """
    if logger is None:
        logger = get_logger("Migration")
    
    events = []
    
    # Perform migration: source -> target
    for sid1, neighbors in topology.items():
        if not neighbors:
            continue  # No neighbors, can't migrate
        
        sp1 = species.get(sid1)
        if sp1 is None or sp1.size <= 2:
            continue  # Can't migrate (too small or doesn't exist)
        
        # Pick random neighbor to migrate to
        sid2 = random.choice(neighbors)
        sp2 = species.get(sid2)
        if sp2 is None or sp2.leader.embedding is None:
            continue
        
        # Select migrant (non-leader)
        migrant = select_migrant(sp1.members, sp2.leader.embedding, selection_method)
        if migrant is None:
            continue
        
        # Execute migration
        sp1.remove_member(migrant)
        sp2.add_member(migrant)
        
        event = {"generation": current_generation, "from": sid1, "to": sid2, "migrant_id": migrant.id}
        
        # Handle capacity overflow in target (max 100 per species)
        if sp2.size > max_capacity:
            # Eject weakest non-leader member
            weakest = min([m for m in sp2.members if m != sp2.leader], key=lambda x: x.fitness, default=None)
            if weakest:
                sp2.remove_member(weakest)
                event["ejected_id"] = weakest.id
                # Send ejected to cluster 0
                if cluster0:
                    cluster0.add(weakest, current_generation)
        
        events.append(event)
        logger.debug(f"Migration: {migrant.id} from {sid1} to {sid2}")
    
    if events:
        logger.info(f"Performed {len(events)} migration(s)")
    
    return events


def process_migrations(
    species: Dict[int, Species],
    current_generation: int,
    migration_frequency: int = 5,
    k_neighbors: int = 3,
    max_capacity: int = 100,
    selection_method: str = "most_unique",
    cluster0: Optional["Cluster0"] = None,
    logger=None
) -> Tuple[Dict[int, Species], List[Dict]]:
    """
    Full migration processing pipeline.
    
    Orchestrates the complete migration process:
    1. Check if migration should happen this generation (every migration_frequency gens)
    2. Build semantic topology (k-nearest neighbors)
    3. Execute migration on topology
    
    Args:
        species: Dict of all current species (modified in-place)
        current_generation: Current generation number
        migration_frequency: Perform migration every N generations
        k_neighbors: Number of nearest neighbors for topology
        max_capacity: Maximum members per species (default: 100)
        selection_method: Migrant selection strategy
        cluster0: Optional cluster 0 for ejected members
        logger: Optional logger instance
    
    Returns:
        Tuple of (species, migration_events)
    """
    if logger is None:
        logger = get_logger("Migration")
    
    # Check if migration should happen this generation
    if current_generation % migration_frequency != 0:
        return species, []
    
    # Build topology and perform migration
    topology = compute_semantic_topology(species, k_neighbors)
    events = perform_migration(species, topology, current_generation, 1, max_capacity, selection_method, cluster0, logger)
    
    return species, events


def get_topology_statistics(species: Dict[int, Species], topology: Dict[int, List[int]]) -> Dict:
    """
    Get statistics about the migration topology.
    
    Analyzes the island migration network:
    - Number of islands
    - Average number of neighbors per island
    - Number of isolated islands (no neighbors)
    
    Useful for understanding the connectivity of the population structure.
    
    Args:
        species: Dict of all current species
        topology: Migration topology (species_id -> neighbor IDs)
    
    Returns:
        Dict with topology statistics
    """
    if not topology:
        return {"n_islands": 0, "avg_neighbors": 0.0, "isolated": 0}
    
    # Count neighbors for each island
    counts = [len(n) for n in topology.values()]
    return {
        "n_islands": len(topology),
        "avg_neighbors": sum(counts) / len(counts) if counts else 0.0,
        "isolated": sum(1 for c in counts if c == 0)  # Islands with no neighbors
    }

