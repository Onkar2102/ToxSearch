"""
leader_follower.py

Leader-Follower clustering algorithm for Plan A+ speciation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

from .island import Individual, Species, generate_species_id
from .embeddings import semantic_distance, semantic_distances_batch

if TYPE_CHECKING:
    from .limbo import LimboBuffer

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def leader_follower_clustering(
    population: List[Individual],
    theta_sim: float,
    viability_baseline: float = 0.3,
    limbo: Optional["LimboBuffer"] = None,
    current_generation: int = 0,
    existing_species: Optional[Dict[int, Species]] = None,
    logger=None
) -> Tuple[Dict[int, Species], List[Individual]]:
    """
    Assign individuals to species using Leader-Follower clustering.
    
    Algorithm:
        1. Sort population by fitness (descending)
        2. First individual becomes first leader
        3. For each remaining: if d_min < theta_sim → follower, else → limbo or new leader
    """
    if logger is None:
        logger = get_logger("LeaderFollowerClustering")
    
    if not population:
        return {}, []
    
    valid_population = [ind for ind in population if ind.embedding is not None]
    if not valid_population:
        logger.error("No individuals with embeddings")
        return {}, []
    
    sorted_pop = sorted(valid_population, key=lambda x: x.fitness, reverse=True)
    
    species: Dict[int, Species] = {}
    leaders: List[Tuple[int, np.ndarray]] = []
    limbo_candidates: List[Individual] = []
    
    # First individual becomes first leader
    first = sorted_pop[0]
    first_species_id = generate_species_id()
    first_species = Species(
        id=first_species_id, leader=first, members=[first],
        radius=theta_sim, created_at=current_generation, last_improvement=current_generation
    )
    species[first_species_id] = first_species
    leaders.append((first_species_id, first.embedding))
    
    # Process remaining
    for ind in sorted_pop[1:]:
        min_dist = float('inf')
        nearest_leader_id = None
        
        if len(leaders) > 1:
            leader_embeddings = np.array([emb for _, emb in leaders])
            distances = semantic_distances_batch(ind.embedding, leader_embeddings)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            nearest_leader_id = leaders[min_idx][0]
        elif len(leaders) == 1:
            min_dist = semantic_distance(ind.embedding, leaders[0][1])
            nearest_leader_id = leaders[0][0]
        
        if min_dist < theta_sim:
            species[nearest_leader_id].add_member(ind)
        else:
            if ind.fitness > viability_baseline:
                limbo_candidates.append(ind)
            else:
                new_species_id = generate_species_id()
                new_species = Species(
                    id=new_species_id, leader=ind, members=[ind],
                    radius=theta_sim, created_at=current_generation, last_improvement=current_generation
                )
                species[new_species_id] = new_species
                leaders.append((new_species_id, ind.embedding))
    
    logger.info(f"Leader-Follower clustering: {len(valid_population)} individuals → {len(species)} species, {len(limbo_candidates)} limbo candidates")
    return species, limbo_candidates


def find_nearest_leader(embedding: np.ndarray, species: Dict[int, Species]) -> Tuple[Optional[int], float]:
    """Find the nearest species leader to an embedding."""
    if not species:
        return None, float('inf')
    
    leaders = [(sid, sp.leader.embedding) for sid, sp in species.items() if sp.leader.embedding is not None]
    if not leaders:
        return None, float('inf')
    
    if len(leaders) > 1:
        leader_ids = [sid for sid, _ in leaders]
        leader_embeddings = np.array([emb for _, emb in leaders])
        distances = semantic_distances_batch(embedding, leader_embeddings)
        min_idx = np.argmin(distances)
        return leader_ids[min_idx], distances[min_idx]
    else:
        sid, emb = leaders[0]
        return sid, semantic_distance(embedding, emb)


def update_species_leaders(species: Dict[int, Species]) -> None:
    """Update leaders for all species."""
    for sp in species.values():
        sp.update_leader()


def incremental_clustering(
    new_individuals: List[Individual],
    existing_species: Dict[int, Species],
    theta_sim: float,
    viability_baseline: float = 0.3,
    current_generation: int = 0,
    logger=None
) -> Tuple[Dict[int, Species], List[Individual]]:
    """Incrementally add new individuals to existing species."""
    if logger is None:
        logger = get_logger("IncrementalClustering")
    
    limbo_candidates = []
    
    for ind in new_individuals:
        if ind.embedding is None:
            continue
        
        nearest_id, min_dist = find_nearest_leader(ind.embedding, existing_species)
        
        if nearest_id is not None and min_dist < theta_sim:
            existing_species[nearest_id].add_member(ind)
        elif ind.fitness > viability_baseline:
            limbo_candidates.append(ind)
        else:
            new_id = generate_species_id()
            new_sp = Species(id=new_id, leader=ind, members=[ind], radius=theta_sim,
                           created_at=current_generation, last_improvement=current_generation)
            existing_species[new_id] = new_sp
    
    return existing_species, limbo_candidates


def reassign_to_species(individual: Individual, species: Dict[int, Species], theta_sim: float) -> Optional[int]:
    """Find suitable species for an individual."""
    if individual.embedding is None or not species:
        return None
    nearest_id, min_dist = find_nearest_leader(individual.embedding, species)
    return nearest_id if min_dist < theta_sim else None

