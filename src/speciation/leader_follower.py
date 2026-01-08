"""
leader_follower.py

Leader-Follower clustering algorithm for speciation.
Reads genomes directly from temp.json and performs clustering.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from pathlib import Path

from .island import Individual, Species, generate_species_id
from .distance import semantic_distance, semantic_distances_batch

if TYPE_CHECKING:
    from .limbo import LimboBuffer

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def leader_follower_clustering(
    temp_path: str,
    theta_sim: float,
    viability_baseline: float = 0.3,
    limbo: Optional["LimboBuffer"] = None,
    current_generation: int = 0,
    existing_species: Optional[Dict[int, Species]] = None,
    logger=None) -> Tuple[Dict[int, Species], List[Individual]]:
    """
    Assign individuals to species using Leader-Follower clustering algorithm.
    
    This function reads genomes directly from temp.json, converts them to Individuals,
    and performs clustering. It is the core clustering algorithm for speciation.
    It assigns individuals to species based on semantic similarity to existing leaders,
    with fitness-based prioritization.
    
    Algorithm:
        1. Read genomes from temp.json (with prompt_embedding field)
        2. Convert genomes to Individual objects
        3. Sort population by fitness (descending) - fitness determines processing order
        4. First individual becomes first leader (creates first species)
        5. For each remaining individual:
           a. Find nearest leader (minimum semantic distance)
           b. If distance < theta_sim → assign as follower to that species
           c. Else:
              - If fitness > viability_baseline → send to limbo (high-fitness outlier)
              - Else → create new species with this individual as leader
    
    Key properties:
    - Fitness-based ordering ensures high-quality leaders
    - Semantic distance threshold (theta_sim) controls species granularity
    - High-fitness outliers go to limbo for potential speciation
    - Low-fitness outliers create new species (exploration)
    
    Args:
        temp_path: Path to temp.json file with genomes (must have prompt_embedding field)
        theta_sim: Semantic distance threshold for species assignment
        viability_baseline: Minimum fitness to enter limbo (vs creating new species)
        limbo: Optional limbo buffer (for tracking, not modified here)
        current_generation: Current generation number (for species metadata)
        existing_species: Optional existing species dict (for incremental clustering)
        logger: Optional logger instance
    
    Returns:
        Tuple of:
        - species: Dict mapping species_id -> Species
        - limbo_candidates: List of individuals that should enter limbo
    """
    if logger is None:
        logger = get_logger("LeaderFollowerClustering")
    
    # Read genomes from temp.json
    temp_path_obj = Path(temp_path)
    if not temp_path_obj.exists():
        logger.error(f"Temp file not found: {temp_path}")
        return {}, []
    
    with open(temp_path_obj, 'r', encoding='utf-8') as f:
        genomes = json.load(f)
    
    if not genomes:
        logger.warning("No genomes found in temp.json")
        return {}, []
    
    # Convert genomes to Individual objects (embeddings from prompt_embedding field)
    population = [Individual.from_genome(genome) for genome in genomes]
    
    # Handle empty population
    if not population:
        logger.error("No individuals found in population in temp.json")
        return {}, []
    
    # Filter out individuals without embeddings (can't cluster without embeddings)
    valid_population = [ind for ind in population if ind.embedding is not None]
    if not valid_population:
        logger.error("No individuals with embeddings")
        return {}, []
    
    # Sort by fitness (descending) - highest fitness processed first
    sorted_pop = sorted(valid_population, key=lambda x: x.fitness, reverse=True)
    
    species: Dict[int, Species] = {}
    leaders: List[Tuple[int, np.ndarray]] = []  # (species_id, leader_embedding) for fast lookup
    limbo_candidates: List[Individual] = []
    
    # Step 1: First individual becomes first leader (creates first species)
    first = sorted_pop[0]
    first_species_id = generate_species_id()
    first_species = Species(
        id=first_species_id, leader=first, members=[first],
        radius=theta_sim, created_at=current_generation, last_improvement=current_generation
    )
    species[first_species_id] = first_species
    leaders.append((first_species_id, first.embedding))
    
    # Step 2: Process remaining individuals
    for ind in sorted_pop[1:]:
        min_dist = float('inf')
        nearest_leader_id = None
        
        # Find nearest leader (optimized for multiple leaders)
        if len(leaders) > 1:
            # Vectorized batch computation for efficiency
            leader_embeddings = np.array([emb for _, emb in leaders])
            distances = semantic_distances_batch(ind.embedding, leader_embeddings)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            nearest_leader_id = leaders[min_idx][0]
        elif len(leaders) == 1:
            # Single leader case (direct computation)
            min_dist = semantic_distance(ind.embedding, leaders[0][1])
            nearest_leader_id = leaders[0][0]
        
        # Decision: assign to species, limbo, or create new species
        if min_dist < theta_sim:
            # Within threshold → assign as follower
            species[nearest_leader_id].add_member(ind)
        else:
            # Outside threshold → decide based on fitness
            if ind.fitness > viability_baseline:
                # High-fitness outlier → preserve in limbo
                limbo_candidates.append(ind)
            else:
                # Low-fitness outlier → create new species (exploration)
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
    """
    Find the nearest species leader to a given embedding.
    
    Used for:
    - Incremental clustering (assigning new individuals to existing species)
    - Migration (finding target species)
    - Reassignment after species changes
    
    Args:
        embedding: Query embedding vector (L2-normalized)
        species: Dict of existing species
    
    Returns:
        Tuple of (species_id, distance):
        - species_id: ID of nearest species (None if no species)
        - distance: Semantic distance to nearest leader (inf if no species)
    """
    if not species:
        return None, float('inf')
    
    # Collect all leader embeddings (filter out None)
    leaders = [(sid, sp.leader.embedding) for sid, sp in species.items() if sp.leader.embedding is not None]
    if not leaders:
        return None, float('inf')
    
    # Optimize based on number of leaders
    if len(leaders) > 1:
        # Multiple leaders: use vectorized batch computation
        leader_ids = [sid for sid, _ in leaders]
        leader_embeddings = np.array([emb for _, emb in leaders])
        distances = semantic_distances_batch(embedding, leader_embeddings)
        min_idx = np.argmin(distances)
        return leader_ids[min_idx], distances[min_idx]
    else:
        # Single leader: direct computation
        sid, emb = leaders[0]
        return sid, semantic_distance(embedding, emb)


def update_species_leaders(species: Dict[int, Species]) -> None:
    """
    Update leaders for all species to highest-fitness members.
    
    Should be called after fitness updates or member changes to ensure
    leaders accurately represent species centers. Leader updates affect:
    - Species center for distance computations
    - Leader-follower clustering decisions
    - Migration topology
    
    Args:
        species: Dict of species to update
    """
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
    """
    Incrementally add new individuals to existing species.
    
    This is used when new individuals are generated (e.g., offspring) and need
    to be assigned to existing species. Unlike leader_follower_clustering(),
    this doesn't create a new clustering from scratch - it assigns to existing
    species or creates new ones if needed.
    
    Algorithm:
        For each new individual:
        1. Find nearest existing leader
        2. If distance < theta_sim → assign to that species
        3. Else if fitness > viability_baseline → send to limbo
        4. Else → create new species
    
    Args:
        new_individuals: List of new individuals to assign
        existing_species: Dict of existing species (modified in-place)
        theta_sim: Semantic distance threshold
        viability_baseline: Minimum fitness for limbo
        current_generation: Current generation number
        logger: Optional logger instance
    
    Returns:
        Tuple of (existing_species, limbo_candidates)
        Note: existing_species is modified in-place, returned for convenience
    """
    if logger is None:
        logger = get_logger("IncrementalClustering")
    
    limbo_candidates = []
    
    for ind in new_individuals:
        # Skip individuals without embeddings
        if ind.embedding is None:
            continue
        
        # Find nearest existing leader
        nearest_id, min_dist = find_nearest_leader(ind.embedding, existing_species)
        
        # Decision logic (same as leader_follower_clustering)
        if nearest_id is not None and min_dist < theta_sim:
            # Within threshold → assign to existing species
            existing_species[nearest_id].add_member(ind)
        elif ind.fitness > viability_baseline:
            # High-fitness outlier → limbo
            limbo_candidates.append(ind)
        else:
            # Low-fitness outlier → new species
            new_id = generate_species_id()
            new_sp = Species(id=new_id, leader=ind, members=[ind], radius=theta_sim,
                           created_at=current_generation, last_improvement=current_generation)
            existing_species[new_id] = new_sp
    
    return existing_species, limbo_candidates


def reassign_to_species(individual: Individual, species: Dict[int, Species], theta_sim: float) -> Optional[int]:
    """
    Find suitable species for an individual (without assigning).
    
    Used to check if an individual can be assigned to a species without
    actually performing the assignment. Useful for validation or conditional logic.
    
    Args:
        individual: Individual to check
        species: Dict of existing species
        theta_sim: Semantic distance threshold
    
    Returns:
        species_id if suitable species found (distance < theta_sim), else None
    """
    if individual.embedding is None or not species:
        return None
    nearest_id, min_dist = find_nearest_leader(individual.embedding, species)
    return nearest_id if min_dist < theta_sim else None

