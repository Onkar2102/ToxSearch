"""
extinction.py

Island extinction and repopulation for speciation.
"""

from typing import Dict, List, Optional, Tuple, Callable, TYPE_CHECKING

from .species import Individual, Species, SpeciesMode, IslandMode, generate_species_id

if TYPE_CHECKING:
    from .reserves import Cluster0

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def should_extinct(island: Species, max_stagnation: int = 20, min_size: int = 2) -> bool:
    """
    Check if an island should be extinguished (removed from evolution).
    
    An island is marked for extinction if either:
    1. Size < min_size: Too small to be viable (can't evolve effectively)
    2. Stagnation >= max_stagnation AND mode == EXPLORE: No improvement and exploring
    
    The mode check for EXPLORE is important because:
    - EXPLOIT mode islands might still improve (focused on best solutions)
    - EXPLORE mode islands that don't improve are clearly stuck and should be replaced
    - DEFAULT mode islands are given more time
    
    Args:
        island: Species to check for extinction
        max_stagnation: Maximum generations without improvement before extinction
        min_size: Minimum population size to remain viable
    
    Returns:
        True if island should be extinguished, False otherwise
    """
    # Check if too small to be viable
    if island.size < min_size:
        return True
    # Check if stagnant in EXPLORE mode (no improvement and exploring)
    if island.stagnation_counter >= max_stagnation and island.mode == IslandMode.EXPLORE:
        return True
    return False


def detect_extinction_candidates(species: Dict[int, Species], max_stagnation: int = 20,
                                  min_size: int = 2, logger=None) -> List[int]:
    """
    Find all species that should be extinguished in this generation.
    
    Filters species using should_extinct() criteria to identify extinction candidates.
    Multiple species can be extinct in the same generation if they meet criteria.
    
    Args:
        species: Dict of all current species
        max_stagnation: Maximum stagnation generations (passed to should_extinct)
        min_size: Minimum viable population size (passed to should_extinct)
        logger: Optional logger instance
    
    Returns:
        List of species IDs to extinguish (empty if none)
    """
    if logger is None:
        logger = get_logger("Extinction")
    # Identify all species meeting extinction criteria
    candidates = [sid for sid, sp in species.items() if should_extinct(sp, max_stagnation, min_size)]
    return candidates


def repopulate_from_cluster0(
    cluster0: "Cluster0",
    current_generation: int,
    theta_sim: float = 0.4,
    population_size: int = 20,
    mutate_fn: Optional[Callable] = None,
    mutation_rate: float = 0.3,
    logger=None
) -> Optional[Species]:
    """
    Create a new species from the best individual in cluster 0.
    
    When a species goes extinct, repopulation gives high-fitness outliers (in cluster 0)
    a second chance to form their own species. This:
    - Preserves diversity (cluster 0 contains novel high-fitness solutions)
    - Maintains genetic material across extinctions
    - Avoids losing promising unexplored regions of the search space
    
    Process:
    1. Pop best individual from cluster 0 as leader
    2. Mutate to create population_size-1 new individuals
    3. Create new species with cluster_origin="natural" (repopulation is a natural process)
    4. If cluster 0 is empty or mutation fails, returns None
    
    Args:
        cluster0: Cluster 0 (source for repopulation)
        current_generation: Current generation number
        theta_sim: Constant radius for new species
        population_size: Size of new population to create
        mutate_fn: Mutation function (applied to leader to create new individuals)
        mutation_rate: Mutation rate parameter passed to mutate_fn
        logger: Optional logger instance
    
    Returns:
        New Species created from cluster 0, or None if cluster 0 is empty
    """
    if logger is None:
        logger = get_logger("Repopulation")
    
    if cluster0.size == 0:
        return None
    
    # Pop best individual from cluster 0 to be the leader
    leader = cluster0.pop_best()
    if leader is None:
        return None
    
    # Start with leader as first member
    members = [leader]
    # Create additional members via mutation
    if mutate_fn:
        for _ in range(population_size - 1):
            try:
                members.append(mutate_fn(leader, mutation_rate))
            except Exception:
                # Mutation failed for this individual, skip it
                pass
    
    # Create new species with these members
    new_sp = Species(
        id=generate_species_id(),
        leader=leader,
        members=members,
        radius=theta_sim,  # Constant radius
        created_at=current_generation,
        last_improvement=current_generation,
        cluster_origin="natural",  # Repopulation is a natural process
        parent_ids=None,
        parent_id=None
    )
    logger.info(f"Repopulated from cluster 0: species {new_sp.id}")
    return new_sp


def repopulate_from_global_best(
    global_best: Individual,
    current_generation: int,
    theta_sim: float = 0.4,
    population_size: int = 20,
    mutate_fn: Optional[Callable] = None,
    mutation_rate: float = 0.3,
    logger=None
) -> Species:
    """
    Create a new species from the global best individual (fallback repopulation).
    
    Used when extinction requires repopulation but cluster 0 is empty. Starting from
    the global best ensures the new species has high fitness. The population is
    created via mutation of the global best, exploring the local neighborhood.
    
    This is less diverse than repopulation from cluster 0 (which are diverse outliers),
    but better than creating completely random new individuals.
    
    Process:
    1. Create copy of global best as leader
    2. Mutate global best to create population_size-1 new individuals
    3. Create new species with cluster_origin="natural"
    
    Args:
        global_best: Global best individual (source for new species)
        current_generation: Current generation number
        theta_sim: Constant radius for new species
        population_size: Size of new population to create
        mutate_fn: Mutation function (applied to global_best to create new individuals)
        mutation_rate: Mutation rate parameter passed to mutate_fn
        logger: Optional logger instance
    
    Returns:
        New Species created from global best
    """
    if logger is None:
        logger = get_logger("Repopulation")
    
    # Create a copy of global_best as the leader
    leader = Individual(
        id=global_best.id,
        prompt=global_best.prompt,
        fitness=global_best.fitness,
        embedding=global_best.embedding,
        generation=current_generation,
        genome_data=global_best.genome_data.copy() if global_best.genome_data else None
    )
    
    # Start with leader as first member
    members = [leader]
    # Create additional members via mutation
    if mutate_fn:
        for _ in range(population_size - 1):
            try:
                members.append(mutate_fn(leader, mutation_rate))
            except Exception:
                # Mutation failed for this individual, skip it
                pass
    
    # Create new species with these members
    new_sp = Species(
        id=generate_species_id(),
        leader=leader,
        members=members,
        radius=theta_sim,  # Constant radius
        created_at=current_generation,
        last_improvement=current_generation,
        cluster_origin="natural",  # Repopulation is a natural process
        parent_ids=None,
        parent_id=None
    )
    logger.info(f"Repopulated from global best: species {new_sp.id}")
    return new_sp


def process_extinctions(
    species: Dict[int, Species],
    cluster0: "Cluster0",
    global_best: Optional[Individual],
    current_generation: int,
    theta_sim: float = 0.4,
    max_stagnation: int = 20,
    min_size: int = 2,
    repopulation_size: int = 20,
    mutation_rate: float = 0.3,
    mutate_fn: Optional[Callable] = None,
    logger=None
) -> Tuple[Dict[int, Species], List[Dict]]:
    """
    Process all species extinctions and repopulation.
    
    Extinction removes stagnant or too-small species to make room for new diversity.
    A species is extinguished if:
    1. Size < min_size (too small to be viable)
    2. Stagnation >= max_stagnation AND mode == EXPLORE (stagnant explorer)
    
    After extinction, a new species is created via repopulation:
    1. First tries to repopulate from cluster 0 (high-fitness outliers)
    2. Falls back to global best if cluster 0 is empty
    
    Repopulation creates new individuals via mutation, introducing fresh diversity.
    All repopulated species have cluster_origin="natural".
    
    Args:
        species: Dict of species (modified in-place)
        cluster0: Cluster 0 (source for repopulation)
        global_best: Global best individual (fallback for repopulation)
        current_generation: Current generation number
        theta_sim: Constant radius for new species
        max_stagnation: Maximum stagnation before extinction
        min_size: Minimum species size
        repopulation_size: Size of new repopulated species
        mutation_rate: Mutation rate for repopulation
        mutate_fn: Mutation function for creating new individuals
        logger: Optional logger instance
    
    Returns:
        Tuple of (updated_species, extinction_events)
    """
    if logger is None:
        logger = get_logger("Extinction")
    
    events = []
    extinct_ids = detect_extinction_candidates(species, max_stagnation, min_size, logger)
    
    for sid in extinct_ids:
        if sid not in species:
            continue
        
        sp = species.pop(sid)
        logger.info(f"Extinguished species {sid}")
        
        event = {
            "generation": current_generation,
            "species_id": sid,
            "stagnation": sp.stagnation_counter
        }
        
        new_sp = None
        if cluster0.size > 0:
            new_sp = repopulate_from_cluster0(
                cluster0, current_generation, theta_sim,
                repopulation_size, mutate_fn, mutation_rate, logger
            )
            if new_sp:
                event["repopulation_source"] = "cluster_0"
        
        if new_sp is None and global_best:
            new_sp = repopulate_from_global_best(
                global_best, current_generation, theta_sim,
                repopulation_size, mutate_fn, mutation_rate, logger
            )
            event["repopulation_source"] = "global_best"
        
        if new_sp:
            species[new_sp.id] = new_sp
            event["new_species_id"] = new_sp.id
            event["cluster_origin"] = "natural"
        
        events.append(event)
    
    return species, events


def find_global_best(species: Dict[int, Species]) -> Optional[Individual]:
    """
    Find the individual with highest fitness across all species.
    
    Used for repopulation: when cluster 0 is empty, global best becomes the seed
    for new species creation. Ensures evolutionary progress is not lost.
    
    Args:
        species: Dict of all current species
    
    Returns:
        Best individual across all species, or None if no species/members
    """
    best, best_fitness = None, float('-inf')
    # Scan all members across all species
    for sp in species.values():
        for m in sp.members:
            if m.fitness > best_fitness:
                best, best_fitness = m, m.fitness
    return best

