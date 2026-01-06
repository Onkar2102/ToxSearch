"""
extinction.py

Island extinction and repopulation for Plan A+ speciation.
"""

from typing import Dict, List, Optional, Tuple, Callable, TYPE_CHECKING

from .island import Individual, Species, IslandMode, generate_species_id

if TYPE_CHECKING:
    from .limbo import LimboBuffer

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def should_extinct(island: Species, max_stagnation: int = 20, min_size: int = 2) -> bool:
    """Check if island should be extinguished."""
    if island.size < min_size:
        return True
    if island.stagnation_counter >= max_stagnation and island.mode == IslandMode.EXPLORE:
        return True
    return False


def detect_extinction_candidates(species: Dict[int, Species], max_stagnation: int = 20,
                                  min_size: int = 2, logger=None) -> List[int]:
    """Find all species to extinguish."""
    if logger is None:
        logger = get_logger("Extinction")
    return [sid for sid, sp in species.items() if should_extinct(sp, max_stagnation, min_size)]


def repopulate_from_limbo(limbo: "LimboBuffer", current_generation: int, population_size: int = 20,
                          mutate_fn: Optional[Callable] = None, mutation_rate: float = 0.3,
                          theta_sim: float = 0.4, logger=None) -> Optional[Species]:
    """Create new island from limbo."""
    if logger is None:
        logger = get_logger("Repopulation")
    
    if limbo.size == 0:
        return None
    
    leader = limbo.pop_best()
    if leader is None:
        return None
    
    members = [leader]
    if mutate_fn:
        for _ in range(population_size - 1):
            try:
                members.append(mutate_fn(leader, mutation_rate))
            except Exception:
                pass
    
    new_sp = Species(id=generate_species_id(), leader=leader, members=members,
                    radius=theta_sim, created_at=current_generation, last_improvement=current_generation)
    logger.info(f"Repopulated from limbo: species {new_sp.id}")
    return new_sp


def repopulate_from_global_best(global_best: Individual, current_generation: int, population_size: int = 20,
                                mutate_fn: Optional[Callable] = None, mutation_rate: float = 0.3,
                                theta_sim: float = 0.4, logger=None) -> Species:
    """Create new island from global best."""
    if logger is None:
        logger = get_logger("Repopulation")
    
    leader = Individual(id=global_best.id, prompt=global_best.prompt, fitness=global_best.fitness,
                       embedding=global_best.embedding, generation=current_generation,
                       genome_data=global_best.genome_data.copy() if global_best.genome_data else None)
    
    members = [leader]
    if mutate_fn:
        for _ in range(population_size - 1):
            try:
                members.append(mutate_fn(leader, mutation_rate))
            except Exception:
                pass
    
    new_sp = Species(id=generate_species_id(), leader=leader, members=members,
                    radius=theta_sim, created_at=current_generation, last_improvement=current_generation)
    logger.info(f"Repopulated from global best: species {new_sp.id}")
    return new_sp


def process_extinctions(species: Dict[int, Species], limbo: "LimboBuffer", global_best: Optional[Individual],
                        current_generation: int, max_stagnation: int = 20, min_size: int = 2,
                        repopulation_size: int = 20, mutation_rate: float = 0.3,
                        mutate_fn: Optional[Callable] = None, theta_sim: float = 0.4,
                        logger=None) -> Tuple[Dict[int, Species], List[Dict]]:
    """Process all extinctions."""
    if logger is None:
        logger = get_logger("Extinction")
    
    events = []
    extinct_ids = detect_extinction_candidates(species, max_stagnation, min_size, logger)
    
    for sid in extinct_ids:
        if sid not in species:
            continue
        
        sp = species.pop(sid)
        logger.info(f"Extinguished species {sid}")
        
        event = {"generation": current_generation, "species_id": sid, "stagnation": sp.stagnation_counter}
        
        new_sp = None
        if limbo.size > 0:
            new_sp = repopulate_from_limbo(limbo, current_generation, repopulation_size,
                                          mutate_fn, mutation_rate, theta_sim, logger)
            if new_sp:
                event["repopulation_source"] = "limbo"
        
        if new_sp is None and global_best:
            new_sp = repopulate_from_global_best(global_best, current_generation, repopulation_size,
                                                mutate_fn, mutation_rate, theta_sim, logger)
            event["repopulation_source"] = "global_best"
        
        if new_sp:
            species[new_sp.id] = new_sp
            event["new_species_id"] = new_sp.id
        
        events.append(event)
    
    return species, events


def find_global_best(species: Dict[int, Species]) -> Optional[Individual]:
    """Find global best individual."""
    best, best_fitness = None, float('-inf')
    for sp in species.values():
        for m in sp.members:
            if m.fitness > best_fitness:
                best, best_fitness = m, m.fitness
    return best

