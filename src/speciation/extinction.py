"""
extinction.py

Species freezing and small species management for speciation.
"""

from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

from .species import Individual, Species
from .reserves import CLUSTER_0_ID

if TYPE_CHECKING:
    from .reserves import Cluster0

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def process_extinctions(
    species: Dict[int, Species],
    cluster0: "Cluster0",
    current_generation: int,
    max_stagnation: int = 20,
    min_size: int = 2,
    logger=None
) -> Tuple[Dict[int, Species], List[Dict]]:
    """
    Process species freezing and move small species to cluster 0.
    
    Actions:
    1. Freeze species with stagnation >= max_stagnation (mark as "frozen", not extinguished)
    2. Move species with count < min_size to cluster 0 (until capacity reached)
    
    No repopulation is needed - clustering will handle new species formation.
    
    Args:
        species: Dict of species (modified in-place)
        cluster0: Cluster 0 (reserves) for small species
        current_generation: Current generation number
        max_stagnation: Maximum stagnation before freezing
        min_size: Minimum species size (below this, move to cluster 0)
        logger: Optional logger instance
    
    Returns:
        Tuple of (updated_species, events)
    """
    if logger is None:
        logger = get_logger("Extinction")
    
    events = []
    
    # Step 1: Freeze species with stagnation >= max_stagnation
    frozen_ids = []
    for sid, sp in species.items():
        if sp.stagnation >= max_stagnation and sp.species_state != "frozen":
            sp.species_state = "frozen"
            frozen_ids.append(sid)
            events.append({
                "generation": current_generation,
                "species_id": sid,
                "action": "frozen",
                "stagnation": sp.stagnation,
                "max_fitness": sp.max_fitness
            })
            logger.info(f"Frozen species {sid} (stagnation={sp.stagnation} >= {max_stagnation})")
    
    # Step 2: Move small species to cluster 0 (until capacity reached)
    small_species_ids = [sid for sid, sp in species.items() if sp.size < min_size]
    
    for sid in small_species_ids:
        if sid not in species:
            continue
        
        # Check cluster 0 capacity
        if cluster0.size >= cluster0.max_capacity:
            logger.debug(f"Cluster 0 at capacity ({cluster0.max_capacity}), cannot move species {sid}")
            continue
        
        sp = species.pop(sid)
        
        # Move all members to cluster 0
        moved_count = 0
        for member in sp.members:
            if cluster0.size >= cluster0.max_capacity:
                break  # Stop if capacity reached
            cluster0.add(member, current_generation)
            moved_count += 1
        
        events.append({
            "generation": current_generation,
            "species_id": sid,
            "action": "moved_to_cluster0",
            "size": sp.size,
            "moved_count": moved_count
        })
        logger.info(f"Moved species {sid} ({sp.size} members) to cluster 0 ({moved_count} moved)")
    
    return species, events
