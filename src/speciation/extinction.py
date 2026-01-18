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
    species_stagnation: int = 20,
    min_size: int = 2,
    elites_path: Optional[str] = None,
    logger=None
) -> Tuple[Dict[int, Species], List[Dict], List[Dict], Dict[int, Species]]:
    """
    Process species freezing and move small species to cluster 0.
    
    Actions:
    1. Freeze species with stagnation >= species_stagnation (EXTINCTION - tracked separately)
    2. Move species with count < min_size to cluster 0 (NOT extinction - tracked separately)
       - Species moved to cluster 0 get state="incubator" and are preserved in speciation_state.json
       - The species ID is considered deceased (new species from cluster 0 get new IDs)
    
    No repopulation is needed - clustering will handle new species formation.
    
    Args:
        species: Dict of species (modified in-place)
        cluster0: Cluster 0 (reserves) for small species
        current_generation: Current generation number
        species_stagnation: Maximum stagnation before freezing
        min_size: Minimum species size (below this, move to cluster 0)
        logger: Optional logger instance
    
    Returns:
        Tuple of (updated_species, extinction_events, moved_to_cluster0_events, incubator_species)
        - extinction_events: Only frozen species (stagnation-based)
        - moved_to_cluster0_events: Species moved to cluster 0 (size-based, NOT extinction)
        - incubator_species: Dict of species moved to incubator state (for preservation in historical_species)
    """
    if logger is None:
        logger = get_logger("Extinction")
    
    extinction_events = []  # Only frozen species (stagnation-based)
    moved_to_cluster0_events = []  # Species moved to cluster 0 (size-based, NOT extinction)
    incubator_species = {}  # Species to be marked as incubator (keep in speciation_state.json)
    
    # Step 1: Freeze species with stagnation >= species_stagnation (EXTINCTION)
    frozen_ids = []
    for sid, sp in species.items():
        if sp.stagnation >= species_stagnation and sp.species_state != "frozen":
            sp.species_state = "frozen"
            frozen_ids.append(sid)
            extinction_events.append({
                "generation": current_generation,
                "species_id": sid,
                "action": "frozen",
                "stagnation": sp.stagnation,
                "max_fitness": sp.max_fitness
            })
            logger.info(f"Frozen species {sid} (stagnation={sp.stagnation} >= {species_stagnation}) - EXTINCTION")
    
    # Step 2: Move small species to cluster 0 (NOT extinction, just reorganization)
    # Species get state="incubator" and are kept in speciation_state.json for reference
    # CRITICAL: Use in-memory size (sp.size), NOT elites.json size
    # elites.json accumulates genomes from ALL generations (cumulative), so it's not accurate for current state
    # We want to move species to incubator based on CURRENT size (after radius cleanup, capacity enforcement)
    # In-memory size reflects the current generation's actual state
    small_species_ids = []
    for sid, sp in species.items():
        # Check both active and frozen species for size < min_size
        # Also check species already marked as incubator (may have been marked by _ensure_unique_leader or duplicate fix)
        # Frozen species can also shrink below min_size and should be moved to incubator
        if sp.species_state not in ["active", "frozen", "incubator"]:
            continue
        # Use in-memory size (current generation state), not elites.json (cumulative across all generations)
        current_size = sp.size
        # If already incubator, process it (may have been marked but not cleaned up yet)
        # If active/frozen and size < min_size, mark for incubator
        if sp.species_state == "incubator" or current_size < min_size:
            if sp.species_state != "incubator":
                small_species_ids.append(sid)
                logger.debug(f"Species {sid}: current size={current_size} (in-memory), min_size={min_size}, state={sp.species_state} -> will move to incubator")
            else:
                # Already marked as incubator but not yet processed - add to list for cleanup
                small_species_ids.append(sid)
                logger.debug(f"Species {sid}: already marked as incubator but not yet processed, will complete cleanup")
    
    for sid in small_species_ids:
        if sid not in species:
            continue
        
        # Check cluster 0 capacity
        if cluster0.size >= cluster0.max_capacity:
            logger.debug(f"Cluster 0 at capacity ({cluster0.max_capacity}), cannot move species {sid}")
            continue
        
        sp = species[sid]  # Don't pop yet - we'll keep it with incubator state
        
        # Store original size before clearing members
        original_size = sp.size
        
        # Move all members to cluster 0
        moved_count = 0
        for member in sp.members:
            if cluster0.size >= cluster0.max_capacity:
                break  # Stop if capacity reached
            cluster0.add(member, current_generation)
            moved_count += 1
        
        # Mark species as incubator (species ID is deceased, but kept for reference)
        sp.species_state = "incubator"
        sp.members = []  # Clear members (they're now in cluster 0)
        incubator_species[sid] = sp
        
        moved_to_cluster0_events.append({
            "generation": current_generation,
            "species_id": sid,
            "action": "moved_to_cluster0",
            "new_state": "incubator",
            "size": original_size,  # Use original size before clearing members
            "moved_count": moved_count
        })
        logger.info(f"Moved species {sid} ({moved_count} members) to cluster 0 - state=incubator (NOT extinction)")
    
    # Remove incubator species from active species dict (they're preserved in historical_species)
    for sid in incubator_species:
        species.pop(sid, None)
    
    return species, extinction_events, moved_to_cluster0_events, incubator_species
