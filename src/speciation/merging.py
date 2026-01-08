"""
merging.py

Island merging logic for speciation.
"""

from typing import Dict, List, Tuple, Optional

from .island import Individual, Species, IslandMode
from .distance import semantic_distance

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def detect_merge_candidates(species: Dict[int, Species], theta_merge: float = 0.2,
                            min_stability_gens: int = 3, current_gen: int = 0, logger=None) -> List[Tuple[int, int]]:
    """
    Find pairs of species that should merge.
    
    Two species are candidates for merging if:
    1. Leader embeddings are semantically close (distance < theta_merge)
    2. Both species are stable (have existed for min_stability_gens generations)
    
    Stability requirement prevents merging of newly created species too quickly,
    giving them time to establish their own identity before combining.
    
    Args:
        species: Dict of all current species
        theta_merge: Semantic distance threshold for merging (must be < theta_sim)
        min_stability_gens: Minimum generations a species must exist before merging
        current_gen: Current generation number
        logger: Optional logger instance
    
    Returns:
        List of (species_id1, species_id2) pairs to merge
    """
    if logger is None:
        logger = get_logger("IslandMerging")
    
    merge_pairs = []
    species_list = list(species.items())
    
    # Check all pairs of species
    for i, (id1, sp1) in enumerate(species_list):
        for j, (id2, sp2) in enumerate(species_list[i + 1:], start=i + 1):
            # Skip pairs with missing embeddings
            if sp1.leader.embedding is None or sp2.leader.embedding is None:
                continue
            
            # Check if leaders are close enough to merge
            dist = semantic_distance(sp1.leader.embedding, sp2.leader.embedding)
            if dist < theta_merge:
                # Check stability: both must have existed long enough
                sp1_stable = (current_gen - sp1.created_at) >= min_stability_gens
                sp2_stable = (current_gen - sp2.created_at) >= min_stability_gens
                if sp1_stable and sp2_stable:
                    merge_pairs.append((id1, id2))
    
    return merge_pairs


def merge_islands(sp1: Species, sp2: Species, current_generation: int,
                  max_capacity: int = 50, logger=None) -> Species:
    """
    Merge two islands into a single species.
    
    Merging combines two similar species into one:
    - Members: All members from both species (deduplicated by prompt)
    - Leader: Highest-fitness individual (usually from one of the two)
    - Mode: Reset to DEFAULT (forces re-adaptation)
    - Radius: Max of the two radii (use larger threshold)
    - Stagnation: Reset to 0 (fresh start for merged species)
    
    If total members exceed max_capacity, keeps highest-fitness individuals.
    
    Args:
        sp1: First species to merge
        sp2: Second species to merge
        current_generation: Current generation number
        max_capacity: Maximum members after merge (trims if exceeded)
        logger: Optional logger instance
    
    Returns:
        New merged Species (uses sp1.id as result ID)
    """
    if logger is None:
        logger = get_logger("IslandMerging")
    
    # Combine members, deduplicating by prompt
    seen = set()
    combined = []
    for m in sp1.members + sp2.members:
        if m.id not in seen:
            combined.append(m)
            seen.add(m.id)
    
    # Sort by fitness and trim to capacity
    combined = sorted(combined, key=lambda x: x.fitness, reverse=True)[:max_capacity]
    # Ensure leader is in the combined list
    new_leader = max([sp1.leader, sp2.leader], key=lambda x: x.fitness)
    
    if new_leader not in combined:
        combined.insert(0, new_leader)
        combined = combined[:max_capacity]
    
    # Create merged species
    merged = Species(
        id=sp1.id, leader=new_leader, members=combined, mode=IslandMode.DEFAULT,
        radius=max(sp1.radius, sp2.radius), stagnation_counter=0,
        created_at=current_generation, last_improvement=current_generation
    )
    
    # Update species assignments
    for m in combined:
        m.species_id = merged.id
    
    logger.info(f"Merged species {sp1.id} + {sp2.id} → {merged.id} ({merged.size} members)")
    return merged


def process_merges(species: Dict[int, Species], theta_merge: float = 0.2,
                   min_stability_gens: int = 3, current_gen: int = 0,
                   max_capacity: int = 50, max_merges_per_gen: int = 3, logger=None) -> Tuple[Dict[int, Species], List[Dict]]:
    """
    Process all species merges for a generation.
    
    Merging combines similar species to prevent excessive fragmentation.
    Two species merge if:
    1. Leader distance < theta_merge (very similar)
    2. Both species are stable (existed for min_stability_gens)
    
    Merged species:
    - Combines all members (deduplicated)
    - Keeps highest-fitness leader
    - Resets to DEFAULT mode
    - Truncates to max_capacity if needed
    
    Process iteratively (up to max_merges_per_gen) until no more merges found.
    
    Args:
        species: Dict of species (modified in-place)
        theta_merge: Merge distance threshold (must be < theta_sim)
        min_stability_gens: Minimum age for species to be mergeable
        current_gen: Current generation number
        max_capacity: Maximum members after merge
        max_merges_per_gen: Maximum merges per generation (prevents excessive merging)
        logger: Optional logger instance
    
    Returns:
        Tuple of (updated_species, merge_events)
    """
    if logger is None:
        logger = get_logger("IslandMerging")
    
    events = []
    merges_done = 0
    
    while merges_done < max_merges_per_gen:
        pairs = detect_merge_candidates(species, theta_merge, min_stability_gens, current_gen, logger)
        if not pairs:
            break
        
        id1, id2 = pairs[0]
        if id1 not in species or id2 not in species:
            continue
        
        sp1, sp2 = species[id1], species[id2]
        merged = merge_islands(sp1, sp2, current_gen, max_capacity, logger)
        
        species.pop(id1, None)
        species.pop(id2, None)
        species[merged.id] = merged
        
        events.append({"generation": current_gen, "merged": (id1, id2), "result_id": merged.id})
        merges_done += 1
    
    return species, events


def should_merge(sp1: Species, sp2: Species, theta_merge: float, min_stability_gens: int, current_gen: int) -> bool:
    """
    Check if two species should merge based on similarity and stability.
    
    Two species merge if:
    1. Leader distance < theta_merge (very similar semantically)
    2. Both species stable (age >= min_stability_gens)
    
    Args:
        sp1: First species
        sp2: Second species
        theta_merge: Merge distance threshold
        min_stability_gens: Minimum stability age
        current_gen: Current generation number
    
    Returns:
        True if species should merge, False otherwise
    """
    # Check for embeddings
    if sp1.leader.embedding is None or sp2.leader.embedding is None:
        return False
    # Check stability requirement
    if (current_gen - sp1.created_at) < min_stability_gens or (current_gen - sp2.created_at) < min_stability_gens:
        return False
    # Check distance threshold
    return semantic_distance(sp1.leader.embedding, sp2.leader.embedding) < theta_merge

