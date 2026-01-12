"""
merging.py

Island merging logic for speciation.
"""

from typing import Dict, List, Tuple, Optional

from .species import Individual, Species, generate_species_id
from .distance import ensemble_distance

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def merge_islands(
    sp1: Species,
    sp2: Species,
    current_generation: int,
    theta_sim: float = 0.4,
    max_capacity: int = 100,
    logger=None
) -> Species:
    """
    Merge two islands into a single species.
    
    Merging combines two similar species into one:
    - Members: All members from both species (deduplicated by ID)
    - Leader: Highest-fitness individual (from either species)
    - Mode: Reset to DEFAULT (forces re-adaptation)
    - Radius: Constant theta_sim (same for all species)
    - Stagnation: Reset to 0 (fresh start for merged species)
    - Origin: "merge" with parent_ids = [sp1.id, sp2.id]
    
    If total members exceed max_capacity (100), keeps highest-fitness individuals.
    
    Note: Creates a NEW species ID (not reusing sp1.id or sp2.id) to avoid confusion.
    
    Args:
        sp1: First species to merge
        sp2: Second species to merge
        current_generation: Current generation number
        theta_sim: Constant radius for the merged species (default: 0.4)
        max_capacity: Maximum members after merge (default: 100)
        logger: Optional logger instance
    
    Returns:
        New merged Species with cluster_origin="merge" and parent_ids=[sp1.id, sp2.id]
    """
    if logger is None:
        logger = get_logger("IslandMerging")
    
    # Combine members, deduplicating by ID
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
    
    # Create merged species with NEW ID (not reusing old IDs)
    merged = Species(
        id=generate_species_id(),  # New ID for clarity
        leader=new_leader,
        members=combined,
        radius=theta_sim,  # Constant radius for all species
        stagnation=0,
        max_fitness=max(sp1.leader.fitness, sp2.leader.fitness),
        species_state="active",
        created_at=current_generation,
        last_improvement=current_generation,
        cluster_origin="merge",  # Created via merge
        parent_ids=[sp1.id, sp2.id]  # Both parent IDs
    )
    
    # Update species assignments
    for m in combined:
        m.species_id = merged.id
    
    logger.info(f"Merged species {sp1.id} + {sp2.id} -> {merged.id} ({merged.size} members)")
    return merged


def process_merges(
    species: Dict[int, Species],
    theta_merge: float = 0.1,
    theta_sim: float = 0.2,
    min_stability_gens: int = 3,
    current_gen: int = 0,
    max_capacity: int = 100,
    max_merges_per_gen: int = 3,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
    logger=None
) -> Tuple[Dict[int, Species], List[Dict]]:
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
    - Uses constant theta_sim radius
    - Has cluster_origin="merge" and parent_ids=[id1, id2]
    - Truncates to max_capacity if needed
    
    Process iteratively (up to max_merges_per_gen) until no more merges found.
    
    Args:
        species: Dict of species (modified in-place)
        theta_merge: Merge distance threshold (must be < theta_sim)
        theta_sim: Constant radius for merged species
        min_stability_gens: Minimum age for species to be mergeable
        current_gen: Current generation number
        max_capacity: Maximum members after merge (default: 100)
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
        # Find merge candidates: pairs with leader distance <= theta_merge and both stable
        merge_pairs = []
        species_list = list(species.items())
        for i, (id1, sp1) in enumerate(species_list):
            for j, (id2, sp2) in enumerate(species_list[i + 1:], start=i + 1):
                if sp1.leader.embedding is None or sp2.leader.embedding is None:
                    continue
                dist = ensemble_distance(
                    sp1.leader.embedding, sp2.leader.embedding,
                    sp1.leader.phenotype, sp2.leader.phenotype,
                    w_genotype, w_phenotype
                )
                if dist <= theta_merge:
                    sp1_stable = (current_gen - sp1.created_at) >= min_stability_gens
                    sp2_stable = (current_gen - sp2.created_at) >= min_stability_gens
                    if sp1_stable and sp2_stable:
                        merge_pairs.append((id1, id2))
        
        if not merge_pairs:
            break
        
        id1, id2 = merge_pairs[0]
        if id1 not in species or id2 not in species:
            continue
        
        sp1, sp2 = species[id1], species[id2]
        merged = merge_islands(sp1, sp2, current_gen, theta_sim, max_capacity, logger)
        
        # Remove old species and add merged
        species.pop(id1, None)
        species.pop(id2, None)
        species[merged.id] = merged
        
        events.append({
            "generation": current_gen,
            "merged": (id1, id2),
            "result_id": merged.id,
            "cluster_origin": "merge",
            "parent_ids": [id1, id2]
        })
        merges_done += 1
    
    return species, events
