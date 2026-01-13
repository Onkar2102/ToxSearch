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
    theta_sim: float = 0.2,
    max_capacity: int = 100,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
    logger=None
) -> Tuple[Species, List[Individual]]:
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
        theta_sim: Constant radius for the merged species (default: 0.2, matches config.py)
        max_capacity: Maximum members after merge (default: 100)
        logger: Optional logger instance
    
    Returns:
        Tuple of (merged Species, list of outliers outside radius)
        - merged: New merged Species with cluster_origin="merge" and parent_ids=[sp1.id, sp2.id]
        - outliers: List of Individual objects outside radius (to be moved to cluster 0 by caller)
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
    # Select new leader as highest fitness from ALL combined members (not just old leaders)
    new_leader = combined[0] if combined else max([sp1.leader, sp2.leader], key=lambda x: x.fitness)
    
    # Verify all members are within radius of new leader (post-merge radius verification)
    members_within_radius = []
    members_outside_radius = []
    
    for member in combined:
        if member.id == new_leader.id:
            # Leader always stays
            members_within_radius.append(member)
            continue
        
        if member.embedding is None or new_leader.embedding is None:
            # Members without embeddings go to cluster 0 (handled by caller)
            members_outside_radius.append(member)
            continue
        
        dist = ensemble_distance(
            member.embedding, new_leader.embedding,
            member.phenotype, new_leader.phenotype,
            w_genotype, w_phenotype
        )
        
        if dist < theta_sim:
            members_within_radius.append(member)
        else:
            members_outside_radius.append(member)
    
    # Create merged species with only members within radius
    merged = Species(
        id=generate_species_id(),  # New ID for clarity
        leader=new_leader,
        members=members_within_radius,
        radius=theta_sim,  # Constant radius for all species
        stagnation=0,
        max_fitness=new_leader.fitness,
        species_state="active",
        created_at=current_generation,
        last_improvement=current_generation,
        cluster_origin="merge",  # Created via merge
        parent_ids=[sp1.id, sp2.id]  # Both parent IDs
    )
    
    # Update species assignments for members within radius
    for m in members_within_radius:
        m.species_id = merged.id
    
    if members_outside_radius:
        logger.warning(f"Merge {sp1.id}+{sp2.id}->{merged.id}: {len(members_outside_radius)} members outside radius of new leader (will be moved to cluster 0 by caller)")
    
    logger.info(f"Merged species {sp1.id} + {sp2.id} -> {merged.id} ({merged.size} members within radius, {len(members_outside_radius)} outside)")
    return merged, members_outside_radius  # Return outliers for caller to handle


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
) -> Tuple[Dict[int, Species], List[Dict], List[Individual]]:
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
        Tuple of (updated_species, merge_events, outliers)
        - updated_species: Dict of species after merging
        - merge_events: List of merge event dictionaries
        - outliers: List of Individual objects outside radius after merge (to be moved to cluster 0 by caller)
    """
    if logger is None:
        logger = get_logger("IslandMerging")
    
    events = []
    merges_done = 0
    all_outliers = []  # Collect all outliers from merges
    
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
        merged, outliers = merge_islands(sp1, sp2, current_gen, theta_sim, max_capacity, w_genotype, w_phenotype, logger)
        
        # Remove old species and add merged
        species.pop(id1, None)
        species.pop(id2, None)
        species[merged.id] = merged
        
        # Collect outliers for caller to handle
        if outliers:
            all_outliers.extend(outliers)
            logger.debug(f"Merge {id1}+{id2}->{merged.id}: {len(outliers)} outliers need to be moved to cluster 0")
        
        events.append({
            "generation": current_gen,
            "merged": (id1, id2),
            "result_id": merged.id,
            "cluster_origin": "merge",
            "parent_ids": [id1, id2]
        })
        merges_done += 1
    
    return species, events, all_outliers
