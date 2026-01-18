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
    historical_species: Optional[Dict[int, Species]] = None,
    logger=None
) -> Tuple[Dict[int, Species], List[Dict], List[Individual], Dict[int, Species]]:
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
    
    Frozen species can merge with active or other frozen species.
    When species merge, BOTH parent species become extinct (moved to historical_species).
    The merged species is a new species with a new ID.
    
    Process iteratively until no more merge candidates are found.
    All eligible merges happen in a single generation.
    
    Args:
        species: Dict of active and frozen species (modified in-place)
        theta_merge: Merge distance threshold (must be < theta_sim)
        theta_sim: Constant radius for merged species
        min_stability_gens: Minimum age for species to be mergeable
        current_gen: Current generation number
        max_capacity: Maximum members after merge (default: 100)
        max_merges_per_gen: Deprecated parameter (kept for backward compatibility, not used)
        historical_species: Optional dict for storing extinct parent species
        logger: Optional logger instance
    
    Returns:
        Tuple of (updated_species, merge_events, outliers, extinct_parents)
        - updated_species: Dict of species after merging (parents removed, merged species added)
        - merge_events: List of merge event dictionaries
        - outliers: List of Individual objects outside radius after merge (to be moved to cluster 0 by caller)
        - extinct_parents: Dict of parent species that became extinct via merging (to be moved to historical_species)
    """
    if logger is None:
        logger = get_logger("IslandMerging")
    
    events = []
    all_outliers = []  # Collect all outliers from merges
    extinct_parents = {}  # Track parent species that became extinct via merging
    
    # Combine active and frozen species for merge candidate search
    # Frozen species can merge with active or other frozen species
    # Both active and frozen species are "alive" - only difference is parent selection preference
    # Note: Frozen species are now in the active species dict (not historical_species)
    all_species_for_merging = dict(species)  # Start with active and frozen species
    
    # BACKWARD COMPATIBILITY: Check historical_species for any frozen species
    # This handles state files from before the refactoring where frozen species were in historical_species
    # TODO: Remove this after confirming all old state files have been migrated (track usage via logs)
    frozen_in_historical_count = 0
    if historical_species:
        for sid, sp in historical_species.items():
            if sp.species_state == "frozen" and sp.leader and sp.leader.embedding is not None:
                # Frozen species preserve all members from when they were active
                all_species_for_merging[sid] = sp
                frozen_in_historical_count += 1
    
    if frozen_in_historical_count > 0:
        logger.warning(
            f"Found {frozen_in_historical_count} frozen species in historical_species (backward compatibility). "
            f"This indicates an old state file format. Frozen species should now be in the active species dict. "
            f"Consider migrating your state files."
        )
    
    # Continue merging until no more candidates are found
    while True:
        # Find merge candidates: pairs with leader distance < theta_merge and both stable
        merge_pairs = []
        species_list = list(all_species_for_merging.items())
        for i, (id1, sp1) in enumerate(species_list):
            for j, (id2, sp2) in enumerate(species_list[i + 1:], start=i + 1):
                if sp1.leader.embedding is None or sp2.leader.embedding is None:
                    continue
                dist = ensemble_distance(
                    sp1.leader.embedding, sp2.leader.embedding,
                    sp1.leader.phenotype, sp2.leader.phenotype,
                    w_genotype, w_phenotype
                )
                if dist < theta_merge:
                    sp1_stable = (current_gen - sp1.created_at) >= min_stability_gens
                    sp2_stable = (current_gen - sp2.created_at) >= min_stability_gens
                    if sp1_stable and sp2_stable:
                        merge_pairs.append((id1, id2, sp1.species_state, sp2.species_state))
        
        if not merge_pairs:
            break
        
        id1, id2, state1, state2 = merge_pairs[0]
        # Get species from appropriate dict (active or historical)
        sp1 = all_species_for_merging.get(id1)
        sp2 = all_species_for_merging.get(id2)
        
        if not sp1 or not sp2:
            continue
        
        merged, outliers = merge_islands(sp1, sp2, current_gen, theta_sim, max_capacity, w_genotype, w_phenotype, logger)
        
        # Remove old species from active dict
        species.pop(id1, None)
        species.pop(id2, None)
        # Remove from all_species_for_merging too
        all_species_for_merging.pop(id1, None)
        all_species_for_merging.pop(id2, None)
        
        # Add merged species (always active after merge)
        species[merged.id] = merged
        all_species_for_merging[merged.id] = merged
        
        # Mark both parent species as extinct (they will be moved to historical_species by caller)
        sp1.species_state = "extinct"
        sp2.species_state = "extinct"
        
        extinct_parents[id1] = sp1
        extinct_parents[id2] = sp2
        logger.info(f"Parent species {id1} and {id2} became extinct via merge -> new species {merged.id}")
        
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
        logger.info(f"Completed merge {len(events)}: {id1}+{id2}->{merged.id} (total merges so far: {len(events)})")
    
    logger.info(f"Merge process complete: {len(events)} merges performed in generation {current_gen}")
    if extinct_parents:
        logger.info(f"{len(extinct_parents)} parent species became extinct via merging: {sorted(extinct_parents.keys())}")
    return species, events, all_outliers, extinct_parents
