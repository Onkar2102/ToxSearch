"""
run_speciation.py

Main entry point functions for Dynamic Islands speciation framework.
All functionality is provided as module-level functions (no classes).
"""

import json
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path

from .config import SpeciationConfig
from .species import Individual, Species, SpeciesIdGenerator
from .embeddings import compute_and_save_embeddings, remove_embeddings_from_temp, get_embedding_model
from .leader_follower import leader_follower_clustering
from .reserves import Cluster0, CLUSTER_0_ID
from .merging import process_merges
from .extinction import process_extinctions
from .metrics import SpeciationMetricsTracker, log_generation_summary
from .genome_tracker import GenomeTracker
from .validation import validate_speciation_consistency, analyze_distance_distribution

from utils import get_custom_logging
from utils import get_system_utils

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _ = get_system_utils()


# Global state (replaces class instance)
_state: Optional[Dict[str, Any]] = None


def _init_state(config: Optional[SpeciationConfig] = None, logger=None) -> None:
    """Initialize global state."""
    global _state
    if _state is None:
        _state = {
            "config": config or SpeciationConfig(),
            "logger": logger or get_logger("Speciation"),
            "species": {},  # Active species only (state="active")
            "historical_species": {},  # Frozen and incubator species (preserved for reference)
            "cluster0": Cluster0(
                min_cluster_size=(config or SpeciationConfig()).cluster0_min_cluster_size,
                theta_sim=(config or SpeciationConfig()).theta_sim,
                max_capacity=(config or SpeciationConfig()).cluster0_max_capacity,
                logger=logger or get_logger("Speciation")
            ),
            "global_best": None,
            "metrics_tracker": SpeciationMetricsTracker(logger=logger or get_logger("Speciation")),
            "_current_gen_events": {"speciation": 0, "merge": 0, "extinction": 0, "moved_to_cluster0": 0},
            "_embedding_model": None,
            "_archived_count": 0
        }
        _state["logger"].info(f"Speciation initialized: theta_sim={_state['config'].theta_sim}, species_capacity={_state['config'].species_capacity}")


def _get_state() -> Dict[str, Any]:
    """Get global state, initializing if needed."""
    if _state is None:
        _init_state()
    return _state


def _archive_individuals(individuals: List[Individual], generation: int, reason: str) -> None:
    """Archive individuals to archive.json."""
    if not individuals:
        return
    
    state = _get_state()
    state["_archived_count"] += len(individuals)
    logger = state["logger"]
    
    try:
        outputs_path = get_outputs_path()
        archive_path = outputs_path / "archive.json"
        
        # Load existing archive
        if archive_path.exists():
            with open(archive_path, 'r', encoding='utf-8') as f:
                archive = json.load(f)
        else:
            archive = []
        
        # Add new entries
        for ind in individuals:
            if hasattr(ind, 'to_genome'):
                entry = ind.to_genome()
                # Ensure entry has required fields
                if not entry:
                    entry = {}
                if "id" not in entry:
                    entry["id"] = ind.id
                if "prompt" not in entry and hasattr(ind, 'prompt'):
                    entry["prompt"] = ind.prompt
            else:
                # Fallback: create minimal entry
                entry = {"id": ind.id}
                if hasattr(ind, 'prompt'):
                    entry["prompt"] = ind.prompt
            
            entry["archived_at_generation"] = generation
            entry["archive_reason"] = reason
            # Preserve fitness if available
            if hasattr(ind, 'fitness') and "fitness" not in entry:
                entry["fitness"] = ind.fitness
            if hasattr(ind, 'species_id') and "species_id" not in entry:
                entry["species_id"] = ind.species_id
            # Set initial_state for operator effectiveness metrics
            # Genomes archived due to capacity limits are non-elites
            if "initial_state" not in entry:
                if "capacity" in reason.lower():
                    entry["initial_state"] = "non-elite"
                else:
                    # For other reasons (extinction, etc.), preserve existing or default to elite
                    entry["initial_state"] = entry.get("initial_state", "elite")
            archive.append(entry)
        
        # Save updated archive
        with open(archive_path, 'w', encoding='utf-8') as f:
            json.dump(archive, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Archived {len(individuals)} individuals ({reason}) to archive.json")
        
    except Exception as e:
        logger.warning(f"Failed to archive individuals: {e}")


def process_generation(population: List[Dict[str, Any]], current_generation: int,
                       mutate_fn: Optional[Callable] = None, temp_path: Optional[str] = None,
                       config: Optional[SpeciationConfig] = None, logger=None) -> Tuple[Dict[int, Species], Cluster0]:
    """
    Process a single generation with full speciation pipeline.
    
    Pipeline steps:
    1. Compute embeddings (adds "prompt_embedding" field to temp.json)
    2. Leader-Follower clustering (assigns genomes to species, updates leaders immediately)
    3. Enforce capacities (cluster 0 and species capacity enforcement)
    4. Island merging (combine similar species)
    5. Freeze stagnant species and move small species to cluster 0
    6. Record metrics and save state
    
    Args:
        population: List of genome dictionaries (with prompts and fitness)
        current_generation: Current generation number
        mutate_fn: Optional mutation function (for repopulation)
        temp_path: Optional path to temp.json
        config: Optional SpeciationConfig (uses defaults if None)
        logger: Optional logger instance
        
    Returns:
        Tuple of (species_dict, cluster0)
    """
    _init_state(config, logger)
    state = _get_state()
    
    state["logger"].info(f"=== Speciation Generation {current_generation} ===")
    state["_current_gen_events"] = {"speciation": 0, "merge": 0, "extinction": 0, "moved_to_cluster0": 0}
    state["_archived_count"] = 0
    
    # Initialize genome tracker for audit trail
    genome_tracker = GenomeTracker(current_generation, logger=state["logger"])
    state["_genome_tracker"] = genome_tracker
    
    # Auto-load previous state if not first generation
    if current_generation > 0:
        outputs_path = get_outputs_path()
        state_path = str(outputs_path / "speciation_state.json")
        if Path(state_path).exists():
            load_state(state_path)
            state["logger"].info("Restored speciation state from previous generation")
    
    # Step 1: Compute and save embeddings to temp.json
    if temp_path is None:
        outputs_path = get_outputs_path()
        temp_path = str(outputs_path / "temp.json")
    
    compute_and_save_embeddings(
        temp_path=temp_path,
        model_name=state["config"].embedding_model,
        batch_size=state["config"].embedding_batch_size,
        logger=state["logger"]
    )
    
    # Step 2: Leader-Follower clustering
    outputs_path = get_outputs_path()
    speciation_state_path = str(outputs_path / "speciation_state.json")
    
    state["species"], species_with_new_members = leader_follower_clustering(
        temp_path=temp_path,
        speciation_state_path=speciation_state_path,
        theta_sim=state["config"].theta_sim,
        current_generation=current_generation,
        w_genotype=state["config"].w_genotype,
        w_phenotype=state["config"].w_phenotype,
        logger=state["logger"]
    )
    
    # Step 2a: Sync cluster 0 with reserves.json
    # After leader_follower_clustering, outliers that formed species are updated in reserves.json
    # but may still be in the in-memory Cluster0.members. We need to sync them.
    outputs_path = get_outputs_path()
    reserves_path = outputs_path / "reserves.json"
    if reserves_path.exists():
        try:
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves_genomes = json.load(f)
            
            # Get IDs of individuals that are now in species (species_id > 0)
            species_member_ids = set()
            for sp in state["species"].values():
                for member in sp.members:
                    species_member_ids.add(member.id)
            
            # Remove from cluster0.members any that are now in species
            removed_count = 0
            cluster0_members_to_keep = []
            for cm in state["cluster0"].members:
                if cm.individual.id not in species_member_ids:
                    cluster0_members_to_keep.append(cm)
                else:
                    removed_count += 1
            
            state["cluster0"].members = cluster0_members_to_keep
            
            # Add new individuals from reserves.json that aren't tracked yet
            existing_cluster0_ids = {cm.individual.id for cm in state["cluster0"].members}
            added_count = 0
            for genome in reserves_genomes:
                if genome.get("species_id", CLUSTER_0_ID) == CLUSTER_0_ID:
                    genome_id = genome.get("id")
                    if genome_id and genome_id not in existing_cluster0_ids:
                        # Create Individual from genome and add to cluster0
                        outlier_ind = Individual.from_genome(genome)
                        if outlier_ind.embedding is not None:
                            state["cluster0"].add(outlier_ind, current_generation)
                            added_count += 1
            
            if removed_count > 0 or added_count > 0:
                state["logger"].info(f"Synced cluster 0: removed {removed_count} (now in species), added {added_count} (from reserves.json)")
        except Exception as e:
            state["logger"].warning(f"Failed to sync cluster 0 with reserves.json: {e}")
    
    # Step 2b: Post-processing - remove genomes outside radius after all variants processed
    # This applies to BOTH Generation 0 and Generation N:
    # - When leader was updated during clustering, remaining genomes were assigned using updated leader
    # - For Generation N: Species formed from outliers may have new leaders (highest fitness)
    # - For both generations: We need to verify all members are still within radius of the (possibly updated) leader
    # - This ensures species cohesion: all members must be within theta_sim of the current leader
    from .distance import ensemble_distance
    
    for sid in list(species_with_new_members):
        if sid not in state["species"]:
            continue
        sp = state["species"][sid]
        if sp.leader is None or sp.leader.embedding is None:
            continue
        
        # Recalculate distances and remove members outside radius
        members_to_keep = []
        members_to_remove = []
        
        for member in sp.members:
            if member.id == sp.leader.id:
                # Leader always stays
                members_to_keep.append(member)
                continue
            
            if member.embedding is None:
                # Members without embeddings go to cluster 0
                members_to_remove.append(member)
                continue
            
            dist = ensemble_distance(
                member.embedding, sp.leader.embedding,
                member.phenotype, sp.leader.phenotype,
                state["config"].w_genotype, state["config"].w_phenotype
            )
            
            if dist < state["config"].theta_sim:
                members_to_keep.append(member)
            else:
                members_to_remove.append(member)
        
        # Update species members
        if members_to_remove:
            state["logger"].debug(f"Species {sid}: removing {len(members_to_remove)} members outside radius")
            sp.members = members_to_keep
            
            # Move removed members to cluster 0
            for member in members_to_remove:
                if state["cluster0"].size < state["config"].cluster0_max_capacity:
                    state["cluster0"].add(member, current_generation)
            
            # Check if species is now empty (only leader)
            if len(sp.members) <= 1:
                # Species is empty (only leader) - move to incubator state
                sp.species_state = "incubator"
                sp.members = []  # Clear members
                
                # Move leader to cluster 0
                if state["cluster0"].size < state["config"].cluster0_max_capacity:
                    state["cluster0"].add(sp.leader, current_generation)
                
                # Remove from active species (will be preserved in speciation_state.json with incubator state)
                del state["species"][sid]
                state["logger"].info(f"Species {sid} became empty after radius cleanup - moved to incubator")
    
    # Step 3: Enforce capacities directly (species and cluster 0)
    # Only process species that received new members (optimization: unaffected species remain unchanged)
    for sid in species_with_new_members:
        if sid not in state["species"]:
            continue  # Species may have been removed
        sp = state["species"][sid]
        if sp.size > state["config"].species_capacity:
            # Sort members by fitness (descending)
            sp.members.sort(key=lambda x: x.fitness, reverse=True)
            # Remove excess genomes (keep top species_capacity)
            excess = sp.members[state["config"].species_capacity:]
            sp.members = sp.members[:state["config"].species_capacity]
            # Archive removed genomes
            _archive_individuals(excess, current_generation, "species_capacity_exceeded")
            # Track archival events
            if "_genome_tracker" in state:
                for ind in excess:
                    state["_genome_tracker"].log(
                        ind.id, "capacity_archived",
                        {"species_id": sid, "reason": "species_capacity", "capacity": state["config"].species_capacity}
                    )
            # Ensure leader is still in members (should be, but verify)
            if sp.leader not in sp.members and sp.members:
                sp.leader = max(sp.members, key=lambda x: x.fitness)
    
    # For cluster 0: enforce capacity
    if state["cluster0"].size > state["config"].cluster0_max_capacity:
        # Sort by fitness (descending)
        state["cluster0"].members.sort(key=lambda x: x.individual.fitness, reverse=True)
        # Remove excess (keep top cluster0_max_capacity)
        excess_members = state["cluster0"].members[state["config"].cluster0_max_capacity:]
        state["cluster0"].members = state["cluster0"].members[:state["config"].cluster0_max_capacity]
        # Archive removed genomes
        excess_individuals = [m.individual for m in excess_members]
        _archive_individuals(excess_individuals, current_generation, "cluster0_capacity_exceeded")
        # Track archival events
        if "_genome_tracker" in state:
            for ind in excess_individuals:
                state["_genome_tracker"].log(
                    ind.id, "capacity_archived",
                    {"reason": "cluster0_capacity", "capacity": state["config"].cluster0_max_capacity}
                )
    
    # Check for speciation events in cluster 0
    # Loop until no more species can form (handles multiple clusters in cluster 0)
    while True:
        new_species = state["cluster0"].check_speciation(current_generation)
        if new_species:
            state["species"][new_species.id] = new_species
            species_with_new_members.add(new_species.id)  # Add to set so fitness is recorded and radius is checked
            state["_current_gen_events"]["speciation"] += 1
            state["logger"].info(f"Species {new_species.id} formed from cluster 0 ({new_species.size} members)")
            # Track speciation events
            if "_genome_tracker" in state:
                for member in new_species.members:
                    state["_genome_tracker"].log(
                        member.id, "species_formed_from_cluster0",
                        {"species_id": new_species.id, "size": new_species.size}
                    )
        else:
            break
    
    # Step 4: Record fitness for species that received new members (optimization)
    for sid in species_with_new_members:
        if sid in state["species"]:
            state["species"][sid].record_fitness(current_generation)
    
    # Step 5: Island merging
    state["species"], merge_events, merge_outliers = process_merges(
        state["species"], 
        theta_merge=state["config"].theta_merge,
        theta_sim=state["config"].theta_sim,
        current_gen=current_generation,
        max_capacity=state["config"].species_capacity,
        w_genotype=state["config"].w_genotype,
        w_phenotype=state["config"].w_phenotype,
        logger=state["logger"]
    )
    state["_current_gen_events"]["merge"] = len(merge_events)
    
    # Move merge outliers to cluster 0 (post-merge radius verification)
    if merge_outliers:
        state["logger"].info(f"Moving {len(merge_outliers)} outliers from merges to cluster 0")
        for outlier in merge_outliers:
            if state["cluster0"].size < state["config"].cluster0_max_capacity:
                state["cluster0"].add(outlier, current_generation)
                outlier.species_id = CLUSTER_0_ID
                if "_genome_tracker" in state:
                    state["_genome_tracker"].log(
                        outlier.id, "merge_outlier_to_cluster0",
                        {"reason": "outside_radius_after_merge"}
                    )
            else:
                state["logger"].warning(f"Cluster 0 at capacity, cannot add merge outlier {outlier.id}")
    
    # Track merge events
    if "_genome_tracker" in state:
        for merge_event in merge_events:
            merged_ids = merge_event.get("merged", [])
            result_id = merge_event.get("result_id")
            if result_id and result_id in state["species"]:
                for member in state["species"][result_id].members:
                    state["_genome_tracker"].log(
                        member.id, "species_merged",
                        {"from_species": merged_ids, "to_species": result_id}
                    )
    
    # Step 6: Freeze stagnant species (extinction) and move small species to cluster 0 (not extinction)
    # CRITICAL: Pass elites_path so process_extinctions can use actual sizes from elites.json
    # This prevents species from being incorrectly moved to incubator based on stale in-memory sizes
    outputs_path = get_outputs_path()
    elites_path_for_extinction = str(outputs_path / "elites.json")
    state["species"], extinction_events, moved_to_cluster0_events, incubator_species = process_extinctions(
        state["species"],
        state["cluster0"],
        current_generation,
        species_stagnation=state["config"].species_stagnation,
        min_size=state["config"].min_island_size,
        elites_path=elites_path_for_extinction,  # Pass elites_path to use actual sizes
        logger=state["logger"]
    )
    # Only frozen species (stagnation-based) count as extinction events
    # Moving to cluster 0 (size-based) is NOT extinction, just reorganization
    state["_current_gen_events"]["extinction"] = len(extinction_events)
    state["_current_gen_events"]["moved_to_cluster0"] = len(moved_to_cluster0_events)
    
    # Move frozen species to historical_species for preservation
    # (process_extinctions already sets species_state="frozen" but keeps them in species dict)
    frozen_species_to_move = {sid: sp for sid, sp in state["species"].items() if sp.species_state == "frozen"}
    for sid, sp in frozen_species_to_move.items():
        state["historical_species"][sid] = sp
        del state["species"][sid]
        state["logger"].debug(f"Moved frozen species {sid} to historical_species")
    
    # Move incubator species to historical_species for tracking (just IDs, not full data)
    # (incubator species are returned separately by process_extinctions)
    # We keep them in historical_species but they won't be saved with full data - just tracked by ID
    for sid, sp in incubator_species.items():
        state["historical_species"][sid] = sp  # Keep for in-memory tracking
        state["logger"].debug(f"Moved incubator species {sid} to historical_species (will be tracked by ID only in save_state)")
    
    # Step 7: Update c-TF-IDF labels for all species
    from .labeling import update_species_labels
    update_species_labels(
        state["species"],
        current_generation=current_generation,
        n_words=10,
        logger=state["logger"]
    )
    
    # Step 8: Save critical state immediately (before distribution)
    # Save speciation_state.json with current species and cluster0 state
    # This ensures data is persisted to file, not just in-memory
    outputs_path = get_outputs_path()
    state_path = str(outputs_path / "speciation_state.json")
    save_state(state_path)
    state["logger"].debug("Saved speciation state before distribution (in-memory state persisted)")
    
    # Remove embeddings from temp.json after speciation is complete
    remove_embeddings_from_temp(temp_path=temp_path, logger=state["logger"])
    
    # Note: Metrics recording and validation are moved to AFTER distribution in run_speciation()
    # This ensures metrics and validation reflect the final state after genomes are distributed
    
    # Analyze distance distributions (can be done before distribution - uses in-memory species)
    outputs_path = get_outputs_path()
    distance_analysis = analyze_distance_distribution(
        state["species"], state["cluster0"], state["config"], logger=state["logger"]
    )
    
    # Save genome tracker
    if "_genome_tracker" in state:
        state["_genome_tracker"].save()
        tracker_summary = state["_genome_tracker"].get_summary()
        state["logger"].debug(f"Genome tracker: {tracker_summary['total_events']} events for {tracker_summary['unique_genomes']} genomes")
    
    return state["species"], state["cluster0"]


def distribute_genomes(
    temp_path: Optional[str] = None,
    north_star_metric: str = "toxicity",
    current_generation: Optional[int] = None) -> Dict[str, int]:
    """
    Distribute genomes from temp.json to elites.json and reserves.json.
    
    Distribution is based on species_id:
    - Genomes with species_id > 0 → elites.json (all species members)
    - Genomes with species_id == 0 → reserves.json (cluster 0 / reserves)
    - Genomes exceeding capacity → archive.json (non-elites)
    
    Sets initial_state for Table 4 metrics:
    - "elite": Genomes going to elites.json or reserves.json (reserves are elites)
    - "non-elite": Genomes archived due to capacity limits
    
    Args:
        temp_path: Path to temp.json file (defaults to outputs_path / "temp.json")
        north_star_metric: The metric to use for scoring (default: "toxicity")
        current_generation: Current generation number (for tracking archived variants)
        
    Returns:
        dict: Distribution statistics with elites_moved, reserves_moved, total_processed, archived_count
    """
    from utils.population_io import _extract_north_star_score
    from utils.constants import FileConstants
    
    state = _get_state()
    config = state["config"]
    logger = state["logger"]
    
    if temp_path is None:
        outputs_path = get_outputs_path()
        temp_path = str(outputs_path / "temp.json")
    
    temp_path_obj = Path(temp_path)
    if not temp_path_obj.exists():
        logger.warning("temp.json not found for distribution")
        return {"elites_moved": 0, "reserves_moved": 0, "total_processed": 0, "archived_count": 0}
    
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    archive_path = outputs_path / "archive.json"
    
    # Load genomes from temp.json
    with open(temp_path_obj, 'r', encoding='utf-8') as f:
        temp_genomes = json.load(f)
    
    if not temp_genomes:
        logger.debug("No genomes in temp.json to distribute")
        return {"elites_moved": 0, "reserves_moved": 0, "total_processed": 0, "archived_count": 0}
    
    elites_to_move = []
    reserves_to_move = []
    archived_genomes = []
    
    # Get current generation from state if not provided
    if current_generation is None:
        # Try to get from state or default to 0
        current_generation = state.get("_current_generation", 0)
    
    # Distribute genomes based on species_id
    for genome in temp_genomes:
        if not genome or not genome.get("prompt"):
            continue
        
        genome_id = genome.get("id")
        species_id = genome.get("species_id", CLUSTER_0_ID)
        genome_generation = genome.get("generation", current_generation)
        
        # Distribute based on species_id
        if species_id is not None and species_id > 0:
            # Goes to elites.json - all species members are elites
            genome["initial_state"] = "elite"
            elites_to_move.append(genome)
            logger.debug(f"Genome {genome_id} from species {species_id} → elites.json")
        else:
            # Goes to reserves.json initially (reserves are also elites)
            genome["initial_state"] = "elite"  # Reserves are elites
            genome["species_id"] = CLUSTER_0_ID
            reserves_to_move.append(genome)
            logger.debug(f"Genome {genome_id} from cluster 0 → reserves.json")
    
    # Save elites
    if elites_to_move:
        elites_to_save = []
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_to_save = json.load(f)
        
        elites_to_save.extend(elites_to_move)
        
        with open(elites_path, 'w', encoding='utf-8') as f:
            json.dump(elites_to_save, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Moved {len(elites_to_move)} genomes from species to elites.json")
    
    # Save reserves (with capacity limit)
    if reserves_to_move:
        reserves_to_save = []
        if reserves_path.exists():
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves_to_save = json.load(f)
        
        reserves_to_save.extend(reserves_to_move)
        reserves_to_save.sort(key=lambda g: _extract_north_star_score(g, north_star_metric), reverse=True)
        
        # Enforce capacity limit - archive excess genomes
        if len(reserves_to_save) > config.cluster0_max_capacity:
            excess_count = len(reserves_to_save) - config.cluster0_max_capacity
            excess_genomes = reserves_to_save[config.cluster0_max_capacity:]
            reserves_to_save = reserves_to_save[:config.cluster0_max_capacity]
            
            # Mark excess genomes as non-elite and archive them
            for excess_genome in excess_genomes:
                # Only track variants from current generation
                if excess_genome.get("generation") == current_generation:
                    excess_genome["initial_state"] = "non-elite"
                    archived_genomes.append(excess_genome)
                    logger.debug(f"Genome {excess_genome.get('id')} archived due to capacity (non-elite)")
            
            logger.debug(f"Reserves capacity exceeded, keeping top {config.cluster0_max_capacity} genomes, archiving {excess_count}")
        
        with open(reserves_path, 'w', encoding='utf-8') as f:
            json.dump(reserves_to_save, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Moved {len(reserves_to_move)} genomes to reserves.json (capacity: {len(reserves_to_save)})")
    
    # Archive genomes that were removed due to capacity
    if archived_genomes:
        archive_to_save = []
        if archive_path.exists():
            with open(archive_path, 'r', encoding='utf-8') as f:
                archive_to_save = json.load(f)
        
        archive_to_save.extend(archived_genomes)
        
        with open(archive_path, 'w', encoding='utf-8') as f:
            json.dump(archive_to_save, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Archived {len(archived_genomes)} genomes from current generation to archive.json")
    
    # Clear temp.json
    with open(temp_path_obj, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2, ensure_ascii=False)
    
    distribution_stats = {
        "elites_moved": len(elites_to_move),
        "reserves_moved": len(reserves_to_move),
        "total_processed": len(temp_genomes),
        "archived_count": len(archived_genomes)
    }
    
    logger.info(f"Distribution complete: {distribution_stats['total_processed']} genomes → "
                f"{distribution_stats['elites_moved']} elites, "
                f"{distribution_stats['reserves_moved']} reserves, "
                f"{distribution_stats['archived_count']} archived")
    
    return distribution_stats


def _update_genomes_with_species(genomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Update genomes with species IDs (0 for cluster 0, None if not found)."""
    state = _get_state()
    cluster0_ids = {ind.id for ind in state["cluster0"].individuals}
    species_ids = {}
    for sp in state["species"].values():
        for m in sp.members:
            species_ids[m.id] = sp.id
    
    for g in genomes:
        genome_id = g.get("id")
        if genome_id in cluster0_ids:
            g["species_id"] = CLUSTER_0_ID
        else:
            g["species_id"] = species_ids.get(genome_id)
    return genomes


def save_state(path: str) -> None:
    """Save state to file.
    
    Note: Config is NOT saved here as it's passed as project arguments or fixed constants.
    The cluster0 section only contains metadata (size, speciation_events), not full member data
    since reserves.json already stores the complete genome data for cluster 0.
    
    The size in speciation_state.json reflects the current total size of genomes in that species
    from elites.json (all generations), not just the members in the Species object (current generation only).
    
    Species storage strategy:
    - Active species (state="active") - full data saved (participate in evolution)
    - Frozen species (state="frozen") - full data saved (stagnated, excluded from parent selection)
    - Incubator species (state="incubator") - only species ID tracked (moved to cluster 0, just for tracking)
    """
    state = _get_state()
    logger = state["logger"]
    
    # Calculate actual sizes from elites.json for each species
    # This gives the true current size across all generations
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    species_sizes = {}
    elites_genomes = []  # Store for later use in species reconstruction
    
    if elites_path.exists():
        try:
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_genomes = json.load(f)
            
            # Count genomes per species
            for genome in elites_genomes:
                species_id = genome.get("species_id")
                if species_id is not None and species_id > 0:
                    species_sizes[species_id] = species_sizes.get(species_id, 0) + 1
        except Exception as e:
            logger.warning(f"Failed to calculate species sizes from elites.json: {e}")
    
    # Build species dict - only save full data for active and frozen species
    species_dict = {}
    incubator_ids = []  # Just track IDs for incubator species
    
    # Add active species (full data)
    for sid, sp in state["species"].items():
        if sp.species_state == "active":
            sp_dict = sp.to_dict()
            # Update size and member_ids from elites.json (source of truth after distribution)
            actual_size = species_sizes.get(sid, len(sp.members))
            sp_dict["size"] = actual_size
            # Get actual member IDs from elites.json (not from in-memory members which may be stale)
            actual_member_ids = [g.get("id") for g in elites_genomes if g.get("species_id") == sid]
            if actual_member_ids:
                sp_dict["member_ids"] = actual_member_ids
            # If no members found in elites.json but size > 0, keep original member_ids
            # (This shouldn't happen, but handle gracefully)
            species_dict[str(sid)] = sp_dict
        elif sp.species_state == "frozen":
            # Frozen species also get full data
            sp_dict = sp.to_dict()
            actual_size = species_sizes.get(sid, len(sp.members))
            sp_dict["size"] = actual_size
            # Get actual member IDs from elites.json
            actual_member_ids = [g.get("id") for g in elites_genomes if g.get("species_id") == sid]
            if actual_member_ids:
                sp_dict["member_ids"] = actual_member_ids
            species_dict[str(sid)] = sp_dict
    
    # Add historical species - only frozen get full data, incubator just IDs
    for sid, sp in state.get("historical_species", {}).items():
        if str(sid) not in species_dict:  # Avoid duplicates
            if sp.species_state == "frozen":
                # Frozen species get full data
                sp_dict = sp.to_dict()
                actual_size = species_sizes.get(sid, 0)  # Historical frozen species may have members
                sp_dict["size"] = actual_size
                # Get actual member IDs from elites.json
                actual_member_ids = [g.get("id") for g in elites_genomes if g.get("species_id") == sid]
                if actual_member_ids:
                    sp_dict["member_ids"] = actual_member_ids
                species_dict[str(sid)] = sp_dict
            elif sp.species_state == "incubator":
                # Incubator species - just track ID
                incubator_ids.append(sid)
    
    # CRITICAL FIX: Add species that exist in elites.json but are missing from in-memory state
    # This can happen if species were frozen/extinct in previous generations or state wasn't loaded properly
    from .species import Species
    from .distance import ensemble_distance
    
    for species_id, size in species_sizes.items():
        if str(species_id) not in species_dict and species_id not in incubator_ids:
            # Species exists in elites.json but not in state - need to reconstruct basic info
            logger.warning(f"Species {species_id} found in elites.json ({size} genomes) but missing from state - reconstructing basic entry")
            
            # Find leader (highest fitness genome in this species)
            species_genomes = [g for g in elites_genomes if g.get("species_id") == species_id]
            if species_genomes:
                # Sort by fitness to find leader
                def get_fitness(g):
                    if 'north_star_score' in g:
                        return g['north_star_score']
                    elif 'moderation_result' in g and g['moderation_result']:
                        google = g['moderation_result'].get('google', {})
                        if google and 'scores' in google:
                            return google['scores'].get('toxicity', 0)
                    return 0
                
                species_genomes.sort(key=get_fitness, reverse=True)
                leader_genome = species_genomes[0]
                leader_fitness = get_fitness(leader_genome)
                
                # Create minimal species entry (frozen state since it's not in active state)
                # This is a recovery mechanism - species should ideally be in state
                species_dict[str(species_id)] = {
                    "id": species_id,
                    "leader_id": leader_genome.get("id"),
                    "leader_prompt": leader_genome.get("prompt", ""),
                    "leader_embedding": leader_genome.get("prompt_embedding", []),
                    "leader_fitness": round(leader_fitness, 4),
                    "member_ids": [g.get("id") for g in species_genomes],  # All member IDs
                    "radius": state["config"].theta_sim,
                    "stagnation": 0,  # Unknown
                    "max_fitness": round(leader_fitness, 4),
                    "species_state": "frozen",  # Assume frozen since not in active state
                    "created_at": 0,  # Unknown
                    "last_improvement": 0,  # Unknown
                    "fitness_history": [round(leader_fitness, 4)],
                    "labels": [],
                    "label_history": [],
                    "cluster_origin": "unknown",
                    "parent_ids": None,
                    "size": size
                }
                logger.info(f"Reconstructed species {species_id} entry from elites.json (size={size}, state=frozen)")
    
    # Validate no duplicate leader IDs in active/frozen species
    leader_ids = []
    for sid_str, sp_dict in species_dict.items():
        leader_id = sp_dict.get("leader_id")
        if leader_id is not None:
            leader_ids.append(leader_id)
    
    from collections import Counter
    leader_id_counts = Counter(leader_ids)
    duplicates = {lid: count for lid, count in leader_id_counts.items() if count > 1}
    if duplicates:
        logger.warning(f"Duplicate leader IDs detected in active/frozen species: {duplicates}")
    
    # Get actual cluster 0 size from reserves.json (more accurate than in-memory size)
    reserves_path = outputs_path / "reserves.json"
    actual_cluster0_size = state["cluster0"].size  # Fallback to in-memory size
    if reserves_path.exists():
        try:
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves_genomes = json.load(f)
            actual_cluster0_size = len(reserves_genomes)
            logger.debug(f"Cluster 0 size: {actual_cluster0_size} (from reserves.json), in-memory: {state['cluster0'].size}")
        except Exception as e:
            logger.warning(f"Failed to read reserves.json for cluster 0 size: {e}")
    
    # Update cluster0 dict with actual size
    cluster0_dict = state["cluster0"].to_dict()
    cluster0_dict["size"] = actual_cluster0_size
    
    state_dict = {
        "species": species_dict,
        "incubators": sorted(incubator_ids),  # Just list of species IDs
        "cluster0": cluster0_dict,
        "cluster0_size_from_reserves": actual_cluster0_size,  # Store actual size from reserves.json
        "global_best_id": state["global_best"].id if state["global_best"] else None,
        "metrics": state["metrics_tracker"].to_dict()
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=2, ensure_ascii=False)
    
    # Count species by state for logging
    active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
    frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])
    frozen_count += len([sp for sp in state.get("historical_species", {}).values() if sp.species_state == "frozen"])
    incubator_count = len(incubator_ids)
    
    state["logger"].info(f"Saved speciation state to {path}: {active_count} active, {frozen_count} frozen, {incubator_count} incubator (IDs only)")


def load_state(path: str) -> bool:
    """
    Load state from file and restore species, cluster 0, and metrics.
    
    Species are loaded into two dictionaries:
    - state["species"]: Active species (state="active") - participate in evolution
    - state["historical_species"]: Frozen and incubator species (preserved for reference)
    
    Args:
        path: Path to speciation_state.json file
        
    Returns:
        True if loaded successfully, False otherwise
    """
    import numpy as np
    
    state = _get_state()
    logger = state["logger"]
    config = state["config"]
    
    state_path = Path(path)
    if not state_path.exists():
        logger.warning(f"Speciation state file not found: {path}")
        return False
    
    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            loaded_state = json.load(f)
        
        # Restore species - separate active from historical
        state["species"] = {}
        state["historical_species"] = {}
        max_species_id = 0
        
        # Load active and frozen species (full data)
        for sid_str, sp_dict in loaded_state.get("species", {}).items():
            sid = int(sid_str)
            max_species_id = max(max_species_id, sid)
            
            leader_embedding = None
            if sp_dict.get("leader_embedding"):
                leader_embedding = np.array(sp_dict["leader_embedding"])
            
            leader = Individual(
                id=sp_dict["leader_id"],
                prompt=sp_dict.get("leader_prompt", ""),
                fitness=sp_dict.get("leader_fitness", 0.0),
                embedding=leader_embedding,
                species_id=sid
            )
            
            species = Species(
                id=sid,
                leader=leader,
                members=[leader],
                radius=sp_dict.get("radius", config.theta_sim),
                stagnation=sp_dict.get("stagnation", 0),
                max_fitness=sp_dict.get("max_fitness", leader.fitness),
                species_state=sp_dict.get("species_state", "active"),
                created_at=sp_dict.get("created_at", 0),
                last_improvement=sp_dict.get("last_improvement", 0),
                fitness_history=sp_dict.get("fitness_history", []),
                labels=sp_dict.get("labels", []),
                label_history=sp_dict.get("label_history", []),
                cluster_origin=sp_dict.get("cluster_origin", "natural"),  # Default to "natural" if None
                parent_ids=sp_dict.get("parent_ids"),
                leader_distance=sp_dict.get("leader_distance", 0.0)
            )
            
            # Separate active from historical species
            if species.species_state == "active":
                state["species"][sid] = species
            else:
                # Frozen -> historical (full data preserved)
                state["historical_species"][sid] = species
        
        # Load incubator species IDs (just for tracking, no full data)
        incubator_ids = loaded_state.get("incubators", [])
        if incubator_ids:
            # Track incubator IDs but don't create Species objects
            # They're just for tracking purposes
            for sid in incubator_ids:
                max_species_id = max(max_species_id, sid)
            logger.debug(f"Loaded {len(incubator_ids)} incubator species IDs for tracking: {incubator_ids}")
        
        SpeciesIdGenerator.set_min_id(max_species_id + 1)
        
        active_count = len(state["species"])
        historical_count = len(state["historical_species"])
        logger.info(f"Loaded speciation state from {path}: {active_count} active, {historical_count} historical species")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load speciation state: {e}", exc_info=True)
        return False


def reset_speciation_module() -> None:
    """Reset the global speciation state (for testing or fresh start)."""
    global _state
    if _state is not None:
        config = _state["config"]
        logger = _state["logger"]
        _state["species"] = {}
        _state["historical_species"] = {}  # Also reset historical species
        _state["cluster0"] = Cluster0(
            min_cluster_size=config.cluster0_min_cluster_size,
            theta_sim=config.theta_sim,
            max_capacity=config.cluster0_max_capacity,
            logger=logger
        )
        _state["global_best"] = None
        _state["metrics_tracker"] = SpeciationMetricsTracker(logger=logger)
        _state["_current_gen_events"] = {"speciation": 0, "merge": 0, "extinction": 0, "moved_to_cluster0": 0}
        _state["_archived_count"] = 0
        SpeciesIdGenerator.reset()
        logger.info("Speciation module reset")
    else:
        _state = None


def run_speciation(
    temp_path: Optional[str] = None,
    current_generation: int = 0,
    config: Optional[SpeciationConfig] = None,
    log_file: Optional[str] = None,
    mutate_fn: Optional[Any] = None,
    north_star_metric: str = "toxicity") -> Dict[str, Any]:
    """
    Run speciation processing and distribution for a single generation.
    
    This is the main entry point for speciation, similar to run_evolution().
    
    Args:
        temp_path: Path to temp.json file with evaluated genomes.
        current_generation: Current generation number
        config: Optional SpeciationConfig (uses defaults if None)
        log_file: Optional log file path
        mutate_fn: Optional mutation function for repopulation
        north_star_metric: The metric to use for scoring (default: "toxicity")
        
    Returns:
        Dict with speciation and distribution results
    """
    logger = get_logger("RunSpeciation", log_file)
    logger.info("Starting speciation: generation=%d", current_generation)
    
    if temp_path is None:
        outputs_path = get_outputs_path()
        temp_path = str(outputs_path / "temp.json")
    
    temp_path_obj = Path(temp_path)
    if not temp_path_obj.exists():
        logger.warning("Temp file not found: %s - updating EvolutionTracker with current state", temp_path)
        # Even with no temp file, update EvolutionTracker with current speciation state
        _init_state(config, logger)
        state = _get_state()
        
        # Load previous state if available
        if current_generation > 0:
            outputs_path_state = get_outputs_path()
            state_path = str(outputs_path_state / "speciation_state.json")
            if Path(state_path).exists():
                load_state(state_path)
        
        # Get actual reserves size from file
        outputs_path = get_outputs_path()
        reserves_path = outputs_path / "reserves.json"
        actual_reserves_size = state["cluster0"].size
        if reserves_path.exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                actual_reserves_size = len(reserves_genomes)
            except Exception:
                pass  # Use cluster0.size as fallback
        
        # Calculate species count: active + frozen, but EXCLUDE incubator
        active_count = len(state["species"])
        frozen_count = len([sp for sp in state.get("historical_species", {}).values() if sp.species_state == "frozen"])
        total_species_count = active_count + frozen_count
        
        # Create result with current state
        no_temp_result = {
            "species_count": total_species_count,
            "active_species_count": active_count,
            "reserves_size": actual_reserves_size,
            "speciation_events": 0,
            "merge_events": 0,
            "extinction_events": 0,
            "archived_count": 0,
            "genomes_updated": 0,
            "elites_moved": 0,
            "reserves_moved": 0,
            "success": True,  # No error, just no new genomes
            "error": None
        }
        
        # Update EvolutionTracker with current state
        try:
            outputs_path_tracker = get_outputs_path()
            evolution_tracker_path = str(outputs_path_tracker / "EvolutionTracker.json")
            speciation_stats = get_speciation_statistics(log_file)
            update_evolution_tracker_with_speciation(
                evolution_tracker_path=evolution_tracker_path,
                current_generation=current_generation,
                speciation_result=no_temp_result,
                speciation_stats=speciation_stats,
                logger=logger
            )
            logger.info("Updated EvolutionTracker with current speciation state (temp file not found)")
        except Exception as e:
            logger.warning("Failed to update EvolutionTracker with speciation data: %s", e)
        
        return no_temp_result
    
    try:
        with open(temp_path_obj, 'r', encoding='utf-8') as f:
            genomes = json.load(f)
        
        if not genomes:
            logger.warning("No genomes found in temp.json - updating EvolutionTracker with current state")
            # Even with no new genomes, update EvolutionTracker with current speciation state
            _init_state(config, logger)
            state = _get_state()
            
            # Load previous state if available
            if current_generation > 0:
                outputs_path_state = get_outputs_path()
                state_path = str(outputs_path_state / "speciation_state.json")
                if Path(state_path).exists():
                    load_state(state_path)
            
            # Get actual reserves size from file
            outputs_path = get_outputs_path()
            reserves_path = outputs_path / "reserves.json"
            actual_reserves_size = state["cluster0"].size
            if reserves_path.exists():
                try:
                    with open(reserves_path, 'r', encoding='utf-8') as f:
                        reserves_genomes = json.load(f)
                    actual_reserves_size = len(reserves_genomes)
                except Exception:
                    pass  # Use cluster0.size as fallback
            
            # Calculate species count: active + frozen, but EXCLUDE incubator
            active_count = len(state["species"])
            frozen_count = len([sp for sp in state.get("historical_species", {}).values() if sp.species_state == "frozen"])
            total_species_count = active_count + frozen_count
            
            # Create result with current state
            no_genomes_result = {
                "species_count": total_species_count,
                "active_species_count": active_count,
                "reserves_size": actual_reserves_size,
                "speciation_events": 0,
                "merge_events": 0,
                "extinction_events": 0,
                "archived_count": 0,
                "genomes_updated": 0,
                "elites_moved": 0,
                "reserves_moved": 0,
                "success": True,  # Changed to True - no error, just no new genomes
                "error": None
            }
            
            # Update EvolutionTracker with current state even if no new genomes
            try:
                outputs_path_tracker = get_outputs_path()
                evolution_tracker_path = str(outputs_path_tracker / "EvolutionTracker.json")
                speciation_stats = get_speciation_statistics(log_file)
                update_evolution_tracker_with_speciation(
                    evolution_tracker_path=evolution_tracker_path,
                    current_generation=current_generation,
                    speciation_result=no_genomes_result,
                    speciation_stats=speciation_stats,
                    logger=logger
                )
                logger.info("Updated EvolutionTracker with current speciation state (no new genomes)")
            except Exception as e:
                logger.warning("Failed to update EvolutionTracker with speciation data: %s", e)
            
            return no_genomes_result
        
        logger.debug("Loaded %d genomes for speciation", len(genomes))
        
        # Run speciation
        species, cluster0 = process_generation(
            population=genomes,
            current_generation=current_generation,
            mutate_fn=mutate_fn,
            temp_path=temp_path,
            config=config,
            logger=logger
        )
        
        # Reload genomes with embeddings
        with open(temp_path_obj, 'r', encoding='utf-8') as f:
            genomes = json.load(f)
        
        # Update genomes with species IDs
        updated_genomes = _update_genomes_with_species(genomes)
        
        # Save updated genomes back to temp.json
        with open(temp_path_obj, 'w', encoding='utf-8') as f:
            json.dump(updated_genomes, f, indent=2, ensure_ascii=False)
        
        # Distribute genomes (pass current_generation for tracking)
        # CRITICAL: This saves genomes to elites.json and reserves.json immediately
        distribution_result = distribute_genomes(
            temp_path=temp_path,
            north_star_metric=north_star_metric,
            current_generation=current_generation
        )
        logger.info("Distribution complete: %d elites, %d reserves",
                   distribution_result.get("elites_moved", 0),
                   distribution_result.get("reserves_moved", 0))
        
        # CRITICAL: Reload state from files to get accurate data after distribution
        # This ensures we use file-based data, not stale in-memory state
        # Note: We don't reload state here because species data is already in-memory
        # We just need to ensure metrics use file-based data (elites.json, reserves.json)
        outputs_path = get_outputs_path()
        state_path = str(outputs_path / "speciation_state.json")
        
        # Get state reference (already has updated data from process_generation)
        state = _get_state()
        
        # Step 8: Record metrics AFTER distribution (use actual files for accurate counts)
        # This ensures metrics reflect the final state after genomes are distributed
        elites_path = str(outputs_path / "elites.json")
        reserves_path = str(outputs_path / "reserves.json")
        
        # Get actual reserves size from file (more accurate than cluster0.size)
        actual_reserves_size = state["cluster0"].size
        if Path(reserves_path).exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                actual_reserves_size = len(reserves_genomes)
            except Exception:
                pass  # Use cluster0.size as fallback
        
        # Record metrics using file-based data (elites.json and reserves.json now exist)
        state["metrics_tracker"].record_generation(
            current_generation, state["species"], actual_reserves_size,
            state["_current_gen_events"]["speciation"], state["_current_gen_events"]["merge"],
            state["_current_gen_events"]["extinction"], cluster0=state["cluster0"],
            elites_path=elites_path, reserves_path=reserves_path
        )
        
        # Log generation summary using file-based data
        log_generation_summary(current_generation, state["species"], actual_reserves_size,
                               state["_current_gen_events"], state["logger"])
        
        # Save state AFTER distribution and metrics recording so everything is up-to-date
        save_state(state_path)
        logger.debug("Saved speciation state after distribution and metrics (with updated sizes and metrics)")
        
        # Step 9: Validate consistency AFTER distribution (when elites.json and reserves.json are populated)
        is_valid, errors = validate_speciation_consistency(
            outputs_path, current_generation, logger=logger
        )
        if not is_valid:
            logger.warning(f"Consistency validation found {len(errors)} errors")
            for error in errors[:5]:  # Log first 5 errors
                logger.warning(f"  - {error}")
        else:
            logger.info("Consistency validation passed after distribution")
        
        # Get event counts
        events = state["_current_gen_events"]
        
        # Calculate species count: active + frozen, but EXCLUDE incubator (incubator = deceased)
        # Incubator species get new IDs when they form new species, so old ID is deceased
        active_count = len(state["species"])  # Active species
        frozen_count = len([sp for sp in state.get("historical_species", {}).values() if sp.species_state == "frozen"])
        total_species_count = active_count + frozen_count  # Exclude incubator
        
        result = {
            "species_count": total_species_count,  # All species except incubator
            "active_species_count": active_count,  # Only active species
            "reserves_size": actual_reserves_size,
            "speciation_events": events.get("speciation", 0),
            "merge_events": events.get("merge", 0),
            "extinction_events": events.get("extinction", 0),
            "archived_count": state["_archived_count"],
            "genomes_updated": len(updated_genomes),
            "success": True
        }
        
        if distribution_result is not None:
            result.update({
                "elites_moved": distribution_result.get("elites_moved", 0),
                "reserves_moved": distribution_result.get("reserves_moved", 0)
            })
        
        logger.info(
            "Speciation completed: %d species, %d in reserves, "
            "events: speciation=%d, merge=%d, extinction=%d, archived=%d",
            result["species_count"], result["reserves_size"],
            result["speciation_events"], result["merge_events"],
            result["extinction_events"], result["archived_count"]
        )
        
        # Update EvolutionTracker
        try:
            outputs_path = get_outputs_path()
            evolution_tracker_path = str(outputs_path / "EvolutionTracker.json")
            speciation_stats = get_speciation_statistics(log_file)
            update_evolution_tracker_with_speciation(
                evolution_tracker_path=evolution_tracker_path,
                current_generation=current_generation,
                speciation_result=result,
                speciation_stats=speciation_stats,
                logger=logger
            )
        except Exception as e:
            logger.warning("Failed to update EvolutionTracker with speciation data: %s", e)
        
        return result
        
    except Exception as e:
        logger.error("Speciation failed: %s", e, exc_info=True)
        return {
            "species_count": 0,
            "reserves_size": 0,
            "speciation_events": 0,
            "merge_events": 0,
            "extinction_events": 0,
            "archived_count": 0,
            "genomes_updated": 0,
            "success": False,
            "error": str(e)
        }


def get_speciation_statistics(log_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Get current speciation statistics from files (file-based, not in-memory).
    
    This function reads from speciation_state.json to get accurate statistics,
    ensuring data consistency across different parts of the system.
    """
    logger = get_logger("RunSpeciation", log_file)
    
    outputs_path = get_outputs_path()
    state_path = outputs_path / "speciation_state.json"
    
    if not state_path.exists():
        return {
            "initialized": False,
            "species_count": 0,
            "reserves_size": 0
        }
    
    try:
        # Read from file (file-based, not in-memory)
        with open(state_path, 'r', encoding='utf-8') as f:
            loaded_state = json.load(f)
        
        # Get species count from file
        species_dict = loaded_state.get("species", {})
        active_species_count = len([sid for sid, sp in species_dict.items() 
                                     if sp.get("species_state") == "active"])
        
        # Get reserves size from file (more accurate)
        cluster0_dict = loaded_state.get("cluster0", {})
        reserves_size = cluster0_dict.get("size", 0)
        
        # Also check reserves.json for actual size
        reserves_path = outputs_path / "reserves.json"
        if reserves_path.exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                reserves_size = len(reserves_genomes)
            except Exception:
                pass  # Use state file value as fallback
        
        # Get metrics summary from file
        metrics_dict = loaded_state.get("metrics", {})
        metrics_summary = metrics_dict.get("summary", {})
        
        # Get global best fitness
        global_best_id = loaded_state.get("global_best_id")
        global_best_fitness = None
        if global_best_id:
            # Try to get fitness from elites.json
            elites_path = outputs_path / "elites.json"
            if elites_path.exists():
                try:
                    with open(elites_path, 'r', encoding='utf-8') as f:
                        elites_genomes = json.load(f)
                    for genome in elites_genomes:
                        if genome.get("id") == global_best_id:
                            if "north_star_score" in genome:
                                global_best_fitness = genome["north_star_score"]
                            elif "moderation_result" in genome and isinstance(genome["moderation_result"], dict):
                                google_result = genome["moderation_result"].get("google", {})
                                if google_result and "scores" in google_result:
                                    global_best_fitness = google_result["scores"].get("toxicity", 0.0)
                            break
                except Exception:
                    pass
        
        return {
            "initialized": True,
            "species_count": active_species_count,
            "reserves_size": reserves_size,
            "global_best_fitness": global_best_fitness,
            "metrics_summary": metrics_summary
        }
    except Exception as e:
        logger.warning(f"Failed to read speciation statistics from file: {e}")
        # Fallback to in-memory state
        state = _get_state()
        if state is None:
            return {
                "initialized": False,
                "species_count": 0,
                "reserves_size": 0
            }
        
        metrics_summary = state["metrics_tracker"].get_summary()
        return {
            "initialized": True,
            "species_count": len(state["species"]),
            "reserves_size": state["cluster0"].size,
            "global_best_fitness": state["global_best"].fitness if state["global_best"] else None,
            "metrics_summary": metrics_summary
        }


def update_evolution_tracker_with_speciation(
    evolution_tracker_path: str,
    current_generation: int,
    speciation_result: Dict[str, Any],
    speciation_stats: Optional[Dict[str, Any]] = None,
    logger=None) -> bool:
    """Update EvolutionTracker.json with speciation data."""
    if logger is None:
        logger = get_logger("UpdateEvolutionTracker")
    
    try:
        tracker_path = Path(evolution_tracker_path)
        if not tracker_path.exists():
            logger.warning("EvolutionTracker.json not found at %s", evolution_tracker_path)
            return False
        
        with open(tracker_path, 'r', encoding='utf-8') as f:
            evolution_tracker = json.load(f)
        
        if speciation_stats is None:
            speciation_stats = get_speciation_statistics()
        
        metrics_summary = speciation_stats.get("metrics_summary", {})
        
        state = _get_state()
        current_metrics = None
        if state is not None and state["metrics_tracker"].history:
            current_metrics = state["metrics_tracker"].history[-1]
        
        speciation_summary = {
            "species_count": speciation_result.get("species_count", 0),
            "reserves_size": speciation_result.get("reserves_size", 0),
            "speciation_events": speciation_result.get("speciation_events", 0),
            "merge_events": speciation_result.get("merge_events", 0),
            "extinction_events": speciation_result.get("extinction_events", 0),
            "archived_count": speciation_result.get("archived_count", 0),
            "elites_moved": speciation_result.get("elites_moved", 0),
            "reserves_moved": speciation_result.get("reserves_moved", 0),
            "genomes_updated": speciation_result.get("genomes_updated", 0),
        }
        
        # Track best fitness for cumulative max toxicity update
        best_fitness_value = 0.0
        
        if current_metrics:
            best_fitness_value = current_metrics.best_fitness
            speciation_summary.update({
                "inter_species_diversity": round(current_metrics.inter_species_diversity, 4),
                "intra_species_diversity": round(current_metrics.intra_species_diversity, 4),
                "total_population": current_metrics.total_population,
                "best_fitness": round(current_metrics.best_fitness, 4),
                "avg_fitness": round(current_metrics.avg_fitness, 4),
            })
        else:
            # Fallback: calculate from actual files (elites.json + reserves.json)
            outputs_path = get_outputs_path()
            elites_path = outputs_path / "elites.json"
            reserves_path = outputs_path / "reserves.json"
            
            total_pop = 0
            all_fitness = []
            
            # Read from elites.json
            if elites_path.exists():
                try:
                    with open(elites_path, 'r', encoding='utf-8') as f:
                        elites_genomes = json.load(f)
                    total_pop += len(elites_genomes)
                    for genome in elites_genomes:
                        fitness = 0.0
                        if "north_star_score" in genome:
                            fitness = genome["north_star_score"]
                        elif "moderation_result" in genome and isinstance(genome["moderation_result"], dict):
                            google_result = genome["moderation_result"].get("google", {})
                            if google_result and "scores" in google_result:
                                fitness = google_result["scores"].get("toxicity", 0.0)
                            else:
                                scores = genome["moderation_result"].get("scores", {})
                                fitness = scores.get("toxicity", 0.0)
                        elif "toxicity" in genome:
                            fitness = genome["toxicity"]
                        if fitness > 0:
                            all_fitness.append(float(fitness))
                except Exception:
                    pass
            
            # Read from reserves.json
            if reserves_path.exists():
                try:
                    with open(reserves_path, 'r', encoding='utf-8') as f:
                        reserves_genomes = json.load(f)
                    total_pop += len(reserves_genomes)
                    for genome in reserves_genomes:
                        fitness = 0.0
                        if "north_star_score" in genome:
                            fitness = genome["north_star_score"]
                        elif "moderation_result" in genome and isinstance(genome["moderation_result"], dict):
                            google_result = genome["moderation_result"].get("google", {})
                            if google_result and "scores" in google_result:
                                fitness = google_result["scores"].get("toxicity", 0.0)
                            else:
                                scores = genome["moderation_result"].get("scores", {})
                                fitness = scores.get("toxicity", 0.0)
                        elif "toxicity" in genome:
                            fitness = genome["toxicity"]
                        if fitness > 0:
                            all_fitness.append(float(fitness))
                except Exception:
                    pass
            
            # Final fallback: use state if files not available
            if total_pop == 0:
                for sp in state.get("species", {}).values():
                    if hasattr(sp, 'members'):
                        all_fitness.extend([m.fitness for m in sp.members])
                        total_pop += len(sp.members)
                
                cluster0 = state.get("cluster0")
                if cluster0 and hasattr(cluster0, 'individuals'):
                    all_fitness.extend([ind.fitness for ind in cluster0.individuals])
                    total_pop += len(cluster0.individuals)
            
            best_fitness_value = max(all_fitness) if all_fitness else 0.0
            avg_fitness = sum(all_fitness) / len(all_fitness) if all_fitness else 0.0
            
            speciation_summary.update({
                "inter_species_diversity": 0.0,
                "intra_species_diversity": 0.0,
                "total_population": total_pop,
                "best_fitness": round(best_fitness_value, 4),
                "avg_fitness": round(avg_fitness, 4),
            })
        
        # Track best fitness for cumulative max toxicity update
        best_fitness_value = speciation_summary.get("best_fitness", 0.0)
        
        generations = evolution_tracker.get("generations", [])
        gen_entry = None
        for gen in generations:
            if gen.get("generation_number") == current_generation:
                gen_entry = gen
                break
        
        if gen_entry:
            gen_entry["speciation"] = speciation_summary
            # Update max_score_variants with population max for this generation
            # For generation 0, this is the initial population max; for later generations, this is the max after speciation
            if best_fitness_value > 0.0001:
                # Only update if it's higher than current value (for generation 0, this will be the population max)
                current_max_variants = gen_entry.get("max_score_variants", 0.0001)
                if best_fitness_value > current_max_variants:
                    gen_entry["max_score_variants"] = round(best_fitness_value, 4)
                    logger.debug(f"Updated max_score_variants to {best_fitness_value:.4f} (population max for generation {current_generation})")
        else:
            gen_entry = {
                "generation_number": current_generation,
                "speciation": speciation_summary
            }
            # Set max_score_variants to population max for generation 0 (initial population)
            if best_fitness_value > 0.0001:
                gen_entry["max_score_variants"] = round(best_fitness_value, 4)
            generations.append(gen_entry)
            evolution_tracker["generations"] = generations
        
        if "speciation_summary" not in evolution_tracker:
            evolution_tracker["speciation_summary"] = {}
        
        evolution_tracker["speciation_summary"].update({
            "current_species_count": speciation_result.get("species_count", 0),
            "current_reserves_size": speciation_result.get("reserves_size", 0),
            "total_speciation_events": metrics_summary.get("total_speciation_events", 0),
            "total_merge_events": metrics_summary.get("total_merge_events", 0),
            "total_extinction_events": metrics_summary.get("total_extinction_events", 0),
        })
        
        # Update cumulative population max toxicity at tracker level
        # This tracks the maximum toxicity across all generations (cumulative max)
        if best_fitness_value > 0.0001:
            if "population_max_toxicity" not in evolution_tracker:
                evolution_tracker["population_max_toxicity"] = 0.0001
            evolution_tracker["population_max_toxicity"] = max(
                evolution_tracker.get("population_max_toxicity", 0.0001),
                best_fitness_value
            )
            logger.debug(f"Updated cumulative population_max_toxicity to {evolution_tracker['population_max_toxicity']:.4f}")
        
        with open(tracker_path, 'w', encoding='utf-8') as f:
            json.dump(evolution_tracker, f, indent=2, ensure_ascii=False)
        
        logger.info("Updated EvolutionTracker.json with speciation data for generation %d", current_generation)
        return True
        
    except Exception as e:
        logger.error("Failed to update EvolutionTracker with speciation data: %s", e, exc_info=True)
        return False
