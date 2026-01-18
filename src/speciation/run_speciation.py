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
            "historical_species": {},  # Extinct (merged parents) and incubator species (preserved for reference)
            "cluster0": Cluster0(
                min_cluster_size=(config or SpeciationConfig()).cluster0_min_cluster_size,
                theta_sim=(config or SpeciationConfig()).theta_sim,
                max_capacity=(config or SpeciationConfig()).cluster0_max_capacity,
                min_island_size=(config or SpeciationConfig()).min_island_size,
                w_genotype=(config or SpeciationConfig()).w_genotype,
                w_phenotype=(config or SpeciationConfig()).w_phenotype,
                logger=logger or get_logger("Speciation")
            ),
            "global_best": None,
            "metrics_tracker": SpeciationMetricsTracker(logger=logger or get_logger("Speciation")),
            "_current_gen_events": {"speciation": 0, "merge": 0, "extinction": 0, "moved_to_cluster0": 0},
            "_embedding_model": None,
            "_archived_count": 0,
            "_archived_genomes": []  # Track all archived genomes for final corrected save
        }
        _state["logger"].info(f"Speciation initialized: theta_sim={_state['config'].theta_sim}, species_capacity={_state['config'].species_capacity}")
    else:
        # Update config if provided (ensures command-line arguments are followed)
        if config is not None:
            old_config = _state["config"]
            _state["config"] = config
            # Update cluster0 parameters if config changed
            if (old_config.theta_sim != config.theta_sim or 
                old_config.cluster0_max_capacity != config.cluster0_max_capacity or
                old_config.cluster0_min_cluster_size != config.cluster0_min_cluster_size):
                _state["cluster0"].theta_sim = config.theta_sim
                _state["cluster0"].max_capacity = config.cluster0_max_capacity
                _state["cluster0"].min_cluster_size = config.cluster0_min_cluster_size
                _state["logger"].info(f"Config updated: theta_sim={config.theta_sim}, species_stagnation={config.species_stagnation}, species_capacity={config.species_capacity}")
        # Update logger if provided
        if logger is not None:
            _state["logger"] = logger


def _get_state() -> Dict[str, Any]:
    """Get global state, initializing if needed."""
    if _state is None:
        _init_state()
    return _state


def _validate_active_count(state: Dict[str, Any], calculated_count: int, source: str) -> int:
    """
    Validate and correct active species count.
    
    Compares calculated count with in-memory count and uses in-memory as source of truth.
    
    Args:
        state: Global speciation state
        calculated_count: Count calculated from file or other source
        source: Description of where calculated_count came from (for logging)
        
    Returns:
        Corrected active species count (uses in-memory if mismatch detected)
    """
    in_memory_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
    if calculated_count != in_memory_count:
        state["logger"].warning(
            f"Active count mismatch: calculated={calculated_count} (from {source}), "
            f"in_memory={in_memory_count}, using in_memory as source of truth"
        )
        return in_memory_count
    return calculated_count


def _archive_individuals(individuals: List[Individual], generation: int, reason: str) -> None:
    """
    Archive individuals to archive.json and track them in state for final corrected save.
    
    This function archives individuals and maintains a list of all archived genomes
    in the global state so they can be included in the final corrected save.
    """
    if not individuals:
        return
    
    state = _get_state()
    state["_archived_count"] += len(individuals)
    logger = state["logger"]
    
    # Initialize archived genomes list in state if not exists
    if "_archived_genomes" not in state:
        state["_archived_genomes"] = []
    
    try:
        outputs_path = get_outputs_path()
        archive_path = outputs_path / "archive.json"
        
        # Load existing archive
        if archive_path.exists():
            with open(archive_path, 'r', encoding='utf-8') as f:
                archive = json.load(f)
            # Ensure archive is a list (handle edge cases where it might be a dict)
            if not isinstance(archive, list):
                if isinstance(archive, dict):
                    logger.warning(f"archive.json is a dict (expected list), converting to list")
                    archive = list(archive.values()) if len(archive) > 0 else []
                else:
                    logger.warning(f"archive.json has unexpected format, initializing as empty list")
                    archive = []
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
            # Remove embeddings before archiving (save space, not needed for archived genomes)
            if "prompt_embedding" in entry:
                del entry["prompt_embedding"]
            
            archive.append(entry)
            # Also track in state for final corrected save
            state["_archived_genomes"].append(entry)
        
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
    Process a single generation with full speciation pipeline (NEW FLOW).
    
    New Pipeline (8 Phases):
    Phase 1: Existing Species Processing
      1. Compute embeddings
      2. Process variants against existing species (skip cluster 0 outliers)
      3. Radius cleanup of existing species
      4. Capacity enforcement of existing species
      5. Save intermediate #1
    
    Phase 2: Cluster 0 Speciation (Isolated)
      6. Load cluster 0 from reserves.json
      7. Apply isolated cluster 0 speciation
      8. Radius cleanup of newly formed species
      9. Capacity enforcement of newly formed species
      10. Save intermediate #2
    
    Phase 3: Merging
      11. Merging of all species
      11a. Save intermediate (after merging)
    
    Phase 4: Freeze & Incubator
      12. Freeze stagnant species
      13. Move small species to incubator
      14. Save intermediate #3
    
    Phase 5: Cluster 0 Capacity Enforcement
      15. Enforce cluster 0 capacity at end
      15a. Save intermediate (after cluster 0 capacity enforcement)
      15a. Save intermediate (after cluster 0 capacity enforcement)
    
    Phase 6: Final Corrected Save
      16. Save corrected files (elites, reserves, archives)
    
    Phase 7: Update Metrics & Stats
      17. Update metrics from corrected files
    
    Phase 8: Return (distribution happens in run_speciation)
    
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
    # Initialize archived genomes list if not exists
    if "_archived_genomes" not in state:
        state["_archived_genomes"] = []
    
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
            # Ensure archived genomes list exists after loading state
            if "_archived_genomes" not in state:
                state["_archived_genomes"] = []
    
    # ========================================================================
    # PHASE 1: EXISTING SPECIES PROCESSING
    # ========================================================================
    state["logger"].info("=== Phase 1: Existing Species Processing ===")
    
    # 1. Compute and save embeddings to temp.json
    if temp_path is None:
        outputs_path = get_outputs_path()
        temp_path = str(outputs_path / "temp.json")
    
    compute_and_save_embeddings(
        temp_path=temp_path,
        model_name=state["config"].embedding_model,
        batch_size=state["config"].embedding_batch_size,
        logger=state["logger"]
    )
    
    # 2. Process variants against existing species (skip cluster 0 outliers)
    outputs_path = get_outputs_path()
    speciation_state_path = str(outputs_path / "speciation_state.json")
    
    # Track species count BEFORE clustering
    species_count_before_clustering = len(state["species"])
    
    # Use skip_cluster0_outliers=True to process only against existing species
    state["species"], species_with_new_members = leader_follower_clustering(
        temp_path=temp_path,
        speciation_state_path=speciation_state_path,
        theta_sim=state["config"].theta_sim,
        current_generation=current_generation,
        w_genotype=state["config"].w_genotype,
        w_phenotype=state["config"].w_phenotype,
        min_island_size=state["config"].min_island_size,
        skip_cluster0_outliers=True,  # NEW: Skip cluster 0 outliers during variant processing
        logger=state["logger"]
    )
    
    # Count new species formed during leader_follower_clustering (should be 0 with skip_cluster0_outliers=True)
    species_count_after_clustering = len(state["species"])
    new_species_from_clustering = species_count_after_clustering - species_count_before_clustering
    if new_species_from_clustering > 0:
        state["_current_gen_events"]["speciation"] += new_species_from_clustering
        state["logger"].info(f"Counted {new_species_from_clustering} new species formed during leader-follower clustering (before: {species_count_before_clustering}, after: {species_count_after_clustering})")
    
    # 2a: Sync cluster 0 with reserves.json (add new variants that went to cluster 0)
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
    
    # 3. Radius cleanup of existing species (only species that received new members)
    # Verify all members are still within radius of the current leader
    # This ensures species cohesion: all members must be within theta_sim of the current leader
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
            
            # Check if species is now empty or too small (only leader or below min_island_size)
            if len(sp.members) <= 1:
                # Species is empty (only leader) - move to incubator state
                sp.species_state = "incubator"
                sp.members = []  # Clear members
                
                # Move leader to cluster 0 (if leader exists)
                if sp.leader and state["cluster0"].size < state["config"].cluster0_max_capacity:
                    state["cluster0"].add(sp.leader, current_generation)
                
                # Remove from active species (will be preserved in speciation_state.json with incubator state)
                del state["species"][sid]
                state["logger"].info(f"Species {sid} became empty after radius cleanup - moved to incubator")
            elif len(sp.members) < state["config"].min_island_size:
                # Species too small after cleanup - mark for incubator (will be processed in process_extinctions)
                state["logger"].debug(f"Species {sid} size={len(sp.members)} < min_island_size={state['config'].min_island_size} after radius cleanup - will be moved to incubator in process_extinctions")
                # Keep as active/frozen for now, process_extinctions will handle it
    
    # 4. Capacity enforcement of existing species
    # Check ALL existing species for capacity (ensures no species exceeds capacity, even if no new members)
    for sid, sp in list(state["species"].items()):  # Use list() to avoid dict modification during iteration
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
            state["logger"].info(f"Species {sid}: enforced capacity ({state['config'].species_capacity}), archived {len(excess)} excess members")
    
    # Validate no duplicate leader IDs across existing species
    leader_ids = {}
    duplicate_leaders = []
    for sid, sp in state["species"].items():
        if sp.leader:
            leader_id = sp.leader.id
            if leader_id in leader_ids:
                duplicate_leaders.append((leader_id, [leader_ids[leader_id], sid]))
            else:
                leader_ids[leader_id] = sid
    
    if duplicate_leaders:
        # Fix duplicates by reassigning leaders
        for leader_id, species_ids in duplicate_leaders:
            state["logger"].warning(f"Duplicate leader ID {leader_id} found in species {species_ids}, fixing...")
            # Keep the first species, fix the others
            for sid in species_ids[1:]:
                if sid not in state["species"]:
                    continue
                sp = state["species"][sid]
                # Find the old leader in members and remove it
                old_leader = None
                for member in sp.members:
                    if member.id == leader_id:
                        old_leader = member
                        break
                
                if old_leader:
                    sp.members.remove(old_leader)
                    old_leader.species_id = None  # Will be set when added to new species
                
                if len(sp.members) > 0:
                    # Reassign to next highest fitness from remaining members
                    sp.leader = max(sp.members, key=lambda x: x.fitness)
                    # Ensure new leader is in members (should be, but verify)
                    if sp.leader not in sp.members:
                        sp.members.insert(0, sp.leader)
                    state["logger"].info(f"Reassigned species {sid} leader to genome {sp.leader.id} (fitness={sp.leader.fitness:.4f})")
                else:
                    # No other members - mark for incubator (process_extinctions will handle cleanup)
                    sp.species_state = "incubator"
                    sp.leader = None  # No leader if no members
                    state["logger"].info(f"Species {sid} has no other members, marking as incubator (will be processed by process_extinctions)")
    
    # 5. Save intermediate #1 (existing species processing complete)
    save_to_files_intermediate(
        current_generation=current_generation,
        species=state["species"],
        cluster0=state["cluster0"],
        save_type="existing_species",
        logger=state["logger"]
    )
    
    # ========================================================================
    # PHASE 2: CLUSTER 0 SPECIATION (ISOLATED)
    # ========================================================================
    state["logger"].info("=== Phase 2: Cluster 0 Speciation (Isolated) ===")
    
    # 6. Load cluster 0 from reserves.json and apply isolated speciation
    # Sync cluster 0 with reserves.json first to get all genomes
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
                    if genome_id and genome_id not in existing_cluster0_ids and genome_id not in species_member_ids:
                        # Create Individual from genome and add to cluster0
                        outlier_ind = Individual.from_genome(genome)
                        if outlier_ind.embedding is not None:
                            state["cluster0"].add(outlier_ind, current_generation)
                            added_count += 1
            
            if removed_count > 0 or added_count > 0:
                state["logger"].info(f"Synced cluster 0: removed {removed_count} (now in species), added {added_count} (from reserves.json)")
        except Exception as e:
            state["logger"].warning(f"Failed to sync cluster 0 with reserves.json: {e}")
    
    # 7. Apply isolated cluster 0 speciation
    new_species_from_cluster0 = cluster0_speciation_isolated(
        current_generation=current_generation,
        config=state["config"],
        logger=state["logger"]
    )
    
    # Add newly formed species to state
    newly_formed_species_ids = set()
    for new_species in new_species_from_cluster0:
        state["species"][new_species.id] = new_species
        newly_formed_species_ids.add(new_species.id)
        state["_current_gen_events"]["speciation"] += 1
        state["logger"].info(f"Species {new_species.id} formed from cluster 0 ({new_species.size} members)")
        
        # Track speciation events
        if "_genome_tracker" in state:
            for member in new_species.members:
                state["_genome_tracker"].log(
                    member.id, "species_formed_from_cluster0",
                    {"species_id": new_species.id, "size": new_species.size}
                )
    
    # 8. Radius cleanup of newly formed species
    for sid in newly_formed_species_ids:
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
                members_to_keep.append(member)
                continue
            
            if member.embedding is None:
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
            state["logger"].debug(f"Newly formed species {sid}: removing {len(members_to_remove)} members outside radius")
            sp.members = members_to_keep
            
            # Move removed members to cluster 0
            for member in members_to_remove:
                if state["cluster0"].size < state["config"].cluster0_max_capacity:
                    state["cluster0"].add(member, current_generation)
            
            # Check if species is now too small
            if len(sp.members) < state["config"].min_island_size:
                state["logger"].debug(f"Newly formed species {sid} size={len(sp.members)} < min_island_size={state['config'].min_island_size} after radius cleanup - will be moved to incubator in process_extinctions")
    
    # 9. Capacity enforcement of newly formed species (only if size > min_island_size)
    for sid in newly_formed_species_ids:
        if sid not in state["species"]:
            continue
        sp = state["species"][sid]
        # Only enforce capacity if size > min_island_size
        if sp.size > state["config"].min_island_size and sp.size > state["config"].species_capacity:
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
            # Ensure leader is still in members
            if sp.leader not in sp.members and sp.members:
                sp.leader = max(sp.members, key=lambda x: x.fitness)
            state["logger"].info(f"Newly formed species {sid}: enforced capacity ({state['config'].species_capacity}), archived {len(excess)} excess members")
    
    # 10. Save intermediate #2 (cluster 0 speciation complete)
    save_to_files_intermediate(
        current_generation=current_generation,
        species=state["species"],
        cluster0=state["cluster0"],
        save_type="cluster0_speciation",
        logger=state["logger"]
    )
    
    # ========================================================================
    # PHASE 3: MERGING
    # ========================================================================
    state["logger"].info("=== Phase 3: Merging ===")
    
    # 11. Merging of all species (existing + newly formed)
    # NOTE: record_fitness() is called ONCE per generation in Phase 4 (Freeze & Incubator)
    # to avoid double-incrementing stagnation. We skip it here to prevent calling it twice.
    # This ensures stagnation only increments once per generation, preventing premature freezing.
    
    species_count_before_merge = len(state["species"])
    state["species"], merge_events, merge_outliers, extinct_parents = process_merges(
        state["species"],
        theta_merge=state["config"].theta_merge,
        theta_sim=state["config"].theta_sim,
        current_gen=current_generation,
        max_capacity=state["config"].species_capacity,
        w_genotype=state["config"].w_genotype,
        w_phenotype=state["config"].w_phenotype,
        historical_species=state.get("historical_species", {}),
        logger=state["logger"]
    )
    
    # Move extinct parent species to historical_species (they merged, so they're extinct)
    # Both parent species become extinct when they merge - the merged species is new
    for sid, extinct_sp in extinct_parents.items():
        state["historical_species"][sid] = extinct_sp
        state["logger"].info(f"Parent species {sid} became extinct via merge (moved to historical_species)")
    species_count_after_merge = len(state["species"])
    state["_current_gen_events"]["merge"] = len(merge_events)
    
    # Verify merge logic: species count should decrease by number of merges
    # Each merge combines 2 species into 1, so count decreases by number of merges
    # This applies regardless of whether species are active or frozen (both are alive)
    expected_species_after_merge = species_count_before_merge - len(merge_events)
    if species_count_after_merge != expected_species_after_merge:
        state["logger"].warning(f"Merge count mismatch: before={species_count_before_merge}, after={species_count_after_merge}, merges={len(merge_events)}, expected_after={expected_species_after_merge}")
    
    # Verify parent_ids are set correctly for merged species
    for merge_event in merge_events:
        merged_ids = merge_event.get("merged", [])
        result_id = merge_event.get("result_id")
        if result_id and result_id in state["species"]:
            merged_sp = state["species"][result_id]
            expected_parents = sorted(merged_ids)
            actual_parents = sorted(merged_sp.parent_ids) if merged_sp.parent_ids else []
            if actual_parents != expected_parents:
                state["logger"].error(f"Merge parent_ids mismatch for species {result_id}: expected {expected_parents}, got {actual_parents}")
            elif merged_sp.cluster_origin != "merge":
                state["logger"].error(f"Merge species {result_id} has incorrect cluster_origin: expected 'merge', got '{merged_sp.cluster_origin}'")
            else:
                state["logger"].debug(f"Merge verification passed: species {result_id} from {merged_ids}, parent_ids={actual_parents}, origin={merged_sp.cluster_origin}")
    
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
    
    # Save immediately after merging (major data modification)
    # Merging creates new species, removes parent species, and moves outliers to cluster 0
    save_to_files_intermediate(
        current_generation=current_generation,
        species=state["species"],
        cluster0=state["cluster0"],
        save_type="after_merging_only",
        logger=state["logger"]
    )
    
    # ========================================================================
    # PHASE 4: FREEZE & INCUBATOR
    # ========================================================================
    state["logger"].info("=== Phase 4: Freeze & Incubator ===")
    
    # 12. Record fitness for ALL species (not just those with new members)
    # CRITICAL: Stagnation only increments if species was selected as parent AND no improvement
    # Load parents.json to determine which species were selected as parents
    selected_species_ids = set()
    if current_generation > 0:  # Generation 0 has no parents
        try:
            outputs_path = get_outputs_path()
            parents_path = outputs_path / "parents.json"
            if parents_path.exists():
                with open(parents_path, 'r', encoding='utf-8') as f:
                    parents = json.load(f)
                if isinstance(parents, list):
                    for parent in parents:
                        species_id = parent.get("species_id")
                        if species_id is not None:
                            selected_species_ids.add(int(species_id))
                    state["logger"].debug(f"Loaded {len(selected_species_ids)} species from parents.json: {sorted(selected_species_ids)}")
        except Exception as e:
            state["logger"].warning(f"Failed to load parents.json to determine selected species: {e}")
    
    for sid, sp in state["species"].items():
        was_selected = sid in selected_species_ids
        sp.record_fitness(current_generation, was_selected_as_parent=was_selected)
    
    # 13. Freeze stagnant species and move small species to cluster 0
    # CRITICAL: Use in-memory sizes (sp.size) - elites.json is cumulative across all generations
    # We want to move species to incubator based on CURRENT size (after radius cleanup, capacity enforcement)
    species_count_before_extinction = len(state["species"])
    state["species"], extinction_events, moved_to_cluster0_events, incubator_species = process_extinctions(
        state["species"],
        state["cluster0"],
        current_generation,
        species_stagnation=state["config"].species_stagnation,
        min_size=state["config"].min_island_size,
        elites_path=None,  # Don't use elites.json - it's cumulative, use in-memory size instead
        logger=state["logger"]
    )
    species_count_after_extinction = len(state["species"])
    # Only frozen species (stagnation-based) count as extinction events
    # Moving to cluster 0 (size-based) is NOT extinction, just reorganization
    state["_current_gen_events"]["extinction"] = len(extinction_events)
    state["_current_gen_events"]["moved_to_cluster0"] = len(moved_to_cluster0_events)
    
    # Verify extinction logic
    expected_removed = len(moved_to_cluster0_events)  # Only moved_to_cluster0 removes from active species
    expected_species_after = species_count_before_extinction - expected_removed
    if species_count_after_extinction != expected_species_after:
        state["logger"].warning(f"Extinction count mismatch: before={species_count_before_extinction}, after={species_count_after_extinction}, moved_to_cluster0={len(moved_to_cluster0_events)}, expected_after={expected_species_after}")
    
    # Verify extinction events have correct structure
    for ext_event in extinction_events:
        if "species_id" not in ext_event or "action" not in ext_event:
            state["logger"].error(f"Invalid extinction event structure: {ext_event}")
        elif ext_event["action"] != "frozen":
            state["logger"].error(f"Extinction event has incorrect action: expected 'frozen', got '{ext_event.get('action')}'")
        else:
            state["logger"].debug(f"Extinction verification passed: species {ext_event['species_id']} frozen (stagnation={ext_event.get('stagnation', 'unknown')})")
    
    # Frozen species stay in active species dict (they are still alive, just excluded from parent selection)
    # They are NOT moved to historical_species - only merged/extinct and incubator species go there
    # (process_extinctions already sets species_state="frozen" but keeps them in species dict)
    
    # Move incubator species to historical_species for tracking (just IDs, not full data)
    # (incubator species are returned separately by process_extinctions)
    # We keep them in historical_species but they won't be saved with full data - just tracked by ID
    for sid, sp in incubator_species.items():
        state["historical_species"][sid] = sp  # Keep for in-memory tracking
        state["logger"].debug(f"Moved incubator species {sid} to historical_species (will be tracked by ID only in save_state)")
    
    # 14. Save intermediate #3 (after merging and incubator)
    save_to_files_intermediate(
        current_generation=current_generation,
        species=state["species"],
        cluster0=state["cluster0"],
        save_type="after_merging",
        logger=state["logger"]
    )
    
    # ========================================================================
    # PHASE 5: CLUSTER 0 CAPACITY ENFORCEMENT
    # ========================================================================
    state["logger"].info("=== Phase 5: Cluster 0 Capacity Enforcement ===")
    
    # 15. Enforce cluster 0 capacity at end
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
        state["logger"].info(f"Cluster 0 capacity enforced: archived {len(excess_individuals)} excess members (capacity: {state['config'].cluster0_max_capacity})")
        
        # Save immediately after cluster 0 capacity enforcement (major data modification)
        # Cluster 0 size is reduced and excess genomes are archived
        save_to_files_intermediate(
            current_generation=current_generation,
            species=state["species"],
            cluster0=state["cluster0"],
            save_type="after_cluster0_capacity",
            logger=state["logger"]
        )
    
    # ========================================================================
    # PHASE 6: FINAL CORRECTED SAVE
    # ========================================================================
    state["logger"].info("=== Phase 6: Final Corrected Save ===")
    
    # 16. Save corrected files (elites, reserves, archives)
    archived_genomes = state.get("_archived_genomes", [])
    save_to_files_corrected(
        current_generation=current_generation,
        species=state["species"],
        cluster0=state["cluster0"],
        archived_genomes=archived_genomes,
        logger=state["logger"]
    )
    
    # ========================================================================
    # PHASE 7: UPDATE METRICS & STATS
    # ========================================================================
    state["logger"].info("=== Phase 7: Update Metrics & Stats ===")
    
    # 17. Update metrics from corrected files
    # Update c-TF-IDF labels for all species
    from .labeling import update_species_labels
    update_species_labels(
        state["species"],
        current_generation=current_generation,
        n_words=10,
        logger=state["logger"]
    )
    
    # Record metrics (calculated from corrected files)
    # Files should exist after Phase 6 (Final Corrected Save)
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    
    # Files must exist after Phase 6 (Final Corrected Save)
    if not elites_path.exists():
        raise FileNotFoundError(f"elites.json not found at {elites_path} - required for metrics calculation")
    if not reserves_path.exists():
        state["logger"].warning(f"reserves.json not found at {reserves_path} - using cluster0.size")
    
    state["metrics_tracker"].record_generation(
        generation=current_generation,
        species=state["species"],
        reserves_size=state["cluster0"].size,
        speciation_events=state["_current_gen_events"]["speciation"],
        merge_events=state["_current_gen_events"]["merge"],
        extinction_events=state["_current_gen_events"]["extinction"],
        cluster0=state["cluster0"],
        elites_path=str(elites_path),
        reserves_path=str(reserves_path) if reserves_path.exists() else None
    )
    
    state["logger"].debug(f"Recorded metrics for generation {current_generation} from corrected files")
    
    # Save speciation_state.json with current state
    outputs_path = get_outputs_path()
    state_path = str(outputs_path / "speciation_state.json")
    save_state(state_path)
    state["logger"].debug("Saved speciation state after metrics update")
    
    # Analyze distance distributions
    distance_analysis = analyze_distance_distribution(
        state["species"], state["cluster0"], state["config"], logger=state["logger"]
    )
    
    # Save genome tracker
    if "_genome_tracker" in state:
        state["_genome_tracker"].save()
        tracker_summary = state["_genome_tracker"].get_summary()
        state["logger"].debug(f"Genome tracker: {tracker_summary['total_events']} events for {tracker_summary['unique_genomes']} genomes")
    
    # ========================================================================
    # PHASE 8: RETURN (distribution happens in run_speciation)
    # ========================================================================
    state["logger"].info("=== Phase 8: Complete - Returning to run_speciation for distribution ===")
    
    return state["species"], state["cluster0"]


def cluster0_speciation_isolated(current_generation: int, config: "SpeciationConfig", logger=None) -> List[Species]:
    """
    Apply leader-follower clustering on cluster 0 in complete isolation (like generation 0).
    
    This function loads cluster 0 genomes from reserves.json and applies speciation
    without any exposure to existing species. It behaves exactly like generation 0
    speciation but only processes cluster 0 genomes.
    
    Args:
        current_generation: Current generation number
        config: SpeciationConfig object with parameters
        logger: Optional logger instance
        
    Returns:
        List of newly formed Species (empty list if none formed)
    """
    from .species import Species, Individual, generate_species_id
    from .distance import ensemble_distance, ensemble_distances_batch
    from .reserves import CLUSTER_0_ID
    import numpy as np
    
    if logger is None:
        logger = get_logger("Cluster0SpeciationIsolated")
    
    outputs_path = get_outputs_path()
    reserves_path = outputs_path / "reserves.json"
    
    # Load cluster 0 genomes from reserves.json
    if not reserves_path.exists():
        logger.debug("reserves.json not found, no cluster 0 speciation possible")
        return []
    
    try:
        with open(reserves_path, 'r', encoding='utf-8') as f:
            reserves_genomes = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load reserves.json: {e}")
        return []
    
    if not reserves_genomes:
        logger.debug("No genomes in reserves.json")
        return []
    
    # Filter to genomes in cluster 0 (species_id == 0 or None)
    cluster0_genomes = [
        g for g in reserves_genomes
        if g.get("species_id", CLUSTER_0_ID) == CLUSTER_0_ID and g.get("prompt_embedding")
    ]
    
    if len(cluster0_genomes) < config.cluster0_min_cluster_size:
        logger.debug(f"Cluster 0 has {len(cluster0_genomes)} genomes, need {config.cluster0_min_cluster_size} to attempt speciation")
        return []
    
    # Convert genomes to Individual objects
    individuals = []
    for genome in cluster0_genomes:
        try:
            ind = Individual.from_genome(genome)
            if ind.embedding is not None:
                individuals.append(ind)
        except Exception as e:
            logger.warning(f"Failed to create Individual from genome {genome.get('id')}: {e}")
            continue
    
    if len(individuals) < config.cluster0_min_cluster_size:
        logger.debug(f"Only {len(individuals)} individuals with embeddings, need {config.cluster0_min_cluster_size}")
        return []
    
    # Sort by fitness (descending) - highest fitness processed first
    sorted_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
    
    # Potential leaders: Dict mapping leader_id -> (species_id_or_None, embedding, phenotype, Individual, followers_list)
    potential_leaders: Dict[str, Tuple[Optional[int], np.ndarray, Optional[np.ndarray], Individual, List[Individual]]] = {}
    
    # Track all new species formed
    new_species_list: List[Species] = []
    # Track all individuals to remove (from formed species)
    individuals_to_remove: List[Individual] = []
    
    # First individual becomes first potential leader
    first = sorted_individuals[0]
    potential_leaders[first.id] = (None, first.embedding, first.phenotype, first, [])
    remaining_individuals = sorted_individuals[1:]
    
    # Process remaining individuals
    for ind in remaining_individuals:
        assigned = False
        min_dist = float('inf')
        nearest_leader_id = None
        
        # Check against all potential leaders (excluding those that already formed species)
        active_leaders = {pl_id: data for pl_id, data in potential_leaders.items() if data[0] is None}
        if active_leaders:
            # Collect all leader embeddings and phenotypes
            leader_embeddings = []
            leader_phenotypes = []
            leader_ids = []
            for pl_id, (_, pl_emb, pl_pheno, _, _) in active_leaders.items():
                leader_ids.append(pl_id)
                leader_embeddings.append(pl_emb)
                leader_phenotypes.append(pl_pheno)
            
            if len(leader_embeddings) > 1:
                # Vectorized distance computation
                leader_embeddings_array = np.array(leader_embeddings)
                distances = ensemble_distances_batch(
                    ind.embedding, leader_embeddings_array,
                    ind.phenotype, leader_phenotypes,
                    config.w_genotype, config.w_phenotype
                )
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                nearest_leader_id = leader_ids[min_idx]
            elif len(leader_embeddings) == 1:
                min_dist = ensemble_distance(
                    ind.embedding, leader_embeddings[0],
                    ind.phenotype, leader_phenotypes[0],
                    config.w_genotype, config.w_phenotype
                )
                nearest_leader_id = leader_ids[0]
            
            # If within threshold, add as follower
            if nearest_leader_id is not None and min_dist < config.theta_sim:
                pl_species_id, pl_emb, pl_pheno, pl_ind, followers = potential_leaders[nearest_leader_id]
                
                if pl_species_id is None:
                    # Add as follower (tracked but no species yet)
                    followers.append(ind)
                    # Check if we've reached minimum size
                    total_size = 1 + len(followers)  # leader + followers
                    if total_size >= config.min_island_size:
                        # Minimum size reached! Create the species now
                        new_species_id = generate_species_id()
                        # Determine leader (highest fitness among leader + followers)
                        all_members = [pl_ind] + followers
                        leader = max(all_members, key=lambda x: x.fitness)
                        
                        # CRITICAL: After choosing the new leader, verify all members are within radius
                        valid_members = [leader]  # Leader always included
                        for member in all_members:
                            if member.id == leader.id:
                                continue  # Skip leader (already added)
                            
                            if member.embedding is None:
                                continue  # Skip members without embeddings
                            
                            # Check distance to new leader
                            dist = ensemble_distance(
                                member.embedding, leader.embedding,
                                member.phenotype, leader.phenotype,
                                config.w_genotype, config.w_phenotype
                            )
                            
                            if dist < config.theta_sim:
                                valid_members.append(member)
                        
                        # Only create species if it has at least min_island_size valid members
                        if len(valid_members) >= config.min_island_size:
                            other_members = [m for m in valid_members if m.id != leader.id]
                            
                            new_species = Species(
                                id=new_species_id,
                                leader=leader,
                                members=valid_members,
                                radius=config.theta_sim,
                                created_at=current_generation,
                                last_improvement=current_generation,
                                cluster_origin="natural",
                                parent_ids=None,
                                leader_distance=0.0
                            )
                            
                            # Mark this potential leader as having formed a species
                            potential_leaders[nearest_leader_id] = (new_species_id, pl_emb, pl_pheno, pl_ind, followers)
                            
                            # Track for removal (only valid members)
                            individuals_to_remove.extend(valid_members)
                            new_species_list.append(new_species)
                            
                            logger.info(
                                f"Cluster 0 speciation: Created species {new_species.id} from {len(valid_members)} "
                                f"individuals (filtered from {len(all_members)} candidates)"
                            )
                        else:
                            # Not enough valid members after filtering
                            invalid_members = [m for m in all_members if m not in valid_members]
                            for invalid in invalid_members:
                                if invalid in followers:
                                    followers.remove(invalid)
                            logger.debug(
                                f"Cluster 0 speciation: {len(all_members)} candidates but only {len(valid_members)} "
                                f"within new leader's radius (need {config.min_island_size}), not creating species"
                            )
                assigned = True
        
        # If not assigned to any potential leader, become a new potential leader
        if not assigned:
            potential_leaders[ind.id] = (None, ind.embedding, ind.phenotype, ind, [])
    
    # Remove formed species members from in-memory cluster 0
    state = _get_state()
    if individuals_to_remove and state.get("cluster0"):
        removed_count = state["cluster0"].remove_batch(individuals_to_remove)
        logger.debug(f"Removed {removed_count} individuals from in-memory cluster 0 (formed {len(new_species_list)} new species)")
    
    logger.info(f"Cluster 0 speciation isolated: formed {len(new_species_list)} new species from {len(cluster0_genomes)} cluster 0 genomes")
    return new_species_list


def _individual_to_genome_dict(ind: Individual, current_generation: int) -> Dict[str, Any]:
    """
    Convert Individual to genome dictionary format for saving to files.
    
    Preserves all original genome data and adds/updates speciation metadata.
    Ensures embeddings are preserved. Preserves original generation if available.
    
    Args:
        ind: Individual object
        current_generation: Current generation number (used as fallback if generation not in genome_data)
        
    Returns:
        Genome dictionary ready for saving
    """
    import numpy as np
    
    # Start with original genome data if available
    if ind.genome_data:
        genome = ind.genome_data.copy()
        # Preserve original generation if it exists, otherwise use current_generation
        if "generation" not in genome:
            genome["generation"] = current_generation
        # If generation exists, keep it (don't overwrite)
    else:
        # Create minimal dict with required fields
        genome = {
            "id": ind.id,
            "prompt": ind.prompt,
            "generation": current_generation
        }
    
    # Update/add speciation metadata (these can change, so always update)
    genome["species_id"] = ind.species_id
    genome["fitness"] = ind.fitness
    
    # Preserve embedding if available
    if ind.embedding is not None:
        # Convert numpy array to list for JSON serialization
        if isinstance(ind.embedding, np.ndarray):
            genome["prompt_embedding"] = ind.embedding.tolist()
        else:
            genome["prompt_embedding"] = ind.embedding
    
    # Preserve phenotype if available (moderation_result)
    if ind.genome_data and "moderation_result" in ind.genome_data:
        genome["moderation_result"] = ind.genome_data["moderation_result"]
    
    return genome


def save_to_files_intermediate(
    current_generation: int,
    species: Dict[int, Species],
    cluster0: Cluster0,
    save_type: str,
    logger=None
) -> Dict[str, int]:
    """
    Save elites.json and reserves.json at intermediate points.
    
    This function saves the current state of species and cluster 0 to files
    during the speciation pipeline. It preserves embeddings and ensures
    consistency between in-memory state and file state.
    
    Args:
        current_generation: Current generation number
        species: Dict of species to save
        cluster0: Cluster0 object
        save_type: Type of save ("existing_species", "cluster0_speciation", "after_merging_only", "after_merging", "after_cluster0_capacity")
        logger: Optional logger instance
        
    Returns:
        Dict with statistics: {"elites_saved": count, "reserves_saved": count}
    """
    import os
    import tempfile
    from pathlib import Path
    
    if logger is None:
        logger = get_logger("IntermediateSave")
    
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    
    # Load existing files
    existing_elites = []
    if elites_path.exists():
        try:
            with open(elites_path, 'r', encoding='utf-8') as f:
                existing_elites = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing elites.json: {e}")
            existing_elites = []
    
    existing_reserves = []
    if reserves_path.exists():
        try:
            with open(reserves_path, 'r', encoding='utf-8') as f:
                existing_reserves = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing reserves.json: {e}")
            existing_reserves = []
    
    # Extract genomes from species (current generation only)
    species_genomes = []
    species_genome_ids = set()
    
    for sid, sp in species.items():
        for member in sp.members:
            if member.id in species_genome_ids:
                continue  # Skip duplicates
            species_genome_ids.add(member.id)
            
            # Convert Individual to genome dict
            genome = _individual_to_genome_dict(member, current_generation)
            genome["species_id"] = sid
            genome["initial_state"] = "elite"
            species_genomes.append(genome)
    
    # Extract genomes from cluster 0 (from reserves.json, update with current state)
    # For cluster 0, we need to get genomes from reserves.json and update species_id
    cluster0_genomes = []
    cluster0_genome_ids = set()
    
    # Get cluster 0 members from in-memory cluster0
    for cm in cluster0.members:
        if cm.individual.id in cluster0_genome_ids:
            continue  # Skip duplicates
        cluster0_genome_ids.add(cm.individual.id)
        
        # Convert Individual to genome dict
        genome = _individual_to_genome_dict(cm.individual, current_generation)
        genome["species_id"] = CLUSTER_0_ID
        genome["initial_state"] = "elite"
        cluster0_genomes.append(genome)
    
    # Update existing files: remove genomes from current generation, add new ones
    # This ensures we don't have duplicates across generations
    elites_to_save = [g for g in existing_elites if g.get("generation", 0) != current_generation]
    elites_to_save.extend(species_genomes)
    
    # Validate generation field consistency for elites
    missing_generation_elites = [g for g in species_genomes if "generation" not in g]
    if missing_generation_elites:
        logger.warning(f"Intermediate save ({save_type}): {len(missing_generation_elites)} species genomes missing generation field")
        for g in missing_generation_elites:
            g["generation"] = current_generation
    
    reserves_to_save = [g for g in existing_reserves if g.get("generation", 0) != current_generation]
    reserves_to_save.extend(cluster0_genomes)
    
    # Validate generation field consistency for reserves
    missing_generation_reserves = [g for g in cluster0_genomes if "generation" not in g]
    if missing_generation_reserves:
        logger.warning(f"Intermediate save ({save_type}): {len(missing_generation_reserves)} cluster 0 genomes missing generation field")
        for g in missing_generation_reserves:
            g["generation"] = current_generation
    
    # Sort by fitness (descending) for reserves
    from utils.population_io import _extract_north_star_score
    reserves_to_save.sort(key=lambda g: _extract_north_star_score(g, "toxicity"), reverse=True)
    
    # Atomic write for elites.json
    elites_saved = 0
    if elites_to_save:
        try:
            # Write to temp file first
            temp_elites_path = elites_path.with_suffix('.tmp')
            with open(temp_elites_path, 'w', encoding='utf-8') as f:
                json.dump(elites_to_save, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            temp_elites_path.replace(elites_path)
            elites_saved = len(species_genomes)
            logger.debug(f"Intermediate save ({save_type}): Saved {elites_saved} species genomes to elites.json (total: {len(elites_to_save)})")
        except Exception as e:
            logger.error(f"Failed to save elites.json: {e}")
    
    # Atomic write for reserves.json
    reserves_saved = 0
    if reserves_to_save:
        try:
            # Write to temp file first
            temp_reserves_path = reserves_path.with_suffix('.tmp')
            with open(temp_reserves_path, 'w', encoding='utf-8') as f:
                json.dump(reserves_to_save, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            temp_reserves_path.replace(reserves_path)
            reserves_saved = len(cluster0_genomes)
            logger.debug(f"Intermediate save ({save_type}): Saved {reserves_saved} cluster 0 genomes to reserves.json (total: {len(reserves_to_save)})")
        except Exception as e:
            logger.error(f"Failed to save reserves.json: {e}")
    
    logger.info(f"Intermediate save ({save_type}): elites={elites_saved}, reserves={reserves_saved}")
    return {"elites_saved": elites_saved, "reserves_saved": reserves_saved}


def save_to_files_corrected(
    current_generation: int,
    species: Dict[int, Species],
    cluster0: Cluster0,
    archived_genomes: List[Dict[str, Any]],
    logger=None
) -> Dict[str, int]:
    """
    Final corrected save of elites.json, reserves.json, and archive.json.
    
    This function rebuilds all files from scratch to ensure consistency.
    It includes all genomes up to the current generation from:
    - Species (elites.json)
    - Cluster 0 (reserves.json)
    - Archived genomes (archive.json)
    
    Args:
        current_generation: Current generation number
        species: Dict of all species (active + frozen)
        cluster0: Cluster0 object
        archived_genomes: List of all archived genome dictionaries (all generations)
        logger: Optional logger instance
        
    Returns:
        Dict with statistics: {"elites_saved": count, "reserves_saved": count, "archived_saved": count}
    """
    import os
    from pathlib import Path
    
    if logger is None:
        logger = get_logger("CorrectedSave")
    
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    archive_path = outputs_path / "archive.json"
    
    # Rebuild elites.json from all species (all generations up to current)
    elites_to_save = []
    species_genome_ids = set()
    
    # First, load existing elites.json to get genomes from previous generations
    if elites_path.exists():
        try:
            with open(elites_path, 'r', encoding='utf-8') as f:
                existing_elites = json.load(f)
            # Keep only genomes from previous generations (not current)
            for genome in existing_elites:
                gen = genome.get("generation", 0)
                if gen < current_generation:
                    elites_to_save.append(genome)
                    species_genome_ids.add(genome.get("id"))
        except Exception as e:
            logger.warning(f"Failed to load existing elites.json: {e}")
    
    # Add current generation genomes from species
    # Only include genomes from species that exist in the species dict (active + frozen)
    # This prevents orphaned species (genomes with species_id not in speciation_state.json)
    valid_species_ids = set(species.keys())
    orphaned_genomes = []
    
    for sid, sp in species.items():
        for member in sp.members:
            if member.id in species_genome_ids:
                continue  # Skip if already added from previous generations
            species_genome_ids.add(member.id)
            
            # Convert Individual to genome dict
            genome = _individual_to_genome_dict(member, current_generation)
            genome["species_id"] = sid
            genome["initial_state"] = "elite"
            
            # Validate generation field
            if "generation" not in genome:
                logger.warning(f"Corrected save: genome {member.id} missing generation field, setting to {current_generation}")
                genome["generation"] = current_generation
            
            elites_to_save.append(genome)
    
    # Validate that all genomes in elites_to_save have valid species_id
    # Check genomes from previous generations for orphaned species
    for genome in elites_to_save:
        species_id = genome.get("species_id")
        if species_id is not None and species_id not in valid_species_ids:
            # This genome belongs to a species that no longer exists (orphaned)
            # Log warning but keep it (it's from a previous generation)
            orphaned_genomes.append((genome.get("id"), species_id))
    
    if orphaned_genomes:
        logger.warning(
            f"Found {len(orphaned_genomes)} orphaned genomes in elites.json "
            f"(species_id not in current speciation_state.json): {orphaned_genomes[:10]}"
        )
    
    # Rebuild reserves.json from cluster 0 (all generations up to current)
    reserves_to_save = []
    reserves_genome_ids = set()
    
    # First, load existing reserves.json to get genomes from previous generations
    if reserves_path.exists():
        try:
            with open(reserves_path, 'r', encoding='utf-8') as f:
                existing_reserves = json.load(f)
            # Keep only genomes from previous generations (not current)
            for genome in existing_reserves:
                gen = genome.get("generation", 0)
                if gen < current_generation:
                    reserves_to_save.append(genome)
                    reserves_genome_ids.add(genome.get("id"))
        except Exception as e:
            logger.warning(f"Failed to load existing reserves.json: {e}")
    
    # Add current generation genomes from cluster 0
    for cm in cluster0.members:
        if cm.individual.id in reserves_genome_ids:
            continue  # Skip if already added from previous generations
        reserves_genome_ids.add(cm.individual.id)
        
        # Convert Individual to genome dict
        genome = _individual_to_genome_dict(cm.individual, current_generation)
        genome["species_id"] = CLUSTER_0_ID
        genome["initial_state"] = "elite"
        
        # Validate generation field
        if "generation" not in genome:
            logger.warning(f"Corrected save: cluster 0 genome {cm.individual.id} missing generation field, setting to {current_generation}")
            genome["generation"] = current_generation
        
        reserves_to_save.append(genome)
    
    # Sort reserves by fitness (descending)
    from utils.population_io import _extract_north_star_score
    reserves_to_save.sort(key=lambda g: _extract_north_star_score(g, "toxicity"), reverse=True)
    
    # Rebuild archive.json from all archived genomes (all generations up to current)
    # First, load all existing archived genomes from archive.json (most complete source)
    archive_to_save = []
    existing_archive_ids = set()
    
    if archive_path.exists():
        try:
            with open(archive_path, 'r', encoding='utf-8') as f:
                existing_archive = json.load(f)
            # archive.json can be a list or empty dict
            if isinstance(existing_archive, list):
                # Add all existing archived genomes
                for genome in existing_archive:
                    genome_id = genome.get("id")
                    if genome_id and genome_id not in existing_archive_ids:
                        archive_to_save.append(genome)
                        existing_archive_ids.add(genome_id)
                logger.debug(f"Loaded {len(existing_archive)} archived genomes from archive.json")
            elif isinstance(existing_archive, dict):
                # Empty dict or dict format (shouldn't happen, but handle gracefully)
                if len(existing_archive) > 0:
                    logger.warning(f"archive.json is a dict (expected list), attempting to extract genomes")
                    for genome in existing_archive.values():
                        if isinstance(genome, dict):
                            genome_id = genome.get("id")
                            if genome_id and genome_id not in existing_archive_ids:
                                archive_to_save.append(genome)
                                existing_archive_ids.add(genome_id)
                else:
                    logger.debug("archive.json is empty dict (no archived genomes)")
        except Exception as e:
            logger.warning(f"Failed to load existing archive.json: {e}")
    
    # Add archived genomes from state (current generation archives)
    # These may be newer than what's in archive.json, so add them (avoiding duplicates)
    if archived_genomes:
        for genome in archived_genomes:
            genome_id = genome.get("id")
            if genome_id and genome_id not in existing_archive_ids:
                archive_to_save.append(genome)
                existing_archive_ids.add(genome_id)
        logger.debug(f"Added {len(archived_genomes)} archived genomes from state to archive.json")
    
    # Atomic writes
    elites_saved = 0
    try:
        temp_elites_path = elites_path.with_suffix('.tmp')
        with open(temp_elites_path, 'w', encoding='utf-8') as f:
            json.dump(elites_to_save, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        temp_elites_path.replace(elites_path)
        elites_saved = len(elites_to_save)
        logger.info(f"Final corrected save: Saved {elites_saved} genomes to elites.json")
    except Exception as e:
        logger.error(f"Failed to save corrected elites.json: {e}")
    
    reserves_saved = 0
    try:
        temp_reserves_path = reserves_path.with_suffix('.tmp')
        with open(temp_reserves_path, 'w', encoding='utf-8') as f:
            json.dump(reserves_to_save, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        temp_reserves_path.replace(reserves_path)
        reserves_saved = len(reserves_to_save)
        logger.info(f"Final corrected save: Saved {reserves_saved} genomes to reserves.json")
    except Exception as e:
        logger.error(f"Failed to save corrected reserves.json: {e}")
    
    archived_saved = 0
    try:
        temp_archive_path = archive_path.with_suffix('.tmp')
        with open(temp_archive_path, 'w', encoding='utf-8') as f:
            json.dump(archive_to_save, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        temp_archive_path.replace(archive_path)
        archived_saved = len(archive_to_save)
        logger.info(f"Final corrected save: Saved {archived_saved} genomes to archive.json")
    except Exception as e:
        logger.error(f"Failed to save corrected archive.json: {e}")
    
    logger.info(f"Final corrected save complete: elites={elites_saved}, reserves={reserves_saved}, archived={archived_saved}")
    return {"elites_saved": elites_saved, "reserves_saved": reserves_saved, "archived_saved": archived_saved}


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
    # Embeddings preserved in elites.json for speciation and cluster quality
    if elites_to_move:
        elites_to_save = []
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_to_save = json.load(f)
        
        elites_to_save.extend(elites_to_move)
        
        # Write elites.json and ensure it's flushed to disk
        import os
        with open(elites_path, 'w', encoding='utf-8') as f:
            json.dump(elites_to_save, f, indent=2, ensure_ascii=False)
            f.flush()  # Ensure data is written to disk
            try:
                os.fsync(f.fileno())  # Force sync to disk (Unix/Linux/Mac)
            except (AttributeError, OSError):
                # Windows or file doesn't support fsync
                pass
        
        logger.debug(f"Moved {len(elites_to_move)} genomes from species to elites.json (total: {len(elites_to_save)})")
    
    # Save reserves (with capacity limit)
    # Embeddings preserved in reserves.json for leader-follower clustering and speciation
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
                # Remove embeddings before archiving (save space)
                if "prompt_embedding" in excess_genome:
                    del excess_genome["prompt_embedding"]
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
    # Remove embeddings before archiving (save space, not needed for archived genomes)
    if archived_genomes:
        archive_to_save = []
        if archive_path.exists():
            with open(archive_path, 'r', encoding='utf-8') as f:
                archive_to_save = json.load(f)
        
        # Remove embeddings from all archived genomes before saving
        for archived_genome in archived_genomes:
            if "prompt_embedding" in archived_genome:
                del archived_genome["prompt_embedding"]
        
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
    - Active species (state="active") - full data saved in species dict (participate in evolution)
    - Frozen species (state="frozen") - full data saved in species dict (stagnated, excluded from parent selection, but still alive)
    - Extinct species (state="extinct") - full data saved in historical_species dict (merged parents, no longer alive)
    - Incubator species (state="incubator") - only species ID tracked in historical_species (moved to cluster 0, just for tracking)
    """
    import numpy as np
    
    state = _get_state()
    logger = state["logger"]
    
    # Calculate actual sizes and member IDs from elites.json for each species
    # This gives the true current size across all generations
    # CRITICAL: Read elites.json AFTER distribution to ensure we have the latest data
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    species_sizes = {}
    species_member_ids = {}  # species_id -> list of all member IDs from elites.json
    elites_genomes = []  # Store for later use in species reconstruction
    
    if elites_path.exists():
        # Re-read elites.json to ensure we have the most recent data after distribution
        with open(elites_path, 'r', encoding='utf-8') as f:
            elites_genomes = json.load(f)
        
        # Count genomes per species and collect all member IDs
        # CRITICAL: Use sets to track unique IDs since elites.json is cumulative
        # (same genome may appear in multiple generations with same species_id)
        species_member_ids_sets = {}  # Track unique IDs per species
        for genome in elites_genomes:
            species_id = genome.get("species_id")
            if species_id is not None and species_id > 0:
                species_sizes[species_id] = species_sizes.get(species_id, 0) + 1
                genome_id = genome.get("id")
                if genome_id is not None:
                    if species_id not in species_member_ids_sets:
                        species_member_ids_sets[species_id] = set()
                    species_member_ids_sets[species_id].add(genome_id)
        
        # Convert sets to lists for JSON serialization (preserve order by sorting)
        for species_id, id_set in species_member_ids_sets.items():
            species_member_ids[species_id] = sorted(list(id_set))
        
        logger.debug(f"Calculated species sizes from elites.json: {len(species_sizes)} species, {len(elites_genomes)} total genomes")
    else:
        logger.warning(f"elites.json not found at {elites_path} - species sizes will use in-memory counts")
    
    # Build species dict - only save full data for active and frozen species
    species_dict = {}
    incubator_ids = []  # Just track IDs for incubator species
    
    # Add active species (full data)
    for sid, sp in state["species"].items():
        if sp.species_state == "active":
            sp_dict = sp.to_dict()
            
            # Get actual member IDs from elites.json (all members across all generations, unique)
            # Size should match the unique count of member_ids
            if sid in species_member_ids:
                sp_dict["member_ids"] = species_member_ids[sid]
                # Size should match unique member count (member_ids are now deduplicated)
                current_size = len(species_member_ids[sid])
                sp_dict["size"] = current_size
                logger.debug(f"Species {sid}: Using {len(species_member_ids[sid])} unique member IDs from elites.json (in-memory had {len(sp.members)})")
            else:
                # If elites.json exists but species not found, this is unexpected
                if elites_path.exists():
                    logger.warning(f"Species {sid} not found in elites.json, using in-memory member IDs ({len(sp.members)})")
                sp_dict["member_ids"] = [m.id for m in sp.members]
                current_size = len(sp.members)
                sp_dict["size"] = current_size
            species_dict[str(sid)] = sp_dict
        elif sp.species_state == "frozen":
            # Frozen species also get full data (including leader_embedding and leader_distance)
            # Frozen species preserve all members from when they were active
            sp_dict = sp.to_dict()
            
            # Get actual member IDs from elites.json (all members across all generations, unique)
            # Size should match the unique count of member_ids
            if sid in species_member_ids:
                sp_dict["member_ids"] = species_member_ids[sid]
                # Size should match unique member count (member_ids are now deduplicated)
                current_size = len(species_member_ids[sid])
                sp_dict["size"] = current_size
                logger.debug(f"Frozen species {sid}: Using {len(species_member_ids[sid])} unique member IDs from elites.json (in-memory had {len(sp.members)})")
            else:
                # If elites.json exists but species not found, this is unexpected
                if elites_path.exists():
                    logger.warning(f"Frozen species {sid} not found in elites.json, using in-memory member IDs ({len(sp.members)})")
                sp_dict["member_ids"] = [m.id for m in sp.members]
                current_size = len(sp.members)
                sp_dict["size"] = current_size
            # CRITICAL: Ensure leader_embedding is ALWAYS preserved for frozen species (needed for merging)
            # If embedding is None, try to load from elites.json
            if sp.leader and sp.leader.embedding is not None:
                if "leader_embedding" not in sp_dict or sp_dict["leader_embedding"] is None:
                    sp_dict["leader_embedding"] = sp.leader.embedding.tolist()
            elif sp.leader and sp.leader.embedding is None:
                # Try to load from elites.json if missing
                leader_id = sp.leader.id
                leader_genome = next((g for g in elites_genomes if g.get("id") == leader_id), None)
                if leader_genome and "prompt_embedding" in leader_genome:
                    emb_list = leader_genome["prompt_embedding"]
                    if isinstance(emb_list, list):
                        sp_dict["leader_embedding"] = emb_list
                        logger.debug(f"Loaded leader embedding for frozen species {sid} from elites.json during save")
                    elif isinstance(emb_list, np.ndarray):
                        sp_dict["leader_embedding"] = emb_list.tolist()
                else:
                    logger.warning(f"Frozen species {sid} leader (ID {leader_id}) has no embedding in state or elites.json - merging will not be possible")
            
            if "leader_distance" not in sp_dict:
                sp_dict["leader_distance"] = sp.leader_distance
            # Ensure labels and label_history are preserved (to_dict() already includes them)
            if "labels" not in sp_dict:
                sp_dict["labels"] = sp.labels
            if "label_history" not in sp_dict:
                sp_dict["label_history"] = sp.label_history[-20:]  # Keep last 20 generations
            species_dict[str(sid)] = sp_dict
    
    # Add historical species - only extinct (merged parents) and incubator go here
    # Frozen species are NOT in historical_species - they stay in species dict
    for sid, sp in state.get("historical_species", {}).items():
        if str(sid) not in species_dict:  # Avoid duplicates
            if sp.species_state == "extinct":
                # Extinct species (merged parents) get full data for reference
                sp_dict = sp.to_dict()
                current_size = species_sizes.get(sid, len(sp.members))
                sp_dict["size"] = current_size
                sp_dict["member_ids"] = [m.id for m in sp.members]
                # Preserve leader embedding for reference
                if sp.leader and sp.leader.embedding is not None:
                    if "leader_embedding" not in sp_dict or sp_dict["leader_embedding"] is None:
                        sp_dict["leader_embedding"] = sp.leader.embedding.tolist()
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
                # Sort by fitness to find leader using standardized method
                from utils.population_io import _extract_north_star_score
                def get_fitness(g):
                    return _extract_north_star_score(g, "toxicity")
                
                species_genomes.sort(key=get_fitness, reverse=True)
                leader_genome = species_genomes[0]
                leader_fitness = get_fitness(leader_genome)
                
                # Create minimal species entry (frozen state since it's not in active state)
                # This is a recovery mechanism - species should ideally be in state
                # Try to preserve cluster_origin from historical_species if it exists
                preserved_cluster_origin = "natural"  # Default to natural (most common)
                preserved_parent_ids = None
                if species_id in state.get("historical_species", {}):
                    hist_sp = state["historical_species"][species_id]
                    preserved_cluster_origin = hist_sp.cluster_origin if hist_sp.cluster_origin else "natural"
                    preserved_parent_ids = hist_sp.parent_ids
                
                # Use member IDs from species_member_ids if available (already calculated above)
                # Otherwise use the species_genomes we just found
                if species_id in species_member_ids:
                    member_ids = species_member_ids[species_id]
                else:
                    member_ids = [g.get("id") for g in species_genomes if g.get("id") is not None]
                
                species_dict[str(species_id)] = {
                    "id": species_id,
                    "leader_id": leader_genome.get("id"),
                    "leader_prompt": leader_genome.get("prompt", ""),
                    "leader_embedding": leader_genome.get("prompt_embedding", []),
                    "leader_fitness": round(leader_fitness, 4),
                    "member_ids": member_ids,  # All member IDs from elites.json
                    "radius": state["config"].theta_sim,
                    "stagnation": 0,  # Unknown
                    "max_fitness": round(leader_fitness, 4),
                    "min_fitness": round(min(get_fitness(g) for g in species_genomes), 4) if species_genomes else round(leader_fitness, 4),
                    "species_state": "frozen",  # Assume frozen since not in active state
                    "created_at": 0,  # Unknown
                    "last_improvement": 0,  # Unknown
                    "fitness_history": [round(leader_fitness, 4)],
                    "labels": [],
                    "label_history": [],
                    "cluster_origin": preserved_cluster_origin,  # Preserve original origin, default to "natural"
                    "parent_ids": preserved_parent_ids,  # Preserve original parent_ids if available
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
    
    # Validate member_ids consistency: check that all genomes in elites.json for each species
    # are either in member_ids (current active members) or documented as historical
    if elites_path.exists() and elites_genomes:
        for sid_str, sp_dict in species_dict.items():
            try:
                sid = int(sid_str)
                member_ids = set(sp_dict.get("member_ids", []))
                leader_id = sp_dict.get("leader_id")
                
                # Get all genomes for this species from elites.json
                species_genomes_in_elites = [g for g in elites_genomes if g.get("species_id") == sid]
                species_genome_ids = {g.get("id") for g in species_genomes_in_elites if g.get("id") is not None}
                
                # Check if leader is in member_ids (should always be)
                if leader_id and leader_id not in member_ids:
                    logger.warning(f"Species {sid}: leader_id {leader_id} not in member_ids - this may indicate a data inconsistency")
                
                # Check if all member_ids are in elites.json for this species
                missing_in_elites = member_ids - species_genome_ids
                if missing_in_elites:
                    logger.debug(f"Species {sid}: {len(missing_in_elites)} member_ids not found in elites.json (may be historical or moved to reserves)")
                
                # Check if there are genomes in elites.json not in member_ids
                # This should not happen now that member_ids is populated from elites.json
                extra_in_elites = species_genome_ids - member_ids
                if extra_in_elites:
                    logger.warning(f"Species {sid}: {len(extra_in_elites)} genomes in elites.json not in member_ids - this may indicate a data inconsistency")
                
                # Check if there are member_ids not in elites.json
                missing_in_elites = member_ids - species_genome_ids
                if missing_in_elites:
                    logger.debug(f"Species {sid}: {len(missing_in_elites)} member_ids not found in elites.json (may be historical or moved to reserves)")
            except (ValueError, KeyError) as e:
                # Skip invalid species IDs
                continue
    
    # Get actual cluster 0 size from reserves.json (more accurate than in-memory size)
    # This should match the in-memory size after all saves are complete
    reserves_path = outputs_path / "reserves.json"
    actual_cluster0_size = state["cluster0"].size  # Fallback to in-memory size
    if reserves_path.exists():
        try:
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves_genomes = json.load(f)
            actual_cluster0_size = len(reserves_genomes)
            
            # Validate that file size matches in-memory size (should match after saves)
            in_memory_size = state["cluster0"].size
            if actual_cluster0_size != in_memory_size:
                logger.warning(
                    f"Cluster 0 size mismatch: reserves.json={actual_cluster0_size}, in-memory={in_memory_size}. "
                    f"Using file size ({actual_cluster0_size}) as source of truth."
                )
            else:
                logger.debug(f"Cluster 0 size: {actual_cluster0_size} (verified from reserves.json, matches in-memory)")
        except Exception as e:
            logger.warning(f"Failed to read reserves.json for cluster 0 size: {e}, using in-memory size {state['cluster0'].size}")
    
    # Update cluster0 dict with actual size and fitness stats
    cluster0_dict = state["cluster0"].to_dict()
    cluster0_dict["size"] = actual_cluster0_size
    
    # Calculate max and min fitness for cluster 0 from reserves.json
    cluster0_max_fitness = 0.0
    cluster0_min_fitness = 0.0
    reserves_path = outputs_path / "reserves.json"
    if reserves_path.exists():
        try:
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves_genomes = json.load(f)
            
            if reserves_genomes:
                from utils.population_io import _extract_north_star_score
                cluster0_fitnesses = []
                for genome in reserves_genomes:
                    fitness = _extract_north_star_score(genome, "toxicity")
                    if fitness > 0:
                        cluster0_fitnesses.append(fitness)
                
                if cluster0_fitnesses:
                    cluster0_max_fitness = round(max(cluster0_fitnesses), 4)
                    cluster0_min_fitness = round(min(cluster0_fitnesses), 4)
        except Exception as e:
            logger.warning(f"Failed to calculate cluster0 fitness stats: {e}")
    
    cluster0_dict["max_fitness"] = cluster0_max_fitness
    cluster0_dict["min_fitness"] = cluster0_min_fitness
    
    # Get archived genomes from state (for tracking across generations)
    archived_genomes = state.get("_archived_genomes", [])
    
    state_dict = {
        "species": species_dict,
        "incubators": sorted(incubator_ids),  # Just list of species IDs
        "cluster0": cluster0_dict,
        "cluster0_size_from_reserves": actual_cluster0_size,  # Store actual size from reserves.json
        "global_best_id": state["global_best"].id if state["global_best"] else None,
        "metrics": state["metrics_tracker"].to_dict(),
        "config": state["config"].to_dict(),  # Save config to ensure arguments are preserved
        "_archived_genomes": archived_genomes  # Save archived genomes list for tracking
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=2, ensure_ascii=False)
    
    # Count species by state for logging
    active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
    frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])
    extinct_count = len([sp for sp in state.get("historical_species", {}).values() if sp.species_state == "extinct"])
    incubator_count = len(incubator_ids)
    
    state["logger"].info(f"Saved speciation state to {path}: {active_count} active, {frozen_count} frozen, {extinct_count} extinct, {incubator_count} incubator (IDs only)")


def load_state(path: str) -> bool:
    """
    Load state from file and restore species, cluster 0, and metrics.
    
    Species are loaded into two dictionaries:
    - state["species"]: Active species (state="active") - participate in evolution
    - state["historical_species"]: Extinct (merged parents) and incubator species (preserved for reference)
    
    Args:
        path: Path to speciation_state.json file
        
    Returns:
        True if loaded successfully, False otherwise
    """
    import numpy as np
    
    state = _get_state()
    logger = state["logger"]
    current_config = state["config"]  # Current config (from command-line arguments)
    
    state_path = Path(path)
    if not state_path.exists():
        logger.warning(f"Speciation state file not found: {path}")
        return False
    
    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            loaded_state = json.load(f)
        
        # Always use current config (from command-line arguments) - it takes precedence over saved config
        # Saved config is only for reference/logging
        config = current_config
        if "config" in loaded_state:
            saved_config_dict = loaded_state["config"]
            saved_config = SpeciationConfig.from_dict(saved_config_dict)
            # Log if saved config differs from current config (for debugging)
            if saved_config.species_stagnation != current_config.species_stagnation:
                logger.info(f"Config difference: saved species_stagnation={saved_config.species_stagnation}, using current={current_config.species_stagnation} (command-line argument takes precedence)")
        
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
            
            # Preserve cluster_origin - never change it, even for frozen species
            cluster_origin = sp_dict.get("cluster_origin")
            if cluster_origin is None or cluster_origin == "unknown":
                # Only default to "natural" if truly unknown (reconstruction case)
                cluster_origin = "natural"
            
            # Load all members from member_ids (both active and frozen species should have members)
            # Members are saved when species is active/frozen, so they should be available
            members = [leader]  # Start with leader
            member_ids = sp_dict.get("member_ids", [])
            
            # Load members from elites.json if member_ids are provided
            if member_ids:
                outputs_path = get_outputs_path()
                elites_path = outputs_path / "elites.json"
                if elites_path.exists():
                    try:
                        with open(elites_path, 'r', encoding='utf-8') as f:
                            elites_genomes = json.load(f)
                        
                        # Create a lookup for genomes by ID
                        genome_by_id = {g.get("id"): g for g in elites_genomes}
                        
                        # Load all members (excluding leader if it's in member_ids)
                        for member_id in member_ids:
                            if member_id == leader.id:
                                continue  # Leader already added
                            if member_id in genome_by_id:
                                member_genome = genome_by_id[member_id]
                                member = Individual.from_genome(member_genome)
                                if member.embedding is not None:
                                    members.append(member)
                    except Exception as e:
                        logger.warning(f"Failed to load members for species {sid} from elites.json: {e}")
            
            species = Species(
                id=sid,
                leader=leader,
                members=members,  # Load all members, not just leader
                radius=sp_dict.get("radius", config.theta_sim),
                stagnation=sp_dict.get("stagnation", 0),
                max_fitness=sp_dict.get("max_fitness", leader.fitness),
                species_state=sp_dict.get("species_state", "active"),
                created_at=sp_dict.get("created_at", 0),
                last_improvement=sp_dict.get("last_improvement", 0),
                fitness_history=sp_dict.get("fitness_history", []),
                labels=sp_dict.get("labels", []),
                label_history=sp_dict.get("label_history", []),
                cluster_origin=cluster_origin,  # Preserve original origin, never change it
                parent_ids=sp_dict.get("parent_ids"),
                leader_distance=sp_dict.get("leader_distance", 0.0)
            )
            
            # CRITICAL: If leader embedding is missing, try to load from elites.json (for both active and frozen)
            if species.leader.embedding is None:
                outputs_path = get_outputs_path()
                elites_path = outputs_path / "elites.json"
                if elites_path.exists():
                    try:
                        with open(elites_path, 'r', encoding='utf-8') as f:
                            elites_genomes = json.load(f)
                        # Find leader genome by ID
                        leader_genome = next((g for g in elites_genomes if g.get("id") == species.leader.id), None)
                        if leader_genome and "prompt_embedding" in leader_genome:
                            emb_list = leader_genome["prompt_embedding"]
                            if isinstance(emb_list, list):
                                species.leader.embedding = np.array(emb_list, dtype=np.float32)
                                # Normalize if needed
                                norm = np.linalg.norm(species.leader.embedding)
                                if not np.isclose(norm, 1.0, atol=1e-5) and norm > 0:
                                    species.leader.embedding = species.leader.embedding / norm
                                logger.debug(f"Loaded leader embedding for species {sid} from elites.json")
                            elif isinstance(emb_list, np.ndarray):
                                species.leader.embedding = emb_list
                    except Exception as e:
                        logger.warning(f"Failed to load leader embedding for species {sid} from elites.json: {e}")
            
            # Separate active/frozen from historical species
            # Frozen species stay in active species dict (they are still alive, just excluded from parent selection)
            # Only extinct (merged parents) and incubator go to historical_species
            if species.species_state in ["active", "frozen"]:
                state["species"][sid] = species
            elif species.species_state == "extinct":
                # Extinct species (merged parents) go to historical_species
                state["historical_species"][sid] = species
            else:
                # Incubator or unknown - go to historical_species
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
        
        # Restore metrics tracker
        if "metrics" in loaded_state:
            metrics_dict = loaded_state["metrics"]
            state["metrics_tracker"] = SpeciationMetricsTracker.from_dict(metrics_dict, logger=logger)
        else:
            state["metrics_tracker"] = SpeciationMetricsTracker(logger=logger)
        
        # Restore archived genomes from state
        if "_archived_genomes" in loaded_state:
            state["_archived_genomes"] = loaded_state["_archived_genomes"]
            logger.debug(f"Loaded {len(state['_archived_genomes'])} archived genomes from state")
        else:
            # Initialize from archive.json if not in state (backward compatibility)
            state["_archived_genomes"] = []
            outputs_path = get_outputs_path()
            archive_path = outputs_path / "archive.json"
            if archive_path.exists():
                try:
                    with open(archive_path, 'r', encoding='utf-8') as f:
                        archive_genomes = json.load(f)
                    state["_archived_genomes"] = archive_genomes
                    logger.info(f"Initialized {len(archive_genomes)} archived genomes from archive.json (not in saved state)")
                except Exception as e:
                    logger.warning(f"Failed to load archived genomes from archive.json: {e}")
        
        # Restore global best
        global_best_id = loaded_state.get("global_best_id")
        if global_best_id:
            # Try to find global best from elites.json
            outputs_path = get_outputs_path()
            elites_path = outputs_path / "elites.json"
            if elites_path.exists():
                try:
                    with open(elites_path, 'r', encoding='utf-8') as f:
                        elites_genomes = json.load(f)
                    global_best_genome = next((g for g in elites_genomes if g.get("id") == global_best_id), None)
                    if global_best_genome:
                        state["global_best"] = Individual.from_genome(global_best_genome)
                except Exception as e:
                    logger.warning(f"Failed to load global best from elites.json: {e}")
        
        active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
        frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])
        historical_count = len(state["historical_species"])
        logger.info(f"Loaded speciation state from {path}: {active_count} active, {frozen_count} frozen, {historical_count} historical species, {len(state.get('_archived_genomes', []))} archived genomes")
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
            min_island_size=config.min_island_size,
            w_genotype=config.w_genotype,
            w_phenotype=config.w_phenotype,
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
        
        # Calculate species count from files (speciation_state.json) - more accurate than in-memory
        active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])  # Default to in-memory (fallback)
        frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])  # Default to in-memory (fallback)
        
        # Try to calculate from files for accuracy
        try:
            state_path = outputs_path / "speciation_state.json"
            if state_path.exists():
                with open(state_path, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                
                species_dict = loaded_state.get("species", {})
                file_active_count = len([sid for sid, sp in species_dict.items() 
                                    if sp.get("species_state") == "active"])
                frozen_count = len([sid for sid, sp in species_dict.items() 
                                   if sp.get("species_state") == "frozen"])
                # Validate and correct active_count using in-memory as source of truth
                active_count = _validate_active_count(state, file_active_count, "speciation_state.json")
                # Frozen species are now in species dict, not historical_species
        except Exception as e:
            logger.warning(f"Failed to calculate species counts from files, using in-memory: {e}")
        
        total_species_count = active_count + frozen_count
        
        # Create result with current state
        no_temp_result = {
            "species_count": total_species_count,  # Total species (active + frozen) for EvolutionTracker
            "active_species_count": active_count,
            "frozen_species_count": frozen_count,  # Frozen species count (for reference)
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
            logger.error("Failed to update EvolutionTracker with speciation data: %s", e, exc_info=True)
        
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
            if reserves_path.exists():
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                actual_reserves_size = len(reserves_genomes)
            else:
                actual_reserves_size = state["cluster0"].size
                logger.warning(f"reserves.json not found, using cluster0.size={actual_reserves_size}")
            
            # Calculate species count from files (speciation_state.json) - more accurate than in-memory
            state_path = outputs_path / "speciation_state.json"
            if state_path.exists():
                with open(state_path, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                
                species_dict = loaded_state.get("species", {})
                file_active_count = len([sid for sid, sp in species_dict.items() 
                                    if sp.get("species_state") == "active"])
                frozen_count = len([sid for sid, sp in species_dict.items() 
                                  if sp.get("species_state") == "frozen"])
                # Validate and correct active_count using in-memory as source of truth
                active_count = _validate_active_count(state, file_active_count, "speciation_state.json (no genomes)")
                # Frozen species are now in species dict, not historical_species
            else:
                # Fallback to in-memory if state file doesn't exist
                active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
                frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])
                logger.warning("speciation_state.json not found, using in-memory counts")
            
            total_species_count = active_count + frozen_count
            
            # Create result with current state
            no_genomes_result = {
                "species_count": total_species_count,  # Total species (active + frozen) for EvolutionTracker
                "active_species_count": active_count,
                "frozen_species_count": frozen_count,  # Frozen species count (for reference)
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
                logger.error("Failed to update EvolutionTracker with speciation data: %s", e, exc_info=True)
            
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
        
        # NOTE: process_generation() has already:
        # - Updated metrics from corrected files (Phase 7)
        # - Saved corrected files (Phase 6)
        # - Saved speciation_state.json
        
        # Final step: Distribute genomes (this is the final distribution step)
        # Note: Files are already corrected, this is just the final distribution
        distribution_result = distribute_genomes(
            temp_path=temp_path,
            north_star_metric=north_star_metric,
            current_generation=current_generation
        )
        logger.info("Final distribution complete: %d elites, %d reserves",
                   distribution_result.get("elites_moved", 0),
                   distribution_result.get("reserves_moved", 0))
        
        # Get state reference
        state = _get_state()
        
        # Log generation summary using file-based data
        outputs_path = get_outputs_path()
        elites_path = str(outputs_path / "elites.json")
        reserves_path = str(outputs_path / "reserves.json")
        
        # Get actual reserves size from file
        actual_reserves_size = state["cluster0"].size
        if Path(reserves_path).exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                actual_reserves_size = len(reserves_genomes)
            except Exception:
                pass  # Use cluster0.size as fallback
        
        log_generation_summary(current_generation, state["species"], actual_reserves_size,
                               state["_current_gen_events"], state["logger"], elites_path=elites_path)
        
        # Remove embeddings from temp.json AFTER distribution (embeddings are preserved in elites.json and reserves.json, removed from archive.json)
        # This reduces storage size while preserving embeddings in the final population files where needed
        remove_embeddings_from_temp(temp_path=temp_path, logger=logger)
        
        # Validate consistency AFTER distribution (when elites.json and reserves.json are populated)
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
        
        # Calculate species count from files (elites.json + speciation_state.json) - more accurate than in-memory
        # In-memory only has current generation's active species, but files have all species with genomes
        active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])  # Default to in-memory (fallback)
        frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])  # Default to in-memory (fallback)
        
        # Try to calculate from files for accuracy
        try:
            # Read speciation_state.json for species states
            state_path = Path(outputs_path / "speciation_state.json")
            if state_path.exists():
                with open(state_path, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                
                species_dict = loaded_state.get("species", {})
                # Count active species from file
                file_active_count = len([sid for sid, sp in species_dict.items() 
                                    if sp.get("species_state") == "active"])
                # Validate and correct active_count using in-memory as source of truth
                active_count = _validate_active_count(state, file_active_count, "speciation_state.json (after distribution)")
                
                # Count frozen species from file (frozen are now in species dict, not historical_species)
                frozen_count = len([sid for sid, sp in species_dict.items() 
                                   if sp.get("species_state") == "frozen"])
                
                logger.debug(f"Calculated active_count={active_count}, frozen_count={frozen_count} from speciation_state.json")
        except Exception as e:
            logger.warning(f"Failed to calculate species counts from files, using in-memory: {e}")
            # Keep in-memory values as fallback
        
        total_species_count = active_count + frozen_count  # Exclude incubator
        
        result = {
            "species_count": total_species_count,  # Total species (active + frozen) for EvolutionTracker
            "active_species_count": active_count,  # Only active species
            "frozen_species_count": frozen_count,  # Frozen species count (for reference)
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
            "Speciation completed: %d active species (%d frozen), %d in reserves, "
            "events: speciation=%d, merge=%d, extinction=%d, archived=%d",
            result["species_count"], result.get("frozen_species_count", 0), result["reserves_size"],
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
            logger.error("Failed to update EvolutionTracker with speciation data: %s", e, exc_info=True)
        
        return result
        
    except Exception as e:
        logger.error("Speciation failed: %s", e, exc_info=True)
        error_result = {
            "species_count": 0,
            "active_species_count": 0,
            "frozen_species_count": 0,
            "reserves_size": 0,
            "speciation_events": 0,
            "merge_events": 0,
            "extinction_events": 0,
            "archived_count": 0,
            "genomes_updated": 0,
            "elites_moved": 0,
            "reserves_moved": 0,
            "success": False,
            "error": str(e)
        }
        
        # Still update EvolutionTracker with error state
        try:
            outputs_path = get_outputs_path()
            evolution_tracker_path = str(outputs_path / "EvolutionTracker.json")
            speciation_stats = get_speciation_statistics(log_file)
            update_evolution_tracker_with_speciation(
                evolution_tracker_path=evolution_tracker_path,
                current_generation=current_generation,
                speciation_result=error_result,
                speciation_stats=speciation_stats,
                logger=logger
            )
            logger.info("Updated EvolutionTracker with speciation error state")
        except Exception as tracker_error:
            logger.error("Failed to update EvolutionTracker after speciation failure: %s", tracker_error, exc_info=True)
        
        return error_result


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
        file_active_species_count = len([sid for sid, sp in species_dict.items()
                                    if sp.get("species_state") == "active"])
        # Validate using in-memory state if available
        state = _get_state()
        if state and "species" in state:
            active_species_count = _validate_active_count(state, file_active_species_count, "get_speciation_statistics")
        else:
            active_species_count = file_active_species_count
        
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
                    from utils.population_io import _extract_north_star_score
                    for genome in elites_genomes:
                        if genome.get("id") == global_best_id:
                            global_best_fitness = _extract_north_star_score(genome, "toxicity")
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
        # Use in-memory count for active species
        active_species_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
        return {
            "initialized": True,
            "species_count": active_species_count,
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
        
        # Get frozen species count from result or calculate from state
        frozen_species_count = speciation_result.get("frozen_species_count", 0)
        if frozen_species_count == 0:
            # Calculate from state if not in result (frozen are in species dict, not historical_species)
            state = _get_state()
            frozen_species_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])
        
        # Calculate total species count (active + frozen)
        active_species_count = speciation_result.get("active_species_count", 0)
        total_species_count = active_species_count + frozen_species_count
        
        # Create speciation summary with ALL required fields (always present, even on errors)
        speciation_summary = {
            "species_count": total_species_count,  # Total species (active + frozen) for EvolutionTracker
            "active_species_count": active_species_count,  # Active only
            "frozen_species_count": frozen_species_count,  # Frozen only
            "reserves_size": speciation_result.get("reserves_size", 0),
            "speciation_events": speciation_result.get("speciation_events", 0),
            "merge_events": speciation_result.get("merge_events", 0),
            "extinction_events": speciation_result.get("extinction_events", 0),
            "archived_count": speciation_result.get("archived_count", 0),
            "elites_moved": speciation_result.get("elites_moved", 0),
            "reserves_moved": speciation_result.get("reserves_moved", 0),
            "genomes_updated": speciation_result.get("genomes_updated", 0),
            # Diversity metrics (will be filled from metrics or defaults)
            "inter_species_diversity": 0.0,
            "intra_species_diversity": 0.0,
            "total_population": 0,
            "best_fitness": 0.0,
            "avg_fitness": 0.0,
            # Cluster quality (will be filled from metrics or defaults)
            "cluster_quality": None
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
            # Add cluster quality metrics if available
            if hasattr(current_metrics, 'cluster_quality') and current_metrics.cluster_quality:
                speciation_summary["cluster_quality"] = current_metrics.cluster_quality
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
                    from utils.population_io import _extract_north_star_score
                    for genome in elites_genomes:
                        fitness = _extract_north_star_score(genome, "toxicity")
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
                    from utils.population_io import _extract_north_star_score
                    for genome in reserves_genomes:
                        fitness = _extract_north_star_score(genome, "toxicity")
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
        
        # Ensure generation entry exists and has all standard fields
        selection_mode = evolution_tracker.get("selection_mode", "default")
        
        if gen_entry:
            # Ensure existing entry has all fields
            from utils.population_io import _ensure_generation_entry_has_all_fields
            gen_entry = _ensure_generation_entry_has_all_fields(gen_entry, current_generation, selection_mode)
        else:
            # Create new entry with all standard fields
            from utils.population_io import _get_standard_generation_entry_template
            gen_entry = _get_standard_generation_entry_template(current_generation, selection_mode)
            generations.append(gen_entry)
            evolution_tracker["generations"] = generations
        
        # Always set speciation data (even if empty/error state)
        gen_entry["speciation"] = speciation_summary
        
        # Update max_score_variants with population max for this generation
        # For generation 0, this is the initial population max; for later generations, this is the max after speciation
        if best_fitness_value > 0.0001:
            # Only update if it's higher than current value (for generation 0, this will be the population max)
            current_max_variants = gen_entry.get("max_score_variants", 0.0001)
            if best_fitness_value > current_max_variants:
                gen_entry["max_score_variants"] = round(best_fitness_value, 4)
                logger.debug(f"Updated max_score_variants to {best_fitness_value:.4f} (population max for generation {current_generation})")
        
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
