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
from .events_tracker import EventsTracker
from .genome_tracker import GenomeTracker
from .validation import validate_speciation_consistency, analyze_distance_distribution, validate_flow2_speciation, validate_metrics_from_files

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
            "_archived_count": 0
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


def _save_tracker_if_dirty(state: Dict[str, Any]) -> None:
    """Save tracker if it has unsaved changes."""
    if "_genome_tracker" in state:
        tracker = state["_genome_tracker"]
        if tracker._dirty:
            tracker.save()
            state["logger"].debug("Saved genome tracker after critical operation")


def _validate_tracker_consistency(state: Dict[str, Any], phase_name: str) -> None:
    """Validate tracker consistency after a phase."""
    if "_genome_tracker" not in state:
        return
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    archive_path = outputs_path / "archive.json"
    
    is_consistent, errors = state["_genome_tracker"].validate_consistency(
        elites_path, reserves_path, archive_path, load_archive=False
    )
    if not is_consistent:
        state["logger"].warning(f"Tracker consistency check failed after {phase_name}: {len(errors)} errors")
        for error in errors[:5]:
            state["logger"].warning(f"  - {error}")


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
    Archive individuals to archive.json.
    
    This function archives individuals to archive.json. The archive.json file is the
    authoritative source for archived genomes. Genome tracking is handled by genome_tracker.json,
    and speciation_state.json only contains species metadata, not full genome details.
    """
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
            # Preserve generation field (original generation when genome was created)
            if "generation" not in entry and hasattr(ind, 'generation'):
                entry["generation"] = ind.generation
            elif "generation" not in entry:
                # Fallback: use archived_at_generation if generation not available
                entry["generation"] = generation
            # Preserve fitness if available
            if hasattr(ind, 'fitness') and "fitness" not in entry:
                entry["fitness"] = ind.fitness
            # Always set species_id=-1 for archived genomes (don't preserve ind.species_id)
            entry["species_id"] = -1
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
        
        # Save updated archive
        with open(archive_path, 'w', encoding='utf-8') as f:
            json.dump(archive, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Archived {len(individuals)} individuals ({reason}) to archive.json")
        
    except Exception as e:
        logger.warning(f"Failed to archive individuals: {e}")


def phase1_compute_embeddings(temp_path: Optional[str] = None, current_generation: int = 0) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Phase 1: Prompt to Embedding
    
    Steps:
    1. Read temp.json, sort by fitness (descending)
    2. Compute embeddings, add to genome structure
    3. Check if elites.json and reserves.json are empty to determine generation type
    4. Initialize genome tracker, register all genomes with species_id=0 (temporary)
    
    Args:
        temp_path: Path to temp.json
        current_generation: Current generation number
        
    Returns:
        Tuple of (genomes_with_embeddings, is_generation_0)
    """
    state = _get_state()
    logger = state["logger"]
    outputs_path = get_outputs_path()
    
    if temp_path is None:
        temp_path = str(outputs_path / "temp.json")
    
    temp_path_obj = Path(temp_path)
    if not temp_path_obj.exists():
        logger.warning(f"temp.json not found at {temp_path}")
        return [], True
    
    # 1. Read temp.json
    with open(temp_path_obj, 'r', encoding='utf-8') as f:
        genomes = json.load(f)
    
    if not genomes:
        logger.warning("No genomes found in temp.json")
        return [], True
    
    # 2. Sort by fitness (descending)
    from utils.population_io import _extract_north_star_score
    genomes.sort(key=lambda g: _extract_north_star_score(g, "toxicity"), reverse=True)
    
    # 3. Compute embeddings
    compute_and_save_embeddings(
        temp_path=temp_path,
        model_name=state["config"].embedding_model,
        batch_size=state["config"].embedding_batch_size,
        logger=logger
    )
    
    # Reload genomes with embeddings
    with open(temp_path_obj, 'r', encoding='utf-8') as f:
        genomes = json.load(f)
    
    # 4. Determine generation type
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    
    elites_empty = True
    reserves_empty = True
    
    if elites_path.exists():
        try:
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_genomes = json.load(f)
                elites_empty = len(elites_genomes) == 0
        except Exception:
            elites_empty = True
    
    if reserves_path.exists():
        try:
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves_genomes = json.load(f)
                reserves_empty = len(reserves_genomes) == 0
        except Exception:
            reserves_empty = True
    
    # Generation 0 if both empty OR elites.json is empty
    is_generation_0 = (elites_empty and reserves_empty) or elites_empty
    
    logger.info(f"Phase 1: Computed embeddings for {len(genomes)} genomes, is_generation_0={is_generation_0}")
    
    # 5. Register all genomes in tracker with species_id=0 (temporary, will be updated in Phase 2/3)
    if "_genome_tracker" in state:
        genome_tracker = state["_genome_tracker"]
        for genome in genomes:
            genome_id = genome.get("id")
            if genome_id:
                genome_tracker.register(str(genome_id), 0, current_generation)
    
    return genomes, is_generation_0


def phase8_redistribute_genomes(temp_path: Optional[str] = None, current_generation: int = 0) -> Dict[str, int]:
    """
    Phase 7: Redistribution of Genomes
    
    Function name is phase8_redistribute_genomes() but called as Phase 7 within process_generation().
    Uses genome tracker as source of truth for distribution.
    Called within process_generation() as Phase 7, before Phase 8 (metrics).
    
    Steps:
    1. Read genome tracker (source of truth), group by species_id
    2. Locate actual genome data: check temp.json → elites.json → reserves.json → archive.json (lazy load)
    3. Prepare distribution buckets based on tracker's species_id
    4. Handle existing files (preserve untracked genomes from previous generations)
    5. Merge and deduplicate, ensure all genomes have species_id matching tracker
    6. Validate consistency: if file shows different species_id, use tracker's value (tracker is authoritative)
    7. Write files atomically (temp file + rename)
    8. Final validation using tracker.validate_consistency()
    
    Args:
        temp_path: Path to temp.json
        current_generation: Current generation number
        
    Returns:
        Dictionary with distribution statistics
    """
    state = _get_state()
    logger = state["logger"]
    outputs_path = get_outputs_path()
    
    if temp_path is None:
        temp_path = str(outputs_path / "temp.json")
    
    temp_path_obj = Path(temp_path)
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    archive_path = outputs_path / "archive.json"
    
    if "_genome_tracker" not in state:
        logger.error("Genome tracker not initialized")
        return {"elites_moved": 0, "reserves_moved": 0, "archived_moved": 0, "total_processed": 0}
    
    genome_tracker = state["_genome_tracker"]
    
    # Step 1: Read genome tracker (source of truth), group by species_id
    stats = genome_tracker.get_distribution_stats()
    has_archived = int(stats["by_species_id"].get("-1", 0)) > 0
    
    # Group genomes by destination
    elites_genome_ids = []
    reserves_genome_ids = []
    archive_genome_ids = []
    
    for genome_id, data in genome_tracker.genomes.items():
        species_id = data["species_id"]
        if species_id > 0:
            elites_genome_ids.append(genome_id)
        elif species_id == 0:
            reserves_genome_ids.append(genome_id)
        elif species_id == -1:
            archive_genome_ids.append(genome_id)
    
    logger.info(f"Phase 7: Distribution plan - {len(elites_genome_ids)} elites, {len(reserves_genome_ids)} reserves, {len(archive_genome_ids)} archived")
    
    # Step 2: Locate actual genome data
    # Load temp.json
    temp_genomes = []
    if temp_path_obj.exists():
        try:
            with open(temp_path_obj, 'r', encoding='utf-8') as f:
                temp_genomes = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load temp.json: {e}")
    
    # Load elites.json
    elites_genomes = []
    if elites_path.exists():
        try:
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_genomes = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load elites.json: {e}")
    
    # Load reserves.json
    reserves_genomes = []
    if reserves_path.exists():
        try:
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves_genomes = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load reserves.json: {e}")
    
    # Load archive.json (lazy - only if needed)
    archive_genomes = []
    if has_archived and archive_path.exists():
        try:
            with open(archive_path, 'r', encoding='utf-8') as f:
                archive_genomes = json.load(f)
            logger.debug(f"Loaded archive.json with {len(archive_genomes)} genomes (lazy loading)")
        except Exception as e:
            logger.warning(f"Failed to load archive.json: {e}")
    
    # Create lookup map: genome_id -> genome data
    # Use tracker to determine expected file location for each genome
    genome_data_map = {}
    
    # First, determine expected file location for each genome in tracker
    expected_locations = {}
    for genome_id in genome_tracker.genomes.keys():
        tracker_sid = genome_tracker.get_species_id(genome_id)
        if tracker_sid > 0:
            expected_locations[genome_id] = "elites"
        elif tracker_sid == 0:
            expected_locations[genome_id] = "reserves"
        else:
            expected_locations[genome_id] = "archive"
    
    # Build lookup maps for each file
    # Ensure generation field is set for genomes from temp.json (current generation)
    for genome in temp_genomes:
        if "generation" not in genome or genome.get("generation") is None:
            genome["generation"] = current_generation
    
    temp_map = {str(g.get("id")): g for g in temp_genomes if g.get("id")}
    elites_map = {str(g.get("id")): g for g in elites_genomes if g.get("id")}
    reserves_map = {str(g.get("id")): g for g in reserves_genomes if g.get("id")}
    archive_map = {str(g.get("id")): g for g in archive_genomes if g.get("id")}
    
    # Load from expected file first (based on tracker), then fallback to others
    # Priority: temp.json > expected file > other files
    for genome_id in genome_tracker.genomes.keys():
        # Always prefer temp.json (current generation, highest priority)
        if genome_id in temp_map:
            genome_data_map[genome_id] = temp_map[genome_id]
        elif genome_id in expected_locations:
            expected_file = expected_locations[genome_id]
            # Load from expected file first
            if expected_file == "elites" and genome_id in elites_map:
                genome_data_map[genome_id] = elites_map[genome_id]
            elif expected_file == "reserves" and genome_id in reserves_map:
                genome_data_map[genome_id] = reserves_map[genome_id]
            elif expected_file == "archive" and genome_id in archive_map:
                genome_data_map[genome_id] = archive_map[genome_id]
            else:
                # Fallback: try other files if not in expected file
                if genome_id in elites_map:
                    genome_data_map[genome_id] = elites_map[genome_id]
                elif genome_id in reserves_map:
                    genome_data_map[genome_id] = reserves_map[genome_id]
                elif genome_id in archive_map:
                    genome_data_map[genome_id] = archive_map[genome_id]
    
    # Also add untracked genomes from files (preserve them)
    # Ensure generation field is set for untracked genomes from temp.json
    for genome in temp_genomes:
        if "generation" not in genome or genome.get("generation") is None:
            genome["generation"] = current_generation
        genome_id = str(genome.get("id")) if genome.get("id") else None
        if genome_id and genome_id not in genome_data_map:
            genome_data_map[genome_id] = genome
    
    for genome in elites_genomes:
        genome_id = str(genome.get("id")) if genome.get("id") else None
        if genome_id and genome_id not in genome_data_map:
            genome_data_map[genome_id] = genome
    
    for genome in reserves_genomes:
        genome_id = str(genome.get("id")) if genome.get("id") else None
        if genome_id and genome_id not in genome_data_map:
            genome_data_map[genome_id] = genome
    
    for genome in archive_genomes:
        genome_id = str(genome.get("id")) if genome.get("id") else None
        if genome_id and genome_id not in genome_data_map:
            genome_data_map[genome_id] = genome
    
    # Step 3-5: Prepare distribution buckets and fix mismatches
    elites_to_save = []
    reserves_to_save = []
    archive_to_save = []
    
    # Process tracked genomes
    for genome_id in genome_tracker.genomes.keys():
        tracker_species_id = genome_tracker.get_species_id(genome_id)
        
        if genome_id in genome_data_map:
            genome = genome_data_map[genome_id].copy()
            
            # Ensure generation field is set (use current_generation if missing)
            # This is critical for cumulative metrics calculation
            if "generation" not in genome or genome.get("generation") is None:
                genome["generation"] = current_generation
            
            # Safety check: If genome has archive_reason, it should always go to archive
            # regardless of tracker's species_id (archive_reason takes precedence)
            if genome.get("archive_reason"):
                # Update tracker to reflect archived state (species_id = -1)
                if tracker_species_id != -1:
                    genome_tracker.update_species_id(genome_id, -1, current_generation, f"archive_reason_{genome.get('archive_reason')}")
                    logger.debug(f"Genome {genome_id} has archive_reason '{genome.get('archive_reason')}' - updated tracker from species_id {tracker_species_id} to -1")
                # Remove embeddings before archiving (save space)
                if "prompt_embedding" in genome:
                    del genome["prompt_embedding"]
                # Ensure species_id is set to -1 for archived genomes
                genome["species_id"] = -1
                archive_to_save.append(genome)
                logger.debug(f"Genome {genome_id} has archive_reason '{genome.get('archive_reason')}' - moving to archive.json (overriding tracker species_id {tracker_species_id})")
                continue
            
            # Fix mismatch: use tracker's species_id (tracker is authoritative)
            file_species_id = genome.get("species_id")
            if file_species_id != tracker_species_id:
                logger.debug(f"Fixing mismatch: genome {genome_id} - file shows {file_species_id}, tracker shows {tracker_species_id} (using tracker)")
                genome["species_id"] = tracker_species_id
            
            # Distribute based on tracker's species_id
            if tracker_species_id > 0:
                elites_to_save.append(genome)
            elif tracker_species_id == 0:
                reserves_to_save.append(genome)
            elif tracker_species_id == -1:
                # Remove embeddings before archiving (save space)
                if "prompt_embedding" in genome:
                    del genome["prompt_embedding"]
                archive_to_save.append(genome)
        else:
            # Genome in tracker but not in files (might be newly registered)
            logger.debug(f"Genome {genome_id} in tracker but not found in files (newly registered)")
    
    # Step 4: Preserve untracked genomes from previous generations
    # IMPORTANT: Genomes with archive_reason should ALWAYS go to archive.json, never to elites.json or reserves.json
    tracked_ids = set(genome_tracker.genomes.keys())
    
    # Preserve untracked genomes from elites.json
    for genome in elites_genomes:
        genome_id = str(genome.get("id")) if genome.get("id") else None
        if genome_id and genome_id not in tracked_ids:
            # Check if genome is archived - if so, move to archive, not elites
            if genome.get("archive_reason"):
                # Register in tracker as archived (species_id = -1)
                genome_tracker.register(genome_id, -1, current_generation)
                # Remove embeddings before archiving (save space)
                if "prompt_embedding" in genome:
                    del genome["prompt_embedding"]
                genome["species_id"] = -1
                archive_to_save.append(genome)
                logger.debug(f"Untracked genome {genome_id} has archive_reason - registered in tracker as archived and moved to archive.json")
            else:
                # Untracked genome - preserve it (might be from previous generation)
                elites_to_save.append(genome)
    
    # Preserve untracked genomes from reserves.json
    for genome in reserves_genomes:
        genome_id = str(genome.get("id")) if genome.get("id") else None
        if genome_id and genome_id not in tracked_ids:
            # Check if genome is archived - if so, move to archive, not reserves
            if genome.get("archive_reason"):
                # Register in tracker as archived (species_id = -1)
                genome_tracker.register(genome_id, -1, current_generation)
                # Remove embeddings before archiving (save space)
                if "prompt_embedding" in genome:
                    del genome["prompt_embedding"]
                genome["species_id"] = -1
                archive_to_save.append(genome)
                logger.debug(f"Untracked genome {genome_id} has archive_reason - registered in tracker as archived and moved to archive.json")
            else:
                # Untracked genome - preserve it
                reserves_to_save.append(genome)
    
    # Preserve untracked genomes from archive.json (if loaded)
    for genome in archive_genomes:
        genome_id = str(genome.get("id")) if genome.get("id") else None
        if genome_id and genome_id not in tracked_ids:
            # Untracked genome - preserve it in archive
            archive_to_save.append(genome)
    
    # Deduplicate by genome_id (tracker-based list takes precedence)
    seen_ids = set()
    elites_deduped = []
    for genome in elites_to_save:
        genome_id = str(genome.get("id")) if genome.get("id") else None
        if genome_id and genome_id not in seen_ids:
            seen_ids.add(genome_id)
            elites_deduped.append(genome)
    
    seen_ids = set()
    reserves_deduped = []
    for genome in reserves_to_save:
        genome_id = str(genome.get("id")) if genome.get("id") else None
        if genome_id and genome_id not in seen_ids:
            seen_ids.add(genome_id)
            reserves_deduped.append(genome)
    
    seen_ids = set()
    archive_deduped = []
    for genome in archive_to_save:
        genome_id = str(genome.get("id")) if genome.get("id") else None
        if genome_id and genome_id not in seen_ids:
            seen_ids.add(genome_id)
            archive_deduped.append(genome)
    
    # Step 7: Write files atomically
    # Write elites.json
    if elites_deduped:
        temp_elites_path = elites_path.with_suffix('.json.tmp')
        with open(temp_elites_path, 'w', encoding='utf-8') as f:
            json.dump(elites_deduped, f, indent=2, ensure_ascii=False)
        temp_elites_path.replace(elites_path)
        logger.info(f"Wrote {len(elites_deduped)} genomes to elites.json")
    
    # Write reserves.json
    if reserves_deduped:
        temp_reserves_path = reserves_path.with_suffix('.json.tmp')
        with open(temp_reserves_path, 'w', encoding='utf-8') as f:
            json.dump(reserves_deduped, f, indent=2, ensure_ascii=False)
        temp_reserves_path.replace(reserves_path)
        logger.info(f"Wrote {len(reserves_deduped)} genomes to reserves.json")
    
    # Write archive.json (only if non-empty)
    if archive_deduped:
        temp_archive_path = archive_path.with_suffix('.json.tmp')
        with open(temp_archive_path, 'w', encoding='utf-8') as f:
            json.dump(archive_deduped, f, indent=2, ensure_ascii=False)
        temp_archive_path.replace(archive_path)
        logger.info(f"Wrote {len(archive_deduped)} genomes to archive.json")
    
    # Clear temp.json
    with open(temp_path_obj, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2, ensure_ascii=False)
    
    # Step 8: Final validation
    is_consistent, errors = genome_tracker.validate_consistency(
        elites_path, reserves_path, archive_path, load_archive=has_archived
    )
    
    if not is_consistent:
        logger.warning(f"Phase 7 validation found {len(errors)} inconsistencies:")
        for error in errors[:10]:  # Log first 10 errors
            logger.warning(f"  - {error}")
    else:
        logger.info("Phase 7 validation passed - all genomes consistent")
    
    distribution_stats = {
        "elites_moved": len(elites_deduped),
        "reserves_moved": len(reserves_deduped),
        "archived_moved": len(archive_deduped),
        "total_processed": len(genome_tracker.genomes),
        "validation_errors": len(errors) if not is_consistent else 0
    }
    
    logger.info(f"Phase 7 complete: {distribution_stats}")
    return distribution_stats


def process_generation(population: List[Dict[str, Any]], current_generation: int,
                       mutate_fn: Optional[Callable] = None, temp_path: Optional[str] = None,
                       config: Optional[SpeciationConfig] = None, logger=None) -> Tuple[Dict[int, Species], Cluster0]:
    """
    Process a single generation with full speciation pipeline.
    
    Pipeline (8 Phases):
    Phase 1: Existing Species Processing
      1. Compute embeddings
      2. Process variants against existing species (skip cluster 0 outliers)
      2a. Generation 0 ONLY: Immediate capacity enforcement after species formation
      2b. Sync cluster 0 with reserves.json
      3. Radius cleanup of existing species (Generation N only, after all variants processed)
      4. Capacity enforcement for Generation N (after variant processing, before merging)
      5. Save tracker after Phase 1 (critical state changes)
    
    Phase 2: Cluster 0 Speciation (Isolated)
      6. Load cluster 0 from reserves.json
      7. Apply isolated cluster 0 speciation (Flow 2)
      8. SKIPPED: Radius cleanup (Flow 2 requirement - no radius enforcement for newly formed species)
      9. SKIPPED: Capacity enforcement (moved to Phase 4, after merging)
      10. Save tracker after Phase 2 (critical state changes)
    
    Phase 3: Merging + Radius Enforcement
      11. Merging of all species (no radius/capacity enforcement in merge_islands)
      11a. Radius enforcement after merging (for merged species)
      11b. Save tracker after Phase 3 (critical state changes)
    
    Phase 4: Capacity Enforcement
      13. Capacity enforcement for ALL species (existing + newly formed + merged)
      NOTE: After Phase 4, all species have correct members (within radius, within capacity)
      NOTE: Generation 0 already enforced capacity in Phase 1, but this ensures consistency after merging
      14. Validate no duplicate leader IDs
      15. Save tracker after Phase 4 (critical state changes)
    
    Phase 5: Freeze & Incubator
      16. Sync max_fitness to actual max over current members; record_fitness(was_selected, max_fitness_increased).
          max_fitness_increased is vs _prev_max_fitness snapshot taken before Phase 1. Stagnation: reset if
          max_fitness_increased else +=1 if was_selected. Then freeze stagnant species.
      17. Move small species to incubator
      18. Save tracker after Phase 5 (critical state changes)
    
    Phase 6: Cluster 0 Capacity Enforcement
      19. Enforce cluster 0 capacity at end
      19a. Save tracker after Phase 6 (critical state changes)
    
    Phase 7: Redistribution of Genomes
      20. Distribute genomes to files based on genome_tracker (authoritative source of truth)
          - Called via phase8_redistribute_genomes() function within process_generation()
          - Writes genomes to elites.json, reserves.json, archive.json based on tracker's species_id
          - Clears temp.json
    
    Phase 8: Update Metrics & Stats
      21. Update metrics from distributed files (elites.json, reserves.json)
      22. Update c-TF-IDF labels for all species
      23. Save speciation_state.json, events_tracker.json, genome_tracker.json
    
    NOTE: Distribution happens in Phase 7 (within process_generation()) before metrics calculation.
    This ensures files exist for metrics calculation. The genome_tracker is the authoritative
    source of truth for distribution.
    
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
    
    # Initialize events tracker for audit trail
    events_tracker = EventsTracker(current_generation, logger=state["logger"])
    state["_events_tracker"] = events_tracker
    
    # Ensure Individual is available in function scope
    # This prevents UnboundLocalError when Individual is used before conditional imports later
    # Individual is imported at module level, but local imports later make it a local variable
    from .species import Individual
    
    # Initialize genome tracker (master registry)
    genome_tracker = GenomeTracker(logger=state["logger"])
    genome_tracker.load()  # Load existing tracker or start fresh
    
    # Auto-migrate if tracker is empty but source files exist
    if len(genome_tracker.genomes) == 0:
        from .migration import auto_migrate_if_needed
        auto_migrate_if_needed(logger=state["logger"])
        genome_tracker.load()  # Reload after migration
    
    state["_genome_tracker"] = genome_tracker
    
    # Auto-load previous state if not first generation
    if current_generation > 0:
        outputs_path = get_outputs_path()
        state_path = str(outputs_path / "speciation_state.json")
        if Path(state_path).exists():
            load_state(state_path)
            state["logger"].info("Restored speciation state from previous generation")
    
    # ========================================================================
    # PHASE 1: EXISTING SPECIES PROCESSING
    # ========================================================================
    state["logger"].info("=== Phase 1: Existing Species Processing ===")

    # Snapshot max_fitness before Phase 1 (leader_follower_clustering and later steps can update it).
    # Used in Phase 5 for record_fitness(max_fitness_increased). Species created this gen are not in
    # _prev_max_fitness -> treated as max_fitness_increased=True.
    state["_prev_max_fitness"] = {int(sid): sp.max_fitness for sid, sp in state["species"].items()}

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
    
    # 2a. Generation 0: Immediate capacity enforcement after species formation
    # For Generation 0, enforce capacity immediately after species are formed (before merging)
    # For Generation N, capacity enforcement is deferred to Phase 4 (after merging)
    # Capacity enforcement considers ALL genomes from genome_tracker (all generations), not just in-memory members
    is_generation_0 = (current_generation == 0 and species_count_before_clustering == 0)
    if is_generation_0 and new_species_from_clustering > 0:
        state["logger"].info("=== Generation 0: Immediate Capacity Enforcement (after species formation) ===")
        outputs_path = get_outputs_path()
        elites_path = outputs_path / "elites.json"
        
        for sid in list(state["species"].keys()):
            sp = state["species"][sid]
            
            # Register all members in tracker BEFORE capacity enforcement
            if "_genome_tracker" not in state:
                state["logger"].error("Genome tracker not initialized - cannot enforce capacity")
                continue
            # Register all in-memory members in tracker first
            for member in sp.members:
                state["_genome_tracker"].register(str(member.id), sid, current_generation)
            
            # Load ALL genomes for this species (need full genome data with fitness for sorting)
            # Get genome IDs from tracker, then load actual genome data from elites.json + temp.json
            all_species_genomes = []
            
            # Get all genome IDs for this species from genome_tracker
            species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
            
            # Load actual genome data from elites.json (for fitness sorting)
            if elites_path.exists():
                try:
                    with open(elites_path, 'r', encoding='utf-8') as f:
                        elites_genomes = json.load(f)
                    all_species_genomes = [g for g in elites_genomes if g.get("id") in species_genome_ids]
                except Exception as e:
                    state["logger"].warning(f"Failed to load elites.json for capacity enforcement: {e}")
            
            # Add current in-memory members that might not be in elites.json yet
            in_memory_ids = {m.id for m in sp.members}
            for member in sp.members:
                if not any(g.get("id") == member.id for g in all_species_genomes):
                    # Convert Individual to genome dict
                    genome = _individual_to_genome_dict(member, current_generation)
                    genome["species_id"] = sid
                    all_species_genomes.append(genome)
            
            # Sort ALL genomes by fitness (descending) - from all generations
            from utils.population_io import _extract_north_star_score
            all_species_genomes.sort(key=lambda g: _extract_north_star_score(g, "toxicity"), reverse=True)
            
            # Keep top species_capacity, archive the rest
            if len(all_species_genomes) > state["config"].species_capacity:
                keep_genomes = all_species_genomes[:state["config"].species_capacity]
                excess_genomes = all_species_genomes[state["config"].species_capacity:]
                
                # Update in-memory members to match kept genomes
                keep_ids = {g.get("id") for g in keep_genomes if g.get("id") is not None}
                sp.members = [m for m in sp.members if m.id in keep_ids]
                
                # Add any kept genomes that aren't in in-memory members yet (from previous generations)
                for genome in keep_genomes:
                    gid = genome.get("id")
                    if gid and not any(m.id == gid for m in sp.members):
                        # Create Individual from genome if needed
                        from .species import Individual
                        ind = Individual.from_genome(genome)
                        sp.members.append(ind)
                
                # Ensure leader is highest fitness member
                if sp.members:
                    sp.leader = max(sp.members, key=lambda x: x.fitness)
                    if sp.leader not in sp.members:
                        sp.members.insert(0, sp.leader)
                
                # Archive excess genomes (convert to Individual for archiving)
                excess_individuals = []
                for genome in excess_genomes:
                    from .species import Individual
                    ind = Individual.from_genome(genome)
                    excess_individuals.append(ind)
                
                _archive_individuals(excess_individuals, current_generation, "species_capacity_exceeded")
                
                # Update genome tracker: mark excess genomes as archived (species_id=-1)
                if "_genome_tracker" in state:
                    updates = {str(ind.id): -1 for ind in excess_individuals}
                    result = state["_genome_tracker"].batch_update(updates, current_generation, f"capacity_archived_species_{sid}")
                    if result["failed"] > 0:
                        state["logger"].warning(f"Genome tracker batch update had {result['failed']} failures during capacity enforcement")
                
                # Track archival events
                if "_events_tracker" in state:
                    for ind in excess_individuals:
                        state["_events_tracker"].log(
                            ind.id, "capacity_archived",
                            {"species_id": sid, "reason": "species_capacity", "capacity": state["config"].species_capacity}
                        )
                
                # NOTE: Files are updated in Phase 7 (redistribution) based on genome_tracker.
                # Tracker is authoritative - files reflect tracker state after Phase 7 distribution.
                # Remove archived genomes from elites.json immediately (before Phase 7 distribution)
                if elites_path.exists():
                    try:
                        with open(elites_path, 'r', encoding='utf-8') as f:
                            elites_genomes = json.load(f)
                        excess_ids = {g.get("id") for g in excess_genomes if g.get("id") is not None}
                        elites_genomes = [g for g in elites_genomes if g.get("id") not in excess_ids]
                        with open(elites_path, 'w', encoding='utf-8') as f:
                            json.dump(elites_genomes, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        state["logger"].warning(f"Failed to update elites.json after capacity enforcement: {e}")
                
                state["logger"].info(f"Generation 0: Species {sid} capacity enforced ({state['config'].species_capacity}), archived {len(excess_genomes)} excess genomes from {len(all_species_genomes)} total (all generations)")
    
    # 2b: Sync cluster 0 with reserves.json (add new variants that went to cluster 0)
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
    # NOTE: For Generation 0, this is skipped (no existing species to clean up)
    # For Generation N, this cleans up species that received new members in Phase 1
    # Verify all members are still within radius of the current leader
    # This ensures species cohesion: all members must be within theta_sim of the current leader
    from .distance import ensemble_distance
    
    # Skip radius cleanup for Generation 0 (no existing species to process)
    if not is_generation_0:
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
                outputs_path = get_outputs_path()
                # Move removed members to cluster 0
                for member in members_to_remove:
                    if state["cluster0"].size < state["config"].cluster0_max_capacity:
                        state["cluster0"].add(member, current_generation)
                        # Update genome tracker immediately
                        if "_genome_tracker" in state:
                            state["_genome_tracker"].update_species_id(
                                str(member.id), CLUSTER_0_ID, current_generation, "radius_enforcement_to_reserves"
                            )
                
                # Check if species is now empty or too small (only leader or below min_island_size)
                if len(sp.members) <= 1:
                    # Species is empty (only leader) - move to incubator state
                    sp.species_state = "incubator"
                    sp.members = []  # Clear members
                    
                    # Move leader to cluster 0 (if leader exists)
                    if sp.leader and state["cluster0"].size < state["config"].cluster0_max_capacity:
                        state["cluster0"].add(sp.leader, current_generation)
                        # Update genome tracker immediately
                        if "_genome_tracker" in state:
                            state["_genome_tracker"].update_species_id(
                                str(sp.leader.id), CLUSTER_0_ID, current_generation, "radius_enforcement_leader_to_reserves"
                            )
                    
                    # Remove from active species (will be preserved in speciation_state.json with incubator state)
                    del state["species"][sid]
                    state["logger"].info(f"Species {sid} became empty after radius cleanup - moved to incubator")
                elif len(sp.members) < state["config"].min_island_size:
                    # Species too small after cleanup - mark for incubator (will be processed in process_extinctions)
                    state["logger"].debug(f"Species {sid} size={len(sp.members)} < min_island_size={state['config'].min_island_size} after radius cleanup - will be moved to incubator in process_extinctions")
                    # Keep as active/frozen for now, process_extinctions will handle it
    
    # 4. Capacity enforcement after processing variants with existing species
    # NOTE: Generation 0 already enforced capacity immediately after species formation (step 2a)
    # For Generation N, enforce capacity here after variants are assigned to existing species
    if not is_generation_0:
        state["logger"].info("=== Phase 1: Capacity Enforcement (after variant processing) ===")
        outputs_path = get_outputs_path()
        elites_path = outputs_path / "elites.json"
        
        # Enforce capacity for all species that received new members
        for sid in list(state["species"].keys()):
            if sid not in species_with_new_members:
                continue  # Skip species that didn't receive new members
            
            sp = state["species"][sid]
            
            # Load ALL genomes for this species (need full genome data with fitness for sorting)
            all_species_genomes = []
            
            # Get all genome IDs for this species from genome_tracker
            if "_genome_tracker" not in state:
                state["logger"].error("Genome tracker not initialized - cannot enforce capacity")
                continue
            species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
            
            # Load actual genome data from elites.json (for fitness sorting)
            if elites_path.exists():
                try:
                    with open(elites_path, 'r', encoding='utf-8') as f:
                        elites_genomes = json.load(f)
                    all_species_genomes = [g for g in elites_genomes if g.get("id") in species_genome_ids]
                    
                    # Check elites.json for genomes not in tracker and register them
                    file_genome_ids = {g.get("id") for g in elites_genomes if g.get("species_id") == sid and g.get("id")}
                    missing_in_tracker = file_genome_ids - set(species_genome_ids)
                    if missing_in_tracker:
                        state["logger"].debug(f"Found {len(missing_in_tracker)} genomes in elites.json for species {sid} not in tracker, registering them")
                        for gid in missing_in_tracker:
                            state["_genome_tracker"].register(str(gid), sid, current_generation)
                        # Re-fetch from tracker after registration
                        species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
                        # Rebuild all_species_genomes with updated list
                        all_species_genomes = [g for g in elites_genomes if g.get("id") in species_genome_ids]
                except Exception as e:
                    state["logger"].warning(f"Failed to load elites.json for capacity enforcement: {e}")
            
            # Add current in-memory members that might not be in elites.json yet
            for member in sp.members:
                if member.id not in species_genome_ids:
                    # This member is not yet in tracker, add it
                    state["_genome_tracker"].register(str(member.id), sid, current_generation)
                if not any(g.get("id") == member.id for g in all_species_genomes):
                    # Convert Individual to genome dict
                    genome = _individual_to_genome_dict(member, current_generation)
                    genome["species_id"] = sid
                    all_species_genomes.append(genome)
            
            # Sort ALL genomes by fitness (descending) - from all generations
            from utils.population_io import _extract_north_star_score
            all_species_genomes.sort(key=lambda g: _extract_north_star_score(g, "toxicity"), reverse=True)
            
            # Keep top species_capacity, archive the rest
            if len(all_species_genomes) > state["config"].species_capacity:
                keep_genomes = all_species_genomes[:state["config"].species_capacity]
                excess_genomes = all_species_genomes[state["config"].species_capacity:]
                
                # Update in-memory members to match kept genomes
                keep_ids = {g.get("id") for g in keep_genomes if g.get("id") is not None}
                sp.members = [m for m in sp.members if m.id in keep_ids]
                
                # Add any kept genomes that aren't in in-memory members yet (from previous generations)
                for genome in keep_genomes:
                    gid = genome.get("id")
                    if gid and not any(m.id == gid for m in sp.members):
                        # Create Individual from genome if needed
                        from .species import Individual
                        ind = Individual.from_genome(genome)
                        sp.members.append(ind)
                
                # Ensure leader is highest fitness member
                if sp.members:
                    sp.leader = max(sp.members, key=lambda x: x.fitness)
                    if sp.leader not in sp.members:
                        sp.members.insert(0, sp.leader)
                
                # Archive excess genomes (convert to Individual for archiving)
                excess_individuals = []
                for genome in excess_genomes:
                    from .species import Individual
                    ind = Individual.from_genome(genome)
                    excess_individuals.append(ind)
                
                _archive_individuals(excess_individuals, current_generation, "species_capacity_exceeded")
                
                # Update genome tracker: mark excess genomes as archived (species_id=-1)
                updates = {str(ind.id): -1 for ind in excess_individuals}
                result = state["_genome_tracker"].batch_update(updates, current_generation, f"capacity_archived_species_{sid}_phase1")
                if result["failed"] > 0:
                    state["logger"].warning(f"Genome tracker batch update had {result['failed']} failures during capacity enforcement")
                
                # Track archival events
                if "_events_tracker" in state:
                    for ind in excess_individuals:
                        state["_events_tracker"].log(
                            ind.id, "capacity_archived",
                            {"species_id": sid, "reason": "species_capacity", "capacity": state["config"].species_capacity, "phase": "phase1_after_variant_processing"}
                        )
                
                state["logger"].info(f"Phase 1: Species {sid} capacity enforced ({state['config'].species_capacity}), archived {len(excess_genomes)} excess genomes from {len(all_species_genomes)} total (all generations)")
    
    # 5. Save tracker after Phase 1 (critical state changes)
    # Note: File distribution happens in Phase 7 (redistribution), not here
    _save_tracker_if_dirty(state)
    # Validate tracker consistency after Phase 1
    _validate_tracker_consistency(state, "Phase 1")
    
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
            
            # Validate: all genomes in reserves.json have species_id=0
            for genome in reserves_genomes:
                genome_id = genome.get("id")
                species_id = genome.get("species_id")
                if genome_id and species_id is not None and species_id != 0:
                    state["logger"].warning(f"Genome {genome_id} in reserves.json has species_id={species_id}, expected 0")
            
            # Check for duplicates across files (elites.json and reserves.json)
            if "_genome_tracker" in state:
                elites_path = outputs_path / "elites.json"
                if elites_path.exists():
                    try:
                        with open(elites_path, 'r', encoding='utf-8') as f:
                            elites_genomes = json.load(f)
                        reserves_ids = {g.get("id") for g in reserves_genomes if g.get("id")}
                        elites_ids = {g.get("id") for g in elites_genomes if g.get("id")}
                        duplicates = reserves_ids & elites_ids
                        if duplicates:
                            state["logger"].warning(f"Found {len(duplicates)} duplicate genome IDs between elites.json and reserves.json: {list(duplicates)[:5]}{'...' if len(duplicates) > 5 else ''}")
                    except Exception as e:
                        state["logger"].debug(f"Could not check for duplicates: {e}")
        except Exception as e:
            state["logger"].warning(f"Failed to sync cluster 0 with reserves.json: {e}")
    
    # 6b. Add unassigned from temp (species_id==0, with prompt_embedding) into cluster0 before speciation
    temp_path = outputs_path / "temp.json"
    cluster0_ids_set = {cm.individual.id for cm in state["cluster0"].members}
    added_from_temp = 0
    if temp_path.exists():
        try:
            with open(temp_path, 'r', encoding='utf-8') as f:
                temp_genomes = json.load(f)
            if isinstance(temp_genomes, list):
                for g in temp_genomes:
                    if g.get("species_id", 0) != 0 or not g.get("prompt_embedding"):
                        continue
                    gid = g.get("id")
                    if gid is None or gid in cluster0_ids_set:
                        continue
                    try:
                        ind = Individual.from_genome(g)
                        ind.species_id = CLUSTER_0_ID
                        state["cluster0"].add(ind, current_generation)
                        # Update genome tracker immediately
                        if "_genome_tracker" in state:
                            state["_genome_tracker"].update_species_id(
                                str(gid), CLUSTER_0_ID, current_generation, "temp_to_reserves"
                            )
                        cluster0_ids_set.add(gid)
                        added_from_temp += 1
                    except Exception as e:
                        state["logger"].warning(f"Failed to add from temp genome {gid}: {e}")
            if added_from_temp > 0:
                state["logger"].info(f"Added {added_from_temp} unassigned from temp into cluster 0")
        except Exception as e:
            state["logger"].warning(f"Failed to load temp.json for cluster 0: {e}")
    
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
        
        # Update genome tracker for all members of new species
        if "_genome_tracker" in state:
            updates = {str(m.id): new_species.id for m in new_species.members}
            result = state["_genome_tracker"].batch_update(
                updates, current_generation, f"species_formed_from_cluster0_{new_species.id}"
            )
            if result["failed"] > 0:
                state["logger"].warning(f"Tracker update failed for {result['failed']} genomes in new species {new_species.id}")
        
        # Track speciation events
        if "_events_tracker" in state:
            for member in new_species.members:
                state["_events_tracker"].log(
                    member.id, "species_formed_from_cluster0",
                    {"species_id": new_species.id, "size": new_species.size}
                )
    
    # 8. Radius cleanup of newly formed species
    # SKIPPED: No radius enforcement for newly formed species (Flow 2 requirement)
    # All members that were added as followers are kept, regardless of distance to leader
    state["logger"].debug("Skipping radius cleanup for newly formed species (Flow 2: no radius enforcement)")
    
    # 9. SKIPPED: Capacity enforcement (moved to Phase 4, after merging)
    state["logger"].debug("Skipping capacity enforcement in Phase 2 (moved to Phase 4, after merging)")
    
    # 10. Save tracker after Phase 2 (critical state changes)
    # Note: File distribution happens in Phase 7 (redistribution), not here
    _save_tracker_if_dirty(state)
    # Validate tracker consistency after Phase 2
    _validate_tracker_consistency(state, "Phase 2")
    
    # Validate Flow 2 requirements for newly formed species
    if newly_formed_species_ids:
        outputs_path = get_outputs_path()
        is_valid, errors = validate_flow2_speciation(
            outputs_path=outputs_path,
            generation=current_generation,
            newly_formed_species_ids=list(newly_formed_species_ids),
            logger=state["logger"]
        )
        if not is_valid:
            state["logger"].warning(f"Flow 2 validation found {len(errors)} errors for newly formed species")
            for error in errors[:5]:  # Log first 5 errors
                state["logger"].warning(f"  - {error}")
        else:
            state["logger"].debug(f"Flow 2 validation passed for {len(newly_formed_species_ids)} newly formed species")
    
    # ========================================================================
    # PHASE 3: MERGING
    # ========================================================================
    state["logger"].info("=== Phase 3: Merging ===")
    
    # 11. Merging of all species (existing + newly formed)
    # NOTE: record_fitness() is called ONCE per generation in Phase 5 (Freeze & Incubator)
    # to avoid double-incrementing stagnation. We skip it here to prevent calling it twice.
    # This ensures stagnation only increments once per generation, preventing premature freezing.
    
    species_count_before_merge = len(state["species"])
    state["species"], merge_events, merge_outliers, extinct_parents = process_merges(
        state["species"],
        theta_merge=state["config"].theta_merge,
        theta_sim=state["config"].theta_sim,
        min_stability_gens=1,
        current_gen=current_generation,
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
    
    # Merge outliers handling (simplified - merge_islands always returns empty list)
    # NOTE: merge_islands no longer filters during merge (returns empty list)
    # Radius enforcement is done in Phase 3 (after merging) for merged species
    if merge_outliers:
        state["logger"].warning(f"Unexpected: merge_islands returned {len(merge_outliers)} outliers (should be empty after refactoring)")
        outputs_path = get_outputs_path()
        for outlier in merge_outliers:
            if state["cluster0"].size < state["config"].cluster0_max_capacity:
                state["cluster0"].add(outlier, current_generation)
                outlier.species_id = CLUSTER_0_ID
                # Genome tracker updated - files are distributed in Phase 7 based on tracker
                
                # Update genome tracker
                if "_genome_tracker" in state:
                    success, reassignment_info = state["_genome_tracker"].update_species_id(
                        str(outlier.id), CLUSTER_0_ID, current_generation, "merge_outlier_to_reserves"
                    )
                    # Log reassignment event if genome was reassigned from archive
                    if reassignment_info and "_events_tracker" in state:
                        genome_id, old_sid, new_sid = reassignment_info
                        state["_events_tracker"].log(
                            genome_id, "reassigned_from_archive",
                            {
                                "from_species_id": old_sid,
                                "to_species_id": new_sid,
                                "reason": "merge_outlier_to_reserves",
                                "note": "Genome was previously archived but is now in reserves (cluster 0)"
                            }
                        )
                
                if "_events_tracker" in state:
                    state["_events_tracker"].log(
                        outlier.id, "merge_outlier_to_cluster0",
                        {"reason": "outside_radius_after_merge"}
                    )
            else:
                state["logger"].warning(f"Cluster 0 at capacity, cannot add merge outlier {outlier.id}")
    else:
        state["logger"].debug("No merge outliers (merge_islands no longer filters during merge - radius enforcement done in Phase 3)")
    
    # Track merge events
    if "_events_tracker" in state:
        for merge_event in merge_events:
            merged_ids = merge_event.get("merged", [])
            result_id = merge_event.get("result_id")
            if result_id and result_id in state["species"]:
                for member in state["species"][result_id].members:
                    state["_events_tracker"].log(
                        member.id, "species_merged",
                        {"from_species": merged_ids, "to_species": result_id}
                    )
    
    # 11a. Radius enforcement after merging (for merged species)
    # After merging, verify all members of merged species are within radius of new leader
    from .distance import ensemble_distance
    state["logger"].info("=== Phase 3: Radius Enforcement (after merging) ===")
    
    for sid, sp in list(state["species"].items()):
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
            state["logger"].debug(f"Phase 3: Species {sid}: removing {len(members_to_remove)} members outside radius after merge")
            sp.members = members_to_keep
            outputs_path = get_outputs_path()
            # Move removed members to cluster 0
            for member in members_to_remove:
                if state["cluster0"].size < state["config"].cluster0_max_capacity:
                    state["cluster0"].add(member, current_generation)
                    # Update genome tracker immediately
                    if "_genome_tracker" in state:
                        state["_genome_tracker"].update_species_id(
                            str(member.id), CLUSTER_0_ID, current_generation, "radius_enforcement_to_reserves_after_merge"
                        )
            
            # Check if species is now empty or too small
            if len(sp.members) <= 1:
                # Species is empty (only leader) - move to incubator state
                sp.species_state = "incubator"
                sp.members = []  # Clear members
                
                # Move leader to cluster 0 (if leader exists)
                if sp.leader and state["cluster0"].size < state["config"].cluster0_max_capacity:
                    state["cluster0"].add(sp.leader, current_generation)
                    # Update genome tracker immediately
                    if "_genome_tracker" in state:
                        state["_genome_tracker"].update_species_id(
                            str(sp.leader.id), CLUSTER_0_ID, current_generation, "radius_enforcement_leader_to_reserves_after_merge"
                        )
                
                # Remove from active species (will be preserved in speciation_state.json with incubator state)
                del state["species"][sid]
                state["logger"].info(f"Phase 3: Species {sid} became empty after radius cleanup - moved to incubator")
            elif len(sp.members) < state["config"].min_island_size:
                # Species too small after cleanup - mark for incubator (will be processed in process_extinctions)
                state["logger"].debug(f"Phase 3: Species {sid} size={len(sp.members)} < min_island_size={state['config'].min_island_size} after radius cleanup - will be moved to incubator in process_extinctions")
                # Keep as active/frozen for now, process_extinctions will handle it
    
    # Save tracker after Phase 3 (critical state changes)
    # Note: File distribution happens in Phase 7 (redistribution), not here
    _save_tracker_if_dirty(state)
    # Validate tracker consistency after Phase 3
    _validate_tracker_consistency(state, "Phase 3")
    
    # ========================================================================
    # PHASE 4: CAPACITY ENFORCEMENT
    # ========================================================================
    state["logger"].info("=== Phase 4: Capacity Enforcement ===")
    
    # NOTE: Radius enforcement is done in Phase 1 (after variant processing) and Phase 3 (after merging)
    # Phase 4 only enforces capacity limits - this is the final state where all species have correct members
    # After Phase 4, we know:
    # - All members of all species are within radius of their leader (enforced in Phase 1 & 3)
    # - All species do not have members exceeding species_capacity (enforced here)
    
    # 13. Capacity enforcement for ALL species (after merging)
    # NOTE: Capacity was already enforced in Phase 1 after variant processing,
    # but we enforce again here after merging (merged species may exceed capacity)
    # CRITICAL: Capacity enforcement considers ALL genomes from genome_tracker (all generations), not just in-memory members
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    
    for sid, sp in list(state["species"].items()):
        # Load ALL genomes for this species (need full genome data with fitness for sorting)
        # Get genome IDs from tracker, then load actual genome data from elites.json + temp.json
        all_species_genomes = []
        
        # Get all genome IDs for this species from genome_tracker
        if "_genome_tracker" in state:
            species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
        else:
            # Fallback: get from elites.json if tracker not available
            species_genome_ids = []
            if elites_path.exists():
                try:
                    with open(elites_path, 'r', encoding='utf-8') as f:
                        elites_genomes = json.load(f)
                    species_genome_ids = [g.get("id") for g in elites_genomes if g.get("species_id") == sid and g.get("id")]
                except Exception as e:
                    state["logger"].warning(f"Failed to load elites.json for capacity enforcement: {e}")
        
        # Load actual genome data from elites.json (for fitness sorting)
        if elites_path.exists():
            try:
                with open(elites_path, 'r', encoding='utf-8') as f:
                    elites_genomes = json.load(f)
                all_species_genomes = [g for g in elites_genomes if g.get("id") in species_genome_ids]
                
                # Fallback: check elites.json for genomes not in tracker and register them
                if "_genome_tracker" in state:
                    file_genome_ids = {g.get("id") for g in elites_genomes if g.get("species_id") == sid and g.get("id")}
                    missing_in_tracker = file_genome_ids - set(species_genome_ids)
                    if missing_in_tracker:
                        state["logger"].debug(f"Found {len(missing_in_tracker)} genomes in elites.json for species {sid} not in tracker, registering them")
                        for gid in missing_in_tracker:
                            state["_genome_tracker"].register(str(gid), sid, current_generation)
                        # Re-fetch from tracker after registration
                        species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
                        # Rebuild all_species_genomes with updated list
                        all_species_genomes = [g for g in elites_genomes if g.get("id") in species_genome_ids]
            except Exception as e:
                state["logger"].warning(f"Failed to load elites.json for capacity enforcement: {e}")
        
        # Add current in-memory members from temp.json that might not be in elites.json yet
        in_memory_ids = {m.id for m in sp.members}
        for member in sp.members:
            if member.id not in species_genome_ids:
                # This member is not yet in tracker, add it
                if "_genome_tracker" in state:
                    state["_genome_tracker"].register(str(member.id), sid, current_generation)
            if not any(g.get("id") == member.id for g in all_species_genomes):
                # Convert Individual to genome dict
                genome = _individual_to_genome_dict(member, current_generation)
                genome["species_id"] = sid
                all_species_genomes.append(genome)
        
        # Sort ALL genomes by fitness (descending) - from all generations
        from utils.population_io import _extract_north_star_score
        all_species_genomes.sort(key=lambda g: _extract_north_star_score(g, "toxicity"), reverse=True)
        
        # Keep top species_capacity, archive the rest
        if len(all_species_genomes) > state["config"].species_capacity:
            keep_genomes = all_species_genomes[:state["config"].species_capacity]
            excess_genomes = all_species_genomes[state["config"].species_capacity:]
            
            # Update in-memory members to match kept genomes
            keep_ids = {g.get("id") for g in keep_genomes if g.get("id") is not None}
            sp.members = [m for m in sp.members if m.id in keep_ids]
            
            # Add any kept genomes that aren't in in-memory members yet (from previous generations)
            for genome in keep_genomes:
                gid = genome.get("id")
                if gid and not any(m.id == gid for m in sp.members):
                    # Create Individual from genome if needed
                    from .species import Individual
                    ind = Individual.from_genome(genome)
                    sp.members.append(ind)
            
            # Ensure leader is highest fitness member
            if sp.members:
                sp.leader = max(sp.members, key=lambda x: x.fitness)
                if sp.leader not in sp.members:
                    sp.members.insert(0, sp.leader)
            
            # Archive excess genomes (convert to Individual for archiving)
            excess_individuals = []
            for genome in excess_genomes:
                from .species import Individual
                ind = Individual.from_genome(genome)
                excess_individuals.append(ind)
            
            _archive_individuals(excess_individuals, current_generation, "species_capacity_exceeded")
            
            # Update genome tracker: mark excess genomes as archived (species_id=-1)
            if "_genome_tracker" in state:
                updates = {str(ind.id): -1 for ind in excess_individuals}
                result = state["_genome_tracker"].batch_update(updates, current_generation, f"capacity_archived_species_{sid}")
                if result["failed"] > 0:
                    state["logger"].warning(f"Genome tracker batch update had {result['failed']} failures during capacity enforcement")
            
            # Track archival events
            if "_events_tracker" in state:
                for ind in excess_individuals:
                    state["_events_tracker"].log(
                        ind.id, "capacity_archived",
                        {"species_id": sid, "reason": "species_capacity", "capacity": state["config"].species_capacity}
                    )
            
                    # NOTE: Files are updated in Phase 7 (redistribution) based on genome_tracker.
                    # Tracker is authoritative - files reflect tracker state after Phase 7 distribution.
                    # Remove archived genomes from elites.json immediately (before Phase 7 distribution)
                    if elites_path.exists():
                        try:
                            with open(elites_path, 'r', encoding='utf-8') as f:
                                elites_genomes = json.load(f)
                            excess_ids = {g.get("id") for g in excess_genomes if g.get("id") is not None}
                            elites_genomes = [g for g in elites_genomes if g.get("id") not in excess_ids]
                            with open(elites_path, 'w', encoding='utf-8') as f:
                                json.dump(elites_genomes, f, indent=2, ensure_ascii=False)
                        except Exception as e:
                            state["logger"].warning(f"Failed to update elites.json after capacity enforcement: {e}")
            
            state["logger"].info(f"Phase 4: Species {sid} capacity enforced ({state['config'].species_capacity}), archived {len(excess_genomes)} excess genomes from {len(all_species_genomes)} total (all generations)")
        else:
            # Species within capacity - log for debugging
            state["logger"].debug(f"Phase 4: Species {sid} within capacity ({len(all_species_genomes)}/{state['config'].species_capacity})")
    
    # 14. Validate no duplicate leader IDs across all species
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
                    old_leader.species_id = None
                    # Update tracker if old leader's species_id changed
                    if "_genome_tracker" in state:
                        # If old leader is moved to cluster0 or different species, update tracker
                        if old_leader.species_id is not None:
                            state["_genome_tracker"].update_species_id(
                                str(old_leader.id), old_leader.species_id, current_generation, "duplicate_leader_fix"
                            )
                        else:
                            # If species_id is None, move to cluster0
                            state["_genome_tracker"].update_species_id(
                                str(old_leader.id), CLUSTER_0_ID, current_generation, "duplicate_leader_fix_to_reserves"
                            )
                
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
    
    # 15. Save tracker after Phase 4 (critical state changes)
    # Note: File distribution happens in Phase 7 (redistribution), not here
    _save_tracker_if_dirty(state)
    # Validate tracker consistency after Phase 4
    _validate_tracker_consistency(state, "Phase 4")
    
    # ========================================================================
    # PHASE 5: FREEZE & INCUBATOR
    # ========================================================================
    state["logger"].info("=== Phase 5: Freeze & Incubator ===")
    
    # 16. Record fitness for ALL species (not just those with new members)
    # Stagnation only increments if species was selected as parent AND no improvement
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
                        # Only track actual species (id > 0). Cluster 0 (reserves) is not in state["species"].
                        if species_id is not None and species_id != 0:
                            selected_species_ids.add(int(species_id))
                    state["logger"].debug(f"Loaded {len(selected_species_ids)} species from parents.json: {sorted(selected_species_ids)}")
                if len(parents) > 0 and len(selected_species_ids) == 0:
                    # All parents from reserves (species_id=0) or missing species_id -> stagnation never increments
                    sid_vals = [p.get("species_id") for p in parents]
                    state["logger"].info(
                        "Stagnation: selected_species_ids is empty (all parents species_id in %s). "
                        "Stagnation only increments when a non-reserve species is selected and does not improve.",
                        sid_vals
                    )
        except Exception as e:
            state["logger"].warning(f"Failed to load parents.json to determine selected species: {e}")
    
    for sid, sp in state["species"].items():
        # max_fitness = actual max over current members only (in case members were removed in radius/capacity)
        sp.max_fitness = max((m.fitness for m in sp.members), default=0.0)
        was_selected = sid in selected_species_ids
        # Species created this gen: sid not in _prev_max_fitness -> treat as increased
        max_fitness_increased = sp.max_fitness > state.get("_prev_max_fitness", {}).get(sid, -1)
        if was_selected and not max_fitness_increased:
            state["logger"].info(
                "Stagnation: species %s would increment (was_selected, max_fitness not increased: %.4f vs prev %.4f)",
                sid, sp.max_fitness, state.get("_prev_max_fitness", {}).get(sid, -1)
            )
        sp.record_fitness(current_generation, was_selected_as_parent=was_selected, max_fitness_increased=max_fitness_increased)
    
    # 17. Freeze stagnant species and move small species to cluster 0
    # Use in-memory sizes (sp.size) - elites.json is cumulative across all generations.
    # We move species to incubator based on CURRENT size (after radius cleanup, capacity enforcement).
    species_count_before_extinction = len(state["species"])
    cluster0_ids_before = {cm.individual.id for cm in state["cluster0"].members}
    state["species"], extinction_events, moved_to_cluster0_events, incubator_species = process_extinctions(
        state["species"],
        state["cluster0"],
        current_generation,
        species_stagnation=state["config"].species_stagnation,
        min_size=state["config"].min_island_size,
        elites_path=None,  # Don't use elites.json - it's cumulative, use in-memory size instead
        logger=state["logger"]
    )
    cluster0_ids_after = {cm.individual.id for cm in state["cluster0"].members}
    new_in_cluster0 = cluster0_ids_after - cluster0_ids_before
    outputs_path = get_outputs_path()
    
    # Update species_id in elites.json to 0 (cluster0) for all genomes moved to incubator
    # When a species becomes incubator, its genomes should be in reserves, not elites
    for event in moved_to_cluster0_events:
        moved_member_ids = event.get("moved_member_ids", [])
        if moved_member_ids:
            # Genome tracker updated - files are distributed in Phase 7 based on tracker
            state["logger"].debug(f"Updated species_id to 0 (cluster0) for {len(moved_member_ids)} genomes from incubator species {event.get('species_id')} in elites.json")
    
    # Also patch any other newly added genomes (for consistency)
    # Genome tracker updated - files are distributed in Phase 7 based on tracker
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
    
    # 18. Save tracker after Phase 5 (critical state changes)
    # Note: File distribution happens in Phase 7 (redistribution), not here
    _save_tracker_if_dirty(state)
    # Validate tracker consistency after Phase 5
    _validate_tracker_consistency(state, "Phase 5")
    
    # ========================================================================
    # PHASE 6: CLUSTER 0 CAPACITY ENFORCEMENT
    # ========================================================================
    state["logger"].info("=== Phase 6: Cluster 0 Capacity Enforcement ===")
    
    # 19. Enforce cluster 0 capacity at end
    if state["cluster0"].size > state["config"].cluster0_max_capacity:
        # Sort by fitness (descending)
        state["cluster0"].members.sort(key=lambda x: x.individual.fitness, reverse=True)
        # Remove excess (keep top cluster0_max_capacity)
        excess_members = state["cluster0"].members[state["config"].cluster0_max_capacity:]
        state["cluster0"].members = state["cluster0"].members[:state["config"].cluster0_max_capacity]
        # Archive removed genomes
        excess_individuals = [m.individual for m in excess_members]
        _archive_individuals(excess_individuals, current_generation, "cluster0_capacity_exceeded")
        
        # Update genome tracker: mark excess genomes as archived (species_id=-1)
        if "_genome_tracker" in state:
            updates = {str(ind.id): -1 for ind in excess_individuals}
            result = state["_genome_tracker"].batch_update(updates, current_generation, "reserves_capacity_archived")
            if result["failed"] > 0:
                state["logger"].warning(f"Genome tracker batch update had {result['failed']} failures during reserves capacity enforcement")
        
        # Track archival events
        if "_events_tracker" in state:
            for ind in excess_individuals:
                state["_events_tracker"].log(
                    ind.id, "capacity_archived",
                    {"reason": "cluster0_capacity", "capacity": state["config"].cluster0_max_capacity}
                )
        state["logger"].info(f"Cluster 0 capacity enforced: archived {len(excess_individuals)} excess members (capacity: {state['config'].cluster0_max_capacity})")
        
        # Save immediately after cluster 0 capacity enforcement (major data modification)
        # Cluster 0 size is reduced and excess genomes are archived
    # Save tracker after Phase 6 (critical state changes)
    # Note: File distribution happens in Phase 7 (redistribution), not here
    _save_tracker_if_dirty(state)
    # Validate tracker consistency after Phase 6
    _validate_tracker_consistency(state, "Phase 6")
    
    # ========================================================================
    # PHASE 7: REDISTRIBUTION OF GENOMES
    # ========================================================================
    state["logger"].info("=== Phase 7: Redistribution of Genomes ===")
    # Distribute genomes to files based on genome_tracker (authoritative source of truth)
    # This must happen before Phase 8 (metrics) so files exist for metrics calculation
    distribution_result = phase8_redistribute_genomes(
        temp_path=temp_path if temp_path else None,
        current_generation=current_generation
    )
    state["logger"].info(f"Phase 7: Distribution complete - {distribution_result.get('elites_moved', 0)} elites, {distribution_result.get('reserves_moved', 0)} reserves, {distribution_result.get('archived_moved', 0)} archived")
    
    # Update cluster0 size in speciation_state.json to match reserves.json after distribution
    outputs_path = get_outputs_path()
    _update_speciation_state_cluster0_size_after_distribution(outputs_path)
    
    # ========================================================================
    # PHASE 8: UPDATE METRICS & STATS
    # ========================================================================
    state["logger"].info("=== Phase 8: Update Metrics & Stats ===")
    # NOTE: This is Phase 8 within process_generation().
    # Phase 7 (redistribution) already completed, so files exist for metrics calculation.
    
    # 21. Update metrics from corrected files
    # Update c-TF-IDF labels for all species
    from .labeling import update_species_labels
    update_species_labels(
        state["species"],
        current_generation=current_generation,
        n_words=10,
        logger=state["logger"]
    )
    
    # Record metrics (calculated from distributed files)
    # Files should exist after Phase 7 (Redistribution)
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    
    # Files must exist after Phase 7 (Redistribution)
    if not elites_path.exists():
        raise FileNotFoundError(f"elites.json not found at {elites_path} - required for metrics calculation (should have been created in Phase 7)")
    if not reserves_path.exists():
        state["logger"].warning(f"reserves.json not found at {reserves_path} - using cluster0.size")
    
    metrics = state["metrics_tracker"].record_generation(
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
    
    # Validate metrics are calculated correctly from files
    is_valid, errors = validate_metrics_from_files(
        outputs_path=outputs_path,
        metrics=metrics.to_dict(),
        logger=state["logger"]
    )
    if not is_valid:
        state["logger"].warning(f"Metrics validation found {len(errors)} errors")
        for error in errors[:5]:  # Log first 5 errors
            state["logger"].warning(f"  - {error}")
    else:
        state["logger"].debug("Metrics validation passed - all metrics match file contents")
    
    # Save speciation_state.json with current state
    outputs_path = get_outputs_path()
    state_path = str(outputs_path / "speciation_state.json")
    save_state(state_path)
    state["logger"].debug("Saved speciation state after metrics update")
    
    # Save events tracker
    if "_events_tracker" in state:
        state["_events_tracker"].save()
        tracker_summary = state["_events_tracker"].get_summary()
        state["logger"].debug(f"Events tracker: {tracker_summary['total_events']} events for {tracker_summary['unique_genomes']} genomes")
    
    # Save genome tracker (master registry)
    if "_genome_tracker" in state:
        state["_genome_tracker"].save()
        stats = state["_genome_tracker"].get_distribution_stats()
        state["logger"].debug(f"Genome tracker: {stats['total_genomes']} genomes tracked")
    
    # ========================================================================
    # PHASE 9: RETURN (distribution already completed in Phase 7)
    # ========================================================================
    state["logger"].info("=== Phase 9: Complete - Distribution already completed in Phase 7 ===")
    
    return state["species"], state["cluster0"]


def cluster0_speciation_isolated(current_generation: int, config: "SpeciationConfig", logger=None) -> List[Species]:
    """
    Apply leader-follower clustering on cluster 0 in complete isolation (like generation 0).
    
    Flow 2: Two-phase approach with no leader update and no radius enforcement.
    - Phase 1: Collect all potential leader groups (no species formation)
    - Phase 2: Form species if group size >= min_island_size (keep all members, no filtering)
    
    Uses in-memory state["cluster0"].individuals (reserves + temp unassigned)
    instead of loading reserves.json. Sync and add-from-temp are done in Phase 2
    before this call.
    
    Args:
        current_generation: Current generation number
        config: SpeciationConfig object with parameters
        logger: Optional logger instance
        
    Returns:
        List of newly formed Species (empty list if none formed)
    """
    from .species import Species, Individual, generate_species_id
    from .distance import ensemble_distance, ensemble_distances_batch
    import numpy as np
    
    if logger is None:
        logger = get_logger("Cluster0SpeciationIsolated")
    
    state = _get_state()
    cluster0 = state.get("cluster0")
    if cluster0 is None:
        logger.debug("cluster0 not in state, no speciation possible")
        return []
    
    # Build individuals from in-memory cluster0 (with embeddings)
    individuals = [ind for ind in cluster0.individuals if getattr(ind, "embedding", None) is not None]
    
    if len(individuals) < config.cluster0_min_cluster_size:
        logger.debug(f"Cluster 0 has {len(individuals)} individuals with embeddings, need {config.cluster0_min_cluster_size} to attempt speciation")
        return []
    
    # Sort by fitness (descending) - highest fitness processed first
    sorted_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
    
    # PHASE 1: Collect all potential leader groups (NO species formation)
    # Potential leaders: Dict mapping leader_id -> (None, embedding, phenotype, Individual, followers_list)
    # Note: species_id is always None in Phase 1 (no species formed yet)
    # Note: leader_id is an integer (ind.id), not a string
    potential_leaders: Dict[int, Tuple[None, np.ndarray, Optional[np.ndarray], Individual, List[Individual]]] = {}
    
    # First individual becomes first potential leader
    first = sorted_individuals[0]
    potential_leaders[first.id] = (None, first.embedding, first.phenotype, first, [])
    remaining_individuals = sorted_individuals[1:]
    
    # Process ALL remaining individuals (NO early species formation)
    for ind in remaining_individuals:
        assigned = False
        min_dist = float('inf')
        nearest_leader_id = None
        
        # Check against ALL potential leaders (all are active in Phase 1)
        if potential_leaders:
            # Collect all leader embeddings and phenotypes
            leader_embeddings = []
            leader_phenotypes = []
            leader_ids = []
            for pl_id, (_, pl_emb, pl_pheno, _, _) in potential_leaders.items():
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
            
            # If within threshold, add as follower (NO species formation check here)
            if nearest_leader_id is not None and min_dist < config.theta_sim:
                _, pl_emb, pl_pheno, pl_ind, followers = potential_leaders[nearest_leader_id]
                # Add as follower (tracked but no species yet)
                followers.append(ind)
                assigned = True
        
        # If not assigned to any potential leader, become a new potential leader
        if not assigned:
            potential_leaders[ind.id] = (None, ind.embedding, ind.phenotype, ind, [])
    
    # PHASE 2: Form species from groups that meet min_island_size
    new_species_list: List[Species] = []
    individuals_to_remove: List[Individual] = []
    
    for pl_id, (_, pl_emb, pl_pheno, pl_ind, followers) in potential_leaders.items():
        all_members = [pl_ind] + followers
        
        if len(all_members) >= config.min_island_size:
            # Create species with original potential leader (NO update)
            # Keep ALL members (NO radius filtering)
            new_species_id = generate_species_id()
            new_species = Species(
                id=new_species_id,
                leader=pl_ind,  # Original potential leader, NO update
                members=all_members,  # ALL members, NO filtering
                radius=config.theta_sim,
                created_at=current_generation,
                last_improvement=current_generation,
                cluster_origin="natural",
                parent_ids=None,
                leader_distance=0.0
            )
            new_species_list.append(new_species)
            individuals_to_remove.extend(all_members)
            logger.info(
                f"Cluster 0 speciation: Created species {new_species.id} from {len(all_members)} "
                f"individuals (leader={pl_ind.id}, followers={len(followers)})"
            )
        else:
            # Below min_island_size → all stay in cluster 0 (no removal)
            logger.debug(
                f"Cluster 0 speciation: group with {len(all_members)} members < "
                f"min_island_size {config.min_island_size} → staying in cluster 0"
            )
    
    # Remove formed species members from in-memory cluster 0
    state = _get_state()
    if individuals_to_remove and state.get("cluster0"):
        removed_count = state["cluster0"].remove_batch(individuals_to_remove)
        logger.debug(f"Removed {removed_count} individuals from in-memory cluster 0 (formed {len(new_species_list)} new species)")
    
    logger.info(f"Cluster 0 speciation isolated: formed {len(new_species_list)} new species from {len(individuals)} cluster 0 individuals")
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


def _update_speciation_state_cluster0_size_after_distribution(outputs_path) -> None:
    """Update cluster0.size in speciation_state.json to match reserves.json after distribution.
    
    This is called after Phase 7 (redistribution) to ensure cluster0.size in speciation_state.json
    matches the actual count in reserves.json after genomes have been distributed.
    """
    outputs_path = Path(outputs_path)
    state_path = outputs_path / "speciation_state.json"
    reserves_path = outputs_path / "reserves.json"
    if not state_path.exists() or not reserves_path.exists():
        return
    try:
        with open(reserves_path, 'r', encoding='utf-8') as f:
            n = len(json.load(f))
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        cluster0 = state.get("cluster0")
        if isinstance(cluster0, dict):
            cluster0["size"] = n
        if "cluster0_size_from_reserves" in state:
            state["cluster0_size_from_reserves"] = n
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        try:
            _log = _get_state().get("logger")
            if _log:
                _log.debug(f"Updated speciation_state cluster0 size to {n} (from reserves.json)")
        except Exception:
            pass
    except Exception as e:
        try:
            _log = _get_state().get("logger")
            if _log:
                _log.warning(f"Failed to update speciation_state cluster0 size after distribution: {e}")
        except Exception:
            pass




def save_state(path: str) -> None:
    """Save state to file.
    
    Note: Config is NOT saved here as it's passed as project arguments or fixed constants.
    The cluster0 section only contains metadata (size, speciation_events), not full member data
    since reserves.json already stores the complete genome data for cluster 0.
    
    Size calculation:
    - For active/frozen species: size = count from member_ids (derived from elites.json after capacity enforcement).
      Capacity enforcement considers ALL genomes from all generations via genome_tracker, sorts by fitness,
      keeps top species_capacity, and archives the rest (species_id=-1 in tracker). The genome_tracker
      is the authoritative source, and elites.json reflects the tracker state after Phase 7 distribution.
    - For extinct species: size = count from elites.json (historical reference).
    
    Species storage strategy:
    - Active species (state="active") - full data saved in species dict (participate in evolution)
    - Frozen species (state="frozen") - full data saved in species dict (stagnated, excluded from parent selection, but still alive)
    - Extinct species (state="extinct") - full data saved in historical_species dict (merged parents, no longer alive)
    - Incubator species (state="incubator") - only species ID tracked in historical_species (moved to cluster 0, just for tracking)
    
    IMPORTANT: Members Storage:
    - Species members are NOT saved as full Individual objects in speciation_state.json
    - Only member_ids (list of genome IDs) are saved for storage efficiency
    - The members list in memory may be empty even if size > 0 - this is EXPECTED
    - Members are reconstructed lazily from elites.json when load_state() is called
    - This is a storage optimization: full member data is in elites.json, only IDs are stored in state
    """
    import numpy as np
    
    state = _get_state()
    logger = state["logger"]
    
    # Calculate actual sizes and member IDs from elites.json for each species
    # This gives the true current size across all generations
    # Read elites.json AFTER distribution to ensure we have the latest data
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
        # Use sets to track unique IDs since elites.json is cumulative
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
        # Use unique count as species_sizes (size = number of distinct genomes, not raw rows)
        for species_id, ids in species_member_ids.items():
            species_sizes[species_id] = len(ids)
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
            
            # Get member IDs from elites.json (which reflects genome_tracker after Phase 7 distribution - only top species_capacity remain)
            # Convert sid to int for lookup (state["species"] keys are strings, but species_member_ids uses int keys)
            # NOTE: Only member_ids are saved, not full member objects (storage optimization)
            # Members are reconstructed from elites.json when load_state() is called
            sid_int = int(sid)
            if sid_int in species_member_ids:
                sp_dict["member_ids"] = species_member_ids[sid_int]
            else:
                if elites_path.exists():
                    logger.warning(f"Species {sid} not found in elites.json, using in-memory member IDs ({len(sp.members)})")
                sp_dict["member_ids"] = [m.id for m in sp.members]
            # Size = count from elites.json (which reflects genome_tracker after Phase 7 - only top species_capacity genomes remain)
            # Validation: size should match member_ids length
            sp_dict["size"] = len(sp_dict["member_ids"])
            if sp_dict["size"] != len(sp_dict["member_ids"]):
                logger.warning(f"Species {sid}: size mismatch - size={sp_dict['size']}, member_ids count={len(sp_dict['member_ids'])}")
            if sid_int in species_member_ids:
                logger.debug(f"Species {sid}: size={sp_dict['size']} from elites.json (reflects genome_tracker after Phase 7), in-memory={len(sp.members)}")
            species_dict[str(sid)] = sp_dict
        elif sp.species_state == "frozen":
            # Frozen species also get full data (including leader_embedding and leader_distance)
            # Frozen species preserve all members from when they were active
            sp_dict = sp.to_dict()
            
            # Get member IDs from elites.json (which reflects genome_tracker after Phase 7 distribution - only top species_capacity remain)
            # Convert sid to int for lookup (state["species"] keys are strings, but species_member_ids uses int keys)
            # NOTE: Only member_ids are saved, not full member objects (storage optimization)
            sid_int = int(sid)
            if sid_int in species_member_ids:
                sp_dict["member_ids"] = species_member_ids[sid_int]
            else:
                if elites_path.exists():
                    logger.warning(f"Frozen species {sid} not found in elites.json, using in-memory member IDs ({len(sp.members)})")
                sp_dict["member_ids"] = [m.id for m in sp.members]
            # Size = count from elites.json (which reflects genome_tracker after Phase 7 - only top species_capacity genomes remain)
            sp_dict["size"] = len(sp_dict["member_ids"])
            # Validation: ensure size matches member_ids count
            if sp_dict["size"] != len(sp_dict["member_ids"]):
                logger.warning(f"Frozen species {sid}: size mismatch - size={sp_dict['size']}, member_ids count={len(sp_dict['member_ids'])}")
            if sid_int in species_member_ids:
                logger.debug(f"Frozen species {sid}: size={sp_dict['size']} from elites.json (after capacity enforcement), in-memory={len(sp.members)}")
            # Ensure leader_embedding is ALWAYS preserved for frozen species (needed for merging)
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
                # Convert sid to int for lookup (historical_species keys may be strings or ints)
                # NOTE: Only member_ids are saved for extinct species too (storage optimization)
                sid_int = int(sid) if isinstance(sid, str) else sid
                if sid_int in species_member_ids:
                    sp_dict["member_ids"] = species_member_ids[sid_int]
                else:
                    sp_dict["member_ids"] = [m.id for m in sp.members]
                sp_dict["size"] = len(sp_dict["member_ids"])
                # Validation: ensure size matches member_ids count
                if sp_dict["size"] != len(sp_dict["member_ids"]):
                    logger.warning(f"Extinct species {sid}: size mismatch - size={sp_dict['size']}, member_ids count={len(sp_dict['member_ids'])}")
                
                # STORAGE OPTIMIZATION: Remove leader_embedding for all extinct species to save storage space
                # Extinct species are no longer active, so embeddings are not needed
                if "leader_embedding" in sp_dict:
                    del sp_dict["leader_embedding"]
                logger.debug(f"Removed leader_embedding for extinct species {sid} (storage optimization)")
                species_dict[str(sid)] = sp_dict
            elif sp.species_state == "incubator":
                # Incubator species - just track ID
                incubator_ids.append(sid)
    
    # Add species that exist in elites.json but are missing from in-memory state
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
                # species_id is already an int from elites.json, so no conversion needed
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
                    "member_ids": member_ids,  # All member IDs from elites.json (unique)
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
                    "size": len(member_ids)  # Must match elites.json unique count
                }
                logger.info(f"Reconstructed species {species_id} entry from elites.json (size={len(member_ids)}, state=frozen)")
    
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
                # This should not happen now that member_ids is populated from elites.json (which reflects genome_tracker after Phase 7)
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

    # Ensure size always equals len(member_ids) (source of truth: genome_tracker via elites.json after Phase 7 distribution)
    min_island_size = state["config"].min_island_size
    for sid_str, sp_dict in species_dict.items():
        mids = sp_dict.get("member_ids", [])
        n = len(mids)
        if sp_dict.get("size") != n:
            logger.warning(f"Species {sid_str}: size={sp_dict.get('size')} != len(member_ids)={n}; setting size={n}")
            sp_dict["size"] = n
        if n < min_island_size:
            logger.debug(
                "Species %s has size %d < min_island_size %d (count from elites.json); "
                "process_extinctions should move such species to incubator when in-memory size drops.",
                sid_str, n, min_island_size
            )

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

    # Round speciation event fitness values for clean storage
    if "speciation_events" in cluster0_dict:
        for evt in cluster0_dict["speciation_events"]:
            if isinstance(evt, dict) and "leader_fitness" in evt:
                try:
                    evt["leader_fitness"] = round(float(evt["leader_fitness"]), 4)
                except Exception:
                    pass
    
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
                    cluster0_fitnesses.append(float(fitness) if fitness is not None else 0.0)
                if cluster0_fitnesses:
                    cluster0_max_fitness = round(max(cluster0_fitnesses), 4)
                    cluster0_min_fitness = round(min(cluster0_fitnesses), 4)
        except Exception as e:
            logger.warning(f"Failed to calculate cluster0 fitness stats: {e}")
    
    cluster0_dict["max_fitness"] = cluster0_max_fitness
    cluster0_dict["min_fitness"] = cluster0_min_fitness
    
    state_dict = {
        "species": species_dict,
        "incubators": sorted(incubator_ids),  # Just list of species IDs
        "cluster0": cluster0_dict,
        "cluster0_size_from_reserves": actual_cluster0_size,  # Store actual size from reserves.json
        "global_best_id": state["global_best"].id if state["global_best"] else None,
        "metrics": state["metrics_tracker"].to_dict(),
        "config": state["config"].to_dict()  # Save config to ensure arguments are preserved
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
            # NOTE: This is lazy loading - members are reconstructed from elites.json
            # The members list in saved state is empty (only member_ids are stored for efficiency)
            # This is EXPECTED behavior: full member data is in elites.json, only IDs in state
            if member_ids:
                outputs_path = get_outputs_path()
                elites_path = outputs_path / "elites.json"
                if elites_path.exists():
                    try:
                        with open(elites_path, 'r', encoding='utf-8') as f:
                            elites_genomes = json.load(f)
                        
                        # Create a lookup for genomes by ID
                        genome_by_id = {g.get("id"): g for g in elites_genomes}
                        
                        # Load all members (excluding leader if it's in member_ids).
                        # Include all that exist in elites so sp.size matches len(member_ids) for process_extinctions.
                        loaded_count = 0
                        for member_id in member_ids:
                            if member_id == leader.id:
                                continue  # Leader already added
                            if member_id in genome_by_id:
                                member_genome = genome_by_id[member_id]
                                member = Individual.from_genome(member_genome)
                                members.append(member)
                                loaded_count += 1
                        
                        # Validation: verify all member_ids were loaded (except leader)
                        expected_count = len(member_ids) - (1 if leader.id in member_ids else 0)
                        if loaded_count != expected_count:
                            logger.warning(f"Species {sid}: member loading incomplete - loaded {loaded_count}/{expected_count} members from elites.json")
                    except Exception as e:
                        logger.warning(f"Failed to load members for species {sid} from elites.json: {e}")
            
            # max_fitness = actual max over current members only, no merge with stored value.
            max_fit = max((m.fitness for m in members), default=0.0)
            species = Species(
                id=sid,
                leader=leader,
                members=members,  # Load all members, not just leader
                radius=sp_dict.get("radius", config.theta_sim),
                stagnation=sp_dict.get("stagnation", 0),
                max_fitness=max_fit,
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
            
            # If leader embedding is missing, try to load from elites.json (for both active and frozen)
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
        logger.info(f"Loaded speciation state from {path}: {active_count} active, {frozen_count} frozen, {historical_count} historical species")
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

    # Ensure global state is clean for each invocation (avoids cross-run contamination)
    reset_speciation_module()
    
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
        
        # NOTE: process_generation() has already:
        # - Distributed genomes to files (Phase 7 in process_generation)
        # - Updated metrics from distributed files (Phase 8 in process_generation)
        # - Saved speciation_state.json with correct member_ids from elites.json (after Phase 7 distribution)
        
        # Distribution is complete, files are ready for any post-processing
        
        # Get state reference
        state = _get_state()
        # Ensure events tracker is saved after distribution
        if "_events_tracker" in state:
            state["_events_tracker"].save()
        
        # Save genome tracker (master registry)
        if "_genome_tracker" in state:
            state["_genome_tracker"].save()
        
        # Log generation summary using file-based data
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
            outputs_path, current_generation, logger=logger, expect_temp_empty=True
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
            "genomes_updated": state["_genome_tracker"].get_distribution_stats()["total_genomes"] if "_genome_tracker" in state else 0,
            "success": True
        }
        
        # Distribution stats are available from genome_tracker
        if "_genome_tracker" in state:
            stats = state["_genome_tracker"].get_distribution_stats()
            result.update({
                "elites_moved": stats["by_species_id"].get(">0", 0),  # Approximate - actual count from elites.json
                "reserves_moved": stats["by_species_id"].get("0", 0)  # Approximate - actual count from reserves.json
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
        
        # best_fitness and avg_fitness are not stored in speciation; they are only at gen level
        # (population_max_toxicity, max_score_variants, avg_fitness).
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
            # Cluster quality (will be filled from metrics or defaults)
            "cluster_quality": None
        }
        
        # best_fitness_value is used for population_max_toxicity and max_score_variants only
        best_fitness_value = 0.0
        
        if current_metrics:
            best_fitness_value = current_metrics.best_fitness
            speciation_summary.update({
                "inter_species_diversity": round(current_metrics.inter_species_diversity, 4),
                "intra_species_diversity": round(current_metrics.intra_species_diversity, 4),
                "total_population": current_metrics.total_population,
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
            
            speciation_summary.update({
                "inter_species_diversity": 0.0,
                "intra_species_diversity": 0.0,
                "total_population": total_pop,
            })
        
        # best_fitness_value used for population_max_toxicity and max_score_variants
        
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
        
        # max_score_variants is NOT updated here - it is correctly calculated in main.py from temp.json
        # BEFORE speciation (representing max fitness among variants created this generation).
        # Updating it here with population max would overwrite the correct value.
        # The population max is tracked separately as population_max_toxicity.
        
        if "speciation_summary" not in evolution_tracker:
            evolution_tracker["speciation_summary"] = {}
        
        evolution_tracker["speciation_summary"].update({
            "current_species_count": speciation_result.get("species_count", 0),
            "current_reserves_size": speciation_result.get("reserves_size", 0),
            "total_speciation_events": metrics_summary.get("total_speciation_events", 0),
            "total_merge_events": metrics_summary.get("total_merge_events", 0),
            "total_extinction_events": metrics_summary.get("total_extinction_events", 0),
        })
        
        # Update cumulative population_max_toxicity at tracker level.
        # population_max_toxicity = max over all generations of (best toxicity in that
        # generation's population, i.e. elites + reserves). Used for Pareto quality axis.
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
