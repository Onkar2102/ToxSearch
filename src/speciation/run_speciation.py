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
            "species": {},
            "cluster0": Cluster0(
                min_cluster_size=(config or SpeciationConfig()).cluster0_min_cluster_size,
                theta_sim=(config or SpeciationConfig()).theta_sim,
                max_capacity=(config or SpeciationConfig()).cluster0_max_capacity,
                logger=logger or get_logger("Speciation")
            ),
            "global_best": None,
            "metrics_tracker": SpeciationMetricsTracker(logger=logger or get_logger("Speciation")),
            "_current_gen_events": {"speciation": 0, "merge": 0, "extinction": 0},
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
            entry = ind.to_genome() if hasattr(ind, 'to_genome') else {"id": ind.id}
            entry["archived_at_generation"] = generation
            entry["archive_reason"] = reason
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
    state["_current_gen_events"] = {"speciation": 0, "merge": 0, "extinction": 0}
    state["_archived_count"] = 0
    
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
    
    state["species"] = leader_follower_clustering(
        temp_path=temp_path,
        speciation_state_path=speciation_state_path,
        theta_sim=state["config"].theta_sim,
        current_generation=current_generation,
        logger=state["logger"]
    )
    
    # Step 3: Enforce capacities directly (species and cluster 0)
    # For species with count > species_capacity: order by fitness, remove excess
    for sp in list(state["species"].values()):
        if sp.size > state["config"].species_capacity:
            # Sort members by fitness (descending)
            sp.members.sort(key=lambda x: x.fitness, reverse=True)
            # Remove excess genomes (keep top species_capacity)
            excess = sp.members[state["config"].species_capacity:]
            sp.members = sp.members[:state["config"].species_capacity]
            # Archive removed genomes
            _archive_individuals(excess, current_generation, "species_capacity_exceeded")
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
    
    # Check for speciation events in cluster 0
    new_species = state["cluster0"].check_speciation(current_generation)
    if new_species:
        state["species"][new_species.id] = new_species
        state["_current_gen_events"]["speciation"] += 1
    
    # Step 4: Record fitness for all species
    for sp in state["species"].values():
        sp.record_fitness(current_generation)
    
    # Step 5: Island merging
    state["species"], merge_events = process_merges(
        state["species"], 
        theta_merge=state["config"].theta_merge,
        theta_sim=state["config"].theta_sim,
        current_gen=current_generation,
        max_capacity=state["config"].species_capacity,
        logger=state["logger"]
    )
    state["_current_gen_events"]["merge"] = len(merge_events)
    
    # Step 6: Freeze stagnant species and move small species to cluster 0
    state["species"], extinction_events = process_extinctions(
        state["species"],
        state["cluster0"],
        current_generation,
        max_stagnation=state["config"].max_stagnation,
        min_size=state["config"].min_island_size,
        logger=state["logger"]
    )
    state["_current_gen_events"]["extinction"] = len(extinction_events)
    
    # Step 7: Record metrics
    state["metrics_tracker"].record_generation(
        current_generation, state["species"], state["cluster0"].size,
        state["_current_gen_events"]["speciation"], state["_current_gen_events"]["merge"],
        state["_current_gen_events"]["extinction"]
    )
    
    log_generation_summary(current_generation, state["species"], state["cluster0"].size,
                           state["_current_gen_events"], state["logger"])
    
    # Auto-save state after processing
    outputs_path = get_outputs_path()
    state_path = str(outputs_path / "speciation_state.json")
    save_state(state_path)
    
    # Remove embeddings from temp.json after speciation is complete
    remove_embeddings_from_temp(temp_path=temp_path, logger=state["logger"])
    
    return state["species"], state["cluster0"]


def distribute_genomes(
    temp_path: Optional[str] = None,
    north_star_metric: str = "toxicity") -> Dict[str, int]:
    """
    Distribute genomes from temp.json to elites.json and reserves.json.
    
    Distribution is based on species_id:
    - Genomes with species_id > 0 → elites.json (all species members)
    - Genomes with species_id == 0 → reserves.json (cluster 0 / reserves)
    
    Args:
        temp_path: Path to temp.json file (defaults to outputs_path / "temp.json")
        north_star_metric: The metric to use for scoring (default: "toxicity")
        
    Returns:
        dict: Distribution statistics with elites_moved, reserves_moved, total_processed
    """
    from utils.population_io import _extract_north_star_score
    
    state = _get_state()
    config = state["config"]
    logger = state["logger"]
    
    if temp_path is None:
        outputs_path = get_outputs_path()
        temp_path = str(outputs_path / "temp.json")
    
    temp_path_obj = Path(temp_path)
    if not temp_path_obj.exists():
        logger.warning("temp.json not found for distribution")
        return {"elites_moved": 0, "reserves_moved": 0, "total_processed": 0}
    
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    
    # Load genomes from temp.json
    with open(temp_path_obj, 'r', encoding='utf-8') as f:
        temp_genomes = json.load(f)
    
    if not temp_genomes:
        logger.debug("No genomes in temp.json to distribute")
        return {"elites_moved": 0, "reserves_moved": 0, "total_processed": 0}
    
    elites_to_move = []
    reserves_to_move = []
    
    # Distribute genomes based on species_id
    for genome in temp_genomes:
        if not genome or not genome.get("prompt"):
            continue
        
        genome_id = genome.get("id")
        species_id = genome.get("species_id", CLUSTER_0_ID)
        
        # Distribute based on species_id
        if species_id is not None and species_id > 0:
            genome["initial_state"] = "elite"
            elites_to_move.append(genome)
            logger.debug(f"Genome {genome_id} from species {species_id} → elites.json")
        else:
            genome["initial_state"] = "reserves"
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
        
        # Enforce capacity limit
        if len(reserves_to_save) > config.cluster0_max_capacity:
            reserves_to_save = reserves_to_save[:config.cluster0_max_capacity]
            logger.debug(f"Reserves capacity exceeded, keeping top {config.cluster0_max_capacity} genomes")
        
        with open(reserves_path, 'w', encoding='utf-8') as f:
            json.dump(reserves_to_save, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Moved {len(reserves_to_move)} genomes to reserves.json (capacity: {len(reserves_to_save)})")
    
    # Clear temp.json
    with open(temp_path_obj, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2, ensure_ascii=False)
    
    distribution_stats = {
        "elites_moved": len(elites_to_move),
        "reserves_moved": len(reserves_to_move),
        "total_processed": len(temp_genomes)
    }
    
    logger.info(f"Distribution complete: {distribution_stats['total_processed']} genomes → "
                f"{distribution_stats['elites_moved']} elites, "
                f"{distribution_stats['reserves_moved']} reserves")
    
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
    """Save state to file."""
    state = _get_state()
    state_dict = {
        "config": state["config"].to_dict(),
        "species": {str(sid): sp.to_dict() for sid, sp in state["species"].items()},
        "cluster0": state["cluster0"].to_dict(),
        "global_best_id": state["global_best"].id if state["global_best"] else None,
        "metrics": state["metrics_tracker"].to_dict()
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=2, ensure_ascii=False)
    state["logger"].info(f"Saved speciation state to {path}")


def load_state(path: str) -> bool:
    """
    Load state from file and restore species, cluster 0, and metrics.
    
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
        
        # Restore species
        state["species"] = {}
        max_species_id = 0
        
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
                cluster_origin=sp_dict.get("cluster_origin"),
                parent_ids=sp_dict.get("parent_ids"),
                parent_id=sp_dict.get("parent_id")
            )
            
            state["species"][sid] = species
        
        SpeciesIdGenerator.set_min_id(max_species_id + 1)
        
        logger.info(f"Loaded speciation state from {path}: {len(state['species'])} species")
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
        _state["cluster0"] = Cluster0(
            min_cluster_size=config.cluster0_min_cluster_size,
            theta_sim=config.theta_sim,
            max_capacity=config.cluster0_max_capacity,
            logger=logger
        )
        _state["global_best"] = None
        _state["metrics_tracker"] = SpeciationMetricsTracker(logger=logger)
        _state["_current_gen_events"] = {"speciation": 0, "merge": 0, "extinction": 0}
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
        logger.error("Temp file not found: %s", temp_path)
        return {
            "species_count": 0,
            "cluster0_size": 0,
            "speciation_events": 0,
            "merge_events": 0,
            "extinction_events": 0,
            "archived_count": 0,
            "genomes_updated": 0,
            "success": False,
            "error": "temp_file_not_found"
        }
    
    try:
        with open(temp_path_obj, 'r', encoding='utf-8') as f:
            genomes = json.load(f)
        
        if not genomes:
            logger.warning("No genomes found in temp.json")
            return {
                "species_count": 0,
                "cluster0_size": 0,
                "speciation_events": 0,
                "merge_events": 0,
                "extinction_events": 0,
                "archived_count": 0,
                "genomes_updated": 0,
                "success": False,
                "error": "no_genomes"
            }
        
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
        
        # Distribute genomes
        distribution_result = distribute_genomes(
            temp_path=temp_path,
            north_star_metric=north_star_metric
        )
        logger.info("Distribution complete: %d elites, %d reserves",
                   distribution_result.get("elites_moved", 0),
                   distribution_result.get("reserves_moved", 0))
        
        # Get event counts
        state = _get_state()
        events = state["_current_gen_events"]
        
        result = {
            "species_count": len(species),
            "cluster0_size": cluster0.size,
            "reserves_size": cluster0.size,
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
            "Speciation completed: %d species, %d in cluster 0, "
            "events: speciation=%d, merge=%d, extinction=%d, archived=%d",
            result["species_count"], result["cluster0_size"],
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
            "cluster0_size": 0,
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
    """Get current speciation statistics."""
    logger = get_logger("RunSpeciation", log_file)
    
    state = _get_state()
    if state is None:
        return {
            "initialized": False,
            "species_count": 0,
            "cluster0_size": 0,
            "reserves_size": 0
        }
    
    metrics_summary = state["metrics_tracker"].get_summary()
    
    return {
        "initialized": True,
        "species_count": len(state["species"]),
        "cluster0_size": state["cluster0"].size,
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
            "cluster0_size": speciation_result.get("cluster0_size", 0),
            "reserves_size": speciation_result.get("reserves_size", speciation_result.get("cluster0_size", 0)),
            "speciation_events": speciation_result.get("speciation_events", 0),
            "merge_events": speciation_result.get("merge_events", 0),
            "extinction_events": speciation_result.get("extinction_events", 0),
            "archived_count": speciation_result.get("archived_count", 0),
            "elites_moved": speciation_result.get("elites_moved", 0),
            "reserves_moved": speciation_result.get("reserves_moved", 0),
            "genomes_updated": speciation_result.get("genomes_updated", 0),
        }
        
        if current_metrics:
            speciation_summary.update({
                "inter_species_diversity": round(current_metrics.inter_species_diversity, 4),
                "intra_species_diversity": round(current_metrics.intra_species_diversity, 4),
                "total_population": current_metrics.total_population,
                "best_fitness": round(current_metrics.best_fitness, 4),
                "avg_fitness": round(current_metrics.avg_fitness, 4),
            })
        
        generations = evolution_tracker.get("generations", [])
        gen_entry = None
        for gen in generations:
            if gen.get("generation_number") == current_generation:
                gen_entry = gen
                break
        
        if gen_entry:
            gen_entry["speciation"] = speciation_summary
        else:
            gen_entry = {
                "generation_number": current_generation,
                "speciation": speciation_summary
            }
            generations.append(gen_entry)
            evolution_tracker["generations"] = generations
        
        if "speciation_summary" not in evolution_tracker:
            evolution_tracker["speciation_summary"] = {}
        
        evolution_tracker["speciation_summary"].update({
            "current_species_count": speciation_result.get("species_count", 0),
            "current_cluster0_size": speciation_result.get("cluster0_size", 0),
            "total_speciation_events": metrics_summary.get("total_speciation_events", 0),
            "total_merge_events": metrics_summary.get("total_merge_events", 0),
            "total_extinction_events": metrics_summary.get("total_extinction_events", 0),
        })
        
        with open(tracker_path, 'w', encoding='utf-8') as f:
            json.dump(evolution_tracker, f, indent=2, ensure_ascii=False)
        
        logger.info("Updated EvolutionTracker.json with speciation data for generation %d", current_generation)
        return True
        
    except Exception as e:
        logger.error("Failed to update EvolutionTracker with speciation data: %s", e, exc_info=True)
        return False
