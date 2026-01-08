"""
SpeciationModule.py

Main SpeciationModule class and entry point functions for Dynamic Islands framework.
Combines the SpeciationModule class with the run_speciation entry point.
"""

import json
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path

from .config import PlanAPlusConfig
from .island import Individual, Species, IslandMode, SpeciesIdGenerator
from .embeddings import compute_and_save_embeddings, get_embedding_model
from .leader_follower import leader_follower_clustering, update_species_leaders
from .limbo import LimboBuffer
from .modes import update_all_island_modes
from .intra_island import select_parents_elite_focused
from .merging import process_merges
from .extinction import process_extinctions, find_global_best
from .migration import process_migrations
from .adaptive_threshold import process_adaptive_thresholds
from .metrics import SpeciationMetricsTracker, log_generation_summary

from utils import get_custom_logging
from utils import get_system_utils

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _ = get_system_utils()


class SpeciationModule:
    """
    Main orchestrator for Dynamic Islands speciation.
    
    Manages:
    1. Embedding computation
    2. Leader-Follower clustering
    3. Limbo buffer management
    4. Mode switching (Explore/Exploit/Default)
    5. Island merging
    6. Extinction and repopulation
    7. Inter-island migration
    8. Adaptive threshold adjustment
    
    Usage:
        >>> config = PlanAPlusConfig()
        >>> module = SpeciationModule(config)
        >>> species, limbo = module.process_generation(genomes, current_generation)
    """
    
    def __init__(self, config: Optional[PlanAPlusConfig] = None, logger=None):
        self.config = config or PlanAPlusConfig()
        self.logger = logger or get_logger("SpeciationModule")
        
        self.species: Dict[int, Species] = {}
        self.limbo = LimboBuffer(
            default_ttl=self.config.limbo_ttl,
            min_cluster_size=self.config.limbo_min_cluster_size,
            theta_sim=self.config.theta_sim,
            logger=self.logger
        )
        self.global_best: Optional[Individual] = None
        self.metrics_tracker = SpeciationMetricsTracker(logger=self.logger)
        
        self._current_gen_events = {"speciation": 0, "merge": 0, "extinction": 0, "migration": 0}
        self._embedding_model = None
        
        self.logger.info(f"SpeciationModule initialized with config: theta_sim={self.config.theta_sim}")
    
    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model(model_name=self.config.embedding_model)
        return self._embedding_model
    
    def process_generation(self, population: List[Dict[str, Any]], current_generation: int,
                           mutate_fn: Optional[Callable] = None, temp_path: Optional[str] = None) -> Tuple[Dict[int, Species], LimboBuffer]:
        """
        Process a single generation with full speciation pipeline.
        
        This is the main orchestration method that runs the complete pipeline:
        
        1. Compute embeddings: Read temp.json, compute embeddings, add "prompt_embedding" field, save back
        2. Leader-Follower clustering: Read temp.json, convert to Individuals, assign to species
        3. Limbo update: Check for speciation events in limbo
        4. Update leaders: Ensure leaders are highest-fitness members
        5. Mode switching: Adapt island modes based on fitness trends
        6. Island merging: Combine similar species
        7. Extinction & repopulation: Remove stagnant species, create new ones
        8. Migration: Exchange individuals between related species
        9. Adaptive thresholds: Adjust radii and split heterogeneous islands
        10. Record metrics: Track generation statistics
        
        Note: The embedding computation modifies temp.json in-place by adding "prompt_embedding"
        field to each genome. If embeddings already exist, computation is skipped.
        
        Args:
            population: List of genome dictionaries (with prompts and fitness)
            current_generation: Current generation number
            mutate_fn: Optional mutation function (for repopulation)
            temp_path: Optional path to temp.json (for embedding computation)
        
        Returns:
            Tuple of (species_dict, limbo_buffer)
        """
        self.logger.info(f"=== Speciation Generation {current_generation} ===")
        
        self._current_gen_events = {"speciation": 0, "merge": 0, "extinction": 0, "migration": 0}
        
        # Step 1: Compute and save embeddings to temp.json
        if temp_path is None:
            outputs_path = get_outputs_path()
            temp_path = str(outputs_path / "temp.json")
        
        compute_and_save_embeddings(
            temp_path=temp_path,
            model_name=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
            logger=self.logger
        )
        
        # Step 2: Leader-Follower clustering (reads directly from temp.json)
        self.species, limbo_candidates = leader_follower_clustering(
            temp_path=temp_path, theta_sim=self.config.theta_sim,
            viability_baseline=self.config.viability_baseline,
            current_generation=current_generation, logger=self.logger
        )
        self.limbo.add_batch(limbo_candidates, current_generation)
        
        # Step 3: Limbo update
        self.limbo.update_ttl(current_generation)
        new_species = self.limbo.check_speciation(current_generation)
        if new_species:
            self.species[new_species.id] = new_species
            self._current_gen_events["speciation"] += 1
        
        # Step 4: Update leaders and record fitness
        update_species_leaders(self.species)
        for sp in self.species.values():
            sp.record_fitness(current_generation)
        
        # Step 5: Mode switching
        update_all_island_modes(self.species, current_generation, self.limbo, self.config, self.logger)
        
        # Step 6: Island merging
        self.species, merge_events = process_merges(
            self.species, self.config.theta_merge, current_gen=current_generation,
            max_capacity=self.config.max_island_capacity, logger=self.logger
        )
        self._current_gen_events["merge"] = len(merge_events)
        
        # Step 7: Extinction and repopulation
        self.global_best = find_global_best(self.species)
        self.species, extinction_events = process_extinctions(
            self.species, self.limbo, self.global_best, current_generation,
            self.config.max_stagnation, self.config.min_island_size,
            self.config.repopulation_size, self.config.repopulation_mutation_rate,
            mutate_fn, self.config.theta_sim, self.logger
        )
        self._current_gen_events["extinction"] = len(extinction_events)
        
        # Step 8: Migration
        self.species, migration_events = process_migrations(
            self.species, current_generation, self.config.migration_frequency,
            self.config.k_neighbors, self.config.max_island_capacity,
            self.config.migration_selection, self.limbo, self.logger
        )
        self._current_gen_events["migration"] = len(migration_events)
        
        # Step 9: Adaptive threshold adjustment
        self.species, _ = process_adaptive_thresholds(
            self.species, current_generation, self.config.max_island_capacity,
            self.config.radius_shrink_factor, self.config.silhouette_threshold,
            self.config.silhouette_check_frequency, self.limbo, self.logger
        )
        
        # Step 10: Record metrics
        self.metrics_tracker.record_generation(
            current_generation, self.species, self.limbo.size,
            self._current_gen_events["speciation"], self._current_gen_events["merge"],
            self._current_gen_events["extinction"], self._current_gen_events["migration"]
        )
        
        log_generation_summary(current_generation, self.species, self.limbo.size,
                               self._current_gen_events, self.logger)
        
        return self.species, self.limbo
    
    def _prepare_individuals(self, population: List[Dict[str, Any]]) -> List[Individual]:
        """
        Convert genome dictionaries to Individual objects with embeddings.
        
        This method assumes embeddings have already been computed and saved to
        the "prompt_embedding" field in each genome (via compute_and_save_embeddings).
        
        Individual.from_genome() will automatically extract embeddings from the
        "prompt_embedding" field in the genome dictionary.
        
        Note: This method is kept for backward compatibility but is no longer
        used in the main pipeline (leader_follower_clustering reads directly from temp.json).
        
        Args:
            population: List of genome dictionaries (with "prompt_embedding" field)
        
        Returns:
            List of Individual objects with embeddings extracted from genomes
        """
        # Individual.from_genome() will extract embeddings from "prompt_embedding" field
        return [Individual.from_genome(genome) for genome in population]
    
    def get_species_for_genome(self, genome_id: int) -> Optional[int]:
        """Get species ID for a genome."""
        for sp in self.species.values():
            for m in sp.members:
                if m.id == genome_id:
                    return sp.id
        return None
    
    def get_parents_for_breeding(self, species_id: Optional[int] = None) -> Tuple[Individual, Individual]:
        """Select parents for breeding."""
        import numpy as np
        
        if not self.species:
            raise ValueError("No species available")
        
        if species_id is None:
            species_list = list(self.species.values())
            weights = [sp.best_fitness + 0.1 for sp in species_list]
            probs = np.array(weights) / sum(weights)
            target = species_list[np.random.choice(len(species_list), p=probs)]
        else:
            if species_id not in self.species:
                raise KeyError(f"Species {species_id} not found")
            target = self.species[species_id]
        
        return select_parents_elite_focused(target)
    
    def add_offspring(self, offspring: List[Dict[str, Any]], current_generation: int, temp_path: Optional[str] = None) -> None:
        """
        Add offspring to species or limbo.
        
        Note: Assumes embeddings are already computed and in "prompt_embedding" field.
        If not, call compute_and_save_embeddings() first.
        
        Args:
            offspring: List of offspring genome dictionaries
            current_generation: Current generation number
            temp_path: Optional path to temp.json (for embedding computation if needed)
        """
        from .leader_follower import find_nearest_leader
        
        # Ensure embeddings are computed
        if temp_path and not all("prompt_embedding" in g for g in offspring):
            compute_and_save_embeddings(
                temp_path=temp_path,
                model_name=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
                logger=self.logger
            )
            # Reload genomes with embeddings
            with open(temp_path, 'r', encoding='utf-8') as f:
                import json
                all_genomes = json.load(f)
                # Match offspring by ID
                offspring_dict = {g.get("id"): g for g in offspring}
                offspring = [all_genomes[i] for i, g in enumerate(all_genomes) if g.get("id") in offspring_dict]
        
        individuals = self._prepare_individuals(offspring)
        for ind in individuals:
            if ind.embedding is None:
                continue
            nearest_id, min_dist = find_nearest_leader(ind.embedding, self.species)
            if nearest_id and min_dist < self.config.theta_sim:
                self.species[nearest_id].add_member(ind)
            elif ind.fitness > self.config.viability_baseline:
                self.limbo.add(ind, current_generation)
    
    def update_genomes_with_species(self, genomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update genomes with species IDs and limbo status."""
        # Get IDs of limbo individuals
        limbo_ids = {ind.id for ind in self.limbo.individuals}
        
        for g in genomes:
            genome_id = g.get("id")
            g["species_id"] = self.get_species_for_genome(genome_id)
            # Mark if in limbo
            g["in_limbo"] = genome_id in limbo_ids
        return genomes
    
    def get_state(self) -> Dict:
        """Get state for serialization."""
        return {
            "config": self.config.to_dict(),
            "species": {sid: sp.to_dict() for sid, sp in self.species.items()},
            "limbo": self.limbo.to_dict(),
            "global_best_id": self.global_best.id if self.global_best else None,
            "metrics": self.metrics_tracker.to_dict()
        }
    
    def save_state(self, path: str) -> None:
        """Save state to file."""
        with open(path, 'w') as f:
            json.dump(self.get_state(), f, indent=2)
        self.logger.info(f"Saved speciation state to {path}")
    
    def reset(self) -> None:
        """Reset all state."""
        self.species = {}
        self.limbo = LimboBuffer(self.config.limbo_ttl, self.config.limbo_min_cluster_size,
                                 self.config.theta_sim, self.logger)
        self.global_best = None
        self.metrics_tracker = SpeciationMetricsTracker(self.logger)
        SpeciesIdGenerator.reset()
        self.logger.info("Speciation module reset")
    
    def __repr__(self):
        return f"SpeciationModule(species={len(self.species)}, limbo={self.limbo.size})"


# Global speciation module instance (singleton pattern)
_speciation_module: Optional[SpeciationModule] = None


def get_speciation_module(config: Optional[PlanAPlusConfig] = None, logger=None) -> SpeciationModule:
    """
    Get or create the global speciation module instance.
    
    Uses singleton pattern to maintain state across generations.
    
    Args:
        config: Optional configuration (uses defaults if None)
        logger: Optional logger instance
        
    Returns:
        SpeciationModule instance
    """
    global _speciation_module
    
    if _speciation_module is None:
        _speciation_module = SpeciationModule(config=config, logger=logger)
    
    return _speciation_module


def reset_speciation_module() -> None:
    """Reset the global speciation module (for testing or fresh start)."""
    global _speciation_module
    if _speciation_module is not None:
        _speciation_module.reset()
    _speciation_module = None


def run_speciation(
    temp_path: Optional[str] = None,
    current_generation: int = 0,
    config: Optional[PlanAPlusConfig] = None,
    log_file: Optional[str] = None,
    mutate_fn: Optional[Any] = None) -> Dict[str, Any]:
    """
    Run speciation processing for a single generation.
    
    This is the main entry point for speciation, similar to run_evolution().
    It handles:
    1. Loading genomes from temp.json (or provided path)
    2. Running speciation pipeline
    3. Updating genomes with species_id
    4. Saving updated genomes back to temp.json
    
    Args:
        temp_path: Path to temp.json file with evaluated genomes.
                   If None, uses default outputs_path / "temp.json"
        current_generation: Current generation number
        config: Optional PlanAPlusConfig (uses defaults if None)
        log_file: Optional log file path
        mutate_fn: Optional mutation function for repopulation
        
    Returns:
        Dict with speciation results:
        {
            "species_count": int,
            "limbo_size": int,
            "speciation_events": int,
            "merge_events": int,
            "extinction_events": int,
            "migration_events": int,
            "genomes_updated": int,
            "success": bool
        }
        
    Example:
        >>> result = run_speciation(
        ...     temp_path="data/outputs/temp.json",
        ...     current_generation=1
        ... )
        >>> print(f"Created {result['species_count']} species")
    """
    logger = get_logger("RunSpeciation", log_file)
    logger.info("Starting speciation: generation=%d", current_generation)
    
    # Determine temp path
    if temp_path is None:
        outputs_path = get_outputs_path()
        temp_path = str(outputs_path / "temp.json")
    
    temp_path_obj = Path(temp_path)
    if not temp_path_obj.exists():
        logger.error("Temp file not found: %s", temp_path)
        return {
            "species_count": 0,
            "limbo_size": 0,
            "speciation_events": 0,
            "merge_events": 0,
            "extinction_events": 0,
            "migration_events": 0,
            "genomes_updated": 0,
            "success": False,
            "error": "temp_file_not_found"
        }
    
    try:
        # Load genomes from temp.json
        with open(temp_path_obj, 'r', encoding='utf-8') as f:
            genomes = json.load(f)
        
        if not genomes:
            logger.warning("No genomes found in temp.json")
            return {
                "species_count": 0,
                "limbo_size": 0,
                "speciation_events": 0,
                "merge_events": 0,
                "extinction_events": 0,
                "migration_events": 0,
                "genomes_updated": 0,
                "success": False,
                "error": "no_genomes"
            }
        
        logger.debug("Loaded %d genomes for speciation", len(genomes))
        
        # Get speciation module
        speciation_module = get_speciation_module(config=config, logger=logger)
        
        # Run speciation (embeddings will be computed and saved to temp.json)
        # leader_follower_clustering reads directly from temp.json
        species, limbo = speciation_module.process_generation(
            population=genomes,
            current_generation=current_generation,
            mutate_fn=mutate_fn,
            temp_path=temp_path
        )
        
        # Reload genomes with embeddings (they were updated by compute_and_save_embeddings)
        with open(temp_path_obj, 'r', encoding='utf-8') as f:
            genomes = json.load(f)
        
        # Update genomes with species IDs
        updated_genomes = speciation_module.update_genomes_with_species(genomes)
        
        # Save updated genomes back to temp.json
        with open(temp_path_obj, 'w', encoding='utf-8') as f:
            json.dump(updated_genomes, f, indent=2, ensure_ascii=False)
        
        # Get event counts from module
        events = speciation_module._current_gen_events
        
        result = {
            "species_count": len(species),
            "limbo_size": limbo.size,
            "speciation_events": events.get("speciation", 0),
            "merge_events": events.get("merge", 0),
            "extinction_events": events.get("extinction", 0),
            "migration_events": events.get("migration", 0),
            "genomes_updated": len(updated_genomes),
            "success": True
        }
        
        logger.info(
            "Speciation completed: %d species, %d in limbo, "
            "events: speciation=%d, merge=%d, extinction=%d, migration=%d",
            result["species_count"], result["limbo_size"],
            result["speciation_events"], result["merge_events"],
            result["extinction_events"], result["migration_events"]
        )
        
        return result
        
    except Exception as e:
        logger.error("Speciation failed: %s", e, exc_info=True)
        return {
            "species_count": 0,
            "limbo_size": 0,
            "speciation_events": 0,
            "merge_events": 0,
            "extinction_events": 0,
            "migration_events": 0,
            "genomes_updated": 0,
            "success": False,
            "error": str(e)
        }


def get_speciation_statistics(log_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Get current speciation statistics from the module.
    
    Args:
        log_file: Optional log file path
        
    Returns:
        Dict with current speciation state and metrics
    """
    logger = get_logger("RunSpeciation", log_file)
    
    global _speciation_module
    if _speciation_module is None:
        return {
            "initialized": False,
            "species_count": 0,
            "limbo_size": 0
        }
    
    metrics_summary = _speciation_module.metrics_tracker.get_summary()
    
    return {
        "initialized": True,
        "species_count": len(_speciation_module.species),
        "limbo_size": _speciation_module.limbo.size,
        "global_best_fitness": _speciation_module.global_best.fitness if _speciation_module.global_best else None,
        "metrics_summary": metrics_summary
    }


def save_speciation_state(output_path: Optional[str] = None, log_file: Optional[str] = None) -> bool:
    """
    Save current speciation state to file.
    
    Args:
        output_path: Optional path to save state (defaults to outputs_path / "speciation_state.json")
        log_file: Optional log file path
        
    Returns:
        True if saved successfully, False otherwise
    """
    logger = get_logger("RunSpeciation", log_file)
    
    global _speciation_module
    if _speciation_module is None:
        logger.warning("No speciation module to save")
        return False
    
    try:
        if output_path is None:
            outputs_path = get_outputs_path()
            output_path = str(outputs_path / "speciation_state.json")
        
        _speciation_module.save_state(output_path)
        logger.info("Speciation state saved to %s", output_path)
        return True
    except Exception as e:
        logger.error("Failed to save speciation state: %s", e)
        return False
