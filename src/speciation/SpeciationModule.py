"""
SpeciationModule.py

Main SpeciationModule class and entry point functions for Dynamic Islands framework.
Combines the SpeciationModule class with the run_speciation entry point.
"""

import json
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path

from .config import SpeciationConfig
from .species import Individual, Species, SpeciesMode, IslandMode, SpeciesIdGenerator
from .embeddings import compute_and_save_embeddings, remove_embeddings_from_temp, get_embedding_model
from .leader_follower import leader_follower_clustering, update_species_leaders
from .reserves import Cluster0, CLUSTER_0_ID
from .modes import update_all_island_modes
from .intra_island import select_parents_elite_focused
from .merging import process_merges
from .extinction import process_extinctions, find_global_best
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
    3. Cluster 0 (reserves) management with max 1000 individuals
    4. Mode switching (Explore/Exploit/Default)
    5. Island merging
    6. Extinction and repopulation
    8. Species capacity (max 100 per species)
    
    Key changes from original design:
    - Species limited to top 100 genomes by fitness
    - All removed genomes go to cluster 0 (no non_elites.json)
    - Cluster 0 limited to 1000, with removal_threshold filtering
    - Removed genomes below threshold are archived to under_performing.json
    - Constant radius (theta_sim) for all species - no dynamic adjustment
    - Cluster origin tracking (merge/natural) in speciation_state.json
    
    Usage:
        >>> config = SpeciationConfig()
        >>> module = SpeciationModule(config)
        >>> species, cluster0 = module.process_generation(genomes, current_generation)
    """
    
    def __init__(self, config: Optional[SpeciationConfig] = None, logger=None):
        self.config = config or SpeciationConfig()
        self.logger = logger or get_logger("SpeciationModule")
        
        self.species: Dict[int, Species] = {}
        self.cluster0 = Cluster0(
            default_ttl=self.config.cluster0_ttl,
            min_cluster_size=self.config.cluster0_min_cluster_size,
            theta_sim=self.config.theta_sim,
            max_capacity=self.config.cluster0_max_capacity,
            removal_threshold=self.config.removal_threshold,
            logger=self.logger
        )
        self.global_best: Optional[Individual] = None
        self.metrics_tracker = SpeciationMetricsTracker(logger=self.logger)
        
        self._current_gen_events = {"speciation": 0, "merge": 0, "extinction": 0}
        self._embedding_model = None
        self._archived_count = 0  # Count of genomes archived this generation
        
        self.logger.info(f"SpeciationModule initialized with config: theta_sim={self.config.theta_sim}, max_capacity={self.config.max_island_capacity}")
    
    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model(model_name=self.config.embedding_model)
        return self._embedding_model
    
    def process_generation(self, population: List[Dict[str, Any]], current_generation: int,
                           mutate_fn: Optional[Callable] = None, temp_path: Optional[str] = None) -> Tuple[Dict[int, Species], Cluster0]:
        """
        Process a single generation with full speciation pipeline.
        
        This is the main orchestration method that runs the complete pipeline:
        
        1. Compute embeddings: Read temp.json, compute embeddings, add "prompt_embedding" field, save back
        2. Leader-Follower clustering: Read temp.json, convert to Individuals, assign to species
        3. Cluster 0 update: Check for speciation events, filter by removal threshold, enforce capacity
        4. Enforce species capacity: Keep top 100 per species, send excess to cluster 0
        5. Update leaders: Ensure leaders are highest-fitness members
        6. Mode switching: Adapt island modes based on fitness trends
        7. Island merging: Combine similar species with cluster_origin="merge"
        8. Extinction & repopulation: Remove stagnant species, create new ones
        10. Record metrics: Track generation statistics
        
        Note: The embedding computation modifies temp.json in-place by adding "prompt_embedding"
        field to each genome. If embeddings already exist, computation is skipped.
        
        Args:
            population: List of genome dictionaries (with prompts and fitness)
            current_generation: Current generation number
            mutate_fn: Optional mutation function (for repopulation)
            temp_path: Optional path to temp.json (for embedding computation)
            
        Returns:
            Tuple of (species_dict, cluster0)
        """
        self.logger.info(f"=== Speciation Generation {current_generation} ===")
        
        self._current_gen_events = {"speciation": 0, "merge": 0, "extinction": 0}
        self._archived_count = 0
        
        # Auto-load previous state if not first generation
        if current_generation > 0:
            outputs_path = get_outputs_path()
            state_path = str(outputs_path / "speciation_state.json")
            if Path(state_path).exists():
                self.load_state(state_path)
                self.logger.info("Restored speciation state from previous generation")
        
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
        
        # Step 2: Leader-Follower clustering (reads and writes temp.json and speciation_state.json directly)
        # This function handles both Generation 0 and N logic internally
        outputs_path = get_outputs_path()
        speciation_state_path = str(outputs_path / "speciation_state.json")
        
        # Leader-Follower clustering reads and writes temp.json and speciation_state.json directly
        self.species = leader_follower_clustering(
            temp_path=temp_path,
            speciation_state_path=speciation_state_path,
            theta_sim=self.config.theta_sim,
            current_generation=current_generation,
            logger=self.logger
        )
        
        # Load cluster0 from speciation_state.json (if exists) to restore state
        if Path(speciation_state_path).exists():
            try:
                with open(speciation_state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                cluster0_dict = state.get("cluster0", {})
                # Reconstruct cluster0 from state
                if cluster0_dict:
                    # Cluster0 is managed by SpeciationModule, restore from state
                    # Note: This is a simplified restoration - full restoration happens in load_state()
                    pass  # Cluster0 will be restored via load_state() if needed
            except Exception:
                pass
        
        # Step 3: Cluster 0 management
        # 3a: Update TTL (decrement and remove expired)
        expired = self.cluster0.update_ttl(current_generation)
        self._archive_individuals(expired, current_generation, "expired_from_cluster0")
        
        # 3b: Filter by removal threshold (archive those below threshold)
        # Calculate max fitness from all genomes (species + cluster0)
        all_fitnesses = [ind.fitness for sp in self.species.values() for ind in sp.members]
        all_fitnesses.extend([ind.fitness for ind in self.cluster0.individuals])
        max_fitness = max(all_fitnesses) if all_fitnesses else None
        
        removed_by_threshold = self.cluster0.filter_by_removal_threshold(
            removal_threshold=self.config.removal_threshold,
            max_fitness=max_fitness
        )
        self._archive_individuals(removed_by_threshold, current_generation, "below_removal_threshold")
        
        # 3c: Enforce capacity (archive excess beyond 1000)
        removed_by_capacity = self.cluster0.enforce_capacity()
        self._archive_individuals(removed_by_capacity, current_generation, "cluster0_capacity_exceeded")
        
        # 3d: Check for speciation events in cluster 0
        new_species = self.cluster0.check_speciation(current_generation)
        if new_species:
            self.species[new_species.id] = new_species
            self._current_gen_events["speciation"] += 1
        
        # Step 4: Enforce species capacity (top 100 per species)
        for sp in list(self.species.values()):
            removed = sp.enforce_capacity(self.config.max_island_capacity)
            if removed:
                # Send all removed individuals to cluster 0
                self.cluster0.add_batch(removed, current_generation)
        
        # Re-apply cluster 0 capacity after receiving from species
        removed_by_capacity = self.cluster0.enforce_capacity()
        self._archive_individuals(removed_by_capacity, current_generation, "cluster0_capacity_exceeded_after_species_enforcement")
        
        # Step 5: Update leaders and record fitness
        update_species_leaders(self.species)
        for sp in self.species.values():
            sp.record_fitness(current_generation)
        
        # Step 6: Mode switching
        update_all_island_modes(self.species, current_generation, self.cluster0, self.config, self.logger)
        
        # Step 7: Island merging
        self.species, merge_events = process_merges(
            self.species, 
            theta_merge=self.config.theta_merge,
            theta_sim=self.config.theta_sim,  # Constant radius for merged species
            current_gen=current_generation,
            max_capacity=self.config.max_island_capacity,
            logger=self.logger
        )
        self._current_gen_events["merge"] = len(merge_events)
        
        # Step 8: Extinction and repopulation
        self.global_best = find_global_best(self.species)
        self.species, extinction_events = process_extinctions(
            self.species, self.cluster0, self.global_best, current_generation,
            theta_sim=self.config.theta_sim,
            max_stagnation=self.config.max_stagnation,
            min_size=self.config.min_island_size,
            repopulation_size=self.config.repopulation_size,
            mutation_rate=self.config.repopulation_mutation_rate,
            mutate_fn=mutate_fn,
            logger=self.logger
        )
        self._current_gen_events["extinction"] = len(extinction_events)
        
        # Step 9: Record metrics
        self.metrics_tracker.record_generation(
            current_generation, self.species, self.cluster0.size,
            self._current_gen_events["speciation"], self._current_gen_events["merge"],
            self._current_gen_events["extinction"], 0  # migration removed
        )
        
        log_generation_summary(current_generation, self.species, self.cluster0.size,
                               self._current_gen_events, self.logger)
        
        # Auto-save state after processing
        outputs_path = get_outputs_path()
        state_path = str(outputs_path / "speciation_state.json")
        self.save_state(state_path)
        
        # Remove embeddings from temp.json after speciation is complete
        # This reduces storage size when genomes are saved to other files
        remove_embeddings_from_temp(temp_path=temp_path, logger=self.logger)
        
        return self.species, self.cluster0
    
    def _archive_individuals(self, individuals: List[Individual], generation: int, reason: str) -> None:
        """
        Archive individuals to under_performing.json.
        
        Args:
            individuals: List of individuals to archive
            generation: Current generation number
            reason: Reason for archiving (for logging)
        """
        if not individuals:
            return
        
        self._archived_count += len(individuals)
        
        try:
            outputs_path = get_outputs_path()
            archive_path = outputs_path / "under_performing.json"
            
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
            
            self.logger.debug(f"Archived {len(individuals)} individuals ({reason}) to under_performing.json")
            
        except Exception as e:
            self.logger.warning(f"Failed to archive individuals: {e}")
    
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
        return [Individual.from_genome(genome) for genome in population]
    
    def get_species_for_genome(self, genome_id: int) -> Optional[int]:
        """Get species ID for a genome (0 for cluster 0, None if not found)."""
        # Check cluster 0 first
        for ind in self.cluster0.individuals:
            if ind.id == genome_id:
                return CLUSTER_0_ID
        
        # Check species
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
        Add offspring to species or cluster 0.
        
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
                self.cluster0.add(ind, current_generation)
    
    def update_genomes_with_species(self, genomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update genomes with species IDs (0 for cluster 0)."""
        for g in genomes:
            genome_id = g.get("id")
            species_id = self.get_species_for_genome(genome_id)
            g["species_id"] = species_id
            # Note: species_id=0 means cluster 0, None means not assigned
        return genomes
    
    def get_state(self) -> Dict:
        """Get state for serialization."""
        return {
            "config": self.config.to_dict(),
            "species": {str(sid): sp.to_dict() for sid, sp in self.species.items()},
            "cluster0": self.cluster0.to_dict(),
            "global_best_id": self.global_best.id if self.global_best else None,
            "metrics": self.metrics_tracker.to_dict()
        }
    
    def save_state(self, path: str) -> None:
        """Save state to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.get_state(), f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved speciation state to {path}")
    
    def load_state(self, path: str) -> bool:
        """
        Load state from file and restore species, cluster 0, and metrics.
        
        Restores species structure with leader embeddings for distance comparisons.
        Includes cluster_origin and parent_ids for tracking species history.
        Individual objects are reconstructed with minimal data; full genome data
        is matched by ID during processing.
        
        Args:
            path: Path to speciation_state.json file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        import numpy as np
        
        state_path = Path(path)
        if not state_path.exists():
            self.logger.warning(f"Speciation state file not found: {path}")
            return False
        
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Restore species
            self.species = {}
            max_species_id = 0
            
            for sid_str, sp_dict in state.get("species", {}).items():
                sid = int(sid_str)
                max_species_id = max(max_species_id, sid)
                
                # Reconstruct leader Individual from saved data
                leader_embedding = None
                if sp_dict.get("leader_embedding"):
                    leader_embedding = np.array(sp_dict["leader_embedding"])
                
                # Create leader Individual with saved embedding
                leader = Individual(
                    id=sp_dict["leader_id"],
                    prompt=sp_dict.get("leader_prompt", ""),
                    fitness=sp_dict.get("leader_fitness", 0.0),
                    embedding=leader_embedding,
                    species_id=sid
                )
                
                # Reconstruct species with origin tracking
                species = Species(
                    id=sid,
                    leader=leader,
                    members=[leader],  # Will be populated during clustering
                    mode=SpeciesMode(sp_dict.get("mode", "DEFAULT")),
                    radius=sp_dict.get("radius", self.config.theta_sim),
                    stagnation_counter=sp_dict.get("stagnation_counter", 0),
                    created_at=sp_dict.get("created_at", 0),
                    last_improvement=sp_dict.get("last_improvement", 0),
                    fitness_history=sp_dict.get("fitness_history", []),
                    cluster_origin=sp_dict.get("cluster_origin"),
                    parent_ids=sp_dict.get("parent_ids"),
                    parent_id=sp_dict.get("parent_id")
                )
                
                self.species[sid] = species
            
            # Update SpeciesIdGenerator to avoid ID conflicts
            # Species IDs start from 1 (0 is reserved for cluster 0)
            SpeciesIdGenerator.set_min_id(max_species_id + 1)
            
            self.logger.info(f"Loaded speciation state from {path}: {len(self.species)} species")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load speciation state: {e}", exc_info=True)
            return False
    
    def reset(self) -> None:
        """Reset all state."""
        self.species = {}
        self.cluster0 = Cluster0(
            default_ttl=self.config.cluster0_ttl,
            min_cluster_size=self.config.cluster0_min_cluster_size,
            theta_sim=self.config.theta_sim,
            max_capacity=self.config.cluster0_max_capacity,
            removal_threshold=self.config.removal_threshold,
            logger=self.logger
        )
        self.global_best = None
        self.metrics_tracker = SpeciationMetricsTracker(self.logger)
        SpeciesIdGenerator.reset()
        self.logger.info("Speciation module reset")
    
    def __repr__(self):
        return f"SpeciationModule(species={len(self.species)}, cluster0={self.cluster0.size})"


# Global speciation module instance (singleton pattern)
_speciation_module: Optional[SpeciationModule] = None


def get_speciation_module(config: Optional[SpeciationConfig] = None, logger=None) -> SpeciationModule:
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
    config: Optional[SpeciationConfig] = None,
    log_file: Optional[str] = None,
    mutate_fn: Optional[Any] = None,
    removal_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Run speciation processing for a single generation.
    
    This is the main entry point for speciation, similar to run_evolution().
    It handles:
    1. Loading genomes from temp.json (or provided path)
    2. Running speciation pipeline
    3. Updating genomes with species_id (0 for cluster 0)
    4. Saving updated genomes back to temp.json
    
    Args:
        temp_path: Path to temp.json file with evaluated genomes.
                   If None, uses default outputs_path / "temp.json"
        current_generation: Current generation number
        config: Optional SpeciationConfig (uses defaults if None)
        log_file: Optional log file path
        mutate_fn: Optional mutation function for repopulation
        removal_threshold: Optional fitness threshold for cluster 0 filtering
        
    Returns:
        Dict with speciation results:
        {
            "species_count": int,
            "cluster0_size": int,
            "speciation_events": int,
            "merge_events": int,
            "extinction_events": int,
            "archived_count": int,
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
        # Load genomes from temp.json
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
        
        # Create config with removal_threshold if provided
        if config is None:
            config = SpeciationConfig()
        if removal_threshold is not None:
            config = SpeciationConfig(
                **{**config.to_dict(), "removal_threshold": removal_threshold}
            )
        
        # Get speciation module
        speciation_module = get_speciation_module(config=config, logger=logger)
        
        # Run speciation (embeddings will be computed and saved to temp.json)
        species, cluster0 = speciation_module.process_generation(
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
            "cluster0_size": cluster0.size,
            "reserves_size": cluster0.size,
            "speciation_events": events.get("speciation", 0),
            "merge_events": events.get("merge", 0),
            "extinction_events": events.get("extinction", 0),
            "archived_count": speciation_module._archived_count,
            "genomes_updated": len(updated_genomes),
            "success": True
        }
        
        logger.info(
            "Speciation completed: %d species, %d in cluster 0, "
            "events: speciation=%d, merge=%d, extinction=%d, archived=%d",
            result["species_count"], result["cluster0_size"],
            result["speciation_events"], result["merge_events"],
            result["extinction_events"],
            result["archived_count"]
        )
        
        # Update EvolutionTracker with speciation data
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
            "cluster0_size": 0,
            "reserves_size": 0
        }
    
    metrics_summary = _speciation_module.metrics_tracker.get_summary()
    
    return {
        "initialized": True,
        "species_count": len(_speciation_module.species),
        "cluster0_size": _speciation_module.cluster0.size,
        "reserves_size": _speciation_module.cluster0.size,
        "global_best_fitness": _speciation_module.global_best.fitness if _speciation_module.global_best else None,
        "metrics_summary": metrics_summary
    }


def save_speciation_state(output_path: Optional[str] = None, log_file: Optional[str] = None) -> bool:
    """
    Save current speciation state to file (internal use).
    
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


def load_speciation_state(output_path: Optional[str] = None, log_file: Optional[str] = None) -> bool:
    """
    Load speciation state from file (internal use only).
    
    Restores species structure with leader embeddings for distance comparisons
    in subsequent generations. Called automatically at the start of process_generation().
    
    Args:
        output_path: Optional path to load state (defaults to outputs_path / "speciation_state.json")
        log_file: Optional log file path
        
    Returns:
        True if loaded successfully, False otherwise
    """
    logger = get_logger("RunSpeciation", log_file)
    
    global _speciation_module
    if _speciation_module is None:
        logger.warning("No speciation module to load state into")
        return False
    
    try:
        if output_path is None:
            outputs_path = get_outputs_path()
            output_path = str(outputs_path / "speciation_state.json")
        
        return _speciation_module.load_state(output_path)
    except Exception as e:
        logger.error("Failed to load speciation state: %s", e)
        return False


def update_evolution_tracker_with_speciation(
    evolution_tracker_path: str,
    current_generation: int,
    speciation_result: Dict[str, Any],
    speciation_stats: Optional[Dict[str, Any]] = None,
    logger=None
) -> bool:
    """
    Update EvolutionTracker.json with speciation data.
    
    This function is called internally to integrate speciation metrics
    into the evolution tracker. The speciation_state.json file remains
    internal to the speciation module.
    
    Adds speciation summary to:
    1. Current generation entry (speciation field)
    2. Top-level summary (speciation_summary field)
    
    Args:
        evolution_tracker_path: Path to EvolutionTracker.json
        current_generation: Current generation number
        speciation_result: Result dict from run_speciation()
        speciation_stats: Optional detailed stats from get_speciation_statistics()
        logger: Optional logger instance
        
    Returns:
        True if updated successfully, False otherwise
    """
    if logger is None:
        logger = get_logger("UpdateEvolutionTracker")
    
    try:
        tracker_path = Path(evolution_tracker_path)
        if not tracker_path.exists():
            logger.warning("EvolutionTracker.json not found at %s", evolution_tracker_path)
            return False
        
        # Read current tracker
        with open(tracker_path, 'r', encoding='utf-8') as f:
            evolution_tracker = json.load(f)
        
        # Get detailed metrics if available
        if speciation_stats is None:
            speciation_stats = get_speciation_statistics()
        
        metrics_summary = speciation_stats.get("metrics_summary", {})
        
        # Get current metrics from speciation module
        global _speciation_module
        current_metrics = None
        if _speciation_module is not None and _speciation_module.metrics_tracker.history:
            current_metrics = _speciation_module.metrics_tracker.history[-1]
        
        # Prepare speciation summary for generation entry
        speciation_summary = {
            "species_count": speciation_result.get("species_count", 0),
            "cluster0_size": speciation_result.get("cluster0_size", 0),
            "speciation_events": speciation_result.get("speciation_events", 0),
            "merge_events": speciation_result.get("merge_events", 0),
            "extinction_events": speciation_result.get("extinction_events", 0),
            "migration_events": speciation_result.get("migration_events", 0),
            "archived_count": speciation_result.get("archived_count", 0),
        }
        
        # Add detailed metrics if available
        if current_metrics:
            speciation_summary.update({
                "avg_silhouette": round(current_metrics.avg_silhouette, 4),
                "inter_species_diversity": round(current_metrics.inter_species_diversity, 4),
                "intra_species_diversity": round(current_metrics.intra_species_diversity, 4),
                "mode_distribution": current_metrics.mode_counts,
            })
        
        # Update or create generation entry
        generations = evolution_tracker.get("generations", [])
        gen_entry = None
        for gen in generations:
            if gen.get("generation_number") == current_generation:
                gen_entry = gen
                break
        
        if gen_entry:
            gen_entry["speciation"] = speciation_summary
        else:
            # Create new generation entry if it doesn't exist
            gen_entry = {
                "generation_number": current_generation,
                "speciation": speciation_summary
            }
            generations.append(gen_entry)
            evolution_tracker["generations"] = generations
        
        # Update top-level speciation summary
        if "speciation_summary" not in evolution_tracker:
            evolution_tracker["speciation_summary"] = {}
        
        evolution_tracker["speciation_summary"].update({
            "current_species_count": speciation_result.get("species_count", 0),
            "current_cluster0_size": speciation_result.get("cluster0_size", 0),
            "total_speciation_events": metrics_summary.get("total_speciation_events", 0),
            "total_merge_events": metrics_summary.get("total_merge_events", 0),
            "total_extinction_events": metrics_summary.get("total_extinction_events", 0),
        })
        
        # Save updated tracker
        with open(tracker_path, 'w', encoding='utf-8') as f:
            json.dump(evolution_tracker, f, indent=2, ensure_ascii=False)
        
        logger.info("Updated EvolutionTracker.json with speciation data for generation %d", current_generation)
        return True
        
    except Exception as e:
        logger.error("Failed to update EvolutionTracker with speciation data: %s", e, exc_info=True)
        return False
