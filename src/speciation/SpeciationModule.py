"""
SpeciationModule.py

Main SpeciationModule class that orchestrates the Plan A+ Dynamic Islands framework.
"""

import json
from typing import Dict, List, Optional, Tuple, Callable, Any

from .config import PlanAPlusConfig
from .island import Individual, Species, IslandMode, SpeciesIdGenerator
from .embeddings import compute_embedding, compute_embeddings_batch, get_embedding_model
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
get_logger, _, _, _ = get_custom_logging()


class SpeciationModule:
    """
    Main orchestrator for Plan A+ Dynamic Islands speciation.
    
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
                           mutate_fn: Optional[Callable] = None) -> Tuple[Dict[int, Species], LimboBuffer]:
        """Process a single generation with full speciation pipeline."""
        self.logger.info(f"=== Speciation Generation {current_generation} ===")
        
        self._current_gen_events = {"speciation": 0, "merge": 0, "extinction": 0, "migration": 0}
        
        # Step 1: Convert to Individuals with embeddings
        individuals = self._prepare_individuals(population)
        
        # Step 2: Leader-Follower clustering
        self.species, limbo_candidates = leader_follower_clustering(
            population=individuals, theta_sim=self.config.theta_sim,
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
        """Convert genome dicts to Individuals with embeddings."""
        prompts = [g.get("prompt", "") for g in population]
        embeddings = compute_embeddings_batch(prompts, self.embedding_model, self.config.embedding_batch_size)
        return [Individual.from_genome(genome, embeddings[i]) for i, genome in enumerate(population)]
    
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
    
    def add_offspring(self, offspring: List[Dict[str, Any]], current_generation: int) -> None:
        """Add offspring to species or limbo."""
        from .leader_follower import find_nearest_leader
        
        individuals = self._prepare_individuals(offspring)
        for ind in individuals:
            nearest_id, min_dist = find_nearest_leader(ind.embedding, self.species)
            if nearest_id and min_dist < self.config.theta_sim:
                self.species[nearest_id].add_member(ind)
            elif ind.fitness > self.config.viability_baseline:
                self.limbo.add(ind, current_generation)
    
    def update_genomes_with_species(self, genomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update genomes with species IDs."""
        for g in genomes:
            g["species_id"] = self.get_species_for_genome(g.get("id"))
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
