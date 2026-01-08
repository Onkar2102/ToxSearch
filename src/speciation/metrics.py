"""
metrics.py

Validation metrics for speciation.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .island import Individual, Species
from .distance import semantic_distance
from .adaptive_threshold import compute_all_silhouette_scores

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


@dataclass
class GenerationMetrics:
    """Metrics for a single generation."""
    generation: int
    species_count: int
    total_population: int
    limbo_size: int
    best_fitness: float
    avg_fitness: float
    fitness_std: float
    avg_silhouette: float
    min_silhouette: float
    max_silhouette: float
    mode_counts: Dict[str, int] = field(default_factory=dict)
    speciation_events: int = 0
    merge_events: int = 0
    extinction_events: int = 0
    migration_events: int = 0
    inter_species_diversity: float = 0.0
    intra_species_diversity: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "generation": self.generation, "species_count": self.species_count,
            "total_population": self.total_population, "limbo_size": self.limbo_size,
            "best_fitness": self.best_fitness, "avg_fitness": self.avg_fitness,
            "avg_silhouette": self.avg_silhouette, "mode_counts": self.mode_counts,
            "speciation_events": self.speciation_events, "merge_events": self.merge_events,
            "extinction_events": self.extinction_events, "migration_events": self.migration_events
        }


class SpeciationMetricsTracker:
    """Tracks metrics over evolution."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger("SpeciationMetrics")
        self.history: List[GenerationMetrics] = []
        self.total_speciation = 0
        self.total_merges = 0
        self.total_extinctions = 0
        self.total_migrations = 0
    
    def record_generation(self, generation: int, species: Dict[int, Species], limbo_size: int = 0,
                          speciation_events: int = 0, merge_events: int = 0,
                          extinction_events: int = 0, migration_events: int = 0) -> GenerationMetrics:
        """Record metrics for a generation."""
        species_count = len(species)
        total_pop = sum(sp.size for sp in species.values())
        
        all_fitness = [m.fitness for sp in species.values() for m in sp.members]
        best = max(all_fitness) if all_fitness else 0.0
        avg = np.mean(all_fitness) if all_fitness else 0.0
        std = np.std(all_fitness) if all_fitness else 0.0
        
        silhouettes = compute_all_silhouette_scores(species)
        sil_vals = list(silhouettes.values()) if silhouettes else [0.0]
        
        mode_counts = {"DEFAULT": 0, "EXPLORE": 0, "EXPLOIT": 0}
        for sp in species.values():
            mode_counts[sp.mode.value] += 1
        
        inter_div, intra_div = compute_diversity_metrics(species)
        
        self.total_speciation += speciation_events
        self.total_merges += merge_events
        self.total_extinctions += extinction_events
        self.total_migrations += migration_events
        
        metrics = GenerationMetrics(
            generation=generation, species_count=species_count, total_population=total_pop,
            limbo_size=limbo_size, best_fitness=float(best), avg_fitness=float(avg),
            fitness_std=float(std), avg_silhouette=float(np.mean(sil_vals)),
            min_silhouette=float(np.min(sil_vals)), max_silhouette=float(np.max(sil_vals)),
            mode_counts=mode_counts, speciation_events=speciation_events,
            merge_events=merge_events, extinction_events=extinction_events,
            migration_events=migration_events, inter_species_diversity=float(inter_div),
            intra_species_diversity=float(intra_div)
        )
        
        self.history.append(metrics)
        return metrics
    
    def get_summary(self) -> Dict:
        if not self.history:
            return {}
        return {
            "total_generations": len(self.history),
            "final_species_count": self.history[-1].species_count,
            "best_fitness_ever": max(m.best_fitness for m in self.history),
            "total_speciation_events": self.total_speciation,
            "total_merge_events": self.total_merges,
            "total_extinction_events": self.total_extinctions,
            "avg_silhouette_overall": np.mean([m.avg_silhouette for m in self.history])
        }
    
    def to_dict(self) -> Dict:
        return {"history": [m.to_dict() for m in self.history], "summary": self.get_summary()}


def compute_diversity_metrics(species: Dict[int, Species]) -> tuple:
    """Compute inter/intra species diversity."""
    if not species:
        return 0.0, 0.0
    
    species_list = list(species.values())
    
    # Inter-species
    inter = []
    for i, sp1 in enumerate(species_list):
        for sp2 in species_list[i + 1:]:
            if sp1.leader.embedding is not None and sp2.leader.embedding is not None:
                inter.append(semantic_distance(sp1.leader.embedding, sp2.leader.embedding))
    inter_div = np.mean(inter) if inter else 0.0
    
    # Intra-species
    intra_divs = []
    for sp in species_list:
        members = [m for m in sp.members if m.embedding is not None]
        if len(members) < 2:
            continue
        dists = [semantic_distance(members[i].embedding, members[j].embedding)
                 for i in range(len(members)) for j in range(i + 1, len(members))]
        if dists:
            intra_divs.append(np.mean(dists))
    intra_div = np.mean(intra_divs) if intra_divs else 0.0
    
    return inter_div, intra_div


def compute_solution_diversity(species: Dict[int, Species], fitness_threshold: float = 0.5) -> int:
    """Count distinct high-fitness strategies."""
    high_fitness = [m for sp in species.values() for m in sp.members
                    if m.fitness >= fitness_threshold and m.embedding is not None]
    if not high_fitness:
        return 0
    
    strategies = []
    threshold = 0.3
    for ind in sorted(high_fitness, key=lambda x: x.fitness, reverse=True):
        is_distinct = all(semantic_distance(ind.embedding, s.embedding) >= threshold for s in strategies)
        if is_distinct:
            strategies.append(ind)
    return len(strategies)


def get_species_statistics(species: Dict[int, Species]) -> Dict:
    """Get detailed species statistics."""
    if not species:
        return {"count": 0, "total_population": 0}
    
    sizes = [sp.size for sp in species.values()]
    mode_counts = {"DEFAULT": 0, "EXPLORE": 0, "EXPLOIT": 0}
    for sp in species.values():
        mode_counts[sp.mode.value] += 1
    
    return {
        "count": len(species), "sizes": sizes, "avg_size": np.mean(sizes),
        "total_population": sum(sizes),
        "fitness": {
            "global_best": max(sp.best_fitness for sp in species.values()),
            "global_avg": np.mean([sp.avg_fitness for sp in species.values()])
        },
        "modes": mode_counts
    }


def log_generation_summary(generation: int, species: Dict[int, Species], limbo_size: int = 0,
                           events: Dict[str, int] = None, logger=None) -> None:
    """Log generation summary."""
    if logger is None:
        logger = get_logger("SpeciationMetrics")
    
    stats = get_species_statistics(species)
    events = events or {}
    event_str = ", ".join(f"{k}={v}" for k, v in events.items() if v > 0)
    
    logger.info(f"Gen {generation}: {stats['count']} species, {stats['total_population']} pop, "
                f"limbo={limbo_size}, best={stats['fitness']['global_best']:.4f}, "
                f"avg={stats['fitness']['global_avg']:.4f}" + (f", events: {event_str}" if event_str else ""))

