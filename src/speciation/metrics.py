"""
metrics.py

Validation metrics for speciation.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .species import Individual, Species
from .distance import ensemble_distance

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


@dataclass
class GenerationMetrics:
    """Metrics for a single generation."""
    generation: int
    species_count: int
    total_population: int
    reserves_size: int
    best_fitness: float
    avg_fitness: float
    fitness_std: float
    speciation_events: int = 0
    merge_events: int = 0
    extinction_events: int = 0
    inter_species_diversity: float = 0.0
    intra_species_diversity: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "generation": self.generation, "species_count": self.species_count,
            "total_population": self.total_population, "reserves_size": self.reserves_size,
            "best_fitness": self.best_fitness, "avg_fitness": self.avg_fitness,
            "speciation_events": self.speciation_events,
            "merge_events": self.merge_events, "extinction_events": self.extinction_events
        }


class SpeciationMetricsTracker:
    """Tracks metrics over evolution."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger("SpeciationMetrics")
        self.history: List[GenerationMetrics] = []
        self.total_speciation = 0
        self.total_merges = 0
        self.total_extinctions = 0
    
    def record_generation(self, generation: int, species: Dict[int, Species], reserves_size: int = 0,
                          speciation_events: int = 0, merge_events: int = 0,
                          extinction_events: int = 0, cluster0=None) -> GenerationMetrics:
        """Record metrics for a generation."""
        species_count = len(species)
        total_pop = sum(sp.size for sp in species.values())
        
        # Calculate fitness from species members
        all_fitness = [m.fitness for sp in species.values() for m in sp.members]
        
        # Include reserves (cluster0) fitness if provided
        if cluster0 is not None and hasattr(cluster0, 'individuals'):
            all_fitness.extend([ind.fitness for ind in cluster0.individuals])
            total_pop += len(cluster0.individuals)
        
        best = max(all_fitness) if all_fitness else 0.0
        avg = np.mean(all_fitness) if all_fitness else 0.0
        std = np.std(all_fitness) if all_fitness else 0.0
        
        inter_div, intra_div = compute_diversity_metrics(species, w_genotype=0.7, w_phenotype=0.3)
        
        self.total_speciation += speciation_events
        self.total_merges += merge_events
        self.total_extinctions += extinction_events
        
        metrics = GenerationMetrics(
            generation=generation, species_count=species_count, total_population=total_pop,
            reserves_size=reserves_size, best_fitness=float(best), avg_fitness=float(avg),
            fitness_std=float(std), speciation_events=speciation_events,
            merge_events=merge_events, extinction_events=extinction_events,
            inter_species_diversity=float(inter_div), intra_species_diversity=float(intra_div)
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
            "total_extinction_events": self.total_extinctions
        }
    
    def to_dict(self) -> Dict:
        return {"history": [m.to_dict() for m in self.history], "summary": self.get_summary()}


def compute_diversity_metrics(species: Dict[int, Species], w_genotype: float = 0.7, w_phenotype: float = 0.3) -> tuple:
    """
    Compute inter-species and intra-species diversity metrics.
    
    Diversity metrics measure:
    - Inter-species diversity: Average distance between species leaders
      (high = species are diverse, low = species are similar/redundant)
    - Intra-species diversity: Average distance within each species
      (high = members are diverse, low = members are homogeneous)
    
    Args:
        species: Dict of all current species
    
    Returns:
        Tuple of (inter_species_diversity, intra_species_diversity)
    """
    if not species:
        return 0.0, 0.0
    
    species_list = list(species.values())
    
    # Compute inter-species diversity (distance between leaders)
    inter = []
    for i, sp1 in enumerate(species_list):
        for sp2 in species_list[i + 1:]:
            if sp1.leader.embedding is not None and sp2.leader.embedding is not None:
                inter.append(ensemble_distance(
                    sp1.leader.embedding, sp2.leader.embedding,
                    sp1.leader.phenotype, sp2.leader.phenotype,
                    w_genotype, w_phenotype
                ))
    inter_div = np.mean(inter) if inter else 0.0
    
    # Compute intra-species diversity (distances within each species)
    intra_divs = []
    for sp in species_list:
        members = [m for m in sp.members if m.embedding is not None]
        if len(members) < 2:
            continue
        dists = [ensemble_distance(
            members[i].embedding, members[j].embedding,
            members[i].phenotype, members[j].phenotype,
            w_genotype, w_phenotype
        ) for i in range(len(members)) for j in range(i + 1, len(members))]
        if dists:
            intra_divs.append(np.mean(dists))
    intra_div = np.mean(intra_divs) if intra_divs else 0.0
    
    return inter_div, intra_div


def get_species_statistics(species: Dict[int, Species]) -> Dict:
    """Get detailed species statistics for a generation."""
    if not species:
        return {
            "count": 0, 
            "total_population": 0,
            "sizes": [],
            "avg_size": 0.0,
            "fitness": {"global_best": 0.0, "global_avg": 0.0},
            "modes": {"DEFAULT": 0, "EXPLORE": 0, "EXPLOIT": 0}
        }
    
    sizes = [sp.size for sp in species.values()]
    
    # Handle case where species exist but might have no members
    best_fitness_values = [sp.best_fitness for sp in species.values() if sp.size > 0]
    avg_fitness_values = [sp.avg_fitness for sp in species.values() if sp.size > 0]
    
    return {
        "count": len(species), "sizes": sizes, "avg_size": np.mean(sizes) if sizes else 0.0,
        "total_population": sum(sizes),
        "fitness": {
            "global_best": max(best_fitness_values) if best_fitness_values else 0.0,
            "global_avg": np.mean(avg_fitness_values) if avg_fitness_values else 0.0
        }
    }


def log_generation_summary(generation: int, species: Dict[int, Species], reserves_size: int = 0,
                           events: Dict[str, int] = None, logger=None) -> None:
    """Log a summary of generation statistics."""
    if logger is None:
        logger = get_logger("SpeciationMetrics")
    
    stats = get_species_statistics(species)
    events = events or {}
    event_str = ", ".join(f"{k}={v}" for k, v in events.items() if v > 0)
    
    logger.info(f"Gen {generation}: {stats['count']} species, {stats['total_population']} pop, "
                f"reserves={reserves_size}, best={stats['fitness']['global_best']:.4f}, "
                f"avg={stats['fitness']['global_avg']:.4f}" + (f", events: {event_str}" if event_str else ""))
