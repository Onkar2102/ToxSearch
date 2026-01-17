"""
metrics.py

Validation metrics for speciation.
"""

import numpy as np
from typing import Dict, List, Optional, Any
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
    cluster_quality: Optional[Dict[str, Any]] = None  # Optional cluster quality metrics
    
    def to_dict(self) -> Dict:
        result = {
            "generation": self.generation, "species_count": self.species_count,
            "total_population": self.total_population, "reserves_size": self.reserves_size,
            "best_fitness": round(self.best_fitness, 4), "avg_fitness": round(self.avg_fitness, 4),
            "fitness_std": round(self.fitness_std, 4),
            "speciation_events": self.speciation_events,
            "merge_events": self.merge_events, "extinction_events": self.extinction_events,
            "inter_species_diversity": round(self.inter_species_diversity, 4),
            "intra_species_diversity": round(self.intra_species_diversity, 4)
        }
        # Add cluster quality if available
        if self.cluster_quality:
            result["cluster_quality"] = self.cluster_quality
        return result


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
                          extinction_events: int = 0, cluster0=None, 
                          elites_path: Optional[str] = None, reserves_path: Optional[str] = None) -> GenerationMetrics:
        """Record metrics for a generation.
        
        Args:
            generation: Current generation number
            species: Dict of species objects
            reserves_size: Size of reserves (cluster 0) - should match reserves.json
            speciation_events: Number of speciation events
            merge_events: Number of merge events
            extinction_events: Number of extinction events
            cluster0: Optional Cluster0 object (for backward compatibility)
            elites_path: Optional path to elites.json (for accurate population count)
            reserves_path: Optional path to reserves.json (for accurate reserves count)
        """
        from pathlib import Path
        import json
        
        species_count = len(species)
        
        # Calculate total_population from actual files (elites.json + reserves.json)
        # This is more accurate than using in-memory species.members which may be stale
        total_pop = 0
        all_fitness = []
        
        # Try to read from files first (most accurate)
        if elites_path and Path(elites_path).exists():
            try:
                with open(elites_path, 'r', encoding='utf-8') as f:
                    elites_genomes = json.load(f)
                total_pop += len(elites_genomes)
                # Extract fitness from elites
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
                # Fallback to in-memory species if file read fails
                total_pop = sum(sp.size for sp in species.values())
                all_fitness = [m.fitness for sp in species.values() for m in sp.members]
        else:
            # Fallback: use in-memory species
            total_pop = sum(sp.size for sp in species.values())
            all_fitness = [m.fitness for sp in species.values() for m in sp.members]
        
        # Add reserves from file (more accurate than cluster0.individuals)
        actual_reserves_size = reserves_size  # Initialize with parameter
        if reserves_path and Path(reserves_path).exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                actual_reserves_size = len(reserves_genomes)
                total_pop += actual_reserves_size
                # Extract fitness from reserves
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
                # Fallback: add reserves_size if file read fails
                total_pop += reserves_size
                # Include cluster0 fitness if provided (backward compatibility)
                if cluster0 is not None and hasattr(cluster0, 'individuals'):
                    all_fitness.extend([ind.fitness for ind in cluster0.individuals])
        else:
            # Fallback: add reserves_size and use cluster0 if available
            total_pop += reserves_size
            if cluster0 is not None and hasattr(cluster0, 'individuals'):
                all_fitness.extend([ind.fitness for ind in cluster0.individuals])
        
        best = max(all_fitness) if all_fitness else 0.0
        avg = np.mean(all_fitness) if all_fitness else 0.0
        std = np.std(all_fitness) if all_fitness else 0.0
        
        inter_div, intra_div = compute_diversity_metrics(species, w_genotype=0.7, w_phenotype=0.3, elites_path=elites_path)
        
        # Calculate cluster quality metrics if we have enough species
        cluster_quality = None
        if species_count > 1 and total_pop >= 4:
            try:
                from utils.cluster_quality import calculate_cluster_quality_metrics
                # Use elites_path parent directory as outputs_path
                if elites_path:
                    from pathlib import Path
                    outputs_path = str(Path(elites_path).parent)
                    cluster_quality = calculate_cluster_quality_metrics(outputs_path=outputs_path, logger=self.logger)
            except Exception as e:
                # Cluster quality calculation is optional, don't fail if it errors
                self.logger.debug(f"Failed to calculate cluster quality metrics: {e}")
        
        self.total_speciation += speciation_events
        self.total_merges += merge_events
        self.total_extinctions += extinction_events
        
        metrics = GenerationMetrics(
            generation=generation, species_count=species_count, total_population=total_pop,
            reserves_size=actual_reserves_size, 
            best_fitness=round(float(best), 4), 
            avg_fitness=round(float(avg), 4),
            fitness_std=round(float(std), 4), 
            speciation_events=speciation_events,
            merge_events=merge_events, extinction_events=extinction_events,
            inter_species_diversity=round(float(inter_div), 4), 
            intra_species_diversity=round(float(intra_div), 4),
            cluster_quality=cluster_quality  # Include cluster quality in constructor
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


def compute_diversity_metrics(species: Dict[int, Species], w_genotype: float = 0.7, w_phenotype: float = 0.3,
                               elites_path: Optional[str] = None) -> tuple:
    """
    Compute inter-species and intra-species diversity metrics.
    
    Diversity metrics measure:
    - Inter-species diversity: Average distance between species leaders
      (high = species are diverse, low = species are similar/redundant)
    - Intra-species diversity: Average distance within each species
      (high = members are diverse, low = members are homogeneous)
    
    Args:
        species: Dict of all current species
        w_genotype: Weight for genotype (embedding) distance component
        w_phenotype: Weight for phenotype (toxicity scores) distance component
        elites_path: Optional path to elites.json to load actual members (if species.members is incomplete)
    
    Returns:
        Tuple of (inter_species_diversity, intra_species_diversity)
    """
    from pathlib import Path
    import json
    
    if not species:
        return 0.0, 0.0
    
    # Filter out species without valid leaders (should not happen, but safeguard)
    species_list = [sp for sp in species.values() if sp.leader is not None]
    
    if not species_list:
        return 0.0, 0.0
    
    # Load actual members from elites.json if path provided and species.members is incomplete
    # This fixes the issue where loaded species only have the leader in members list
    elites_genomes_by_species = {}
    logger = get_logger("DiversityMetrics")
    if elites_path and Path(elites_path).exists():
        try:
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_genomes = json.load(f)
            
            # Group genomes by species_id
            for genome in elites_genomes:
                species_id = genome.get("species_id")
                if species_id is not None and species_id > 0:
                    if species_id not in elites_genomes_by_species:
                        elites_genomes_by_species[species_id] = []
                    elites_genomes_by_species[species_id].append(genome)
            
            logger.debug(f"Loaded {len(elites_genomes_by_species)} species from elites.json for diversity calculation")
        except Exception as e:
            # If file read fails, continue with in-memory members
            logger.warning(f"Failed to load elites.json for diversity metrics: {e}")
            pass
    else:
        logger.debug("No elites_path provided, using in-memory species members for diversity calculation")
    
    # Compute inter-species diversity (distance between leaders)
    inter = []
    for i, sp1 in enumerate(species_list):
        for sp2 in species_list[i + 1:]:
            # Only compare if both leaders have embeddings
            if sp1.leader.embedding is not None and sp2.leader.embedding is not None:
                try:
                    # Normalize embeddings if needed (required for ensemble_distance)
                    e1 = sp1.leader.embedding.copy()
                    e2 = sp2.leader.embedding.copy()
                    norm1 = np.linalg.norm(e1)
                    norm2 = np.linalg.norm(e2)
                    if not np.isclose(norm1, 1.0, atol=1e-5):
                        e1 = e1 / norm1
                    if not np.isclose(norm2, 1.0, atol=1e-5):
                        e2 = e2 / norm2
                    
                    # Handle None phenotypes by passing them as-is (ensemble_distance handles None)
                    dist = ensemble_distance(
                        e1, e2,
                        sp1.leader.phenotype, sp2.leader.phenotype,
                        w_genotype, w_phenotype
                    )
                    inter.append(dist)
                except Exception as e:
                    # Skip this pair if distance calculation fails
                    get_logger("DiversityMetrics").debug(f"Failed to compute inter-species distance: {e}")
                    pass
    
    inter_div = np.mean(inter) if inter else 0.0
    
    # Compute intra-species diversity (distances within each species)
    intra_divs = []
    for sp in species_list:
        # Get members: prefer actual members from elites.json if available, otherwise use in-memory members
        members = []
        
        # Try to load from elites.json first (more complete)
        if sp.id in elites_genomes_by_species:
            from .species import Individual
            from .phenotype_distance import extract_phenotype_vector
            
            for genome in elites_genomes_by_species[sp.id]:
                # Extract embedding
                embedding = None
                if "prompt_embedding" in genome:
                    emb_list = genome["prompt_embedding"]
                    if isinstance(emb_list, list):
                        embedding = np.array(emb_list, dtype=np.float32)
                    elif isinstance(emb_list, np.ndarray):
                        embedding = emb_list
                
                # Extract phenotype
                phenotype = extract_phenotype_vector(genome, logger=None)
                
                # Only include if embedding is available
                if embedding is not None:
                    # Normalize embedding if needed
                    norm = np.linalg.norm(embedding)
                    if not np.isclose(norm, 1.0, atol=1e-5):
                        embedding = embedding / norm
                    
                    # Extract fitness
                    fitness = 0.0
                    if "north_star_score" in genome:
                        fitness = genome["north_star_score"]
                    elif "moderation_result" in genome and isinstance(genome["moderation_result"], dict):
                        google_result = genome["moderation_result"].get("google", {})
                        if google_result and "scores" in google_result:
                            fitness = google_result["scores"].get("toxicity", 0.0)
                    
                    member = Individual(
                        id=genome.get("id", 0),
                        prompt=genome.get("prompt", ""),
                        fitness=fitness,
                        embedding=embedding,
                        phenotype=phenotype,
                        species_id=sp.id
                    )
                    members.append(member)
            
            logger.debug(f"Species {sp.id}: Loaded {len(members)} members from elites.json (had {len(sp.members)} in-memory)")
        else:
            # Fallback to in-memory members
            members = [m for m in sp.members if m is not None and m.embedding is not None]
            logger.debug(f"Species {sp.id}: Using {len(members)} in-memory members (not found in elites.json)")
        
        # Need at least 2 members to compute intra-species diversity
        if len(members) < 2:
            continue
        
        # Calculate pairwise distances between members
        dists = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                try:
                    dist = ensemble_distance(
                        members[i].embedding, members[j].embedding,
                        members[i].phenotype, members[j].phenotype,
                        w_genotype, w_phenotype
                    )
                    dists.append(dist)
                except Exception as e:
                    # Skip this pair if distance calculation fails
                    get_logger("DiversityMetrics").debug(f"Failed to compute intra-species distance: {e}")
                    pass
        
        if dists:
            intra_divs.append(np.mean(dists))
    
    intra_div = np.mean(intra_divs) if intra_divs else 0.0
    
    # Round to 4 decimal places before returning
    return round(float(inter_div), 4), round(float(intra_div), 4)


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
