"""
adaptive_threshold.py

Adaptive threshold management and cluster refinement.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

from .island import Individual, Species, generate_species_id
from .distance import semantic_distance

if TYPE_CHECKING:
    from .limbo import LimboBuffer

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def adjust_island_radius(island: Species, max_capacity: int = 50, shrink_factor: float = 0.9,
                         limbo: Optional["LimboBuffer"] = None, current_generation: int = 0,
                         logger=None) -> List[Individual]:
    """
    Adjust island radius when over capacity and eject fringe members.
    
    When an island exceeds max_capacity, it shrinks its radius and ejects
    members that are now outside the new radius (fringe members). This:
    - Maintains island size within capacity
    - Preserves core members (closest to leader)
    - Sends fringe members to limbo (if provided)
    
    Radius adjustment: new_radius = old_radius * shrink_factor
    
    Args:
        island: Species to adjust
        max_capacity: Maximum allowed members
        shrink_factor: Factor to shrink radius (0 < factor < 1)
        limbo: Optional limbo buffer for ejected members
        current_generation: Current generation number
        logger: Optional logger instance
    
    Returns:
        List of ejected individuals
    """
    if logger is None:
        logger = get_logger("AdaptiveThreshold")
    
    ejected = []
    if island.size <= max_capacity:
        return ejected
    
    old_radius = island.radius
    island.radius *= shrink_factor
    
    fringe = []
    for m in island.members:
        if m == island.leader or m.embedding is None or island.leader.embedding is None:
            continue
        dist = semantic_distance(m.embedding, island.leader.embedding)
        if dist > island.radius:
            fringe.append((m, dist))
    
    fringe.sort(key=lambda x: x[1], reverse=True)
    
    for m, _ in fringe:
        if island.size <= max_capacity:
            break
        island.remove_member(m)
        ejected.append(m)
        if limbo:
            limbo.add(m, current_generation)
    
    if ejected:
        logger.debug(f"Island {island.id}: radius {old_radius:.4f} → {island.radius:.4f}, ejected {len(ejected)}")
    
    return ejected


def compute_silhouette_score(island: Species, all_species: Dict[int, Species]) -> float:
    """
    Compute average silhouette score for an island.
    
    Silhouette score measures cluster cohesion:
    - High score (>0.5): Good cohesion (members similar to each other, different from others)
    - Low score (<0.5): Poor cohesion (members too diverse or too similar to other species)
    
    Formula: s = (b - a) / max(a, b)
    where:
    - a = average intra-species distance (to own members)
    - b = average inter-species distance (to nearest other species)
    
    Used to detect when islands should split (low silhouette = heterogeneous).
    
    Args:
        island: Species to compute silhouette for
        all_species: All species (for inter-species distance computation)
    
    Returns:
        Average silhouette score in range [-1, 1] (typically [0, 1])
    """
    if island.size < 2:
        return 1.0
    
    scores = []
    for m in island.members:
        if m.embedding is None:
            continue
        
        intra = [semantic_distance(m.embedding, o.embedding) for o in island.members if o != m and o.embedding is not None]
        if not intra:
            continue
        intra_dist = np.mean(intra)
        
        min_inter = float('inf')
        for sid, sp in all_species.items():
            if sid == island.id or sp.size == 0:
                continue
            inter = [semantic_distance(m.embedding, o.embedding) for o in sp.members if o.embedding is not None]
            if inter:
                min_inter = min(min_inter, np.mean(inter))
        
        if min_inter == float('inf'):
            s = 0.0
        elif max(intra_dist, min_inter) > 0:
            s = (min_inter - intra_dist) / max(intra_dist, min_inter)
        else:
            s = 0.0
        scores.append(s)
    
    return np.mean(scores) if scores else 0.0


def trigger_split_event(island: Species, all_species: Dict[int, Species],
                        silhouette_threshold: float = 0.5, current_generation: int = 0,
                        logger=None) -> List[Species]:
    """Split island if silhouette too low."""
    if logger is None:
        logger = get_logger("AdaptiveThreshold")
    
    silhouette = compute_silhouette_score(island, all_species)
    if silhouette >= silhouette_threshold:
        return []
    
    members = [m for m in island.members if m.embedding is not None]
    if len(members) < 4:
        return []
    
    threshold = island.radius * 0.7
    clusters = _cluster(members, threshold)
    
    valid = [c for c in clusters if len(c) >= 2]
    if len(valid) < 2:
        return []
    
    new_species = []
    for cluster in valid:
        leader = max(cluster, key=lambda x: x.fitness)
        sp = Species(id=generate_species_id(), leader=leader, members=list(cluster),
                    radius=threshold, created_at=current_generation,
                    last_improvement=current_generation, parent_id=island.id)
        new_species.append(sp)
    
    logger.info(f"Split island {island.id} into {len(new_species)} new islands (silhouette={silhouette:.4f})")
    return new_species


def _cluster(individuals: List[Individual], threshold: float) -> List[List[Individual]]:
    if len(individuals) < 2:
        return [individuals] if individuals else []
    
    n = len(individuals)
    embeddings = np.array([ind.embedding for ind in individuals])
    distances = [semantic_distance(embeddings[i], embeddings[j]) for i in range(n) for j in range(i + 1, n)]
    
    if not distances:
        return [individuals]
    
    try:
        Z = linkage(np.array(distances), method='average')
        labels = fcluster(Z, t=threshold, criterion='distance')
    except Exception:
        return [individuals]
    
    clusters: Dict[int, List[Individual]] = {}
    for ind, label in zip(individuals, labels):
        clusters.setdefault(label, []).append(ind)
    return list(clusters.values())


def process_adaptive_thresholds(species: Dict[int, Species], current_generation: int,
                                max_capacity: int = 50, shrink_factor: float = 0.9,
                                silhouette_threshold: float = 0.5,
                                silhouette_check_frequency: int = 10,
                                limbo: Optional["LimboBuffer"] = None,
                                logger=None) -> Tuple[Dict[int, Species], List[Dict]]:
    """Process adaptive threshold adjustments."""
    if logger is None:
        logger = get_logger("AdaptiveThreshold")
    
    events = []
    
    for sid, island in list(species.items()):
        ejected = adjust_island_radius(island, max_capacity, shrink_factor, limbo, current_generation, logger)
        if ejected:
            events.append({"type": "radius", "species_id": sid, "ejected": len(ejected)})
    
    if current_generation % silhouette_check_frequency == 0:
        to_split = [(sid, island, compute_silhouette_score(island, species))
                    for sid, island in species.items()]
        to_split = [(sid, island, s) for sid, island, s in to_split if s < silhouette_threshold]
        
        for sid, island, silhouette in to_split:
            new_list = trigger_split_event(island, species, silhouette_threshold, current_generation, logger)
            if new_list:
                species.pop(sid, None)
                for sp in new_list:
                    species[sp.id] = sp
                events.append({"type": "split", "original_id": sid, "new_ids": [sp.id for sp in new_list]})
    
    return species, events


def compute_all_silhouette_scores(species: Dict[int, Species]) -> Dict[int, float]:
    """Compute silhouette for all species."""
    return {sid: compute_silhouette_score(sp, species) for sid, sp in species.items()}


def get_adaptive_statistics(species: Dict[int, Species]) -> Dict:
    """Get adaptive threshold statistics."""
    if not species:
        return {"avg_radius": 0.0, "avg_silhouette": 0.0}
    
    radii = [sp.radius for sp in species.values()]
    silhouettes = compute_all_silhouette_scores(species)
    
    return {
        "avg_radius": np.mean(radii),
        "min_radius": np.min(radii),
        "max_radius": np.max(radii),
        "avg_silhouette": np.mean(list(silhouettes.values())) if silhouettes else 0.0
    }

