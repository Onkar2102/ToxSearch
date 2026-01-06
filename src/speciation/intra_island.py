"""
intra_island.py

Intra-island evolution: parent selection and survivor selection.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

from .island import Individual, Species, IslandMode
from .embeddings import semantic_distance

if TYPE_CHECKING:
    from .limbo import LimboBuffer

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def select_parents_elite_focused(island: Species, alpha: float = 0.5,
                                  mode: Optional[IslandMode] = None) -> Tuple[Individual, Individual]:
    """Select parents with tiered approach based on island mode."""
    if mode is None:
        mode = island.mode
    
    members = island.get_sorted_members(reverse=True)
    
    if len(members) == 0:
        raise ValueError(f"Island {island.id} has no members")
    if len(members) == 1:
        return members[0], members[0]
    
    if mode == IslandMode.EXPLOIT:
        return _select_exploit_parents(members)
    elif mode == IslandMode.EXPLORE:
        return _select_explore_parents(island, members, alpha)
    else:
        return _select_default_parents(members, alpha)


def _select_default_parents(sorted_members: List[Individual], alpha: float) -> Tuple[Individual, Individual]:
    """DEFAULT mode: elite + diversity."""
    n = len(sorted_members)
    
    # Parent 1: Elite-focused
    ranks = np.arange(1, n + 1)
    probs = np.exp(-alpha * ranks)
    probs = probs / probs.sum()
    parent1 = sorted_members[np.random.choice(n, p=probs)]
    
    # Parent 2: Diversity-focused
    diversity_scores = []
    for ind in sorted_members:
        if ind.embedding is None:
            diversity_scores.append(0.0)
            continue
        sims = [np.dot(ind.embedding, o.embedding) for o in sorted_members if o != ind and o.embedding is not None]
        diversity_scores.append(1.0 - np.mean(sims) if sims else 0.0)
    
    div_scores = np.maximum(np.array(diversity_scores), 0.01)
    div_probs = div_scores / div_scores.sum()
    parent2 = sorted_members[np.random.choice(n, p=div_probs)]
    
    return parent1, parent2


def _select_exploit_parents(sorted_members: List[Individual]) -> Tuple[Individual, Individual]:
    """EXPLOIT mode: hyper-elitist."""
    n = len(sorted_members)
    top_k = min(3, n)
    top = sorted_members[:top_k]
    weights = np.exp(-2.0 * np.arange(top_k))
    probs = weights / weights.sum()
    return top[np.random.choice(top_k, p=probs)], top[np.random.choice(top_k, p=probs)]


def _select_explore_parents(island: Species, sorted_members: List[Individual], alpha: float) -> Tuple[Individual, Individual]:
    """EXPLORE mode: relaxed selection with external parent."""
    n = len(sorted_members)
    explore_alpha = alpha * 0.5
    ranks = np.arange(1, n + 1)
    probs = np.exp(-explore_alpha * ranks)
    probs = probs / probs.sum()
    parent1 = sorted_members[np.random.choice(n, p=probs)]
    
    if island.external_parent is not None:
        parent2 = island.external_parent
        island.external_parent = None
    else:
        uniform = np.ones(n) / n
        blend = 0.5 * probs + 0.5 * uniform
        blend = blend / blend.sum()
        parent2 = sorted_members[np.random.choice(n, p=blend)]
    
    return parent1, parent2


def survivor_selection(island: Species, offspring: List[Individual], max_capacity: int = 50,
                       limbo: Optional["LimboBuffer"] = None, logger=None) -> List[Individual]:
    """Select survivors within island after breeding."""
    if logger is None:
        logger = get_logger("SurvivorSelection")
    
    ejected = []
    leader = island.leader
    survivors = [leader]
    seen = {leader.prompt}
    
    candidates = sorted(island.members + offspring, key=lambda x: x.fitness, reverse=True)
    
    for ind in candidates:
        if ind == leader:
            continue
        if ind.prompt not in seen:
            survivors.append(ind)
            seen.add(ind.prompt)
            if len(survivors) >= max_capacity:
                break
    
    if len(survivors) > max_capacity:
        ejected = survivors[max_capacity:]
        survivors = survivors[:max_capacity]
    
    island.members = survivors
    for m in island.members:
        m.species_id = island.id
    
    return ejected


def select_parents_from_species(species: Dict[int, Species], species_id: int, alpha: float = 0.5) -> Tuple[Individual, Individual]:
    """Select parents from a specific species."""
    if species_id not in species:
        raise KeyError(f"Species {species_id} not found")
    return select_parents_elite_focused(species[species_id], alpha)


def compute_breeding_budget(island: Species, base_budget: int = 5, fitness_bonus_factor: float = 0.2) -> int:
    """Compute breeding budget based on performance."""
    budget = base_budget
    bonus = int(fitness_bonus_factor * island.avg_fitness * base_budget)
    bonus = min(bonus, base_budget // 2)
    
    if island.stagnation_counter > 5:
        budget = max(2, budget - 1)
    
    if island.mode == IslandMode.EXPLOIT:
        budget = int(budget * 1.2)
    elif island.mode == IslandMode.EXPLORE:
        budget = int(budget * 0.9)
    
    return max(1, budget + bonus)

