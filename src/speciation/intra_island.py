"""
intra_island.py

Intra-island evolution: parent selection and survivor selection.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

from .island import Individual, Species, IslandMode
from .distance import semantic_distance

if TYPE_CHECKING:
    from .limbo import LimboBuffer

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def select_parents_elite_focused(island: Species, alpha: float = 0.5,
                                  mode: Optional[IslandMode] = None) -> Tuple[Individual, Individual]:
    """
    Select parents for breeding with mode-adaptive selection strategy.
    
    Parent selection varies by island mode:
    - DEFAULT: Balanced elite + diversity selection
    - EXPLOIT: Hyper-elitist (top 3 only)
    - EXPLORE: Relaxed selection with potential external parent from limbo
    
    This ensures each mode has appropriate selection pressure:
    - EXPLOIT focuses on best solutions (exploitation)
    - EXPLORE allows more diversity (exploration)
    - DEFAULT balances both
    
    Args:
        island: Species to select parents from
        alpha: Selection pressure parameter (higher = more elite-focused)
        mode: Optional mode override (uses island.mode if None)
    
    Returns:
        Tuple of (parent1, parent2) for breeding
    """
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
    """
    DEFAULT mode parent selection: elite + diversity.
    
    Parent1: Elite-focused (exponential distribution over ranked members)
    Parent2: Diversity-focused (selected based on semantic dissimilarity to others)
    
    This balances selection pressure with diversity preservation.
    
    Args:
        sorted_members: Members sorted by fitness (descending)
        alpha: Selection pressure parameter (higher = more elite-focused)
    
    Returns:
        Tuple of (parent1, parent2)
    """
    n = len(sorted_members)
    
    # Parent 1: Elite-focused using exponential ranking
    ranks = np.arange(1, n + 1)
    # Exponential distribution: higher-ranked (lower rank numbers) are more likely
    probs = np.exp(-alpha * ranks)
    probs = probs / probs.sum()
    parent1 = sorted_members[np.random.choice(n, p=probs)]
    
    # Parent 2: Diversity-focused (select based on dissimilarity from others)
    diversity_scores = []
    for ind in sorted_members:
        if ind.embedding is None:
            diversity_scores.append(0.0)
            continue
        # Compute average similarity to all other members
        sims = [np.dot(ind.embedding, o.embedding) for o in sorted_members if o != ind and o.embedding is not None]
        # Diversity = 1 - average similarity (higher = more different)
        diversity_scores.append(1.0 - np.mean(sims) if sims else 0.0)
    
    # Use diversity scores as selection probabilities
    div_scores = np.maximum(np.array(diversity_scores), 0.01)
    div_probs = div_scores / div_scores.sum()
    parent2 = sorted_members[np.random.choice(n, p=div_probs)]
    
    return parent1, parent2


def _select_exploit_parents(sorted_members: List[Individual]) -> Tuple[Individual, Individual]:
    """
    EXPLOIT mode parent selection: hyper-elitist.
    
    Only selects from top-k members (default k=3) with exponential bias
    toward the best. This ensures EXPLOIT mode focuses on the very best solutions.
    
    Args:
        sorted_members: Members sorted by fitness (descending)
    
    Returns:
        Tuple of (parent1, parent2), both from top-k
    """
    n = len(sorted_members)
    # Consider only top 3 members
    top_k = min(3, n)
    top = sorted_members[:top_k]
    # Exponential weights with stronger bias to best
    weights = np.exp(-2.0 * np.arange(top_k))
    probs = weights / weights.sum()
    # Select two parents from top-k with replacement
    return top[np.random.choice(top_k, p=probs)], top[np.random.choice(top_k, p=probs)]


def _select_explore_parents(island: Species, sorted_members: List[Individual], alpha: float) -> Tuple[Individual, Individual]:
    """
    EXPLORE mode parent selection: relaxed selection with external parent.
    
    Parent1: Relaxed selection (weaker exponential bias than DEFAULT)
    Parent2: Either external parent from limbo (if available) or blended selection
    
    This enables exploration by allowing lower-fitness members to breed
    and potentially introducing novel genetic material from limbo.
    
    Args:
        island: Species to select parents from
        sorted_members: Members sorted by fitness (descending)
        alpha: Base selection pressure parameter
    
    Returns:
        Tuple of (parent1, parent2)
    """
    n = len(sorted_members)
    # Relaxed selection: reduce alpha for weaker bias
    explore_alpha = alpha * 0.5
    ranks = np.arange(1, n + 1)
    probs = np.exp(-explore_alpha * ranks)
    probs = probs / probs.sum()
    parent1 = sorted_members[np.random.choice(n, p=probs)]
    
    # Parent 2: External parent from limbo if available, else blended selection
    if island.external_parent is not None:
        parent2 = island.external_parent
        island.external_parent = None  # One-time use
    else:
        # Blend probabilistic and uniform distributions (more diversity)
        uniform = np.ones(n) / n
        blend = 0.5 * probs + 0.5 * uniform  # 50% elite, 50% random
        blend = blend / blend.sum()
        parent2 = sorted_members[np.random.choice(n, p=blend)]
    
    return parent1, parent2


def survivor_selection(island: Species, offspring: List[Individual], max_capacity: int = 50,
                       limbo: Optional["LimboBuffer"] = None, logger=None) -> List[Individual]:
    """
    Select survivors for island after breeding.
    
    Survivor selection determines which individuals remain in the island for the next
    generation. The process:
    1. Always keep the leader (elitism)
    2. Add unique offspring (by prompt) sorted by fitness
    3. Add remaining high-fitness members from current population
    4. Truncate to max_capacity
    5. Eject lowest-fitness individuals over capacity to limbo
    
    This maintains genetic material while enforcing population constraints.
    
    Args:
        island: Species to perform survivor selection on
        offspring: New offspring generated from breeding
        max_capacity: Maximum members to keep
        limbo: Optional limbo buffer for rejected members
        logger: Optional logger instance
    
    Returns:
        List of ejected individuals (sent to limbo)
    """
    if logger is None:
        logger = get_logger("SurvivorSelection")
    
    ejected = []
    leader = island.leader
    # Start survivors with leader (elitism)
    survivors = [leader]
    seen = {leader.prompt}
    
    # Combine current members and offspring, sort by fitness
    candidates = sorted(island.members + offspring, key=lambda x: x.fitness, reverse=True)
    
    # Add unique candidates (not already in survivors)
    for ind in candidates:
        if ind == leader:
            continue
        if ind.prompt not in seen:
            survivors.append(ind)
            seen.add(ind.prompt)
            if len(survivors) >= max_capacity:
                break
    
    # If still over capacity, trim to max_capacity
    if len(survivors) > max_capacity:
        ejected = survivors[max_capacity:]
        survivors = survivors[:max_capacity]
    
    # Update island members
    island.members = survivors
    # Ensure all survivors know their species assignment
    for m in island.members:
        m.species_id = island.id
    
    return ejected


def select_parents_from_species(species: Dict[int, Species], species_id: int, alpha: float = 0.5) -> Tuple[Individual, Individual]:
    """
    Select parents from a specific species by ID.
    
    Wrapper around select_parents_elite_focused for convenience when you have a species ID
    instead of a Species object.
    
    Args:
        species: Dict of all species
        species_id: ID of species to select parents from
        alpha: Selection pressure parameter
    
    Returns:
        Tuple of (parent1, parent2)
    
    Raises:
        KeyError: If species_id not found
    """
    if species_id not in species:
        raise KeyError(f"Species {species_id} not found")
    return select_parents_elite_focused(species[species_id], alpha)


def compute_breeding_budget(island: Species, base_budget: int = 5, fitness_bonus_factor: float = 0.2) -> int:
    """
    Compute breeding budget for an island (number of offspring to create).
    
    Breeding budget is adaptive based on:
    - Base budget (default 5 offspring)
    - Fitness bonus: High-fitness islands breed more (+up to 0.5 * base_budget)
    - Stagnation penalty: Stagnant islands breed less (-1 if stagnated > 5 gens)
    - Mode adjustment: EXPLOIT breeds more (+20%), EXPLORE breeds less (-10%)
    
    This ensures successful species expand while struggling species are reduced.
    
    Args:
        island: Species to compute budget for
        base_budget: Base number of offspring (default 5)
        fitness_bonus_factor: Multiplier for fitness bonus (default 0.2)
    
    Returns:
        Number of offspring to create (minimum 1)
    """
    budget = base_budget
    # Fitness bonus: High-fitness islands produce more offspring
    bonus = int(fitness_bonus_factor * island.avg_fitness * base_budget)
    # Cap bonus at 50% of base budget
    bonus = min(bonus, base_budget // 2)
    
    # Stagnation penalty: Long-stagnant islands produce fewer offspring
    if island.stagnation_counter > 5:
        budget = max(2, budget - 1)
    
    # Mode adjustment: EXPLOIT produces more, EXPLORE produces less
    if island.mode == IslandMode.EXPLOIT:
        budget = int(budget * 1.2)  # 20% increase
    elif island.mode == IslandMode.EXPLORE:
        budget = int(budget * 0.9)  # 10% decrease
    
    # Final budget = base + bonus, minimum 1
    return max(1, budget + bonus)

