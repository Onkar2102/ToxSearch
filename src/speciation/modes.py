"""
modes.py

Explore/Exploit/Default mode switching logic for islands.
"""

import random
from typing import Dict, List, Optional, TYPE_CHECKING

from .island import Individual, Species, IslandMode

if TYPE_CHECKING:
    from .limbo import LimboBuffer
    from .config import PlanAPlusConfig

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def detect_stagnation(island: Species, window: int = 5, current_gen: int = 0) -> bool:
    """Check if island is stagnant."""
    return current_gen - island.last_improvement >= window


def enter_explore_mode(island: Species, limbo: Optional["LimboBuffer"] = None,
                       explore_mutation_multiplier: float = 2.0,
                       explore_selection_pressure: float = 0.7, logger=None) -> None:
    """Switch to EXPLORE mode."""
    if logger is None:
        logger = get_logger("ModeSwitch")
    
    old_mode = island.mode
    island.mode = IslandMode.EXPLORE
    island.mutation_rate = explore_mutation_multiplier
    
    if limbo and limbo.size > 0:
        candidates = limbo.individuals
        if candidates:
            island.external_parent = random.choice(candidates)
    
    if old_mode != IslandMode.EXPLORE:
        logger.info(f"Island {island.id}: {old_mode.value} → EXPLORE")


def enter_exploit_mode(island: Species, exploit_mutation_multiplier: float = 0.5, logger=None) -> None:
    """Switch to EXPLOIT mode."""
    if logger is None:
        logger = get_logger("ModeSwitch")
    
    old_mode = island.mode
    island.mode = IslandMode.EXPLOIT
    island.mutation_rate = exploit_mutation_multiplier
    island.external_parent = None
    
    if old_mode != IslandMode.EXPLOIT:
        logger.info(f"Island {island.id}: {old_mode.value} → EXPLOIT")


def enter_default_mode(island: Species, logger=None) -> None:
    """Switch to DEFAULT mode."""
    if logger is None:
        logger = get_logger("ModeSwitch")
    
    old_mode = island.mode
    island.mode = IslandMode.DEFAULT
    island.mutation_rate = 1.0
    island.external_parent = None
    
    if old_mode != IslandMode.DEFAULT:
        logger.debug(f"Island {island.id}: {old_mode.value} → DEFAULT")


def update_island_mode(island: Species, current_gen: int, limbo: Optional["LimboBuffer"] = None,
                       window: int = 5, improvement_slope_threshold: float = 0.01,
                       decline_slope_threshold: float = -0.001,
                       explore_mutation_multiplier: float = 2.0,
                       exploit_mutation_multiplier: float = 0.5,
                       explore_selection_pressure: float = 0.7, logger=None) -> IslandMode:
    """
    Update island mode based on fitness trend analysis.
    
    This is the core adaptive mode switching logic. Islands dynamically switch
    between DEFAULT, EXPLORE, and EXPLOIT modes based on their fitness trajectory:
    
    - EXPLOIT: Triggered when fitness slope > improvement_slope_threshold
      (island is improving → focus on exploitation)
    
    - EXPLORE: Triggered when fitness slope < decline_slope_threshold
      (island is declining → need diversity)
    
    - DEFAULT: Triggered when fitness slope is between thresholds
      (island is stable → balanced strategy)
    
    Mode switching affects:
    - Mutation rate (EXPLORE: higher, EXPLOIT: lower)
    - Selection pressure (EXPLORE: relaxed, EXPLOIT: elite-focused)
    - External parent selection (EXPLORE: may use limbo individuals)
    
    Args:
        island: Species to update
        current_gen: Current generation number
        limbo: Optional limbo buffer (for EXPLORE mode external parents)
        window: Number of generations to analyze for slope
        improvement_slope_threshold: Slope threshold for EXPLOIT mode
        decline_slope_threshold: Slope threshold for EXPLORE mode
        explore_mutation_multiplier: Mutation multiplier for EXPLORE mode
        exploit_mutation_multiplier: Mutation multiplier for EXPLOIT mode
        explore_selection_pressure: Selection pressure for EXPLORE mode
        logger: Optional logger instance
    
    Returns:
        Updated IslandMode
    """
    if logger is None:
        logger = get_logger("ModeSwitch")
    
    if len(island.fitness_history) < window:
        return island.mode
    
    slope = island.get_fitness_slope(window)
    
    if slope > improvement_slope_threshold:
        if island.mode != IslandMode.EXPLOIT:
            enter_exploit_mode(island, exploit_mutation_multiplier, logger)
    elif slope < decline_slope_threshold:
        if island.mode != IslandMode.EXPLORE:
            enter_explore_mode(island, limbo, explore_mutation_multiplier, explore_selection_pressure, logger)
    else:
        if island.mode != IslandMode.DEFAULT:
            enter_default_mode(island, logger)
    
    return island.mode


def update_all_island_modes(species: Dict[int, Species], current_gen: int,
                            limbo: Optional["LimboBuffer"] = None,
                            config: Optional["PlanAPlusConfig"] = None, logger=None) -> Dict[str, int]:
    """Update modes for all islands."""
    if logger is None:
        logger = get_logger("ModeSwitch")
    
    window = config.stagnation_window if config else 5
    improvement_slope = config.improvement_slope_threshold if config else 0.01
    decline_slope = config.decline_slope_threshold if config else -0.001
    explore_mult = config.explore_mutation_multiplier if config else 2.0
    exploit_mult = config.exploit_mutation_multiplier if config else 0.5
    explore_pressure = config.explore_selection_pressure if config else 0.7
    
    mode_counts = {"DEFAULT": 0, "EXPLORE": 0, "EXPLOIT": 0}
    
    for sp in species.values():
        mode = update_island_mode(sp, current_gen, limbo, window, improvement_slope, decline_slope,
                                  explore_mult, exploit_mult, explore_pressure, logger)
        mode_counts[mode.value] += 1
    
    return mode_counts


def get_mode_statistics(species: Dict[int, Species]) -> Dict:
    """Get mode statistics."""
    import numpy as np
    mode_counts = {"DEFAULT": 0, "EXPLORE": 0, "EXPLOIT": 0}
    mode_fitness = {"DEFAULT": [], "EXPLORE": [], "EXPLOIT": []}
    
    for sp in species.values():
        mode_counts[sp.mode.value] += 1
        mode_fitness[sp.mode.value].append(sp.avg_fitness)
    
    avg_fitness = {m: (np.mean(f) if f else 0.0) for m, f in mode_fitness.items()}
    return {"counts": mode_counts, "avg_fitness_by_mode": avg_fitness, "total_species": sum(mode_counts.values())}

