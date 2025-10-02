## @file src/utils/__init__.py
# @author Onkar Shelar (os9660@rit.edu)
# @brief Utility functions and helper modules.
#
# This package provides:
#  - custom_logging: Logging configuration and performance tracking
#  - population_io: Population loading, saving, and management
#  - config: Configuration management utilities
#  - m3_optimizer: M3 Mac optimization utilities

# Lazy imports to prevent circular import issues
def get_custom_logging():
    """Lazy import of custom_logging functions"""
    from .custom_logging import get_logger, get_log_filename, log_system_info, PerformanceLogger
    return get_logger, get_log_filename, log_system_info, PerformanceLogger

def get_population_io():
    """Lazy import of population_io functions to avoid circular imports"""
    from .population_io import (
        load_and_initialize_population, 
        get_population_files_info, 
        load_population, 
        save_population, 
        sort_population_json, 
        load_genome_by_id,
        consolidate_generations_to_single_file,
        migrate_from_split_to_single,
        # Steady state population management
        sort_population_by_elite_criteria,
        redistribute_population_after_evaluation,
        redistribute_elites_to_population,
        load_elites,
        save_elites,
        add_variants_to_elites,
        get_population_stats_steady_state
    )
    return (
        load_and_initialize_population, 
        get_population_files_info, 
        load_population, 
        save_population, 
        sort_population_json, 
        load_genome_by_id,
        consolidate_generations_to_single_file,
        migrate_from_split_to_single,
        # Steady state population management
        sort_population_by_elite_criteria,
        redistribute_population_after_evaluation,
        redistribute_elites_to_population,
        load_elites,
        save_elites,
        add_variants_to_elites,
        get_population_stats_steady_state
    )

def get_config():
    """Lazy import of config functions"""
    from .config import load_config, save_config
    return load_config, save_config

__all__ = [
    "get_custom_logging",
    "get_population_io", 
    "get_config",
]
