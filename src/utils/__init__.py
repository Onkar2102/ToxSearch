## @file src/utils/__init__.py
# @brief Utility functions and helper modules.
#
# This package provides:
#  - custom_logging: Logging configuration and performance tracking
#  - population_io: Population loading, saving, and management
#  - m3_optimizer: System optimization utilities for Apple Silicon

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
        load_elites,
        save_elites,
        get_population_stats_steady_state,
        finalize_initial_population
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
        load_elites,
        save_elites,
        get_population_stats_steady_state,
        finalize_initial_population
    )

def get_system_utils():
    """Lazy import of system utility functions"""
    from .population_io import (
        get_project_root,
        get_config_path,
        get_data_path,
        get_outputs_path,
        _extract_north_star_score,
        initialize_system
    )
    return (
        get_project_root,
        get_config_path,
        get_data_path,
        get_outputs_path,
        _extract_north_star_score,
        initialize_system
    )

__all__ = [
    "get_custom_logging",
    "get_population_io",
    "get_system_utils",
]
