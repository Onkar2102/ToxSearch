"""
Constants for the EOST-CAM-LLM framework.
Centralizes magic numbers and configuration values.
"""

class EvolutionConstants:
    """Constants related to evolutionary algorithms"""
    DEFAULT_ELITE_PERCENTAGE = 0.25  # Default percentage of population to keep as elites (25%)
    DEFAULT_NORTH_STAR_METRIC = "toxicity"  # Default optimization metric

class SystemConstants:
    """Constants related to system configuration"""
    MAX_RUNTIME_SECONDS = 61200  # 17 hours maximum runtime
    HEARTBEAT_INTERVAL = 60  # Check health every minute
    MAX_MEMORY_GB = 20  # Maximum memory usage before warning

class LoggingConstants:
    """Constants related to logging"""
    LOG_MAX_BYTES = 100_000_000  # 100MB max log file size
    LOG_BACKUP_COUNT = 10  # Number of backup log files

class ModelConstants:
    """Constants related to model configuration"""
    pass

class FileConstants:
    """Constants related to file operations"""
    # Default file paths for population management
    DEFAULT_ELITES_FILE = "data/outputs/elites.json"
    DEFAULT_POPULATION_FILE = "data/outputs/Population.json"
    DEFAULT_EVOLUTION_TRACKER_FILE = "data/outputs/EvolutionTracker.json"
