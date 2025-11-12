"""
Comprehensive Population I/O Utility

This module provides unified population loading, saving, and management functionality
that automatically handles both monolithic and split file formats for optimal memory efficiency.
"""

from typing import List, Dict, Any, Optional, Union
import os
import json
from pathlib import Path
from collections import defaultdict
import time
from utils import get_custom_logging
from utils.constants import EvolutionConstants, FileConstants
from utils import get_population_io
from gne import get_ResponseGenerator, get_PromptGenerator
from datetime import datetime

# Import pandas directly - it's required for this module
import pandas as pd

# Get the functions at module level to avoid repeated calls
get_logger, _, _, PerformanceLogger = get_custom_logging()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_project_root():
    """Get the absolute path to the project root directory"""
    script_dir = Path(__file__).parent
    # Go up two levels to get to the project root (from src/utils/ to project root)
    project_root = script_dir.parent.parent
    return project_root.resolve()

def get_config_path():
    """Get the absolute path to the config directory"""
    return get_project_root() / "config"

def get_data_path():
    """Get the absolute path to the data directory"""
    return get_project_root() / "data" / "prompt.csv"

# Global variable to store the outputs path for the current run
_current_outputs_path = None

def get_outputs_path():
    """Get the absolute path to the outputs directory"""
    global _current_outputs_path
    
    # If we already have a path for this run, use it
    if _current_outputs_path is not None:
        return _current_outputs_path
    
    # Create timestamp string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Create the full path: data/outputs/YYYYMMDD_HHMM/
    outputs_dir = get_project_root() / "data" / "outputs" / timestamp
    
    # Ensure the directory exists
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Store the path for this run
    _current_outputs_path = outputs_dir
    
    return outputs_dir

def _extract_north_star_score(genome, metric="toxicity"):
    """Extract the north star metric score from a genome using only moderation_result.google.scores.
    
    Returns minimum 0.0001 for consistency across the project.
    """
    if not genome:
        return 0.0001
    
    if "moderation_result" in genome:
        moderation_result = genome["moderation_result"]
        
        # Check if moderation_result is not None before using 'in' operator
        if moderation_result is not None and "google" in moderation_result:
            google_scores = moderation_result["google"]
            if "scores" in google_scores:
                score = google_scores["scores"].get(metric, 0.0001)
                if score is not None and score > 0:
                    return round(float(score), 4)
    
    return 0.0001


# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================

def initialize_system(logger, log_file, seed_file="data/prompt.csv"):
    """Initialize the system components and create gen0 if needed
    
    Args:
        logger: Logger instance
        log_file: Log file path
        seed_file: Path to CSV file with seed prompts (must have 'questions' column).
                   Default: data/prompt.csv
    """
    from utils.device_utils import device_manager
    device = device_manager.get_optimal_device()
    
    logger.debug("Initializing pipeline for device: %s", device)
    
    # Import required modules
    population_io_functions = get_population_io()
    
    # Get population IO functions
    load_and_initialize_population, get_population_files_info, load_population, save_population, sort_population_json, load_genome_by_id, consolidate_generations_to_single_file, migrate_from_split_to_single, sort_population_by_elite_criteria, load_elites, save_elites, get_population_stats_steady_state, finalize_initial_population = get_population_io()
    
    # Initialize Response Generator (for generating responses to prompts)
    ResponseGenerator = get_ResponseGenerator()
    response_generator = ResponseGenerator(model_key="response_generator", config_path="config/RGConfig.yaml", log_file=log_file)
    logger.debug("Response generator initialized")
    
    # Initialize Prompt Generator (for operators and evolutionary algorithms)
    PromptGenerator = get_PromptGenerator()
    prompt_generator = PromptGenerator(model_key="prompt_generator", config_path="config/PGConfig.yaml", log_file=log_file)
    logger.debug("Prompt generator initialized")
    
    # Set the global generators for different purposes
    from ea.evolution_engine import set_global_generators
    set_global_generators(response_generator, prompt_generator)
    logger.debug("Global generators set")
    
    # Check if population already exists (steady state: check elites.json)
    population_file = get_outputs_path() / "elites.json"
    
    # Check if population file exists and has content, and avoid double-loading
    population_content = None
    if not population_file.exists():
        should_initialize = True
    else:
        try:
            with open(population_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            should_initialize = len(content) == 0 or content == '[]'
            if not should_initialize:
                # Parse content once for later use
                import json
                population_content = json.loads(content)
        except Exception:
            should_initialize = True

    if should_initialize:
        try:
            # Resolve seed_file path (can be relative or absolute)
            seed_path = Path(seed_file)
            if not seed_path.is_absolute():
                # If relative, resolve from project root
                seed_path = get_project_root() / seed_path
            input_path = str(seed_path)
            logger.info("Initializing population from seed file: %s", input_path)
            
            load_and_initialize_population(
                input_path=input_path,
                output_path=str(get_outputs_path()),
                log_file=log_file
            )
            logger.debug("Population successfully initialized and saved.")
        except Exception as e:
            logger.error("Failed to initialize population: %s", e, exc_info=True)
            raise
    else:
        logger.info("Existing elites file found. Skipping initialization.")
        # Use already loaded content for info
        try:
            population = population_content if population_content is not None else []
            logger.info("Loaded %d genomes from existing elites.json", len(population))
            generations = set(g.get("generation", 0) for g in population if g)
            logger.debug("Available generations: %s", sorted(generations))
        except Exception as e:
            logger.warning("Could not read existing population info: %s", e)
    
    return response_generator, prompt_generator


def clean_population(population: List[Dict[str, Any]], *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Remove None genomes and invalid entries from population.
    
    Parameters
    ----------
    population : List[Dict[str, Any]]
        Population to clean.
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created.
    log_file : str | None
        Optional log-file path when a new logger is created.
        
    Returns
    -------
    List[Dict[str, Any]]
        Cleaned population with None genomes removed.
    """
    _logger = logger or get_logger("population_io", log_file)
    
    original_count = len(population)
    cleaned_population = [g for g in population if g is not None]
    removed_count = original_count - len(cleaned_population)
    
    if removed_count > 0:
        _logger.warning("Removed %d None genomes from population", removed_count)
    
    _logger.info("Population cleaned: %d → %d genomes", original_count, len(cleaned_population))
    return cleaned_population


# ============================================================================
# Split File Management Functions
# ============================================================================

def get_population_files_info(base_dir: str = "outputs") -> Dict[str, Any]:
    """Get information about population files including non_elites.json and elites.json"""
    
    base_path = Path(base_dir).resolve()
    population_file = base_path / "non_elites.json"
    elites_file = base_path / "elites.json"
    evolution_tracker_file = base_path / "EvolutionTracker.json"
    
    info = {
        "total_generations": 0,
        "generation_counts": {},
    }
    
    # Try to get metadata from EvolutionTracker.json first
    if evolution_tracker_file.exists():
        try:
            with open(evolution_tracker_file, 'r', encoding='utf-8') as f:
                tracker = json.load(f)
            
            # Calculate total_generations from the actual generations array
            # This ensures it's always up-to-date with the actual generation count
            if "generations" in tracker and tracker["generations"]:
                # Get the maximum generation number from the generations array
                max_gen_num = max(gen.get("generation_number", 0) for gen in tracker["generations"])
                info["total_generations"] = max_gen_num + 1  # +1 because generation 0 counts as 1 generation
            else:
                # Fallback: use tracker value or 0 if no generations exist
                info["total_generations"] = tracker.get("total_generations", 0)
            
            return info
                
        except Exception as e:
            # Fall back to file scanning if tracker read fails
            pass
    
    # Count genomes in non_elites.json
    if population_file.exists():
        try:
            with open(population_file, 'r', encoding='utf-8') as f:
                population = json.load(f)
            
            
            # Count genomes by generation
            for genome in population:
                if genome and "generation" in genome:
                    gen_num = str(genome["generation"])
                    info["generation_counts"][gen_num] = info["generation_counts"].get(gen_num, 0) + 1
            
        except Exception as e:
            # Silently fail if we can't read the file
            pass
    
    # Count genomes in elites.json
    if elites_file.exists():
        try:
            with open(elites_file, 'r', encoding='utf-8') as f:
                elites = json.load(f)
            
            
            # Add elite genomes to generation counts
            for genome in elites:
                if genome and "generation" in genome:
                    gen_num = str(genome["generation"])
                    info["generation_counts"][gen_num] = info["generation_counts"].get(gen_num, 0) + 1
            
        except Exception as e:
            # Silently fail if we can't read the file
            pass
    
    # Calculate total genomes and generations
    
    # Ensure all generations from 0 to max are represented, even if they have 0 variants
    if info["generation_counts"]:
        max_generation = max(int(k) for k in info["generation_counts"].keys())
        # Fill in missing generations with 0 counts
        for gen_num in range(max_generation + 1):
            if str(gen_num) not in info["generation_counts"]:
                info["generation_counts"][str(gen_num)] = 0
        info["total_generations"] = max_generation + 1
    else:
        info["total_generations"] = 0
    
    return info


def update_population_index_single_file(base_dir: str, total_genomes: int, *, logger=None, log_file: Optional[str] = None):
    """Update the population metadata in EvolutionTracker.json for single file mode"""
    
    _logger = logger or get_logger("update_population_index", log_file)
    
    try:
        base_dir = str(Path(base_dir).resolve())
        info = get_population_files_info(base_dir)
        evolution_tracker_file = Path(base_dir) / "EvolutionTracker.json"
        
        # Load existing EvolutionTracker.json or create new structure
        if evolution_tracker_file.exists():
            try:
                with open(evolution_tracker_file, 'r', encoding='utf-8') as f:
                    tracker = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                tracker = {
                    "status": "not_complete",
                    "total_generations": 1,
                    "generations_since_improvement": 0,
                    "avg_fitness_history": [],
                    "slope_of_avg_fitness": 0.0,
                    "selection_mode": "default",
                    "generations": []
                }
        else:
                tracker = {
                    "status": "not_complete",
                    "total_generations": 1,
                    "generations_since_improvement": 0,
                    "avg_fitness_history": [],
                    "slope_of_avg_fitness": 0.0,
                    "selection_mode": "default",
                    "generations": []
                }
        
        # Update population counts (flattened from population_metadata)
        
        # Update total generations
        tracker["total_generations"] = info["total_generations"]
        
        # Save updated tracker
        with open(evolution_tracker_file, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, indent=2)
        
        _logger.debug("Updated EvolutionTracker population metadata: single file mode, "
                     "total_generations: %d", info['total_generations'])
        
    except Exception as e:
        _logger.warning("Failed to update EvolutionTracker population metadata: %s", e)


def load_population_generation(generation: int, base_dir: str = "outputs", 
                              *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load genomes for a specific generation from the single non_elites.json file"""
    
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Load Generation {generation} from Single File"):
        try:
            # Load entire population
            all_genomes = load_population(base_dir, logger=_logger, log_file=log_file)
            
            # Filter by generation
            generation_genomes = [g for g in all_genomes if g and g.get("generation") == generation]
            
            _logger.info("Loaded generation %d: %d genomes from non_elites.json", generation, len(generation_genomes))
            return generation_genomes
            
        except Exception as e:
            _logger.error("Failed to load generation %d: %s", generation, e, exc_info=True)
            return []


def load_population_range(start_gen: int, end_gen: int, base_dir: str = "outputs",
                         *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load multiple generations from the single non_elites.json file"""
    
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Load Generations {start_gen}-{end_gen} from Single File"):
        try:
            # Load entire population
            all_genomes = load_population(base_dir, logger=_logger, log_file=log_file)
            
            # Filter by generation range
            range_genomes = [g for g in all_genomes if g and start_gen <= g.get("generation", 0) <= end_gen]
            
            _logger.info("Loaded generations %d-%d: %d genomes from non_elites.json", start_gen, end_gen, len(range_genomes))
            return range_genomes
            
        except Exception as e:
            _logger.error("Failed to load generation range: %s", e, exc_info=True)
            return []


def load_population_lazy(base_dir: str = "outputs", max_gens: Optional[int] = None,
                        *, logger=None, log_file: Optional[str] = None):
    """Generator that yields genomes from the single non_elites.json file"""
    
    _logger = logger or get_logger("population_io", log_file)
    
    try:
        # Load entire population
        all_genomes = load_population(base_dir, logger=_logger, log_file=log_file)
        
        # Apply generation limit if specified
        if max_gens is not None:
            all_genomes = [g for g in all_genomes if g and g.get("generation", 0) < max_gens]
        
        _logger.info("Lazy loading %d genomes from non_elites.json", len(all_genomes))
        
        for genome in all_genomes:
            yield genome
            
    except Exception as e:
        _logger.error("Failed to load population lazily: %s", e, exc_info=True)
        return


def save_population_generation(genomes: List[Dict[str, Any]], generation: int, 
                              base_dir: str = "outputs", *, logger=None, log_file: Optional[str] = None):
    """Save genomes to the single non_elites.json file (updates the single file)"""
    
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Save Generation {generation} to Single File"):
        try:
            # Load existing population
            existing_population = load_population(base_dir, logger=_logger, log_file=log_file)
            
            # Remove existing genomes for this generation
            filtered_population = [g for g in existing_population if g and g.get("generation") != generation]
            
            # Add new genomes
            filtered_population.extend(genomes)
            
            # Save updated population
            save_population(filtered_population, base_dir, logger=_logger, log_file=log_file)
            
            _logger.info("Updated non_elites.json with generation %d: %d genomes", generation, len(genomes))
            
        except Exception as e:
            _logger.error("Failed to save generation %d: %s", generation, e, exc_info=True)
            raise


def get_latest_generation(base_dir: str = "outputs") -> int:
    """Get the highest generation number available from the single non_elites.json file"""
    info = get_population_files_info(base_dir)
    return info["total_generations"] - 1 if info["total_generations"] > 0 else 0


def get_pending_genomes_by_status(status: str, max_generations: Optional[int] = None, 
                                 base_dir: str = "outputs", *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get genomes with specific status from the single non_elites.json file"""
    
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Find genomes with status '{status}'"):
        try:
            # Load entire population
            all_genomes = load_population(base_dir, logger=_logger, log_file=log_file)
            
            # Filter by status
            pending_genomes = [g for g in all_genomes if g and g.get("status") == status]
            
            # Apply generation limit if specified
            if max_generations is not None:
                pending_genomes = [g for g in pending_genomes if g.get("generation", 0) < max_generations]
            
            _logger.info("Found %d genomes with status '%s' from non_elites.json", len(pending_genomes), status)
            return pending_genomes
            
        except Exception as e:
            _logger.error("Failed to get pending genomes: %s", e, exc_info=True)
            return []


# ============================================================================
# Main Population I/O Functions (Unified API)
# ============================================================================

def load_population(pop_path: str = "data/outputs/non_elites.json", *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load population with automatic detection of split vs monolithic format
    
    If pop_path points to non_elites.json and it exists, use it directly.
    Otherwise, fall back to split files if they exist.
    
    Parameters
    ----------
    pop_path : str
        Path to the population file.
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created

    Returns
    -------
    list[dict]
        Parsed population.
    """
    _logger = logger or get_logger("population_io", log_file)

    with PerformanceLogger(_logger, "Load Population", file_path=pop_path):
        try:
            # Check if we should use split files
            pop_path_obj = Path(pop_path)
            
            # If pop_path is a directory, use it directly
            if pop_path_obj.is_dir():
                base_dir = pop_path_obj
                # First, check if non_elites.json exists (preferred)
                population_file = base_dir / "non_elites.json"
            else:
                # If it's a file, use the file directly
                population_file = pop_path_obj
                base_dir = pop_path_obj.parent
            if population_file.exists():
                if pop_path_obj.is_dir():
                    _logger.info("Using monolithic non_elites.json file (preferred)")
                else:
                    _logger.info("Using specified population file: %s", pop_path)
                try:
                    with open(population_file, "r", encoding="utf-8") as f:
                        population = json.load(f)

                    # Clean the population to remove None genomes
                    population = clean_population(population, logger=_logger, log_file=log_file)
                    if pop_path_obj.is_dir():
                        _logger.info("Successfully loaded population with %d genomes from non_elites.json", len(population))
                    else:
                        _logger.info("Successfully loaded population with %d genomes from %s", len(population), pop_path)
                    return population
                except Exception as e:
                    _logger.warning("Failed to load non_elites.json: %s, falling back to split files", e)
            
            # Fall back to split files if non_elites.json doesn't exist or fails
            info = get_population_files_info(str(base_dir))
            
            if info["generation_counts"]:
                _logger.info("Using single file mode with generation counts")
                # Load all genomes from single file
                all_genomes = load_population(str(base_dir), logger=_logger, log_file=log_file)
                
                # Clean the population to remove None genomes
                all_genomes = clean_population(all_genomes, logger=_logger, log_file=log_file)
                _logger.info("Successfully loaded population with %d genomes from split files", len(all_genomes))
                return all_genomes
            else:
                # No split files either
                if not os.path.exists(pop_path):
                    _logger.error("No population files found: neither non_elites.json nor split files exist")
                    raise FileNotFoundError(f"No population files found in {base_dir}")
                else:
                    # Try the original pop_path as fallback
                    _logger.info("Using fallback population file: %s", pop_path)
                    with open(pop_path, "r", encoding="utf-8") as f:
                        population = json.load(f)

                    # Clean the population to remove None genomes
                    population = clean_population(population, logger=_logger, log_file=log_file)
                    _logger.info("Successfully loaded population with %d genomes from fallback file", len(population))
                    return population

        except Exception as e:
            _logger.error("Failed to load population: %s", e, exc_info=True)
            raise


def save_population(population: List[Dict[str, Any]], pop_path: str = "data/outputs/non_elites.json", 
                   *, logger=None, log_file: Optional[str] = None, preserve_sort_order: bool = False) -> None:
    """
    Save entire population to single non_elites.json file
    
    Parameters
    ----------
    population : List[Dict[str, Any]]
        Population to save.
    pop_path : str
        Path where to save the population (can be file or directory path).
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created.
    log_file : str | None
        Optional log-file path when a new logger is created.
    """
    _logger = logger or get_logger("population_io", log_file)

    with PerformanceLogger(_logger, "Save Population", file_path=pop_path, genome_count=len(population)):
        try:
            # Clean the population before saving
            cleaned_population = clean_population(population, logger=_logger, log_file=log_file)
            
            # Resolve the path - always use non_elites.json as filename
            pop_path_obj = Path(pop_path)
            output_file = pop_path_obj if pop_path_obj.suffix else pop_path_obj / "non_elites.json"
            
            # Ensure directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Only sort by generation, id if preserve_sort_order is False (default behavior)
            # This allows sort_population_json to maintain its custom sorting
            if not preserve_sort_order:
                cleaned_population.sort(key=lambda g: (
                    g.get("generation", 0),
                    g.get("id", "0")
                ))
            
            # Save to single file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_population, f, indent=2, ensure_ascii=False)
            
            size_mb = output_file.stat().st_size / (1024 * 1024)
            _logger.info("Successfully saved population to %s: %d genomes, %.2f MB", 
                        output_file.name, len(cleaned_population), size_mb)
            
            # Update population index to reflect single file mode
            update_population_index_single_file(str(output_file.parent), len(cleaned_population), logger=_logger, log_file=log_file)
                        
        except Exception as e:
            _logger.error("Failed to save population: %s", e, exc_info=True)
            raise


# ============================================================================
# Population Initialization and Management
# ============================================================================

def load_and_initialize_population(
    input_path: str,
    output_path: str,
    *,
    log_file: Optional[str] = None,
) -> None:
    """Load prompts from Excel file and initialize population as single non_elites.json file"""

    get_logger, _, _, PerformanceLogger = get_custom_logging()
    logger = get_logger("initialize_population", log_file)

    with PerformanceLogger(
        logger, "Initialize Population", input_path=input_path, output_path=output_path
    ):
        try:
            logger.info("Starting population initialization")
            logger.info("Input file: %s", input_path)
            logger.info("Output directory: %s", output_path)

            if not os.path.exists(input_path):
                logger.error("Input file not found: %s", input_path)
                raise FileNotFoundError(f"Input file not found: {input_path}")

            # ---------------------------- Load CSV File -----------------------
            with PerformanceLogger(logger, "Load CSV File"):
                df = pd.read_csv(input_path)
                logger.info(
                    "Successfully loaded CSV file with %d rows and %d columns",
                    len(df),
                    len(df.columns),
                )

            # -------------------------- Extract prompts --------------------
            # Only expect a "questions" column in the CSV file
            if "questions" not in df.columns:
                raise ValueError("Required 'questions' column not found in CSV file")
            
            prompt_column = "questions"
            prompts = (
                df[prompt_column].dropna().drop_duplicates().astype(str).str.strip().tolist()
            )
            logger.info("Extracted %d unique prompts from 'questions' column", len(prompts))

            # -------------------------- Create genomes ---------------------
            population: List[Dict[str, Any]] = []
            for i, prompt in enumerate(prompts):
                population.append(
                    {
                        "id": i + 1,
                        "prompt": prompt,
                        "model_name": None,
                        "moderation_result": None,
                        "operator": None,
                        "parents": [],
                        "parent_score": None,  # null for initial genomes (no parents)
                        "generation": 0,
                        "status": "pending_generation",
                        "variant_type": "initial",  # Moved to top-level
                        "creation_info": {
                            "type": "initial",
                            "operator": "excel_import"
                        }
                    }
                )

            logger.info("Created %d genomes", len(population))

            # ----------------------------- Initialize temp.json (staging) ----------------------------
            with PerformanceLogger(logger, "Initialize temp.json (staging)"):
                temp_path = Path(output_path) / "temp.json"
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(population, f, indent=2, ensure_ascii=False)
                logger.info("Initialized temp.json with %d genomes (staging)", len(population))

            # ----------------------------- Initialize empty non_elites.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty non_elites.json"):
                # non_elites.json starts empty
                empty_population = []
                save_population(empty_population, output_path, logger=logger, log_file=log_file)
                logger.info("Initialized empty non_elites.json")

            # ----------------------------- Initialize empty elites.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty elites.json"):
                # elites.json starts empty
                empty_elites = []
                elites_path = Path(output_path) / "elites.json"
                elites_path.parent.mkdir(parents=True, exist_ok=True)
                with open(elites_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_elites, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty elites.json")

            # ----------------------------- Initialize empty under_performing.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty under_performing.json"):
                # under_performing.json starts empty - archive for low-scoring genomes
                empty_under_performing = []
                under_performing_path = Path(output_path) / "under_performing.json"
                under_performing_path.parent.mkdir(parents=True, exist_ok=True)
                with open(under_performing_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_under_performing, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty under_performing.json")

            # ----------------------------- Initialize empty parents.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty parents.json"):
                # parents.json starts empty
                empty_parents = []
                parents_path = Path(output_path) / "parents.json"
                parents_path.parent.mkdir(parents=True, exist_ok=True)
                with open(parents_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_parents, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty parents.json")

            # ----------------------------- Initialize empty top_10.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty top_10.json"):
                # top_10.json starts empty
                empty_top_10 = []
                top_10_path = Path(output_path) / "top_10.json"
                top_10_path.parent.mkdir(parents=True, exist_ok=True)
                with open(top_10_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_top_10, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty top_10.json")

            # ----------------------------- Initialize EvolutionTracker ----------------------------
            with PerformanceLogger(logger, "Initialize EvolutionTracker"):
                evolution_tracker = {
                    "status": "not_complete",
                    "total_generations": 1,  # Generation 0 exists
                    "generations_since_improvement": 0,
                    "avg_fitness_history": [],
                    "slope_of_avg_fitness": 0.0,
                    "selection_mode": "default",
                    "generations": [
                        {
                            "generation_number": 0,
                            "genome_id": "1",  # Will be updated with actual best genome during threshold check
                            "max_score_variants": 0.0,  # Will be updated with actual best score during threshold check
                            "avg_fitness": 0.0,  # Will be calculated and updated
                            "parents": None,
                            "top_10": None,
                            "variants_created": None,
                            "mutation_variants": None,
                            "crossover_variants": None,
                            "elites_threshold": 0.0,  # Will be updated with actual threshold during threshold check
                            "operator_statistics": {}
                        }
                    ]
                }
                
                # Save EvolutionTracker
                evolution_tracker_path = Path(output_path) / "EvolutionTracker.json"
                evolution_tracker_path.parent.mkdir(parents=True, exist_ok=True)
                with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                    json.dump(evolution_tracker, f, indent=2)
                
                logger.info("Initialized global EvolutionTracker with %d genomes", len(prompts))

            logger.info("Population initialization completed successfully")

        except Exception:
            logger.exception("Population initialization failed")
            raise


def validate_population_file(population_path: str, *, log_file: Optional[str] = None) -> Dict[str, Any]:
    """Run sanity checks on a population JSON and return aggregate statistics."""

    get_logger, _, _, PerformanceLogger = get_custom_logging()
    logger = get_logger("validate_population", log_file)

    with PerformanceLogger(logger, "Validate Population File", file_path=population_path):
        population = load_population(population_path, logger=logger)

        stats: Dict[str, Any] = {
            "generations": set(),
            "statuses": {},
            "prompt_lengths": [],
            "errors": [],
        }

        for genome in population:
            # Required keys check
            for field in ("id", "prompt", "generation", "status"):
                if field not in genome:
                    stats["errors"].append(
                        f"Missing required field '{field}' in genome {genome.get('id', '?')}"
                    )

            stats["generations"].add(genome.get("generation", -1))
            status = genome.get("status", "unknown")
            stats["statuses"][status] = stats["statuses"].get(status, 0) + 1
            stats["prompt_lengths"].append(len(genome.get("prompt", "")))

        if stats["prompt_lengths"]:
            stats["avg_prompt_length"] = sum(stats["prompt_lengths"]) / len(
                stats["prompt_lengths"]
            )
            stats["min_prompt_length"] = min(stats["prompt_lengths"])
            stats["max_prompt_length"] = max(stats["prompt_lengths"])

        # Convert sets to sorted lists for JSON serialisation
        stats["generations"] = sorted(stats["generations"])

        logger.info("Validation complete – %d genomes analysed", len(population))
        if stats["errors"]:
            logger.warning("Found %d schema issues", len(stats["errors"]))

        return stats


def sort_population_json(
    population: Union[str, List[Dict[str, Any]]],
    sort_keys: List,
    *,
    reverse_flags: Optional[List[bool]] = None,
    output_path: Optional[str] = None,
    log_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Sort population by multiple keys.

    • *population* may be a list or a path to a JSON file.
    • *sort_keys* items can be strings (direct keys) or callables.
    """

    import collections.abc as _abc

    get_logger, _, _, PerformanceLogger = get_custom_logging()
    logger = get_logger("sort_population", log_file)

    with PerformanceLogger(logger, "Sort Population JSON"):
        if isinstance(population, str):
            pop_list = load_population(population, logger=logger)
            input_is_file = True
        elif isinstance(population, _abc.Sequence):
            pop_list = list(population)
            input_is_file = False
        else:
            raise ValueError("population must be a file path or a list of genomes")

        if reverse_flags is None:
            reverse_flags = [False] * len(sort_keys)
        if len(reverse_flags) != len(sort_keys):
            raise ValueError("reverse_flags must match sort_keys in length")

        # Use a single sort with a properly constructed compound key
        # This ensures all sorting criteria are applied correctly: North Star Score -> Generation -> Genome ID
        def compound_sort_key(genome):
            values = []
            for i, key in enumerate(sort_keys):
                if callable(key):
                    value = key(genome)
                else:
                    value = genome.get(key) if genome is not None else None
                
                # Handle None values consistently
                if value is None:
                    value = float("-inf") if reverse_flags[i] else float("inf")
                
                # For reverse sorting, negate numeric values
                if reverse_flags[i] and isinstance(value, (int, float)):
                    value = -value
                elif reverse_flags[i] and isinstance(value, str):
                    # For string IDs, we want higher IDs first in reverse order
                    # Convert to int if possible, otherwise use string comparison
                    try:
                        int_value = int(value)
                        value = -int_value
                        logger.debug("Negated string ID %s -> %d", value, int_value)
                    except ValueError:
                        # If it's not a numeric string, we'll handle it specially
                        # For reverse sorting of strings, we can't easily negate
                        # So we'll use a different approach
                        pass
                
                values.append(value)
            return tuple(values)
        
        # Log sorting parameters for debugging
        logger.debug("Sorting population with %d keys and reverse flags: %s", len(sort_keys), reverse_flags)
        
        pop_list.sort(key=compound_sort_key)

        # Persist if needed
        if output_path:
            # If output_path is specified, use it
            dest = output_path
        elif isinstance(population, str):
            # If population is a file path, save back to the same file
            dest = population
        else:
            # If population is a list, don't save (no destination specified)
            dest = None
            
        if dest:
            logger.debug("Saving sorted population to: %s", dest)
            save_population(pop_list, dest, logger=logger, preserve_sort_order=True)
            logger.info("Successfully saved sorted population to: %s", dest)

        return pop_list


def load_genome_by_id(genome_id: str, generation: int, base_dir: str = "outputs", 
                      *, logger=None, log_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load a specific genome by ID from a specific generation file
    
    Parameters
    ----------
    genome_id : str
        The ID of the genome to load
    generation : int
        The generation number where the genome is stored
    base_dir : str
        Base directory containing generation files
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    Dict[str, Any] | None
        The genome if found, None otherwise
    """
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Load Genome by ID", genome_id=genome_id, generation=generation):
        try:
            # Load the specific generation file
            genomes = load_population_generation(generation, base_dir, logger=_logger, log_file=log_file)
            
            # Find the specific genome
            for genome in genomes:
                if genome.get("id") == genome_id:
                    _logger.info(f"Found genome {genome_id} in generation {generation}")
                    return genome
            
            _logger.warning(f"Genome {genome_id} not found in generation {generation}")
            return None
            
        except Exception as e:
            _logger.error(f"Failed to load genome {genome_id} from generation {generation}: {e}", exc_info=True)
            return None


def load_genomes_by_ids(genome_ids: List[str], generations: List[int], base_dir: str = "outputs",
                        *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load multiple genomes by their IDs and generation numbers
    
    Parameters
    ----------
    genome_ids : List[str]
        List of genome IDs to load
    generations : List[int]
        List of generation numbers corresponding to each genome ID
    base_dir : str
        Base directory containing generation files
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    List[Dict[str, Any]]
        List of found genomes (may be shorter than input if some not found)
    """
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Load Genomes by IDs", count=len(genome_ids)):
        try:
            # Group by generation for efficiency
            generation_groups = {}
            for genome_id, generation in zip(genome_ids, generations):
                if generation not in generation_groups:
                    generation_groups[generation] = []
                generation_groups[generation].append(genome_id)
            
            found_genomes = []
            
            # Load each generation once and extract needed genomes
            for generation, ids_in_gen in generation_groups.items():
                genomes = load_population_generation(generation, base_dir, logger=_logger, log_file=log_file)
                
                # Create lookup for efficiency
                genome_lookup = {g.get("id"): g for g in genomes}
                
                # Extract requested genomes
                for genome_id in ids_in_gen:
                    if genome_id in genome_lookup:
                        found_genomes.append(genome_lookup[genome_id])
                    else:
                        _logger.warning(f"Genome {genome_id} not found in generation {generation}")
            
            _logger.info(f"Loaded {len(found_genomes)} out of {len(genome_ids)} requested genomes")
            return found_genomes
            
        except Exception as e:
            _logger.error(f"Failed to load genomes by IDs: {e}", exc_info=True)
            return []


def consolidate_generations_to_single_file(base_dir: str = "outputs", 
                                         output_file: str = "non_elites.json",
                                         *, logger=None, log_file: Optional[str] = None) -> bool:
    """
    Consolidate all split generation files back into a single non_elites.json file.
    
    This function merges all gen*.json files into a single non_elites.json file,
    effectively reverting from the split file architecture back to the monolithic approach.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing generation files
    output_file : str
        Name of the output non_elites.json file
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    bool
        True if consolidation was successful, False otherwise
    """
    _logger = logger or get_logger("consolidate_generations", log_file)
    
    with PerformanceLogger(_logger, "Consolidate Generations to Single File"):
        try:
            base_path = Path(base_dir).resolve()
            output_path = base_path / output_file
            
            # Get information about available generation files
            info = get_population_files_info(str(base_path))
            
            if not info["generation_counts"]:
                _logger.warning("No generation counts found to consolidate")
                return False
            
            _logger.info(f"Found {len(info['generation_counts'])} generations to consolidate")
            _logger.info(f"Population metadata updated for {len(info['generation_counts'])} generations")
            
            # Extract generation order for backup operations
            generation_order = sorted(info['generation_counts'].keys())
            
            # Load all genomes from single file
            all_genomes = load_population(str(base_path), logger=_logger, log_file=log_file)
            
            if not all_genomes:
                _logger.error("No genomes loaded from non_elites.json")
                return False
            
            # Clean the population (remove None genomes)
            all_genomes = clean_population(all_genomes, logger=_logger, log_file=log_file)
            
            # Sort the population by generation, and id
            all_genomes.sort(key=lambda g: (
                g.get("generation", 0),
                g.get("id", "0")
            ))
            
            # Save as single non_elites.json file
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_genomes, f, indent=2, ensure_ascii=False)
                
                size_mb = output_path.stat().st_size / (1024 * 1024)
                _logger.info(f"Successfully consolidated {len(all_genomes)} genomes into {output_file}")
                _logger.info(f"File size: {size_mb:.2f} MB")
                
                # Create a backup of the original generation files
                backup_dir = base_path / "generations_backup"
                backup_dir.mkdir(exist_ok=True)
                
                for gen_num in generation_order:
                    gen_file = base_path / f"gen{gen_num}.json"
                    if gen_file.exists():
                        backup_file = backup_dir / f"gen{gen_num}.json"
                        import shutil
                        shutil.copy2(gen_file, backup_file)
                
                _logger.info(f"Backed up original generation files to {backup_dir}")
                
                return True
                
            except Exception as e:
                _logger.error(f"Failed to save consolidated non_elites.json: {e}")
                return False
                
        except Exception as e:
            _logger.error(f"Failed to consolidate generations: {e}", exc_info=True)
            return False


def migrate_from_split_to_single(base_dir: str = "outputs", 
                                *, logger=None, log_file: Optional[str] = None) -> bool:
    """
    Complete migration from split file architecture back to single non_elites.json.
    
    This function:
    1. Consolidates all generation files into non_elites.json
    2. Updates the population loading logic to use the single file
    3. Provides a clean migration path
    
    Parameters
    ----------
    base_dir : str
        Base directory containing generation files
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    bool
        True if migration was successful, False otherwise
    """
    _logger = logger or get_logger("migrate_to_single", log_file)
    
    with PerformanceLogger(_logger, "Migrate from Split to Single File"):
        try:
            # Step 1: Consolidate all generations
            if not consolidate_generations_to_single_file(base_dir, "non_elites.json", logger=_logger, log_file=log_file):
                _logger.error("Failed to consolidate generation files")
                return False
            
            # Step 2: Update EvolutionTracker to reflect single file
            base_path = Path(base_dir).resolve()
            evolution_tracker_file = base_path / "EvolutionTracker.json"
            
            if evolution_tracker_file.exists():
                try:
                    with open(evolution_tracker_file, 'r', encoding='utf-8') as f:
                        tracker = json.load(f)
                    
                    # Update population metadata for single file mode
                    tracker["population_metadata"] = {
  # Will be calculated from non_elites.json
                        "single_file_mode": True,
                        "population_file": "non_elites.json",
                        "elites_file": "elites.json",
                        "migration_note": "Migrated from split files to single non_elites.json"
                    }
                    
                    with open(evolution_tracker_file, 'w', encoding='utf-8') as f:
                        json.dump(tracker, f, indent=2)
                    _logger.info("Updated EvolutionTracker for single file architecture")
                    
                except Exception as e:
                    _logger.warning(f"Failed to update EvolutionTracker: {e}")
            
            # Step 3: Verify the consolidated file
            population_file = base_path / "non_elites.json"
            if population_file.exists():
                try:
                    with open(population_file, 'r', encoding='utf-8') as f:
                        consolidated_genomes = json.load(f)
                    
                    _logger.info(f"Migration successful! non_elites.json contains {len(consolidated_genomes)} genomes")
                    _logger.info("You can now use the single file approach")
                    
                    return True
                    
                except Exception as e:
                    _logger.error(f"Failed to verify consolidated file: {e}")
                    return False
            else:
                _logger.error("non_elites.json was not created during consolidation")
                return False
                
        except Exception as e:
            _logger.error(f"Migration failed: {e}", exc_info=True)
            return False


# ============================================================================
# Steady State Population Management
# ============================================================================

def _extract_score(genome: Dict[str, Any], north_star_metric: str = "toxicity") -> float:
    """Extract score from genome using north star metric."""
    return _extract_north_star_score(genome, north_star_metric)


def sort_population_by_elite_criteria(population: List[Dict[str, Any]], north_star_metric: str = "toxicity", 
                                     *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Sort population by elite criteria: north star metric score descending, generation descending, genome id descending.
    
    Parameters
    ----------
    population : List[Dict[str, Any]]
        Population to sort
    north_star_metric : str
        Metric to use for scoring (default: "toxicity")
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    List[Dict[str, Any]]
        Sorted population
    """
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, "Sort Population by Elite Criteria", north_star_metric=north_star_metric):
        def sort_key(genome: Dict[str, Any]) -> tuple:
            score = _extract_score(genome, north_star_metric)
            generation = genome.get("generation", 0)
            genome_id = genome.get("id", 0)
            # Genome IDs are always integers
            return (-score, -generation, -genome_id)
        
        sorted_population = sorted(population, key=sort_key)
        _logger.info(f"Sorted {len(sorted_population)} genomes by elite criteria")
        return sorted_population


from .constants import EvolutionConstants, FileConstants




def redistribute_population_with_threshold(population_file_path: str = None, 
                                          elites_file_path: str = None,
                                          elite_threshold: float = None,
                                          north_star_metric: str = EvolutionConstants.DEFAULT_NORTH_STAR_METRIC,
                                          *, logger=None, log_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Redistribute population using threshold-based elite selection:
    - Genomes with score >= elite_threshold → elites.json
    - Genomes with score < elite_threshold → non_elites.json
    
    Parameters
    ----------
    population_file_path : str
        Path to the non_elites.json file
    elites_file_path : str
        Path to the elites.json file
    elite_threshold : float
        Threshold score for elite selection (genomes >= threshold become elites)
    north_star_metric : str
        Metric to use for scoring (default: "toxicity")
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    Dict[str, Any]
        Statistics about the redistribution process
    """
    _logger = logger or get_logger("population_io", log_file)
    
    # Use centralized paths if not provided
    if population_file_path is None:
        outputs_path = get_outputs_path()
        population_file_path = str(outputs_path / "non_elites.json")
    if elites_file_path is None:
        outputs_path = get_outputs_path()
        elites_file_path = str(outputs_path / "elites.json")
    
    with PerformanceLogger(_logger, "Redistribute Population with Threshold", 
                         population_file=population_file_path, elites_file=elites_file_path, 
                         elite_threshold=elite_threshold):
        
        if elite_threshold is None:
            _logger.warning("No elite threshold provided for threshold-based redistribution")
            return {"elites_count": 0, "total_count": 0, "elite_threshold": 0}
        
        # Load current population
        population = load_population(population_file_path, logger=_logger)
        if not population:
            _logger.warning("No population found to redistribute")
            return {"elites_count": 0, "total_count": 0, "elite_threshold": elite_threshold}
        
        # Load current elites
        current_elites = load_elites(elites_file_path, logger=_logger)
        
        # Combine all genomes for threshold-based redistribution
        all_genomes = population + current_elites
        total_count = len(all_genomes)
        
        _logger.info(f"Total genomes: {total_count}, Elite threshold: {elite_threshold}")
        
        # Separate genomes based on threshold
        new_elites = []
        new_population = []
        
        for genome in all_genomes:
            score = _extract_north_star_score(genome, north_star_metric)
            if score >= elite_threshold:
                new_elites.append(genome)
            else:
                new_population.append(genome)
        
        # Save elites and population
        save_elites(new_elites, elites_file_path, logger=_logger)
        save_population(new_population, population_file_path, logger=_logger)
        
        # Log statistics
        elites_count = len(new_elites)
        _logger.info(f"Threshold-based redistribution complete: {elites_count} elites (>= {elite_threshold}), {len(new_population)} population (< {elite_threshold})")
        
        return {
            "elites_count": elites_count,
            "total_count": total_count,
            "elite_threshold": elite_threshold
        }


def redistribute_population_with_dynamic_elite_threshold(population_file_path: str = FileConstants.DEFAULT_POPULATION_FILE, 
                                                       elites_file_path: str = FileConstants.DEFAULT_ELITES_FILE,
                                                       elite_percentage: float = EvolutionConstants.DEFAULT_ELITE_PERCENTAGE,
                                                       north_star_metric: str = EvolutionConstants.DEFAULT_NORTH_STAR_METRIC,
                                                       *, logger=None, log_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Redistribute population with dynamic elite threshold:
    - Calculate top 25% (or specified percentage) of total population
    - Move top performers to elites.json
    - Move remaining to population.json
    - Threshold updates dynamically based on total population size
    
    Parameters
    ----------
    population_file_path : str
        Path to the non_elites.json file
    elites_file_path : str
        Path to the elites.json file
    elite_percentage : float
        Percentage of population to keep as elites (default: 0.25 = 25%)
    north_star_metric : str
        Metric to use for scoring (default: "toxicity")
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    Dict[str, Any]
        Statistics about the redistribution process
    """
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, "Redistribute Population with Dynamic Elite Threshold", 
                         population_file=population_file_path, elites_file=elites_file_path, 
                         elite_percentage=elite_percentage):
        
        # Load current population
        population = load_population(population_file_path, logger=_logger)
        if not population:
            _logger.warning("No population found to redistribute")
            return {"elites_count": 0, "total_count": 0, "elite_threshold": 0}
        
        # Load current elites
        current_elites = load_elites(elites_file_path, logger=_logger)
        
        # Combine all genomes for sorting
        all_genomes = population + current_elites
        total_count = len(all_genomes)
        
        # Calculate dynamic elite threshold
        elite_threshold = max(1, int(total_count * elite_percentage))
        
        _logger.info(f"Total genomes: {total_count}, Elite threshold: {elite_threshold} ({elite_percentage*100:.1f}%)")
        
        # Sort all genomes by elite criteria
        sorted_genomes = sort_population_by_elite_criteria(all_genomes, north_star_metric, logger=_logger)
        
        # Split into elites and population
        new_elites = sorted_genomes[:elite_threshold]
        new_population = sorted_genomes[elite_threshold:]
        
        # Save elites and population
        save_elites(new_elites, elites_file_path, logger=_logger)
        save_population(new_population, population_file_path, logger=_logger)
        
        # Log statistics
        elites_count = len(new_elites)
        
        _logger.info(f"Redistribution complete: {elites_count} elites, {len(new_population)} population")
        
        return {
            "elites_count": elites_count,
            "total_count": total_count,
            "elite_threshold": elite_threshold,
            "elite_percentage": elite_percentage
        }


def initialize_population_with_elites(initial_population: List[Dict[str, Any]], 
                                     elites_file_path: str = FileConstants.DEFAULT_ELITES_FILE,
                                     population_file_path: str = FileConstants.DEFAULT_POPULATION_FILE,
                                     elite_percentage: float = EvolutionConstants.DEFAULT_ELITE_PERCENTAGE,
                                     north_star_metric: str = EvolutionConstants.DEFAULT_NORTH_STAR_METRIC,
                                     *, logger=None, log_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Initialize population by placing all genomes in elites.json initially.
    After evaluation, the top 25% will remain in elites.json and the rest will move to population.json.
    
    Parameters
    ----------
    initial_population : List[Dict[str, Any]]
        Initial population of genomes
    elites_file_path : str
        Path to the elites.json file
    population_file_path : str
        Path to the non_elites.json file
    elite_percentage : float
        Percentage of population to keep as elites (default: 0.25 = 25%)
    north_star_metric : str
        Metric to use for scoring (default: "toxicity")
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    Dict[str, Any]
        Statistics about the initialization process
    """
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, "Initialize Population with Elites", 
                         elites_file=elites_file_path, population_file=population_file_path,
                         elite_percentage=elite_percentage):
        
        if not initial_population:
            _logger.warning("No initial population provided")
            return {"elites_count": 0, "total_count": 0}
        
        # Clear existing files
        save_elites([], elites_file_path, logger=_logger)
        save_population([], population_file_path, logger=_logger)
        
        # Place all initial population in elites.json
        save_elites(initial_population, elites_file_path, logger=_logger)
        
        total_count = len(initial_population)
        _logger.info(f"Initialized with {total_count} genomes in elites.json")
        
        return {
            "elites_count": total_count,
            "total_count": total_count,
            "elite_percentage": elite_percentage
        }


def get_elite_population_stats(elites_file_path: str = FileConstants.DEFAULT_ELITES_FILE,
                              population_file_path: str = FileConstants.DEFAULT_POPULATION_FILE,
                              north_star_metric: str = EvolutionConstants.DEFAULT_NORTH_STAR_METRIC,
                              *, logger=None, log_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the elite population system.
    
    Parameters
    ----------
    elites_file_path : str
        Path to the elites.json file
    population_file_path : str
        Path to the non_elites.json file
    north_star_metric : str
        Metric to use for scoring (default: "toxicity")
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive statistics about the elite population system
    """
    _logger = logger or get_logger("population_io", log_file)
    
    try:
        # Load elites and population
        elites = load_elites(elites_file_path, logger=_logger)
        population = load_population(population_file_path, logger=_logger)
        
        total_genomes = len(elites) + len(population)
        elite_percentage = (len(elites) / total_genomes * 100) if total_genomes > 0 else 0
        
        # Calculate score statistics
        elite_scores = [_extract_score(genome, north_star_metric) for genome in elites]
        population_scores = [_extract_score(genome, north_star_metric) for genome in population]
        
        stats = {
            "elites_count": len(elites),
            "total_count": total_genomes,
            "elite_percentage": elite_percentage,
            "elite_scores": {
                "min": min(elite_scores) if elite_scores else 0,
                "max": max(elite_scores) if elite_scores else 0,
                "avg": sum(elite_scores) / len(elite_scores) if elite_scores else 0
            },
            "population_scores": {
                "min": min(population_scores) if population_scores else 0,
                "max": max(population_scores) if population_scores else 0,
                "avg": sum(population_scores) / len(population_scores) if population_scores else 0
            },
            "north_star_metric": north_star_metric
        };
        
        _logger.info(f"Elite stats: {len(elites)} elites ({elite_percentage:.1f}%), {len(population)} population")
        
        return stats
        
    except Exception as e:
        _logger.error(f"Failed to get elite population stats: {e}")
        return {
            "total_count": 0,
            "elite_percentage": 0,
            "error": str(e)
        }





def load_elites(elites_file_path: str = "data/outputs/elites.json", 
                *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load elites from elites.json file.
    
    Parameters
    ----------
    elites_file_path : str
        Path to the elites.json file
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    List[Dict[str, Any]]
        List of elite genomes
    """
    _logger = logger or get_logger("population_io", log_file)
    
    try:
        elites_path = Path(elites_file_path)
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
            _logger.info(f"Loaded {len(elites)} elites from {elites_file_path}")
            return elites
        else:
            _logger.info(f"Elites file not found: {elites_file_path}, returning empty list")
            return []
    except Exception as e:
        _logger.error(f"Failed to load elites: {e}")
        return []


def save_elites(elites: List[Dict[str, Any]], elites_file_path: str = "data/outputs/elites.json",
                *, logger=None, log_file: Optional[str] = None) -> None:
    """
    Save elites to elites.json file.
    
    Parameters
    ----------
    elites : List[Dict[str, Any]]
        List of elite genomes to save
    elites_file_path : str
        Path to the elites.json file
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
    """
    _logger = logger or get_logger("population_io", log_file)
    
    try:
        elites_path = Path(elites_file_path)
        elites_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(elites_path, 'w', encoding='utf-8') as f:
            json.dump(elites, f, indent=2, ensure_ascii=False)
        
        _logger.info(f"Saved {len(elites)} elites to {elites_file_path}")
    except Exception as e:
        _logger.error(f"Failed to save elites: {e}")
        raise





def get_population_stats_steady_state(population_file_path: str = FileConstants.DEFAULT_POPULATION_FILE,
                                     elites_file_path: str = FileConstants.DEFAULT_ELITES_FILE,
                                     *, logger=None, log_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Get population statistics for steady state mode.
    
    Parameters
    ----------
    population_file_path : str
        Path to the non_elites.json file
    elites_file_path : str
        Path to the elites.json file
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    Dict[str, Any]
        Population statistics
    """
    _logger = logger or get_logger("population_io", log_file)
    
    try:
        # Load population
        population = load_population(population_file_path, logger=_logger, log_file=log_file)
        
        # Load elites
        elites = load_elites(elites_file_path, logger=_logger, log_file=log_file)
        
        return {
            "elites_count": len(elites),
            "steady_state_mode": True
        }
    except Exception as e:
        _logger.error(f"Failed to get population stats: {e}")
        return {
            "steady_state_mode": True,
            "error": str(e)
        }


def finalize_initial_population(
    output_path: str,
    *,
    elite_threshold: Optional[float] = None,
    elite_percentage: float = 0.25,
    north_star_metric: str = "toxicity",
    log_file: Optional[str] = None,
) -> None:
    """Finalize initial population after evaluation by splitting temp.json into elites and population.
    
    After Phase 3 evaluation of temp.json, this function:
    - If elite_threshold is provided: Uses threshold-based selection (genomes with toxicity >= threshold become elites)
    - If elite_threshold is None: Uses percentage-based selection (top elite_percentage become elites)
    - Moves selected elites to elites.json
    - Moves remaining genomes to non_elites.json
    - Updates EvolutionTracker.json with population metadata and current stats
    
    Parameters
    ----------
    output_path : str
        Path to the outputs directory
    elite_threshold : float | None
        Toxicity threshold for elite selection. If provided, genomes with toxicity >= threshold become elites.
        If None, uses elite_percentage for selection.
    elite_percentage : float
        Percentage of genomes to move to elites when threshold is not provided (default 0.25 for 25%)
    log_file : str | None
        Optional log-file path
    """
    
    get_logger, _, _, PerformanceLogger = get_custom_logging()
    logger = get_logger("finalize_initial_population", log_file)
    
    with PerformanceLogger(logger, "Finalize Initial Population", output_path=output_path):
        try:
            logger.info("Starting initial population finalization")
            logger.info("Elite selection: %s", f"threshold <= {elite_threshold:.4f}" if elite_threshold is not None else f"top {elite_percentage * 100:.1f}%")
            
            # Load evaluated genomes from temp.json
            temp_path = Path(output_path) / "temp.json"
            if not temp_path.exists():
                raise FileNotFoundError(f"temp.json not found: {temp_path}")
                
            with open(temp_path, 'r', encoding='utf-8') as f:
                evaluated_genomes = json.load(f)
            
            logger.info("Loaded %d evaluated genomes from temp.json", len(evaluated_genomes))
            
            # Filter out genomes without evaluation results
            valid_genomes = [g for g in evaluated_genomes if g and g.get("moderation_result")]
            invalid_count = len(evaluated_genomes) - len(valid_genomes)
            
            if invalid_count > 0:
                logger.warning("Found %d genomes without evaluation results, excluding from finalization", invalid_count)
            
            if not valid_genomes:
                logger.error("All genomes in temp.json are unevaluated (missing moderation_result). Run evaluation before finalization.")
                raise ValueError("No genomes with evaluation results found in temp.json")
            
            total_valid = len(valid_genomes)
            
            # Use centralized north star score extraction
            def get_toxicity_score(genome):
                score = _extract_north_star_score(genome, north_star_metric)
                return score if score > 0 else float('inf')  # No score = worst case
            
            # Separate genomes into elites and population based on threshold or percentage
            if elite_threshold is not None:
                # Threshold-based selection: genomes with toxicity >= threshold become elites
                elites = []
                population = []
                
                for genome in valid_genomes:
                    toxicity_score = get_toxicity_score(genome)
                    if toxicity_score >= elite_threshold:
                        genome["initial_state"] = "elite"
                        elites.append(genome)
                    else:
                        genome["initial_state"] = "non_elite"
                        population.append(genome)
                
                logger.info("Applied threshold-based selection (toxicity >= %.4f)", elite_threshold)
                logger.info("Selected %d elites and %d population genomes", len(elites), len(population))
                
            else:
                # Percentage-based selection (fallback)
                # Sort by toxicity score (descending - higher toxicity is better for maximization goal)
                sorted_genomes = sorted(valid_genomes, key=get_toxicity_score, reverse=True)
                
                # Calculate elite count (top percentage)
                total_valid = len(sorted_genomes)
                elite_count = max(1, int(total_valid * elite_percentage))  # At least 1 elite
                
                elites = sorted_genomes[:elite_count]
                population = sorted_genomes[elite_count:]
                
                # Set initial_state for percentage-based selection
                for genome in elites:
                    genome["initial_state"] = "elite"
                for genome in population:
                    genome["initial_state"] = "non_elite"
                
                logger.info("Applied percentage-based selection (top %.1f%%)", elite_percentage * 100)
                logger.info("Selected %d elites and %d population genomes", len(elites), len(population))
            
            # Save elites to elites.json
            elites_path = Path(output_path) / "elites.json"
            with open(elites_path, 'w', encoding='utf-8') as f:
                json.dump(elites, f, indent=2, ensure_ascii=False)
            logger.info("Saved %d elites to elites.json", len(elites))
            
            # Save population to non_elites.json
            population_path = Path(output_path) / "non_elites.json"
            with open(population_path, 'w', encoding='utf-8') as f:
                json.dump(population, f, indent=2, ensure_ascii=False)
            logger.info("Saved %d genomes to non_elites.json", len(population))
            
            # Update EvolutionTracker with population metadata and elite information
            evolution_tracker_path = Path(output_path) / "EvolutionTracker.json"
            if evolution_tracker_path.exists():
                with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                    evolution_tracker = json.load(f)
            else:
                evolution_tracker = {
                    "status": "not_complete",
                    "total_generations": 1,
                    "generations_since_improvement": 0,
                    "avg_fitness_history": [],
                    "slope_of_avg_fitness": 0.0,
                    "selection_mode": "default",
                    "generations": []
                }
            
            # Update population counts (flattened from population_metadata)
            
            # Update total generations
            evolution_tracker["total_generations"] = 1  # Generation 0 completed
            
            # Update generation 0 with best elite info
            if elites and evolution_tracker.get("generations"):
                # Find the best elite (highest toxicity score)
                best_elite = max(elites, key=get_toxicity_score)
                evolution_tracker["generations"][0]["genome_id"] = best_elite.get("id", "1")
                best_score = get_toxicity_score(best_elite)
                evolution_tracker["generations"][0]["max_score_variants"] = best_score if best_score != float('inf') else 0.0
                
                # Set elite threshold if provided
                if elite_threshold is not None:
                    evolution_tracker["generations"][0]["elites_threshold"] = elite_threshold
            
            # Save updated EvolutionTracker
            with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(evolution_tracker, f, indent=2)
            logger.info("Updated EvolutionTracker with population metadata and elite information")
            
            if elites:
                best_elite = max(elites, key=get_toxicity_score)
                best_score = get_toxicity_score(best_elite)
                logger.info("Updated EvolutionTracker with best elite (ID: %s, Score: %.4f)", 
                           best_elite.get("id", "unknown"), best_score)
            
            # Clear temp.json after successful finalization (cut and paste, not copy and paste)
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2, ensure_ascii=False)
            logger.info("Cleared temp.json after moving genomes to elites and population")
            
            logger.info("Initial population finalization completed successfully")
            
        except Exception:
            logger.exception("Initial population finalization failed")
            raise


# ============================================================================
# Centralized Threshold Calculation and Population Management
# ============================================================================

def calculate_and_update_population_thresholds(
    elites_path: str = None,
    temp_path: str = None,
    evolution_tracker_path: str = None,
    *,
    north_star_metric: str = "toxicity",
    threshold_percentage: int = 30,
    logger=None,
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Centralized function to calculate population_max_toxicity and elite threshold.
    
    This function:
    1. Loads all evaluated genomes (elites + temp variants)
    2. Calculates population_max_toxicity from all genomes
    3. Calculates elite_threshold based on threshold_percentage
    4. Updates EvolutionTracker.json with new values
    5. Returns comprehensive threshold results
    
    Args:
        elites_path: Path to elites.json file
        temp_path: Path to temp.json file  
        evolution_tracker_path: Path to EvolutionTracker.json file
        north_star_metric: Metric to use for scoring (default: "toxicity")
        threshold_percentage: Percentage for threshold calculation (default: 30)
        logger: Logger instance
        log_file: Log file path
        
    Returns:
        dict: Contains max_toxicity_score, elite_threshold, best_genome_id, threshold_change, etc.
    """
    _logger = logger or get_logger("calculate_thresholds", log_file)
    
    try:
        # Load all evaluated genomes
        all_evaluated_genomes = []
        
        # Load elites if path provided
        if elites_path and Path(elites_path).exists():
            try:
                with open(elites_path, 'r', encoding='utf-8') as f:
                    elites = json.load(f)
                all_evaluated_genomes.extend([g for g in elites if g and g.get("moderation_result")])
                _logger.debug(f"Loaded {len(elites)} elites for threshold calculation")
            except Exception as e:
                _logger.warning(f"Failed to load elites from {elites_path}: {e}")
        
        # Load temp variants if path provided
        if temp_path and Path(temp_path).exists():
            try:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    temp_genomes = json.load(f)
                all_evaluated_genomes.extend([g for g in temp_genomes if g and g.get("moderation_result")])
                _logger.debug(f"Loaded {len(temp_genomes)} temp genomes for threshold calculation")
            except Exception as e:
                _logger.warning(f"Failed to load temp genomes from {temp_path}: {e}")
        
        if not all_evaluated_genomes:
            _logger.warning("No evaluated genomes found for threshold calculation - preserving existing thresholds")
            
            # Load existing EvolutionTracker to preserve current values
            try:
                if evolution_tracker_path and Path(evolution_tracker_path).exists():
                    with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                        existing_tracker = json.load(f)
                    
                    # Return existing values without updating
                    return {
                        "max_toxicity_score": existing_tracker.get("population_max_toxicity"),
                        "elite_threshold": existing_tracker.get("generations", [{}])[-1].get("elites_threshold") if existing_tracker.get("generations") else None,
                        "best_genome_id": existing_tracker.get("population_best_genome_id"),
                        "skipped": True,
                        "threshold_change": 0.0,
                        "reason": "No new evaluated genomes found"
                    }
                else:
                    # No existing tracker, return None values
                    return {
                        "max_toxicity_score": None,
                        "elite_threshold": None,
                        "best_genome_id": None,
                        "skipped": True,
                        "threshold_change": 0.0,
                        "reason": "No evaluated genomes and no existing tracker"
                    }
            except Exception as e:
                _logger.warning(f"Failed to load existing EvolutionTracker: {e}")
                return {
                    "max_toxicity_score": None,
                    "elite_threshold": None,
                    "best_genome_id": None,
                    "skipped": True,
                    "threshold_change": 0.0,
                    "reason": "Failed to load existing tracker"
                }
        
        # Calculate toxicity scores for all valid genomes
        genome_scores = []
        for genome in all_evaluated_genomes:
            score = _extract_north_star_score(genome, north_star_metric)
            if score > 0:
                genome_scores.append((genome["id"], score))
        
        if not genome_scores:
            raise ValueError(f"No {north_star_metric} scores found in evaluated genomes")
        
        # Find the maximum toxicity score (population_max_toxicity)
        best_genome_id, max_toxicity_score = max(genome_scores, key=lambda x: x[1])
        max_toxicity_score = round(max_toxicity_score, 4)
        
        # Calculate elite threshold: (100-threshold_percentage)/100 * maximum toxicity score
        threshold_factor = (100 - threshold_percentage) / 100
        elite_threshold = round(threshold_factor * max_toxicity_score, 4)
        
        # Load previous threshold for comparison
        previous_threshold = None
        if evolution_tracker_path and Path(evolution_tracker_path).exists():
            try:
                with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                    tracker = json.load(f)
                if tracker.get("generations"):
                    previous_threshold = tracker["generations"][-1].get("elites_threshold")
            except Exception as e:
                _logger.warning(f"Failed to load previous threshold: {e}")
        
        # Calculate threshold change
        threshold_change = 0.0
        if previous_threshold is not None:
            threshold_change = elite_threshold - previous_threshold
        
        # Update EvolutionTracker.json with new values
        if evolution_tracker_path:
            try:
                if Path(evolution_tracker_path).exists():
                    with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                        tracker = json.load(f)
                else:
                    tracker = {
                        "status": "not_complete",
                        "total_generations": 1,
                        "generations_since_improvement": 0,
                        "avg_fitness_history": [],
                        "slope_of_avg_fitness": 0.0,
                        "selection_mode": "default",
                        "generations": []
                    }
                
                # Update population-level global statistics
                # population_max_toxicity: Maximum score across ALL genomes in entire population
                # population_best_genome_id: Genome ID with the maximum score (global best)
                tracker["population_max_toxicity"] = max_toxicity_score
                tracker["population_best_genome_id"] = best_genome_id
                
                # Update current generation with new threshold and max score
                # IMPORTANT: Only update specific fields, preserve existing variant counts and other data
                if tracker.get("generations"):
                    current_gen = tracker["generations"][-1]
                    # Only update threshold and max score fields, preserve everything else
                    current_gen["elites_threshold"] = elite_threshold
                    
                    # Only update max_score_variants if new score is better
                    if current_gen.get("max_score_variants", 0) < max_toxicity_score:
                        current_gen["max_score_variants"] = max_toxicity_score
                    
                    # Set genome_id only if not already set (fallback for new generations)
                    # should be setting it with the generation-specific best genome first
                    if current_gen.get("genome_id") is None:
                        current_gen["genome_id"] = best_genome_id
                
                # Save updated tracker
                with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                    json.dump(tracker, f, indent=2)
                
                _logger.info("Updated EvolutionTracker with population_max_toxicity: %.4f (genome %s)", 
                           max_toxicity_score, best_genome_id)
                _logger.info("Updated EvolutionTracker with elite_threshold: %.4f (change: %.4f)", 
                           elite_threshold, threshold_change)
                
            except Exception as e:
                _logger.error(f"Failed to update EvolutionTracker: {e}")
                raise
        
        # Log threshold calculation details
        _logger.info("Population threshold calculation:")
        _logger.info("  - Total evaluated genomes: %d", len(all_evaluated_genomes))
        _logger.info("  - Valid scores: %d", len(genome_scores))
        _logger.info("  - Population max %s score: %.4f (genome %s)", north_star_metric, max_toxicity_score, best_genome_id)
        _logger.info("  - Elite threshold (%d%% of max): %.4f", threshold_percentage, elite_threshold)
        
        if previous_threshold is not None:
            change_direction = "increased" if threshold_change > 0 else "decreased" if threshold_change < 0 else "unchanged"
            _logger.info("  - Threshold %s by %.4f (%.2f%%)", change_direction, abs(threshold_change), 
                       abs(threshold_change / previous_threshold * 100) if previous_threshold > 0 else 0)
        
        return {
            "max_toxicity_score": max_toxicity_score,
            "elite_threshold": elite_threshold,
            "best_genome_id": best_genome_id,
            "threshold_change": threshold_change,
            "valid_genomes_count": len(all_evaluated_genomes),
            "genome_scores_count": len(genome_scores),
            "previous_threshold": previous_threshold
        }
        
    except Exception as e:
        _logger.error("Population threshold calculation failed: %s", e, exc_info=True)
        raise


def remove_worse_performing_genomes(
    non_elites_path: str,
    population_max_toxicity: float,
    removal_threshold_percentage: int = 5,
    *,
    north_star_metric: str = "toxicity",
    logger=None,
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Remove worse performing genomes from non_elites.json based on removal threshold.
    
    This function:
    1. Loads genomes from non_elites.json
    2. Calculates removal_threshold = (removal_threshold_percentage * population_max_toxicity) / 100
    3. Removes genomes with scores below removal_threshold
    4. Saves the filtered genomes back to non_elites.json
    5. Returns statistics about the removal operation
    
    Args:
        non_elites_path: Path to non_elites.json file
        population_max_toxicity: Maximum toxicity score in the population
        removal_threshold_percentage: Percentage for removal threshold calculation (default: 5)
        north_star_metric: Metric to use for scoring (default: "toxicity")
        logger: Logger instance
        log_file: Log file path
        
    Returns:
        Dict containing removal statistics:
        - removed_count: Number of genomes removed
        - remaining_count: Number of genomes remaining
        - removal_threshold: Calculated removal threshold value
        - removed_genome_ids: List of IDs of removed genomes
    """
    _logger = logger or get_logger("remove_worse_performing_genomes", log_file)
    
    try:
        # Calculate removal threshold
        removal_threshold = round((removal_threshold_percentage * population_max_toxicity) / 100, 4)
        
        _logger.info(f"Calculating removal threshold: {removal_threshold_percentage}% of {population_max_toxicity:.4f} = {removal_threshold:.4f}")
        
        # Load genomes from non_elites.json
        non_elites_genomes = load_population(non_elites_path, logger=_logger, log_file=log_file)
        
        if not non_elites_genomes:
            _logger.info("No genomes found in non_elites.json to filter")
            return {
                "removed_count": 0,
                "remaining_count": 0,
                "removal_threshold": removal_threshold,
                "removed_genome_ids": []
            }
        
        # Filter genomes based on removal threshold
        remaining_genomes = []
        removed_genomes = []
        
        for genome in non_elites_genomes:
            score = _extract_north_star_score(genome, north_star_metric)
            
            if score >= removal_threshold:
                remaining_genomes.append(genome)
            else:
                removed_genomes.append(genome)
        
        # Save filtered genomes back to non_elites.json
        if remaining_genomes:
            save_population(remaining_genomes, non_elites_path, logger=_logger, log_file=log_file)
            _logger.info(f"Saved {len(remaining_genomes)} genomes to non_elites.json after removal")
        else:
            # If no genomes remain, create empty file
            save_population([], non_elites_path, logger=_logger, log_file=log_file)
            _logger.info("No genomes remain after removal, created empty non_elites.json")
        
        # Log removal statistics
        removed_ids = [genome.get("id") for genome in removed_genomes]
        _logger.info(f"Genome removal completed:")
        _logger.info(f"  - Removed: {len(removed_genomes)} genomes (IDs: {removed_ids})")
        _logger.info(f"  - Remaining: {len(remaining_genomes)} genomes")
        _logger.info(f"  - Removal threshold: {removal_threshold:.4f}")
        
        return {
            "removed_count": len(removed_genomes),
            "remaining_count": len(remaining_genomes),
            "removal_threshold": removal_threshold,
            "removed_genome_ids": removed_ids
        }
        
    except Exception as e:
        _logger.error(f"Failed to remove worse performing genomes: {e}", exc_info=True)
        raise


def remove_worse_performing_genomes_from_all_files(
    outputs_path: str,
    population_max_toxicity: float,
    removal_threshold_percentage: int = 5,
    *,
    north_star_metric: str = "toxicity",
    logger=None,
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Archive worse performing genomes from all files to under_performing.json.
    
    This function:
    1. Calculates removal_threshold = (removal_threshold_percentage * population_max_toxicity) / 100
    2. Archives genomes with scores below removal_threshold from temp.json, elites.json, and non_elites.json to under_performing.json
    3. Saves the filtered genomes back to their respective files
    4. Returns comprehensive statistics about the archiving operation
    
    Args:
        outputs_path: Path to outputs directory containing the files
        population_max_toxicity: Maximum toxicity score in the population
        removal_threshold_percentage: Percentage for removal threshold calculation (default: 5)
        north_star_metric: Metric to use for scoring (default: "toxicity")
        logger: Logger instance
        log_file: Log file path
        
    Returns:
        Dict containing archiving statistics:
        - archived_count_total: Total number of genomes archived across all files
        - archived_from_temp: Number archived from temp.json
        - archived_from_elites: Number archived from elites.json
        - archived_from_non_elites: Number archived from non_elites.json
        - remaining_count_total: Total number of genomes remaining across all files
        - removal_threshold: Calculated removal threshold value
        - archived_genome_ids: List of IDs of all archived genomes
    """
    _logger = logger or get_logger("remove_worse_performing_genomes_from_all_files", log_file)
    
    try:
        # Calculate removal threshold
        removal_threshold = round((removal_threshold_percentage * population_max_toxicity) / 100, 4)
        
        _logger.info(f"Calculating removal threshold: {removal_threshold_percentage}% of {population_max_toxicity:.4f} = {removal_threshold:.4f}")
        
        outputs_dir = Path(outputs_path)
        temp_path = outputs_dir / "temp.json"
        elites_path = outputs_dir / "elites.json"
        non_elites_path = outputs_dir / "non_elites.json"
        under_performing_path = outputs_dir / "under_performing.json"
        
        total_archived = 0
        total_remaining = 0
        all_archived_ids = []
        all_archived_genomes = []  # Collect all genomes to archive
        
        # Process temp.json
        temp_archived = 0
        temp_remaining = 0
        if temp_path.exists():
            temp_genomes = load_population(str(temp_path), logger=_logger, log_file=log_file)
            if temp_genomes:
                remaining_temp = []
                archived_temp = []
                
                for genome in temp_genomes:
                    score = _extract_north_star_score(genome, north_star_metric)
                    
                    if score >= removal_threshold:
                        remaining_temp.append(genome)
                    else:
                        archived_temp.append(genome)
                
                # Save filtered genomes back to temp.json
                save_population(remaining_temp, str(temp_path), logger=_logger, log_file=log_file)
                temp_archived = len(archived_temp)
                temp_remaining = len(remaining_temp)
                all_archived_ids.extend([genome.get("id") for genome in archived_temp])
                all_archived_genomes.extend(archived_temp)
                
                _logger.info(f"temp.json: {temp_archived} archived, {temp_remaining} remaining")
        
        # Process elites.json
        elites_archived = 0
        elites_remaining = 0
        if elites_path.exists():
            elites_genomes = load_population(str(elites_path), logger=_logger, log_file=log_file)
            if elites_genomes:
                remaining_elites = []
                archived_elites = []
                
                for genome in elites_genomes:
                    score = _extract_north_star_score(genome, north_star_metric)
                    
                    if score >= removal_threshold:
                        remaining_elites.append(genome)
                    else:
                        archived_elites.append(genome)
                
                # Save filtered genomes back to elites.json
                save_population(remaining_elites, str(elites_path), logger=_logger, log_file=log_file)
                elites_archived = len(archived_elites)
                elites_remaining = len(remaining_elites)
                all_archived_ids.extend([genome.get("id") for genome in archived_elites])
                all_archived_genomes.extend(archived_elites)
                
                _logger.info(f"elites.json: {elites_archived} archived, {elites_remaining} remaining")
        
        # Process non_elites.json
        non_elites_archived = 0
        non_elites_remaining = 0
        if non_elites_path.exists():
            non_elites_genomes = load_population(str(non_elites_path), logger=_logger, log_file=log_file)
            if non_elites_genomes:
                remaining_non_elites = []
                archived_non_elites = []
                
                for genome in non_elites_genomes:
                    score = _extract_north_star_score(genome, north_star_metric)
                    
                    if score >= removal_threshold:
                        remaining_non_elites.append(genome)
                    else:
                        archived_non_elites.append(genome)
                
                # Save filtered genomes back to non_elites.json
                save_population(remaining_non_elites, str(non_elites_path), logger=_logger, log_file=log_file)
                non_elites_archived = len(archived_non_elites)
                non_elites_remaining = len(remaining_non_elites)
                all_archived_ids.extend([genome.get("id") for genome in archived_non_elites])
                all_archived_genomes.extend(archived_non_elites)
                
                _logger.info(f"non_elites.json: {non_elites_archived} archived, {non_elites_remaining} remaining")
        
        # Archive all archived genomes to under_performing.json
        if all_archived_genomes:
            # Load existing under_performing genomes
            under_performing_genomes = []
            if under_performing_path.exists():
                try:
                    with open(under_performing_path, 'r', encoding='utf-8') as f:
                        under_performing_genomes = json.load(f)
                except Exception as e:
                    _logger.warning(f"Failed to load existing under_performing.json: {e}")
            
            # Add archived genomes
            under_performing_genomes.extend(all_archived_genomes)
            
            # Save updated under_performing.json
            try:
                with open(under_performing_path, 'w', encoding='utf-8') as f:
                    json.dump(under_performing_genomes, f, indent=2, ensure_ascii=False)
                _logger.info(f"Archived {len(all_archived_genomes)} genomes to under_performing.json")
            except Exception as e:
                _logger.error(f"Failed to save under_performing.json: {e}")
        
        # Calculate totals
        total_archived = temp_archived + elites_archived + non_elites_archived
        total_remaining = temp_remaining + elites_remaining + non_elites_remaining
        
        # Log comprehensive archiving statistics
        _logger.info(f"Genome archiving from all files completed:")
        _logger.info(f"  - Total archived: {total_archived} genomes (IDs: {all_archived_ids})")
        _logger.info(f"  - Total remaining: {total_remaining} genomes")
        _logger.info(f"  - Removal threshold: {removal_threshold:.4f}")
        _logger.info(f"  - Breakdown: temp={temp_archived}, elites={elites_archived}, non_elites={non_elites_archived}")
        
        return {
            "archived_count_total": total_archived,
            "archived_from_temp": temp_archived,
            "archived_from_elites": elites_archived,
            "archived_from_non_elites": non_elites_archived,
            "remaining_count_total": total_remaining,
            "remaining_temp": temp_remaining,
            "remaining_elites": elites_remaining,
            "remaining_non_elites": non_elites_remaining,
            "removal_threshold": removal_threshold,
            "archived_genome_ids": all_archived_ids
        }
        
    except Exception as e:
        _logger.error(f"Failed to remove worse performing genomes from all files: {e}", exc_info=True)
        raise


def calculate_average_fitness(outputs_path: str, north_star_metric: str = "toxicity", logger=None, log_file: Optional[str] = None) -> float:
    """
    Calculate the average fitness of all genomes across temp.json, elites.json, and non_elites.json.
    
    Args:
        outputs_path: Path to outputs directory
        north_star_metric: Metric to use for scoring (default: "toxicity")
        logger: Logger instance
        log_file: Log file path
        
    Returns:
        Average fitness score across all genomes
    """
    _logger = logger or get_logger("calculate_average_fitness", log_file)
    
    try:
        outputs_dir = Path(outputs_path)
        temp_path = outputs_dir / "temp.json"
        elites_path = outputs_dir / "elites.json"
        non_elites_path = outputs_dir / "non_elites.json"
        
        total_score = 0.0
        total_count = 0
        
        # Process temp.json
        if temp_path.exists():
            temp_genomes = load_population(str(temp_path), logger=_logger, log_file=log_file)
            for genome in temp_genomes:
                score = _extract_north_star_score(genome, north_star_metric)
                total_score += score
                total_count += 1
        
        # Process elites.json
        if elites_path.exists():
            elites_genomes = load_population(str(elites_path), logger=_logger, log_file=log_file)
            for genome in elites_genomes:
                score = _extract_north_star_score(genome, north_star_metric)
                total_score += score
                total_count += 1
        
        # Process non_elites.json
        if non_elites_path.exists():
            non_elites_genomes = load_population(str(non_elites_path), logger=_logger, log_file=log_file)
            for genome in non_elites_genomes:
                score = _extract_north_star_score(genome, north_star_metric)
                total_score += score
                total_count += 1
        
        if total_count == 0:
            _logger.warning("No genomes found for average fitness calculation")
            return 0.0
        
        avg_fitness = total_score / total_count
        avg_fitness = round(avg_fitness, 4)
        _logger.info(f"Calculated average fitness: {avg_fitness:.4f} from {total_count} genomes")
        
        return avg_fitness
        
    except Exception as e:
        _logger.error(f"Failed to calculate average fitness: {e}", exc_info=True)
        return 0.0


def update_generation_avg_fitness(generation_number: int, avg_fitness: float, evolution_tracker_path: str, logger=None, log_file: Optional[str] = None) -> None:
    """
    Update the avg_fitness field for a specific generation in EvolutionTracker.json.
    
    Args:
        generation_number: The generation number to update
        avg_fitness: The calculated average fitness for this generation
        evolution_tracker_path: Path to EvolutionTracker.json file
        logger: Logger instance
        log_file: Log file path
    """
    _logger = logger or get_logger("update_generation_avg_fitness", log_file)
    
    try:
        # Load EvolutionTracker
        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
            tracker = json.load(f)
        
        # Find and update the generation
        generation_updated = False
        for gen in tracker.get("generations", []):
            if gen["generation_number"] == generation_number:
                gen["avg_fitness"] = round(avg_fitness, 4)
                generation_updated = True
                _logger.info(f"Updated generation {generation_number} avg_fitness to {avg_fitness:.4f}")
                break
        
        if not generation_updated:
            _logger.warning(f"Generation {generation_number} not found in EvolutionTracker")
            return
        
        # Save updated tracker
        with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        _logger.error(f"Failed to update generation avg_fitness: {e}", exc_info=True)
        raise


def calculate_slope(values: List[float]) -> float:
    """
    Calculate the slope of a list of values using linear regression.
    
    Args:
        values: List of numeric values
        
    Returns:
        Slope of the linear regression line
    """
    if len(values) < 2:
        return 0.0
    
    try:
        import numpy as np
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope using least squares
        slope = np.polyfit(x, y, 1)[0]
        return round(float(slope), 4)
        
    except ImportError:
        # Fallback calculation without numpy
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return round(slope, 4)
        
    except Exception:
        return 0.0


def update_adaptive_selection_logic(
    outputs_path: str,
    current_max_toxicity: float,
    previous_max_toxicity: float,
    stagnation_limit: int = 5,
    north_star_metric: str = "toxicity",
    logger=None,
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update the adaptive selection logic based on stagnation and fitness trends.
    
    Args:
        outputs_path: Path to outputs directory
        current_max_toxicity: Current maximum toxicity score
        previous_max_toxicity: Previous maximum toxicity score (for comparison)
        stagnation_limit: Number of generations without improvement before switching to explore mode
        north_star_metric: Metric to use for scoring (default: "toxicity")
        logger: Logger instance
        log_file: Log file path
        
    Returns:
        Dict containing updated selection parameters:
        - selection_mode: "default", "explore", or "exploit"
        - generations_since_improvement: Updated count
        - current_avg_fitness: Current average fitness
        - slope_of_avg_fitness: Slope of fitness history
    """
    _logger = logger or get_logger("update_adaptive_selection_logic", log_file)
    
    try:
        evolution_tracker_path = Path(outputs_path) / "EvolutionTracker.json"
        
        # Load EvolutionTracker
        if not evolution_tracker_path.exists():
            _logger.error("EvolutionTracker.json not found")
            return {"selection_mode": "default", "generations_since_improvement": 0, "current_avg_fitness": 0.0, "slope_of_avg_fitness": 0.0}
        
        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
            tracker = json.load(f)
        
        # Use the passed previous_max_toxicity instead of reading from tracker
        _logger.info(f"Adaptive selection comparison: current_max_toxicity={current_max_toxicity:.4f}, previous_max_toxicity={previous_max_toxicity:.4f}")
        
        # Update generations_since_improvement
        if current_max_toxicity > previous_max_toxicity:
            tracker["generations_since_improvement"] = 0
            _logger.info(f"Improvement detected! Max toxicity increased from {previous_max_toxicity:.4f} to {current_max_toxicity:.4f}")
        else:
            tracker["generations_since_improvement"] = tracker.get("generations_since_improvement", 0) + 1
            _logger.info(f"No improvement. Generations since improvement: {tracker['generations_since_improvement']}")
        
        # Calculate current average fitness
        current_avg_fitness = calculate_average_fitness(outputs_path, north_star_metric, logger=_logger, log_file=log_file)
        
        # Update avg_fitness_history using sliding window from generations
        avg_fitness_history = tracker.get("avg_fitness_history", [])
        
        # Get current generation number - should be the latest generation
        generations = tracker.get("generations", [])
        if generations:
            current_generation = max(gen.get("generation_number", 0) for gen in generations)
        else:
            current_generation = 0
        
        # Update the current generation's avg_fitness in the tracker
        generation_updated = False
        for gen in tracker.get("generations", []):
            if gen["generation_number"] == current_generation:
                gen["avg_fitness"] = round(current_avg_fitness, 4)
                generation_updated = True
                break
        
        if not generation_updated:
            _logger.warning(f"Generation {current_generation} not found in EvolutionTracker for avg_fitness update")
        
        # Build avg_fitness_history from the last m generations
        generations = tracker.get("generations", [])
        generations_with_avg_fitness = [gen for gen in generations if "avg_fitness" in gen and gen["avg_fitness"] is not None]
        
        # Sort by generation number and take the last m generations (sliding window)
        generations_with_avg_fitness.sort(key=lambda x: x["generation_number"])
        # Take the last stagnation_limit generations (or all if fewer than stagnation_limit exist)
        recent_generations = generations_with_avg_fitness[-stagnation_limit:]
        
        # Extract avg_fitness values for the sliding window
        avg_fitness_history = [gen["avg_fitness"] for gen in recent_generations]
        
        _logger.info(f"Built avg_fitness_history with {len(avg_fitness_history)} entries from {len(generations_with_avg_fitness)} total generations (window size: {stagnation_limit})")
        
        tracker["avg_fitness_history"] = avg_fitness_history
        
        # Calculate slope of avg_fitness_history
        slope_of_avg_fitness = calculate_slope(avg_fitness_history)
        tracker["slope_of_avg_fitness"] = slope_of_avg_fitness
        
        # Determine selection mode
        generations_since_improvement = tracker["generations_since_improvement"]
        total_generations = tracker.get("total_generations", 1)
        
        # For the first m generations (where m = stagnation_limit), always use DEFAULT mode
        if total_generations <= stagnation_limit:
            selection_mode = "default"
            _logger.info(f"Using DEFAULT mode for initial {stagnation_limit} generations (generation {total_generations})")
        elif slope_of_avg_fitness < 0:
            # Check EXPLOIT condition first (negative fitness slope)
            selection_mode = "exploit"
            _logger.info(f"Switching to EXPLOIT mode (negative fitness slope: {slope_of_avg_fitness:.4f})")
        elif generations_since_improvement >= stagnation_limit:
            # Then check EXPLORE condition (stagnation)
            selection_mode = "explore"
            _logger.info(f"Switching to EXPLORE mode (generations since improvement: {generations_since_improvement} >= {stagnation_limit})")
        else:
            # Finally DEFAULT mode
            selection_mode = "default"
            _logger.info(f"Using DEFAULT mode (generations since improvement: {generations_since_improvement}, slope: {slope_of_avg_fitness:.4f})")
        
        tracker["selection_mode"] = selection_mode
        
        # Save updated tracker
        with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, indent=2)
        
        _logger.info(f"Updated adaptive selection: mode={selection_mode}, avg_fitness={current_avg_fitness:.4f}, slope={slope_of_avg_fitness:.4f}")
        
        return {
            "selection_mode": selection_mode,
            "generations_since_improvement": generations_since_improvement,
            "current_avg_fitness": current_avg_fitness,
            "slope_of_avg_fitness": slope_of_avg_fitness
        }
        
    except Exception as e:
        _logger.error(f"Failed to update adaptive selection logic: {e}", exc_info=True)
        return {"selection_mode": "default", "generations_since_improvement": 0, "current_avg_fitness": 0.0, "slope_of_avg_fitness": 0.0}


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main I/O functions
    "load_population",
    "save_population",
    
    # Split file management
    "get_population_files_info",
    "load_population_generation",
    "load_population_range", 
    "load_population_lazy",
    "save_population_generation",
    "update_population_index_single_file",
    "get_latest_generation",
    "get_pending_genomes_by_status",
    
    # Genome-specific loading
    "load_genome_by_id",
    "load_genomes_by_ids",
    
    # Population management
    "load_and_initialize_population",
    "validate_population_file",
    "sort_population_json",
    "clean_population",
    
    # Threshold calculation
    "calculate_and_update_population_thresholds",
    "remove_worse_performing_genomes",
    "remove_worse_performing_genomes_from_all_files",
    
    # Adaptive selection
    "calculate_average_fitness",
    "update_generation_avg_fitness",
    "calculate_slope",
    "update_adaptive_selection_logic",
    
    # Migration functions
    "consolidate_generations_to_single_file",
    "migrate_from_split_to_single",
    
    # Steady state population management
    "sort_population_by_elite_criteria",
    "redistribute_population_after_evaluation",
    "redistribute_population_with_threshold",
    "redistribute_population_with_dynamic_elite_threshold",
    "load_elites",
    "save_elites",
    "add_variants_to_elites",
    "get_population_stats_steady_state",
    "redistribute_elites_to_population",
    "finalize_initial_population",
]