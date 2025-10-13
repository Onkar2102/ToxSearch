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

# Import pandas directly - it's required for this module
import pandas as pd

# Get the functions at module level to avoid repeated calls
get_logger, _, _, PerformanceLogger = get_custom_logging()


# ============================================================================
# UTILITY FUNCTIONS (moved from main.py)
# ============================================================================

def get_project_root():
    """Get the absolute path to the project root directory"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    # Go up two levels to get to the project root (from src/utils/ to project root)
    project_root = script_dir.parent.parent
    return project_root.resolve()

def get_config_path():
    """Get the absolute path to the config directory"""
    return get_project_root() / "config" / "modelConfig_llamacpp.yaml"

def get_data_path():
    """Get the absolute path to the data directory"""
    return get_project_root() / "data" / "prompt.xlsx"

def get_outputs_path():
    """Get the absolute path to the outputs directory"""
    # Create the full path: outputs/
    outputs_dir = get_project_root() / "outputs"
    
    # Ensure the directory exists
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    return outputs_dir

def _extract_north_star_score(genome, metric="toxicity"):
    """Extract the north star metric score from a genome using the configured metric."""
    if not genome:
        return 0.0
    
    # Try different score locations in order of preference
    try:
        # 1. Direct scores field (flattened structure)
        if "scores" in genome:
            scores = genome.get("scores", {})
            if isinstance(scores, dict) and metric in scores:
                score = scores.get(metric, 0.0)
                if score > 0:
                    return float(score)
        
        # 2. moderation_result.google.scores (nested structure)
        if "moderation_result" in genome:
            moderation_result = genome["moderation_result"]
            
            if "google" in moderation_result:
                google_scores = moderation_result["google"]
                if "scores" in google_scores:
                    score = google_scores["scores"].get(metric, 0.0)
                    if score > 0:
                        return float(score)
            
            # 3. Direct scores in moderation_result (flattened)
            if "scores" in moderation_result:
                scores = moderation_result["scores"]
                if isinstance(scores, dict) and metric in scores:
                    score = scores.get(metric, 0.0)
                    if score > 0:
                        return float(score)
        
        # 4. Direct north_star_score field
        if "north_star_score" in genome:
            return float(genome.get("north_star_score", 0.0))
        
        # 5. Fallback to max_score for legacy compatibility
        if "max_score" in genome:
            return float(genome.get("max_score", 0.0))
        
        return 0.0
        
    except Exception:
        return 0.0


# ============================================================================
# SYSTEM INITIALIZATION (moved from main.py)
# ============================================================================

def initialize_system(logger, log_file):
    """Initialize the system components and create gen0 if needed"""
    logger.info("Initializing optimized pipeline for M3 Mac...")
    
    # Import required modules
    from utils import get_population_io
    from gne import get_ResponseGenerator, get_PromptGenerator
    
    # Get population IO functions
    load_and_initialize_population, get_population_files_info, load_population, save_population, sort_population_json, load_genome_by_id, consolidate_generations_to_single_file, migrate_from_split_to_single, sort_population_by_elite_criteria, load_elites, save_elites, get_population_stats_steady_state, finalize_initial_population = get_population_io()
    
    # Initialize Response Generator (for generating responses to prompts)
    ResponseGenerator = get_ResponseGenerator()
    response_generator = ResponseGenerator(model_key="response_generator", config_path="config/RGConfig.yaml", log_file=log_file)
    logger.info("Response generator initialized for response generation")
    
    # Initialize Prompt Generator (for operators and evolutionary algorithms)
    PromptGenerator = get_PromptGenerator()
    prompt_generator = PromptGenerator(model_key="prompt_generator", config_path="config/PGConfig.yaml", log_file=log_file)
    logger.info("Prompt generator initialized for prompt generation")
    
    # Set the global generators for different purposes
    from ea.EvolutionEngine import set_global_generators
    set_global_generators(response_generator, prompt_generator)
    logger.info("Global generators set: response_generator for responses, prompt_generator for operators")
    
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
            if not population_file.exists():
                logger.info("No population file found. Initializing population from prompt.xlsx...")
            else:
                logger.info("Population file exists but is empty. Initializing population from prompt.xlsx...")
            load_and_initialize_population(
                input_path=str(get_data_path()),
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
            logger.info("Available generations: %s", sorted(generations))
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
    """Get information about population files including Population.json, elites.json, and most_toxic.json"""
    
    base_path = Path(base_dir).resolve()
    population_file = base_path / "Population.json"
    elites_file = base_path / "elites.json"
    most_toxic_file = base_path / "most_toxic.json"
    
    info = {
        "total_generations": 0,
        "total_genomes": 0,
        "generation_counts": {},
        "single_file_mode": True,
        "population_file": "Population.json",
        "elites_file": "elites.json",
        "most_toxic_file": "most_toxic.json",
        "elites_count": 0,
        "population_count": 0,
        "most_toxic_count": 0
    }
    
    # Count genomes in Population.json
    if population_file.exists():
        try:
            with open(population_file, 'r', encoding='utf-8') as f:
                population = json.load(f)
            
            info["population_count"] = len(population)
            
            # Count genomes by generation
            generation_counts = {}
            for genome in population:
                if genome and "generation" in genome:
                    gen_num = genome["generation"]
                    generation_counts[gen_num] = generation_counts.get(gen_num, 0) + 1
            
            info["generation_counts"] = generation_counts
            
        except Exception as e:
            # Silently fail if we can't read the file
            pass
    
    # Count genomes in elites.json
    if elites_file.exists():
        try:
            with open(elites_file, 'r', encoding='utf-8') as f:
                elites = json.load(f)
            
            info["elites_count"] = len(elites)
            
            # Add elite genomes to generation counts
            for genome in elites:
                if genome and "generation" in genome:
                    gen_num = genome["generation"]
                    info["generation_counts"][gen_num] = info["generation_counts"].get(gen_num, 0) + 1
            
        except Exception as e:
            # Silently fail if we can't read the file
            pass
    
    # Count genomes in most_toxic.json
    if most_toxic_file.exists():
        try:
            with open(most_toxic_file, 'r', encoding='utf-8') as f:
                most_toxic = json.load(f)
            
            info["most_toxic_count"] = len(most_toxic)
            
            # Add most_toxic genomes to generation counts
            for genome in most_toxic:
                if genome and "generation" in genome:
                    gen_num = genome["generation"]
                    info["generation_counts"][gen_num] = info["generation_counts"].get(gen_num, 0) + 1
            
        except Exception as e:
            # Silently fail if we can't read the file
            pass
    
    # Calculate total genomes and generations
    info["total_genomes"] = info["population_count"] + info["elites_count"] + info["most_toxic_count"]
    info["total_generations"] = max(info["generation_counts"].keys()) + 1 if info["generation_counts"] else 0
    
    return info


def update_population_index_single_file(base_dir: str, total_genomes: int, *, logger=None, log_file: Optional[str] = None):
    """Update the population index file for single file mode"""
    
    _logger = logger or get_logger("update_population_index", log_file)
    
    try:
        base_dir = str(Path(base_dir).resolve())
        info = get_population_files_info(base_dir)
        index_file = Path(base_dir) / "population_index.json"
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
        
        _logger.debug(f"Updated population index: single file mode, {info['total_genomes']} total genomes "
                     f"(population: {info['population_count']}, elites: {info['elites_count']}, "
                     f"most_toxic: {info['most_toxic_count']})")
        
    except Exception as e:
        _logger.warning(f"Failed to update population index: {e}")


def load_population_generation(generation: int, base_dir: str = "outputs", 
                              *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load genomes for a specific generation from the single Population.json file"""
    
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Load Generation {generation} from Single File"):
        try:
            # Load entire population
            all_genomes = load_population(base_dir, logger=_logger, log_file=log_file)
            
            # Filter by generation
            generation_genomes = [g for g in all_genomes if g and g.get("generation") == generation]
            
            _logger.info(f"Loaded generation {generation}: {len(generation_genomes)} genomes from Population.json")
            return generation_genomes
            
        except Exception as e:
            _logger.error(f"Failed to load generation {generation}: {e}", exc_info=True)
            return []


def load_population_range(start_gen: int, end_gen: int, base_dir: str = "outputs",
                         *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load multiple generations from the single Population.json file"""
    
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Load Generations {start_gen}-{end_gen} from Single File"):
        try:
            # Load entire population
            all_genomes = load_population(base_dir, logger=_logger, log_file=log_file)
            
            # Filter by generation range
            range_genomes = [g for g in all_genomes if g and start_gen <= g.get("generation", 0) <= end_gen]
            
            _logger.info(f"Loaded generations {start_gen}-{end_gen}: {len(range_genomes)} genomes from Population.json")
            return range_genomes
            
        except Exception as e:
            _logger.error(f"Failed to load generation range: {e}", exc_info=True)
            return []


def load_population_lazy(base_dir: str = "outputs", max_gens: Optional[int] = None,
                        *, logger=None, log_file: Optional[str] = None):
    """Generator that yields genomes from the single Population.json file"""
    
    _logger = logger or get_logger("population_io", log_file)
    
    try:
        # Load entire population
        all_genomes = load_population(base_dir, logger=_logger, log_file=log_file)
        
        # Apply generation limit if specified
        if max_gens is not None:
            all_genomes = [g for g in all_genomes if g and g.get("generation", 0) < max_gens]
        
        _logger.info(f"Lazy loading {len(all_genomes)} genomes from Population.json")
        
        for genome in all_genomes:
            yield genome
            
    except Exception as e:
        _logger.error(f"Failed to load population lazily: {e}", exc_info=True)
        return


def save_population_generation(genomes: List[Dict[str, Any]], generation: int, 
                              base_dir: str = "outputs", *, logger=None, log_file: Optional[str] = None):
    """Save genomes to the single Population.json file (updates the single file)"""
    
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
            
            _logger.info(f"Updated Population.json with generation {generation}: {len(genomes)} genomes")
            
        except Exception as e:
            _logger.error(f"Failed to save generation {generation}: {e}", exc_info=True)
            raise


def get_latest_generation(base_dir: str = "outputs") -> int:
    """Get the highest generation number available from the single Population.json file"""
    info = get_population_files_info(base_dir)
    return info["total_generations"] - 1 if info["total_generations"] > 0 else 0


def get_pending_genomes_by_status(status: str, max_generations: Optional[int] = None, 
                                 base_dir: str = "outputs", *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get genomes with specific status from the single Population.json file"""
    
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
            
            _logger.info(f"Found {len(pending_genomes)} genomes with status '{status}' from Population.json")
            return pending_genomes
            
        except Exception as e:
            _logger.error(f"Failed to get pending genomes: {e}", exc_info=True)
            return []


# ============================================================================
# Main Population I/O Functions (Unified API)
# ============================================================================

def load_population(pop_path: str = "outputs/Population.json", *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load population with automatic detection of split vs monolithic format
    
    If pop_path points to Population.json and it exists, use it directly.
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
                # First, check if Population.json exists (preferred)
                population_file = base_dir / "Population.json"
            else:
                # If it's a file, use the file directly
                population_file = pop_path_obj
                base_dir = pop_path_obj.parent
            if population_file.exists():
                if pop_path_obj.is_dir():
                    _logger.info("Using monolithic Population.json file (preferred)")
                else:
                    _logger.info("Using specified population file: %s", pop_path)
                try:
                    with open(population_file, "r", encoding="utf-8") as f:
                        population = json.load(f)

                    # Clean the population to remove None genomes
                    population = clean_population(population, logger=_logger, log_file=log_file)
                    if pop_path_obj.is_dir():
                        _logger.info("Successfully loaded population with %d genomes from Population.json", len(population))
                    else:
                        _logger.info("Successfully loaded population with %d genomes from %s", len(population), pop_path)
                    return population
                except Exception as e:
                    _logger.warning("Failed to load Population.json: %s, falling back to split files", e)
            
            # Fall back to split files if Population.json doesn't exist or fails
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
                    _logger.error("No population files found: neither Population.json nor split files exist")
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


def save_population(population: List[Dict[str, Any]], pop_path: str = "outputs/Population.json", 
                   *, logger=None, log_file: Optional[str] = None, preserve_sort_order: bool = False) -> None:
    """
    Save entire population to single Population.json file
    
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
            
            # Resolve the path
            pop_path_obj = Path(pop_path)
            
            # If pop_path is a directory, save as Population.json in it
            if pop_path_obj.is_dir():
                output_file = pop_path_obj / "Population.json"
            else:
                output_file = pop_path_obj
            
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
    """Load prompts from Excel file and initialize population as single Population.json file"""

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

            # ---------------------------- Load Excel -----------------------
            with PerformanceLogger(logger, "Load Excel File"):
                df = pd.read_excel(input_path)
                logger.info(
                    "Successfully loaded Excel file with %d rows and %d columns",
                    len(df),
                    len(df.columns),
                )

            # -------------------------- Extract prompts --------------------
            # Only expect a "questions" column in the Excel file
            if "questions" not in df.columns:
                raise ValueError("Required 'questions' column not found in Excel file")
            
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
                        "generation": 0,
                        "status": "pending_generation",
                        "creation_info": {
                            "type": "initial",
                            "operator": "excel_import",
                            "source_generation": 0,
                            "evolution_cycle": 0
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

            # ----------------------------- Initialize empty Population.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty Population.json"):
                # Population.json starts empty
                empty_population = []
                save_population(empty_population, output_path, logger=logger, log_file=log_file)
                logger.info("Initialized empty Population.json")

            # ----------------------------- Initialize empty elites.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty elites.json"):
                # elites.json starts empty
                empty_elites = []
                elites_path = Path(output_path) / "elites.json"
                elites_path.parent.mkdir(parents=True, exist_ok=True)
                with open(elites_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_elites, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty elites.json")

            # ----------------------------- Initialize empty parents.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty parents.json"):
                # parents.json starts empty
                empty_parents = []
                parents_path = Path(output_path) / "parents.json"
                parents_path.parent.mkdir(parents=True, exist_ok=True)
                with open(parents_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_parents, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty parents.json")

            # ----------------------------- Initialize empty most_toxic.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty most_toxic.json"):
                # most_toxic.json starts empty
                empty_most_toxic = []
                most_toxic_path = Path(output_path) / "most_toxic.json"
                most_toxic_path.parent.mkdir(parents=True, exist_ok=True)
                with open(most_toxic_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_most_toxic, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty most_toxic.json")

            # ----------------------------- Initialize empty top_10.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty top_10.json"):
                # top_10.json starts empty
                empty_top_10 = []
                top_10_path = Path(output_path) / "top_10.json"
                top_10_path.parent.mkdir(parents=True, exist_ok=True)
                with open(top_10_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_top_10, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty top_10.json")

            # ----------------------------- Initialize population_index.json ----------------------------
            with PerformanceLogger(logger, "Initialize population_index.json"):
                # population_index.json starts with default empty structure
                index_info = {
                    "total_generations": 0,
                    "total_genomes": 0,
                    "generation_counts": {},
                    "single_file_mode": True,
                    "population_file": "Population.json",
                    "elites_file": "elites.json",
                    "most_toxic_file": "most_toxic.json",
                    "elites_count": 0,
                    "population_count": 0,
                    "most_toxic_count": 0
                }
                index_path = Path(output_path) / "population_index.json"
                index_path.parent.mkdir(parents=True, exist_ok=True)
                with open(index_path, 'w', encoding='utf-8') as f:
                    json.dump(index_info, f, indent=2)
                logger.info("Initialized population_index.json with default structure")

            # ----------------------------- Initialize EvolutionTracker ----------------------------
            with PerformanceLogger(logger, "Initialize EvolutionTracker"):
                evolution_tracker = {
                    "scope": "global",
                    "status": "not_complete",
                    "total_generations": 1,  # Generation 0 exists
                    "generations": [
                        {
                            "generation_number": 0,
                            "genome_id": "1",  # Will be updated with actual best genome during threshold check
                            "max_score": 0.0,  # Will be updated with actual best score during threshold check
                            "mutation": None,
                            "crossover": None
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
            "total_genomes": len(population),
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

        logger.info("Validation complete – %d genomes analysed", stats["total_genomes"])
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
    # Set log level to DEBUG to see what's happening
    logger.setLevel(10)  # DEBUG level

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
                                         output_file: str = "Population.json",
                                         *, logger=None, log_file: Optional[str] = None) -> bool:
    """
    Consolidate all split generation files back into a single Population.json file.
    
    This function merges all gen*.json files into a single Population.json file,
    effectively reverting from the split file architecture back to the monolithic approach.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing generation files
    output_file : str
        Name of the output Population.json file
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
            _logger.info(f"Total genomes across all generations: {info['total_genomes']}")
            
            # Extract generation order for backup operations
            generation_order = sorted(info['generation_counts'].keys())
            
            # Load all genomes from single file
            all_genomes = load_population(str(base_path), logger=_logger, log_file=log_file)
            
            if not all_genomes:
                _logger.error("No genomes loaded from Population.json")
                return False
            
            # Clean the population (remove None genomes)
            all_genomes = clean_population(all_genomes, logger=_logger, log_file=log_file)
            
            # Sort the population by generation, and id
            all_genomes.sort(key=lambda g: (
                g.get("generation", 0),
                g.get("id", "0")
            ))
            
            # Save as single Population.json file
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
                _logger.error(f"Failed to save consolidated Population.json: {e}")
                return False
                
        except Exception as e:
            _logger.error(f"Failed to consolidate generations: {e}", exc_info=True)
            return False


def migrate_from_split_to_single(base_dir: str = "outputs", 
                                *, logger=None, log_file: Optional[str] = None) -> bool:
    """
    Complete migration from split file architecture back to single Population.json.
    
    This function:
    1. Consolidates all generation files into Population.json
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
            if not consolidate_generations_to_single_file(base_dir, "Population.json", logger=_logger, log_file=log_file):
                _logger.error("Failed to consolidate generation files")
                return False
            
            # Step 2: Update population index to reflect single file
            base_path = Path(base_dir).resolve()
            index_file = base_path / "population_index.json"
            
            if index_file.exists():
                # Update index to show single file
                updated_info = {
                    "total_generations": 1,
                    "total_genomes": 0,  # Will be calculated from Population.json
                    "generation_counts": {"0": 0},  # Will be calculated
                    "migration_note": "Migrated from split files to single Population.json"
                }
                
                try:
                    with open(index_file, 'w', encoding='utf-8') as f:
                        json.dump(updated_info, f, indent=2)
                    _logger.info("Updated population index for single file architecture")
                except Exception as e:
                    _logger.warning(f"Failed to update population index: {e}")
            
            # Step 3: Verify the consolidated file
            population_file = base_path / "Population.json"
            if population_file.exists():
                try:
                    with open(population_file, 'r', encoding='utf-8') as f:
                        consolidated_genomes = json.load(f)
                    
                    _logger.info(f"Migration successful! Population.json contains {len(consolidated_genomes)} genomes")
                    _logger.info("You can now use the single file approach")
                    
                    return True
                    
                except Exception as e:
                    _logger.error(f"Failed to verify consolidated file: {e}")
                    return False
            else:
                _logger.error("Population.json was not created during consolidation")
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




def redistribute_population_with_threshold(population_file_path: str = FileConstants.DEFAULT_POPULATION_FILE, 
                                          elites_file_path: str = FileConstants.DEFAULT_ELITES_FILE,
                                          elite_threshold: float = None,
                                          north_star_metric: str = EvolutionConstants.DEFAULT_NORTH_STAR_METRIC,
                                          *, logger=None, log_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Redistribute population using threshold-based elite selection:
    - Genomes with score >= elite_threshold → elites.json
    - Genomes with score < elite_threshold → Population.json
    
    Parameters
    ----------
    population_file_path : str
        Path to the Population.json file
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
    
    with PerformanceLogger(_logger, "Redistribute Population with Threshold", 
                         population_file=population_file_path, elites_file=elites_file_path, 
                         elite_threshold=elite_threshold):
        
        if elite_threshold is None:
            _logger.warning("No elite threshold provided for threshold-based redistribution")
            return {"elites_count": 0, "population_count": 0, "total_count": 0, "elite_threshold": 0}
        
        # Load current population
        population = load_population(population_file_path, logger=_logger)
        if not population:
            _logger.warning("No population found to redistribute")
            return {"elites_count": 0, "population_count": 0, "total_count": 0, "elite_threshold": elite_threshold}
        
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
        population_count = len(new_population)
        
        _logger.info(f"Threshold-based redistribution complete: {elites_count} elites (>= {elite_threshold}), {population_count} population (< {elite_threshold})")
        
        return {
            "elites_count": elites_count,
            "population_count": population_count,
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
        Path to the Population.json file
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
            return {"elites_count": 0, "population_count": 0, "total_count": 0, "elite_threshold": 0}
        
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
        population_count = len(new_population)
        
        _logger.info(f"Redistribution complete: {elites_count} elites, {population_count} population")
        
        return {
            "elites_count": elites_count,
            "population_count": population_count,
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
        Path to the Population.json file
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
            return {"elites_count": 0, "population_count": 0, "total_count": 0}
        
        # Clear existing files
        save_elites([], elites_file_path, logger=_logger)
        save_population([], population_file_path, logger=_logger)
        
        # Place all initial population in elites.json
        save_elites(initial_population, elites_file_path, logger=_logger)
        
        total_count = len(initial_population)
        _logger.info(f"Initialized with {total_count} genomes in elites.json")
        
        return {
            "elites_count": total_count,
            "population_count": 0,
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
        Path to the Population.json file
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
            "population_count": len(population),
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
            "elites_count": 0,
            "population_count": 0,
            "total_count": 0,
            "elite_percentage": 0,
            "error": str(e)
        }





def load_elites(elites_file_path: str = "outputs/elites.json", 
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


def save_elites(elites: List[Dict[str, Any]], elites_file_path: str = "outputs/elites.json",
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
        Path to the Population.json file
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
            "total_genomes": len(population) + len(elites),
            "population_count": len(population),
            "elites_count": len(elites),
            "steady_state_mode": True
        }
    except Exception as e:
        _logger.error(f"Failed to get population stats: {e}")
        return {
            "total_genomes": 0,
            "population_count": 0,
            "elites_count": 0,
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
    - Moves remaining genomes to Population.json
    - Updates population_index.json with current stats
    
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
                        elites.append(genome)
                    else:
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
                
                logger.info("Applied percentage-based selection (top %.1f%%)", elite_percentage * 100)
                logger.info("Selected %d elites and %d population genomes", len(elites), len(population))
            
            # Save elites to elites.json
            elites_path = Path(output_path) / "elites.json"
            with open(elites_path, 'w', encoding='utf-8') as f:
                json.dump(elites, f, indent=2, ensure_ascii=False)
            logger.info("Saved %d elites to elites.json", len(elites))
            
            # Save population to Population.json
            population_path = Path(output_path) / "Population.json"
            with open(population_path, 'w', encoding='utf-8') as f:
                json.dump(population, f, indent=2, ensure_ascii=False)
            logger.info("Saved %d genomes to Population.json", len(population))
            
            # Update population_index.json
            index_info = {
                "total_generations": 1,  # Generation 0 completed
                "total_genomes": total_valid,
                "generation_files": {},
                "generation_counts": {"0": total_valid},
                "single_file_mode": True,
                "population_file": "Population.json",
                "elites_file": "elites.json",
                "most_toxic_file": "most_toxic.json",
                "elites_count": len(elites),
                "population_count": len(population),
                "most_toxic_count": 0  # No most_toxic genomes in initial population
            }
            
            index_path = Path(output_path) / "population_index.json"
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index_info, f, indent=2)
            logger.info("Updated population_index.json with final stats")
            
            # Update EvolutionTracker with elite information
            evolution_tracker_path = Path(output_path) / "EvolutionTracker.json"
            if evolution_tracker_path.exists():
                with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                    evolution_tracker = json.load(f)
                
                # Update generation 0 with best elite info
                if elites and evolution_tracker.get("generations"):
                    # Find the best elite (highest toxicity score)
                    best_elite = max(elites, key=get_toxicity_score)
                    evolution_tracker["generations"][0]["genome_id"] = best_elite.get("id", "1")
                    best_score = get_toxicity_score(best_elite)
                    evolution_tracker["generations"][0]["max_score"] = best_score if best_score != float('inf') else 0.0
                    
                    with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                        json.dump(evolution_tracker, f, indent=2)
                    
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