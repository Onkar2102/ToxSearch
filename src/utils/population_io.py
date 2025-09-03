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
from . import get_custom_logging

# Only import pandas when needed for initialization
def _get_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError("pandas is required for Excel file loading. Install with: pip install pandas")

# Get the functions at module level to avoid repeated calls
get_logger, _, _, PerformanceLogger = get_custom_logging()


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
    """Get information about the single Population.json file"""
    
    base_path = Path(base_dir).resolve()
    population_file = base_path / "Population.json"
    
    info = {
        "total_generations": 0,
        "total_genomes": 0,
        "generation_files": {},
        "generation_counts": {},
        "single_file_mode": True,
        "population_file": "Population.json"
    }
    
    if population_file.exists():
        try:
            with open(population_file, 'r', encoding='utf-8') as f:
                population = json.load(f)
            
            # Calculate statistics from the single file
            info["total_genomes"] = len(population)
            
            # Count genomes by generation
            generation_counts = {}
            for genome in population:
                if genome and "generation" in genome:
                    gen_num = genome["generation"]
                    generation_counts[gen_num] = generation_counts.get(gen_num, 0) + 1
            
            info["total_generations"] = max(generation_counts.keys()) + 1 if generation_counts else 0
            info["generation_counts"] = generation_counts
            
            # For backward compatibility, create generation_files dict
            for gen_num in range(info["total_generations"]):
                info["generation_files"][gen_num] = f"Population.json (gen{gen_num})"
            
        except Exception as e:
            # Silently fail if we can't read the file
            pass
    
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
        
        _logger.debug(f"Updated population index: single file mode, {total_genomes} genomes")
        
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
            else:
                # If it's a file, use its parent directory
                base_dir = pop_path_obj.parent
                if base_dir.name == "":
                    base_dir = Path("outputs")
            
            # First, check if Population.json exists (preferred)
            population_file = base_dir / "Population.json"
            if population_file.exists():
                _logger.info("Using monolithic Population.json file (preferred)")
                try:
                    with open(population_file, "r", encoding="utf-8") as f:
                        population = json.load(f)

                    # Clean the population to remove None genomes
                    population = clean_population(population, logger=_logger, log_file=log_file)
                    _logger.info("Successfully loaded population with %d genomes from Population.json", len(population))
                    return population
                except Exception as e:
                    _logger.warning("Failed to load Population.json: %s, falling back to split files", e)
            
            # Fall back to split files if Population.json doesn't exist or fails
            info = get_population_files_info(str(base_dir))
            
            if info["generation_files"]:
                _logger.info("Using split population files (fallback mode)")
                # Load all generations (but this is still memory-intensive for large populations)
                all_genomes = []
                for gen in sorted(info["generation_files"].keys()):
                    genomes = load_population_generation(gen, str(base_dir), logger=_logger, log_file=log_file)
                    all_genomes.extend(genomes)
                
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

            # Load pandas only when needed
            pd = _get_pandas()

            # ---------------------------- Load Excel -----------------------
            with PerformanceLogger(logger, "Load Excel File"):
                df = pd.read_excel(input_path)
                logger.info(
                    "Successfully loaded Excel file with %d rows and %d columns",
                    len(df),
                    len(df.columns),
                )

            # -------------------------- Extract prompts --------------------
            prompt_columns = [
                "prompt",
                "text",
                "input",
                "query",
                "instruction",
                "content",
            ]
            prompt_column = next((c for c in prompt_columns if c in df.columns), None)

            if prompt_column is None:
                for col in df.columns:
                    if df[col].dtype == "object" and len(str(df[col].iloc[0])) > 10:
                        prompt_column = col
                        break

            if prompt_column is None:
                raise ValueError("No suitable prompt column found in Excel file")

            prompts = (
                df[prompt_column].dropna().drop_duplicates().astype(str).str.strip().tolist()
            )
            logger.info("Extracted %d unique prompts", len(prompts))

            # -------------------------- Create genomes ---------------------
            population: List[Dict[str, Any]] = []
            for i, prompt in enumerate(prompts):
                population.append(
                    {
                        "id": str(i + 1),
                        "prompt": prompt,
                        "model_provider": None,
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
                        },
                        "generation_timestamp": time.time(),
                        "source_file": input_path,
                        "source_column": prompt_column,
                    }
                )

            logger.info("Created %d genomes", len(population))

            # ----------------------------- Save Population to single file ----------------------------
            save_population(population, output_path, logger=logger, log_file=log_file)
            logger.info("Saved population to single Population.json file")

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
                base_dir = Path(output_path).resolve()
                evolution_tracker_path = base_dir / "EvolutionTracker.json"
                evolution_tracker_path.parent.mkdir(exist_ok=True)
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
            
            if not info["generation_files"]:
                _logger.warning("No generation files found to consolidate")
                return False
            
            _logger.info(f"Found {len(info['generation_files'])} generation files to consolidate")
            _logger.info(f"Total genomes across all generations: {info['total_genomes']}")
            
            # Load all genomes from all generations
            all_genomes = []
            generation_order = sorted(info["generation_files"].keys())
            
            for gen_num in generation_order:
                gen_file = base_path / f"gen{gen_num}.json"
                if gen_file.exists():
                    try:
                        with open(gen_file, 'r', encoding='utf-8') as f:
                            generation_genomes = json.load(f)
                        
                        _logger.info(f"Loaded generation {gen_num}: {len(generation_genomes)} genomes")
                        all_genomes.extend(generation_genomes)
                        
                    except Exception as e:
                        _logger.error(f"Failed to load generation {gen_num}: {e}")
                        continue
            
            if not all_genomes:
                _logger.error("No genomes loaded from any generation files")
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
                    "generation_files": {"0": "Population.json"},
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
] 