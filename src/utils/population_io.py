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
import pandas as pd
import time
from utils.custom_logging import get_logger, PerformanceLogger


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
    """Get information about available population generation files"""
    
    # Always scan for actual generation files to ensure accuracy
    gen_files = {}
    gen_counts = {}
    total_genomes = 0
    
    base_path = Path(base_dir)
    for file_path in base_path.glob("gen*.json"):
        gen_num = int(file_path.stem.replace("gen", ""))
        gen_files[gen_num] = file_path.name
        
        # Quick count
        with open(file_path, 'r', encoding='utf-8') as f:
            genomes = json.load(f)
            gen_counts[gen_num] = len(genomes)
            total_genomes += len(genomes)
    
    info = {
        "total_generations": len(gen_files),
        "total_genomes": total_genomes,
        "generation_files": gen_files,
        "generation_counts": gen_counts
    }
    
    # Update the index file with current information
    index_file = Path(base_dir) / "population_index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    
    return info


def load_population_generation(generation: int, base_dir: str = "outputs", 
                              *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load a specific generation from split files"""
    
    _logger = logger or get_logger("population_io", log_file)
    
    gen_file = Path(base_dir) / f"gen{generation}.json"
    
    with PerformanceLogger(_logger, f"Load Generation {generation}", file_path=str(gen_file)):
        if not gen_file.exists():
            _logger.warning(f"Generation {generation} file not found: {gen_file}")
            return []
        
        try:
            with open(gen_file, 'r', encoding='utf-8') as f:
                genomes = json.load(f)
            
            _logger.info(f"Loaded generation {generation}: {len(genomes)} genomes")
            return genomes
            
        except Exception as e:
            _logger.error(f"Failed to load generation {generation}: {e}", exc_info=True)
            raise


def load_population_range(start_gen: int, end_gen: int, base_dir: str = "outputs",
                         *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load multiple generations at once (memory-efficient for specific ranges)"""
    
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Load Generations {start_gen}-{end_gen}"):
        all_genomes = []
        for gen in range(start_gen, end_gen + 1):
            genomes = load_population_generation(gen, base_dir, logger=_logger, log_file=log_file)
            all_genomes.extend(genomes)
        
        _logger.info(f"Loaded generations {start_gen}-{end_gen}: {len(all_genomes)} total genomes")
        return all_genomes


def load_population_lazy(base_dir: str = "outputs", max_gens: Optional[int] = None,
                        *, logger=None, log_file: Optional[str] = None):
    """Generator that yields genomes one generation at a time (memory-efficient)"""
    
    _logger = logger or get_logger("population_io", log_file)
    
    info = get_population_files_info(base_dir)
    generations = sorted(info["generation_files"].keys())
    
    if max_gens:
        generations = generations[:max_gens]
    
    _logger.info(f"Lazy loading {len(generations)} generations")
    
    for gen in generations:
        genomes = load_population_generation(gen, base_dir, logger=_logger, log_file=log_file)
        for genome in genomes:
            yield genome


def save_population_generation(genomes: List[Dict[str, Any]], generation: int, 
                              base_dir: str = "outputs", *, logger=None, log_file: Optional[str] = None):
    """Save genomes to a specific generation file"""
    
    _logger = logger or get_logger("population_io", log_file)
    
    gen_file = Path(base_dir) / f"gen{generation}.json"
    
    with PerformanceLogger(_logger, f"Save Generation {generation}", file_path=str(gen_file)):
        try:
            # Ensure directory exists
            gen_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(gen_file, 'w', encoding='utf-8') as f:
                json.dump(genomes, f, indent=2, ensure_ascii=False)
            
            size_mb = gen_file.stat().st_size / (1024 * 1024)
            _logger.info(f"Saved generation {generation}: {len(genomes)} genomes, {size_mb:.2f}MB")
            
            # Update index
            update_population_index(base_dir, logger=_logger, log_file=log_file)
            
        except Exception as e:
            _logger.error(f"Failed to save generation {generation}: {e}", exc_info=True)
            raise


def update_population_index(base_dir: str = "outputs", *, logger=None, log_file: Optional[str] = None):
    """Update the population index file after changes"""
    
    _logger = logger or get_logger("population_io", log_file)
    
    info = get_population_files_info(base_dir)
    index_file = Path(base_dir) / "population_index.json"
    
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    
    _logger.debug(f"Updated population index: {info['total_generations']} generations, {info['total_genomes']} genomes")


def get_latest_generation(base_dir: str = "outputs") -> int:
    """Get the highest generation number available"""
    info = get_population_files_info(base_dir)
    return max(info["generation_files"].keys()) if info["generation_files"] else 0


def get_pending_genomes_by_status(status: str, max_generations: Optional[int] = None, 
                                 base_dir: str = "outputs", *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get genomes with specific status across all generations (memory-efficient)"""
    
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Find genomes with status '{status}'"):
        pending_genomes = []
        info = get_population_files_info(base_dir)
        generations = sorted(info["generation_files"].keys())
        
        if max_generations:
            generations = generations[:max_generations]
        
        for gen in generations:
            genomes = load_population_generation(gen, base_dir, logger=_logger, log_file=log_file)
            gen_pending = [g for g in genomes if g.get("status") == status]
            pending_genomes.extend(gen_pending)
            
            if gen_pending:
                _logger.debug(f"Generation {gen}: {len(gen_pending)} genomes with status '{status}'")
        
        _logger.info(f"Found {len(pending_genomes)} genomes with status '{status}' across {len(generations)} generations")
        return pending_genomes


# ============================================================================
# Main Population I/O Functions (Unified API)
# ============================================================================

def load_population(pop_path: str = "outputs/Population.json", *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load population with automatic detection of split vs monolithic format
    
    If pop_path points to Population.json and split files exist, use split files.
    Otherwise, fall back to original behavior.
    
    Parameters
    ----------
    pop_path : str
        Path to the population file.
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created.
    log_file : str | None
        Optional log-file path when a new logger is created.

    Returns
    -------
    list[dict]
        Parsed population.
    """
    _logger = logger or get_logger("population_io", log_file)

    with PerformanceLogger(_logger, "Load Population", file_path=pop_path):
        try:
            # Check if we should use split files
            base_dir = Path(pop_path).parent
            if base_dir.name == "":
                base_dir = Path("outputs")
            
            info = get_population_files_info(str(base_dir))
            
            if info["generation_files"]:
                _logger.info("Using split population files (memory-efficient mode)")
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
                # Fall back to original monolithic file
                _logger.info("Using monolithic Population.json file")
                if not os.path.exists(pop_path):
                    _logger.error("Population file not found: %s", pop_path)
                    raise FileNotFoundError(f"Population file not found: {pop_path}")

                with open(pop_path, "r", encoding="utf-8") as f:
                    population = json.load(f)

                # Clean the population to remove None genomes
                population = clean_population(population, logger=_logger, log_file=log_file)
                _logger.info("Successfully loaded population with %d genomes from monolithic file", len(population))
                return population
                
        except json.JSONDecodeError as e:
            _logger.error("Failed to parse population JSON: %s", e, exc_info=True)
            raise
        except Exception as e:
            _logger.error("Unexpected error loading population: %s", e, exc_info=True)
            raise


def save_population(population: List[Dict[str, Any]], pop_path: str = "outputs/Population.json", 
                   *, logger=None, log_file: Optional[str] = None) -> None:
    """
    Save population using split files by generation (memory-efficient mode)
    
    Automatically splits by generation and saves each generation to separate files.
    
    Parameters
    ----------
    population : List[Dict[str, Any]]
        Population to save.
    pop_path : str
        Path where to save the population (used to determine base directory).
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
            
            base_dir = Path(pop_path).parent
            
            # Group by generation
            generations = defaultdict(list)
            for genome in cleaned_population:
                gen = genome.get('generation', 0)
                generations[gen].append(genome)
            
            # Save each generation
            for gen, genomes in generations.items():
                save_population_generation(genomes, gen, str(base_dir), logger=_logger, log_file=log_file)
            
            _logger.info("Successfully saved population using split files: %d generations, %d total genomes", 
                        len(generations), len(cleaned_population))
                        
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
    """Read prompts from *input_path* (Excel) and create an initial population.

    This function was previously located in *utils.initialize_population* and is
    kept verbatim to maintain behaviour while consolidating population
    utilities into a single module.
    """

    logger = get_logger("initialize_population", log_file)

    with PerformanceLogger(
        logger, "Initialize Population", input_path=input_path, output_path=output_path
    ):
        try:
            logger.info("Starting population initialization")
            logger.info("Input file: %s", input_path)
            logger.info("Output file: %s", output_path)

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
                        "prompt_id": i,
                        "prompt": prompt,
                        "generation": 0,
                        "status": "pending_generation",
                        "created_timestamp": time.time(),
                        "source_file": input_path,
                        "source_column": prompt_column,
                    }
                )

            logger.info("Created %d genomes", len(population))

            # ----------------------------- Save Population ----------------------------
            save_population(population, output_path, logger=logger)

            # ----------------------------- Save Initial Generation File ----------------------------
            with PerformanceLogger(logger, "Save Initial Generation File"):
                gen0_path = Path("outputs/gen0.json")
                gen0_path.parent.mkdir(exist_ok=True)
                with open(gen0_path, 'w', encoding='utf-8') as f:
                    json.dump(population, f, indent=2)
                
                logger.info("Saved initial generation file: %s", gen0_path)

            # ----------------------------- Initialize EvolutionTracker ----------------------------
            with PerformanceLogger(logger, "Initialize EvolutionTracker"):
                evolution_tracker = []
                for i, prompt in enumerate(prompts):
                    tracker_entry = {
                        "prompt_id": i,
                        "status": "not_complete",
                        "total_generations": 0,
                        "generations": [
                            {
                                "generation_number": 0,
                                "genome_id": str(i + 1),
                                "max_score": 0.0,
                                "mutation": None,
                                "crossover": None
                            }
                        ]
                    }
                    evolution_tracker.append(tracker_entry)
                
                # Save EvolutionTracker
                evolution_tracker_path = Path("outputs/EvolutionTracker.json")
                evolution_tracker_path.parent.mkdir(exist_ok=True)
                with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                    json.dump(evolution_tracker, f, indent=2)
                
                logger.info("Initialized EvolutionTracker with %d prompt entries", len(evolution_tracker))

            # ----------------------------- Initialize Population Index ----------------------------
            with PerformanceLogger(logger, "Initialize Population Index"):
                population_index = {
                    "total_generations": 1,  # Generation 0
                    "total_genomes": len(population),
                    "generation_files": {
                        "0": "gen0.json"
                    },
                    "generation_counts": {
                        "0": len(population)
                    }
                }
                
                # Save population index
                index_path = Path("outputs/population_index.json")
                index_path.parent.mkdir(exist_ok=True)
                with open(index_path, 'w', encoding='utf-8') as f:
                    json.dump(population_index, f, indent=2)
                
                logger.info("Initialized population index with %d genomes in generation 0", len(population))

        except Exception:
            logger.exception("Population initialization failed")
            raise


def validate_population_file(population_path: str, *, log_file: Optional[str] = None) -> Dict[str, Any]:
    """Run sanity checks on a population JSON and return aggregate statistics."""

    logger = get_logger("validate_population", log_file)

    with PerformanceLogger(logger, "Validate Population File", file_path=population_path):
        population = load_population(population_path, logger=logger)

        stats: Dict[str, Any] = {
            "total_genomes": len(population),
            "generations": set(),
            "statuses": {},
            "prompt_ids": set(),
            "prompt_lengths": [],
            "errors": [],
        }

        for genome in population:
            # Required keys check
            for field in ("id", "prompt_id", "prompt", "generation", "status"):
                if field not in genome:
                    stats["errors"].append(
                        f"Missing required field '{field}' in genome {genome.get('id', '?')}"
                    )

            stats["generations"].add(genome.get("generation", -1))
            stats["prompt_ids"].add(genome.get("prompt_id", -1))
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
        stats["prompt_ids"] = sorted(stats["prompt_ids"])

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

        def _sort_key(genome):
            out = []
            for i, key in enumerate(sort_keys):
                if callable(key):
                    value = key(genome)
                else:
                    # Handle string keys safely
                    value = genome.get(key) if genome is not None else None
                
                # Handle None values consistently
                if value is None:
                    value = float("-inf") if reverse_flags[i] else float("inf")
                # For reverse sorting, negate numeric values to achieve descending order
                elif reverse_flags[i] and isinstance(value, (int, float)):
                    value = -value
                out.append(value)
            return tuple(out)

        pop_list.sort(key=_sort_key)

        # Persist if needed
        dest = output_path or (population if isinstance(population, str) else None)
        if dest:
            save_population(pop_list, dest, logger=logger)

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
    "update_population_index",
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
] 