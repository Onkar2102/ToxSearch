from typing import List, Dict, Any, Optional
import os
import json
from utils.custom_logging import get_logger, PerformanceLogger


def load_population(pop_path: str, *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load population JSON with uniform logging & error handling.

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
            if not os.path.exists(pop_path):
                _logger.error("Population file not found: %s", pop_path)
                raise FileNotFoundError(f"Population file not found: {pop_path}")

            with open(pop_path, "r", encoding="utf-8") as f:
                population = json.load(f)

            _logger.info("Successfully loaded population with %d genomes", len(population))
            return population
        except json.JSONDecodeError as e:
            _logger.error("Failed to parse population JSON: %s", e, exc_info=True)
            raise
        except Exception as e:
            _logger.error("Unexpected error loading population: %s", e, exc_info=True)
            raise


def save_population(population: List[Dict[str, Any]], pop_path: str, *, logger=None, log_file: Optional[str] = None) -> None:
    """Save population JSON with uniform logging & error handling."""
    _logger = logger or get_logger("population_io", log_file)

    with PerformanceLogger(_logger, "Save Population", file_path=pop_path, genome_count=len(population)):
        try:
            os.makedirs(os.path.dirname(pop_path), exist_ok=True)
            with open(pop_path, "w", encoding="utf-8") as f:
                json.dump(population, f, indent=2, ensure_ascii=False)
            _logger.info("Successfully saved population with %d genomes to %s", len(population), pop_path)
        except Exception as e:
            _logger.error("Failed to save population: %s", e, exc_info=True)
            raise


# ---------------------------------------------------------------------------
# Higher-level helpers (migrated from initialize_population.py)
# ---------------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import time
from typing import Union


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

            # ----------------------------- Save ----------------------------
            save_population(population, output_path, logger=logger)

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
                value = key(genome) if callable(key) else genome.get(key)
                if reverse_flags[i] and isinstance(value, (int, float)):
                    value = -value if value is not None else float("-inf")
                out.append(value)
            return tuple(out)

        pop_list.sort(key=_sort_key)

        # Persist if needed
        dest = output_path or (population if isinstance(population, str) else None)
        if dest:
            save_population(pop_list, dest, logger=logger)

        return pop_list

# ---------------------------------------------------------------------------
# Backwards-compat: re-export names expected from utils.initialize_population
# ---------------------------------------------------------------------------

__all__ = [
    "load_population",
    "save_population",
    "load_and_initialize_population",
    "validate_population_file",
    "sort_population_json",
] 