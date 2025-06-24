# utils/initialize_population.py (legacy shim)
"""Deprecated module kept for backward-compatibility.

All functionality has moved to :pymod:`utils.population_io`.  Import from there
in new code.  This shim simply re-exports the public API so that existing import
paths continue to work without modification.
"""

from utils.population_io import (
    load_and_initialize_population,
    validate_population_file,
    sort_population_json,
)

__all__ = [
    "load_and_initialize_population",
    "validate_population_file",
    "sort_population_json",
]