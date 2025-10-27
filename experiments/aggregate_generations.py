"""Aggregate evolutionary generation JSON files into a single CSV.

This script scans the 'outputs' directory for files matching the generation
filename pattern (gen0.json, gen1.json, ...), flattens genome records,
adds a `generation` column, and writes a single consolidated CSV to 
'data/all_generations.csv'.

Usage:
    python experiments/aggregate_generations.py

Assumptions:
    - Each generation file contains a list of genome-like dict records at the
      top level (which matches your current structure).
    - Each genome has an 'id' field that serves as a unique identifier.
"""

from __future__ import annotations

import json
import os
import re
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


GEN_FILE_REGEX = re.compile(r"gen(\d+)\.json$")


# Configuration - hardcoded for your use case
INPUT_DIR = "outputs"
PATTERN = "gen*.json"
OUTPUT_PATH = os.path.join("data", "all_generations.csv")


def find_generation_files(input_dirs: List[str], pattern: str) -> List[Tuple[str, str]]:
    """Return a list of (input_dir, file_path) tuples for matching generation files.

    Files are not guaranteed to be sorted by generation; sorting happens later.
    """
    results: List[Tuple[str, str]] = []
    for directory in input_dirs:
        if not os.path.isdir(directory):
            continue
        for path in glob(os.path.join(directory, pattern)):
            # Only consider files that match the regex for extracting generation
            if GEN_FILE_REGEX.search(os.path.basename(path)):
                results.append((directory, path))
    return results


def extract_generation_from_filename(file_path: str) -> Optional[int]:
    """Extract generation number from a filename like 'gen7.json'.

    Returns None if the filename doesn't match the expected pattern.
    """
    name = os.path.basename(file_path)
    match = GEN_FILE_REGEX.search(name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def lookup_with_dotted_path(obj: Any, dotted_path: str) -> Any:
    """Traverse a dotted path (e.g., 'a.b.c') within nested dicts/lists.

    For list indices, allow numeric tokens (e.g., 'items.0').
    Returns None on any lookup error.
    """
    current: Any = obj
    for token in dotted_path.split("."):
        if isinstance(current, dict):
            current = current.get(token)
        elif isinstance(current, list) and token.isdigit():
            idx = int(token)
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
        else:
            return None
    return current


def autodetect_records(obj: Any) -> Optional[List[Dict[str, Any]]]:
    """Auto-detect a list of dict records within the JSON object.

    - If top-level is a list of dicts, return it.
    - If top-level is a dict, return the first value that is a list of dicts.
    - Otherwise, return None.
    """
    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
        return obj  # type: ignore[return-value]
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, list) and (len(value) == 0 or isinstance(value[0], dict)):
                return value  # type: ignore[return-value]
    return None


def load_records_from_file(file_path: str, list_key: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Load a JSON file and return the list of genome records.

    If list_key is provided, attempt to locate that path; otherwise auto-detect.
    Returns None if records cannot be found.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return None

    if list_key:
        node = lookup_with_dotted_path(data, list_key)
        if isinstance(node, list) and (len(node) == 0 or isinstance(node[0], dict)):
            return node  # type: ignore[return-value]
        return None

    return autodetect_records(data)


def flatten_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten a list of dict records into a DataFrame.

    Nested dicts are flattened using dot notation. Lists remain as-is and will be
    serialized to strings by pandas when writing CSV.
    """
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records, sep=".")
    return df


def add_leading_columns(
    df: pd.DataFrame,
    generation: int,
    include_record_index: bool,
    run_label: Optional[str] = None,
) -> pd.DataFrame:
    """Reorder columns to put generation first.

    Columns order: ['generation', ...existing columns]
    """
    df = df.copy()
    
    # Since generation already exists in the JSON, just reorder columns
    # to put generation first
    if 'generation' in df.columns:
        # Move generation to first position
        cols = ['generation'] + [col for col in df.columns if col != 'generation']
        df = df[cols]
    
    return df


def aggregate_to_csv() -> Tuple[int, int, int, int]:
    """Aggregate all generations and write to CSV.

    Returns summary tuple: (num_dirs, num_files, num_generations, num_records)
    """
    located = find_generation_files([INPUT_DIR], PATTERN)
    # Sort by generation ascending
    located_with_gen: List[Tuple[str, str, int]] = []
    for directory, path in located:
        gen = extract_generation_from_filename(path)
        if gen is None:
            continue
        located_with_gen.append((directory, path, gen))
    located_with_gen.sort(key=lambda t: t[2])

    dataframes: List[pd.DataFrame] = []
    generations_seen: set[int] = set()
    total_records = 0

    for directory, path, gen in located_with_gen:
        records = load_records_from_file(path, list_key=None)  # Auto-detect since genomes are top-level list
        if records is None:
            # Skip unreadable or incompatible files
            continue
        flat = flatten_records(records)
        if flat.empty:
            continue
        flat = add_leading_columns(
            flat,
            generation=gen,
            include_record_index=False,  # No record index needed since we have 'id'
            run_label=None,  # No run column needed
        )
        dataframes.append(flat)
        generations_seen.add(gen)
        total_records += len(flat)

    if not dataframes:
        # Ensure parent directory exists before attempting to write, even if empty
        os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
        # Write an empty CSV with just headers if no data
        empty_df = pd.DataFrame(columns=["generation"])
        empty_df.to_csv(OUTPUT_PATH, index=False)
        return (len(list({d for d, _ in located})), 0, 0, 0)

    final_df = pd.concat(dataframes, ignore_index=True)

    # Optional: stable column order - put administrative columns first
    admin_cols = [c for c in ["generation", "record_index", "run"] if c in final_df.columns]
    other_cols = [c for c in final_df.columns if c not in admin_cols]
    final_df = final_df[admin_cols + other_cols]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    return (
        len({d for d, _ in located}),
        len(located_with_gen),
        len(generations_seen),
        total_records,
    )


def main() -> None:
    num_dirs, num_files, num_generations, num_records = aggregate_to_csv()
    print(
        (
            f"Processed {num_files} files from {INPUT_DIR}, "
            f"covering {num_generations} generations. "
            f"Total records: {num_records}. "
            f"Wrote CSV to: {OUTPUT_PATH}"
        )
    )


if __name__ == "__main__":
    main()


