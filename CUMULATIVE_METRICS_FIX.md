# Cumulative Metrics Fix

## Problem

Cumulative metrics (like `population_max_toxicity`) should include the entire population up to and including the current generation, including genomes created and distributed in the current generation. However, genomes from the current generation may not have the `generation` field set correctly during distribution, causing them to be excluded from cumulative metrics.

## Root Cause

1. **Missing generation field**: When genomes are distributed from temp.json to elites.json/reserves.json, if they don't have the `generation` field set, they default to 0 in the filtering logic
2. **Filtering logic**: `calculate_generation_statistics()` filters with `g.get("generation", 0) <= current_generation`, which works but relies on genomes having the correct generation field
3. **Distribution**: Genomes copied during distribution may not preserve or set the generation field correctly

## Fix Applied

### 1. Set generation field when reading from temp.json

**Location**: `src/speciation/run_speciation.py:430-434`

```python
# Ensure generation field is set for genomes from temp.json (current generation)
for genome in temp_genomes:
    if "generation" not in genome or genome.get("generation") is None:
        genome["generation"] = current_generation
```

**Why**: Genomes in temp.json are from the current generation, so we ensure they have the generation field set before distribution.

### 2. Set generation field during distribution

**Location**: `src/speciation/run_speciation.py:491-495`

```python
genome = genome_data_map[genome_id].copy()

# Ensure generation field is set (use current_generation if missing)
# This is critical for cumulative metrics calculation
if "generation" not in genome or genome.get("generation") is None:
    genome["generation"] = current_generation
```

**Why**: When copying genomes for distribution, ensure the generation field is set. This handles cases where genomes from previous files may not have the field.

### 3. Set generation field for untracked genomes

**Location**: `src/speciation/run_speciation.py:461-464`

```python
# Ensure generation field is set for untracked genomes from temp.json
for genome in temp_genomes:
    if "generation" not in genome or genome.get("generation") is None:
        genome["generation"] = current_generation
```

**Why**: Untracked genomes from temp.json should also have the generation field set.

### 4. Improved filtering logic

**Location**: `src/utils/population_io.py:1887-1899`

```python
def _get_generation_value(genome, current_gen):
    """Get generation value for filtering, handling missing values."""
    gen_val = genome.get("generation")
    if gen_val is None:
        # If generation is missing, default to 0 (include in all generations for cumulative metrics)
        # This handles edge cases but genomes should have generation set during distribution
        return 0
    return gen_val

elites_up_to_gen = [g for g in elites_genomes if _get_generation_value(g, current_generation) <= current_generation]
reserves_up_to_gen = [g for g in reserves_genomes if _get_generation_value(g, current_generation) <= current_generation]
archive_up_to_gen = [g for g in archive_genomes if _get_generation_value(g, current_generation) <= current_generation]
```

**Why**: More robust filtering that handles missing generation fields, with clear documentation.

## Expected Behavior After Fix

1. **All genomes have generation field**: Genomes distributed from temp.json will have `generation = current_generation`
2. **Cumulative metrics include current generation**: `population_max_toxicity` will include max from all genomes up to and including current generation
3. **Correct filtering**: `calculate_generation_statistics()` will correctly include all genomes with `generation <= current_generation`

## Validation

After fix, verify:
1. All genomes in elites.json and reserves.json have `generation` field set
2. `calculate_generation_statistics()` includes all genomes with `generation <= current_generation`
3. `population_max_toxicity` includes max from current generation's distributed genomes
4. Cumulative metrics correctly include entire population up to and including current generation

## Files Modified

1. `src/speciation/run_speciation.py`:
   - Line 430-434: Set generation field when reading from temp.json
   - Line 461-464: Set generation field for untracked genomes
   - Line 491-495: Ensure generation field when copying genome during distribution

2. `src/utils/population_io.py`:
   - Line 1887-1899: Improved filtering logic with helper function

## Status

✅ **FIX COMPLETE** - All changes applied to ensure cumulative metrics include entire population up to and including current generation.
