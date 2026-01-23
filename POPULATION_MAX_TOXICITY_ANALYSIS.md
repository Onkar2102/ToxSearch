# Population Max Toxicity Update Analysis

## Issue

Checking if `population_max_toxicity` is being updated correctly.

## Current Implementation

### Update Location 1: `update_evolution_tracker_with_statistics()`
- **File**: `src/utils/population_io.py:2159-2171`
- **Source**: `statistics.get("population_max_toxicity")`
- **Logic**: Uses `max()` to keep cumulative max

### Update Location 2: `update_evolution_tracker_with_speciation()`
- **File**: `src/speciation/run_speciation.py:3481-3491`
- **Source**: `best_fitness_value` (from metrics or calculated from elites+reserves)
- **Logic**: Uses `max()` to keep cumulative max

## Value Sources

### Source 1: `calculate_generation_statistics()`
- **File**: `src/utils/population_io.py:1934`
- **Value**: `max(all_scores)` where `all_scores = elite_scores + reserves_scores`
- **Meaning**: Max fitness from elites.json + reserves.json (after distribution)
- **Comment**: "population_max_toxicity (per-gen): max over elites+reserves"

### Source 2: `main.py` override
- **File**: `src/main.py:823`
- **Value**: `max_toxicity` (from temp.json variants)
- **Meaning**: Max fitness from temp.json (variants created this generation)
- **Issue**: This OVERRIDES the value from `calculate_generation_statistics()`

## The Problem

There's a **CONFLICT** between two sources:

1. **`calculate_generation_statistics()`** sets `population_max_toxicity` to max from **elites+reserves** (after distribution)
2. **`main.py`** overrides it with max from **temp.json** (variants before speciation)

According to the comment in `update_evolution_tracker_with_speciation()`:
> "population_max_toxicity = max over all generations of (best toxicity in that generation's population, i.e. elites + reserves)"

**The correct value should be max from elites+reserves, NOT from temp.json variants.**

## Expected Behavior

`population_max_toxicity` should be:
- **Per-generation value**: Max fitness in elites+reserves for that generation
- **Cumulative value**: Max across all generations of the per-generation values

## Current Behavior

1. `calculate_generation_statistics()` calculates max from elites+reserves → `stats["population_max_toxicity"]`
2. `main.py` overrides it with max from temp.json → `gen_stats["population_max_toxicity"] = max_toxicity`
3. `update_evolution_tracker_with_statistics()` uses the overridden value (from temp.json)
4. This is **INCORRECT** because it uses variant max instead of population max

## Fix Required

Remove the override in `main.py` line 823:
```python
# REMOVE THIS LINE:
gen_stats["population_max_toxicity"] = max_toxicity
```

The value from `calculate_generation_statistics()` is already correct (max from elites+reserves).

## Verification

Check the actual value in EvolutionTracker.json:
- Current: `"population_max_toxicity": 0.3271`
- Max from generations: Need to check max of all `max_score_variants` values
- Max from elites+reserves: Should be calculated from actual files

Let me check what the actual max should be...
