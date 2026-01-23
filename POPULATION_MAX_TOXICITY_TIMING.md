# Population Max Toxicity Calculation Timing

## Question
Is `population_max_toxicity` calculated post-distribution (after variants are distributed from temp.json)?

## Answer: **YES** ✅

## Flow Analysis

### 1. Speciation and Distribution (Inside `run_speciation()`)
- **Location**: `src/main.py:625-631`
- **Function**: `run_speciation()` is called
- **What happens**:
  - Phases 1-6: Processing, merging, capacity enforcement
  - **Phase 7**: Redistribution happens INSIDE `process_generation()` (line 1882)
  - Phase 8: Metrics update

### 2. Phase 7 Redistribution (Inside process_generation)
- **Location**: `src/speciation/run_speciation.py:1876-1886`
- **Function**: `phase8_redistribute_genomes()` called at line 1882
- **What happens**:
  - Genomes distributed from temp.json to elites.json, reserves.json, archive.json
  - **temp.json is cleared** (line 627-628 in phase8_redistribute_genomes)
  - Files are written atomically

### 3. calculate_generation_statistics() Call
- **Location**: `src/main.py:808-814`
- **When**: AFTER `run_speciation()` completes (line 808)
- **What it reads**:
  - `elites.json` (line 1849) - **Already updated by Phase 7**
  - `reserves.json` (line 1856) - **Already updated by Phase 7**
  - `temp.json` (line 1862) - **May be empty or cleared by Phase 7**

### 4. population_max_toxicity Calculation
- **Location**: `src/utils/population_io.py:1934`
- **Formula**: `max(all_scores)` where `all_scores = elite_scores + reserves_scores`
- **Source**: Only from elites.json + reserves.json (NOT from temp.json)
- **Timing**: POST-distribution (files already updated)

## Conclusion

✅ **YES, `population_max_toxicity` is calculated POST-distribution**

The calculation happens:
1. **AFTER** Phase 7 redistribution (genomes moved from temp.json to elites/reserves/archive)
2. **FROM** elites.json + reserves.json (which contain the distributed genomes)
3. **NOT FROM** temp.json (which may be empty or cleared)

This is **CORRECT** because:
- `population_max_toxicity` should represent the max fitness in the **distributed population** (elites + reserves)
- It should NOT include variants that are still in temp.json (before distribution)
- It should reflect the final state after all processing and distribution

## Code Evidence

```python
# src/utils/population_io.py:1928-1934
# avg_fitness_generation: mean over elites + reserves only (after distribution)
all_scores = elite_scores + reserves_scores
if all_scores:
    stats["avg_fitness_generation"] = round(sum(all_scores) / len(all_scores), 4)
    # population_max_toxicity (per-gen): max over elites+reserves; cumulative is
    # updated in update_evolution_tracker_with_statistics. Used for Pareto quality.
    stats["population_max_toxicity"] = round(max(all_scores), 4)
```

The comment explicitly states: "mean over elites + reserves only (after distribution)"

## Verification

The flow is:
1. `run_speciation()` → includes Phase 7 (redistribution) → files updated
2. `calculate_generation_statistics()` → reads updated files → calculates `population_max_toxicity`
3. `update_evolution_tracker_with_statistics()` → uses the calculated value

✅ **All correct - calculation happens post-distribution**
