# How `generations_since_improvement` is Calculated

## Current Implementation

### Step-by-Step Flow

#### Step 1: Read Previous Cumulative Max (main.py, lines 806-816)

```python
previous_cumulative_population_max = 0.0001
if generation_count > 0 and evolution_tracker_path.exists():
    with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
        tracker_before = json.load(f)
    previous_cumulative_population_max = tracker_before.get("population_max_toxicity", 0.0001)
```

**Purpose**: Read the cumulative `population_max_toxicity` from the tracker **BEFORE** updating it for the current generation.

**Example for Gen 2**:
- `previous_cumulative_population_max = 0.1984` (cumulative max from Gen 0 and Gen 1)

#### Step 2: Calculate Current Generation Statistics (main.py, line 819)

```python
current_population_max = gen_stats.get("population_max_toxicity", 0.0001)
```

**Where `gen_stats` comes from**: `calculate_generation_statistics()` (population_io.py, lines 1805-1994)

**In `calculate_generation_statistics()`** (lines 1950-1956):
```python
# avg_fitness_generation: mean over elites + reserves only (after distribution)
all_scores = elite_scores + reserves_scores
if all_scores:
    stats["avg_fitness_generation"] = round(sum(all_scores) / len(all_scores), 4)
    # population_max_toxicity (per-gen): max over elites+reserves; cumulative is
    # updated in update_evolution_tracker_with_statistics. Used for Pareto quality.
    # NOTE: This is cumulative max (all genomes with generation <= current_generation)
    max_score = max(all_scores)
    if max_score > 0.0001:
        stats["population_max_toxicity"] = round(max_score, 4)
```

**Key Point**: The comment says "cumulative max (all genomes with generation <= current_generation)", but the code just does `max(all_scores)` where `all_scores` comes from `elite_scores + reserves_scores`.

**How scores are extracted** (lines 1885-1933):
```python
# Extract scores from elites
elite_scores = []
for g in elites_genomes:
    if g.get("generation", 0) <= current_generation:  # Filter by generation
        score = _extract_north_star_score(g, north_star_metric)
        if score > 0.0001:
            elite_scores.append(score)

# Extract scores from reserves
reserves_scores = []
for g in reserves_genomes:
    if g.get("generation", 0) <= current_generation:  # Filter by generation
        score = _extract_north_star_score(g, north_star_metric)
        if score > 0.0001:
            reserves_scores.append(score)
```

**Example for Gen 2**:
- `current_population_max = 0.2084` (max over all elites+reserves with generation <= 2)

#### Step 3: Update Tracker with Statistics (main.py, line 859)

```python
update_evolution_tracker_with_statistics(
    evolution_tracker_path=str(evolution_tracker_path),
    current_generation=generation_count,
    statistics=gen_stats,
    ...
)
```

**In `update_evolution_tracker_with_statistics()`** (lines 2190-2202):
```python
# Update population_max_toxicity at tracker level (cumulative max across all generations).
new_max = statistics.get("population_max_toxicity")
if new_max and new_max > 0.0001:
    if "population_max_toxicity" not in tracker:
        tracker["population_max_toxicity"] = 0.0001
    # Update to cumulative max (always keep the highest value seen)
    tracker["population_max_toxicity"] = max(
        tracker.get("population_max_toxicity", 0.0001),
        new_max
    )
```

**Example for Gen 2**:
- `tracker["population_max_toxicity"] = max(0.1984, 0.2084) = 0.2084`

#### Step 4: Update Adaptive Selection Logic (main.py, lines 874-883)

```python
adaptive_results = update_adaptive_selection_logic(
    outputs_path=outputs_path,
    current_max_toxicity=current_population_max,  # 0.2084
    previous_max_toxicity=previous_cumulative_population_max,  # 0.1984
    ...
)
```

#### Step 5: Compare and Update (update_adaptive_selection_logic, lines 1691-1696)

```python
# Update generations_since_improvement
# Use a small epsilon for comparison to handle floating-point precision
epsilon = 1e-6
if current_max_toxicity > previous_max_toxicity + epsilon:
    tracker["generations_since_improvement"] = 0
    _logger.info(f"Improvement detected! Max toxicity increased from {previous_max_toxicity:.4f} to {current_max_toxicity:.4f}")
else:
    tracker["generations_since_improvement"] = tracker.get("generations_since_improvement", 0) + 1
    _logger.info(f"No improvement. Generations since improvement: {tracker['generations_since_improvement']}")
```

**Example for Gen 2**:
- Comparison: `0.2084 > 0.1984 + 1e-6` → `TRUE`
- Result: `generations_since_improvement = 0`

---

## How It Should Be Calculated

### Correct Logic

1. **`current_max_toxicity`**: Cumulative maximum fitness over all genomes in `elites.json` + `reserves.json` with `generation <= current_generation`
   - This represents the best fitness found so far (including current generation)

2. **`previous_max_toxicity`**: Cumulative maximum fitness from the tracker (before update)
   - This represents the best fitness found in previous generations

3. **Comparison**: `current_max_toxicity > previous_max_toxicity + epsilon`
   - If `TRUE`: Improvement detected → `generations_since_improvement = 0`
   - If `FALSE`: No improvement → `generations_since_improvement = old_value + 1`

### Expected Behavior

**Generation 0**:
- `previous_max_toxicity = 0.0000` (initial)
- `current_max_toxicity = 0.1052` (max from Gen 0 elites+reserves)
- Comparison: `0.1052 > 0.0000` → **IMPROVEMENT**
- Result: `generations_since_improvement = 0`

**Generation 1**:
- `previous_max_toxicity = 0.1052` (cumulative max from Gen 0)
- `current_max_toxicity = 0.1984` (max from Gen 0+1 elites+reserves)
- Comparison: `0.1984 > 0.1052` → **IMPROVEMENT**
- Result: `generations_since_improvement = 0`

**Generation 2**:
- `previous_max_toxicity = 0.1984` (cumulative max from Gen 0+1)
- `current_max_toxicity = 0.2084` (max from Gen 0+1+2 elites+reserves)
- Comparison: `0.2084 > 0.1984` → **IMPROVEMENT**
- Result: `generations_since_improvement = 0`

**Generation 3** (if no improvement):
- `previous_max_toxicity = 0.2084` (cumulative max from Gen 0+1+2)
- `current_max_toxicity = 0.2084` (max from Gen 0+1+2+3 elites+reserves, same as before)
- Comparison: `0.2084 > 0.2084` → **NO IMPROVEMENT**
- Result: `generations_since_improvement = 0 + 1 = 1`

---

## Potential Issues

### Issue 1: Generation Filtering

**Problem**: In `calculate_generation_statistics()`, the code filters genomes by generation:
```python
if g.get("generation", 0) <= current_generation:
```

**Potential Issue**: If genomes are missing the `generation` field, they might be excluded or included incorrectly.

**Solution**: Ensure all genomes have a valid `generation` field set correctly.

### Issue 2: Score Extraction

**Problem**: The `_extract_north_star_score()` function might not extract scores correctly for some genomes.

**Potential Issue**: If scores are not extracted correctly, `all_scores` might be missing valid scores, leading to incorrect `max_score`.

**Solution**: Verify that `_extract_north_star_score()` correctly extracts scores from all genome formats.

### Issue 3: Timing of Comparison

**Problem**: The comparison happens AFTER `update_evolution_tracker_with_statistics()` updates the tracker.

**Current Flow**:
1. Read `previous_max_toxicity` from tracker (BEFORE update) ✅
2. Calculate `current_max_toxicity` from gen_stats ✅
3. Update tracker with new cumulative max ✅
4. Compare `current_max_toxicity` vs `previous_max_toxicity` ✅

**This is CORRECT** - the comparison uses values from before the update.

### Issue 4: Cumulative vs Per-Generation Max

**Problem**: The comment in `calculate_generation_statistics()` says "cumulative max", but the implementation just does `max(all_scores)` where `all_scores` includes all genomes with `generation <= current_generation`.

**This is CORRECT** - since `elites.json` and `reserves.json` are cumulative files (all genomes from all generations), and we filter by `generation <= current_generation`, the max is indeed cumulative.

**However**: If genomes are missing from `elites.json` or `reserves.json` (e.g., archived), they won't be included in the calculation. This is **intentional** - `population_max_toxicity` is the max over the **active population** (elites + reserves), not including archived genomes.

---

## Summary

### Current Implementation

The logic is **correct in principle**, but there might be issues with:
1. **Generation filtering**: Ensuring all genomes have correct `generation` field
2. **Score extraction**: Ensuring all scores are extracted correctly
3. **File state**: Ensuring `elites.json` and `reserves.json` contain all expected genomes

### Expected Behavior

- `generations_since_improvement` should be **0** whenever `current_max_toxicity > `previous_max_toxicity`
- `generations_since_improvement` should **increment** when there's no improvement
- The comparison uses cumulative max values (all generations up to current)

### For the Current Execution (20260123_1854)

**Expected**: `generations_since_improvement = 0` (because Gen 2 had improvement: 0.1984 → 0.2084)

**Actual**: `generations_since_improvement = 2` ❌

**Possible Causes**:
1. `current_population_max` was calculated incorrectly (not including Gen 2 genomes)
2. `previous_cumulative_population_max` was read incorrectly
3. The comparison logic had a bug (now fixed)
4. The values passed to `update_adaptive_selection_logic()` were wrong

---

## Verification Steps

To verify the calculation is working correctly:

1. **Check logs** for "Adaptive selection comparison" messages
2. **Check logs** for "Improvement detected!" or "No improvement" messages
3. **Verify** that `current_max_toxicity` and `previous_max_toxicity` values in logs match expected values
4. **Check** that `gen_stats["population_max_toxicity"]` matches the actual max from `elites.json` + `reserves.json`
