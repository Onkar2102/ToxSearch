# Variant Statistics Calculation Fix

## Summary

Fixed critical issues with variant statistics calculation in EvolutionTracker. The system now correctly calculates `max_score_variants`, `min_score_variants`, and `avg_fitness_variants` from `temp.json` BEFORE distribution.

## Issues Fixed

### ❌ Issue 1: `max_score_variants` was NOT calculated
**Before:**
- Only `min_score_variants` and `avg_fitness_variants` were calculated
- `max_score` was calculated from the entire population AFTER distribution
- No way to track the maximum score of variants created in each generation

**After:**
- ✅ `max_score_variants` is now calculated from temp.json
- ✅ Correctly represents the max score of variants BEFORE distribution
- ✅ Provides accurate tracking of variant performance

### ❌ Issue 2: Zero scores were filtered out
**Before:**
```python
scores = [s for s in scores if s > 0]  # Filter out zero scores
```
- This removed genomes with score 0.0
- Made min/avg calculations inaccurate
- Lost information about poorly performing variants

**After:**
```python
# DO NOT filter out zero scores - we want ALL variant scores
scores = [_extract_north_star_score(v, north_star_metric) for v in temp_variants if v]
```
- ✅ ALL scores are included (even 0.0)
- ✅ Accurate min/avg calculations
- ✅ Complete variant performance data

### ❌ Issue 3: `max_score_variants` was never stored
**Before:**
- Only `min_score` and `avg_fitness_variants` were stored in EvolutionTracker
- No field for maximum variant score
- Incomplete tracking data

**After:**
- ✅ Added `max_score_variants` field to EvolutionTracker
- ✅ Added `min_score_variants` field (clearer naming)
- ✅ Kept `min_score` for backward compatibility
- ✅ Complete variant performance tracking

## Files Modified

### 1. `src/main.py`
**Lines 591-616**: Calculate variant statistics from temp.json
```python
# Calculate variant statistics from temp.json BEFORE distribution
temp_path = get_outputs_path() / "temp.json"
max_score_variants = 0.0001
min_score_variants = 0.0001
avg_fitness_variants = 0.0001
try:
    with open(temp_path, 'r', encoding='utf-8') as f:
        temp_variants = json.load(f)
    
    if temp_variants:
        from utils.population_io import _extract_north_star_score
        scores = [_extract_north_star_score(v, north_star_metric) for v in temp_variants if v]
        # DO NOT filter out zero scores - we want ALL variant scores for accurate statistics
        
        if scores:
            max_score_variants = round(max(scores), 4)
            min_score_variants = round(min(scores), 4)
            avg_fitness_variants = round(sum(scores) / len(scores), 4)
            logger.info(f"Variant statistics from temp.json ({len(scores)} variants): "
                      f"max={max_score_variants:.4f}, min={min_score_variants:.4f}, avg={avg_fitness_variants:.4f}")
```

**Lines 709-714**: Store variant statistics in EvolutionTracker
```python
# Variant statistics from temp.json (BEFORE distribution)
gen["max_score_variants"] = max_score_variants
gen["min_score_variants"] = min_score_variants
gen["avg_fitness_variants"] = avg_fitness_variants
# Keep legacy field for backward compatibility
gen["min_score"] = min_score_variants
```

**Lines 393-396**: Initialize variant fields for generation 0
```python
# No variants generated yet - set all variant stats to default
gen["max_score_variants"] = 0.0001
gen["min_score_variants"] = 0.0001
gen["avg_fitness_variants"] = 0.0001
```

**Lines 721-727**: Enhanced logging with all variant statistics
```python
logger.info(f"Updated generation {generation_count} with comprehensive metrics: "
           f"elites_count={redistribution_result['elites_count']}, "
           f"non_elites_count={redistribution_result['non_elites_count']}, "
           f"removal_threshold={removal_threshold_value:.4f}, "
           f"avg_fitness_elites={avg_fitness_elites:.4f}, "
           f"avg_fitness_non_elites={avg_fitness_non_elites:.4f}, "
           f"variants: max={max_score_variants:.4f}, min={min_score_variants:.4f}, avg={avg_fitness_variants:.4f}")
```

### 2. `src/ea/run_evolution.py`
**Lines 332-335**: Initialize variant fields in generation 0
```python
# Variant statistics from temp.json (before distribution)
"max_score_variants": 0.0001,
"min_score_variants": 0.0001,
"avg_fitness_variants": 0.0001,
```

**Lines 473-476**: Initialize variant fields in new generation entries
```python
# Variant statistics from temp.json (before distribution)
"max_score_variants": 0.0001,
"min_score_variants": 0.0001,
"avg_fitness_variants": 0.0001,
```

### 3. `src/ea/evolution_engine.py`
**Lines 635-638**: Initialize variant fields when creating generation entries
```python
# Variant statistics from temp.json (before distribution)
"max_score_variants": 0.0001,
"min_score_variants": 0.0001,
"avg_fitness_variants": 0.0001,
```

## EvolutionTracker Schema

### New/Modified Fields

Each generation entry in EvolutionTracker now includes:

```json
{
  "generation_number": 1,
  "genome_id": "best_genome_id",
  
  // Population-wide scores (after distribution)
  "max_score": 0.4257,          // Best score in entire population
  "avg_fitness": 0.0593,        // Average fitness of all genomes
  
  // Variant statistics (from temp.json BEFORE distribution)
  "max_score_variants": 0.3456, // ✅ NEW: Max score of variants created this generation
  "min_score_variants": 0.0123, // ✅ NEW: Min score of variants created this generation
  "avg_fitness_variants": 0.1789, // Average score of variants created this generation
  
  // Legacy field (kept for backward compatibility)
  "min_score": 0.0123,          // Same as min_score_variants
  
  // Population statistics (after distribution)
  "avg_fitness_generation": 0.0596,
  "avg_fitness_elites": 0.3858,
  "avg_fitness_non_elites": 0.0578,
  
  // Variant counts
  "variants_created": 10,
  "mutation_variants": 7,
  "crossover_variants": 3,
  
  // ... other fields ...
}
```

## Testing

Run the test script to verify the fixes:

```bash
python3 test_variant_statistics.py
```

**Expected output:**
```
✅ PASS: max_score_variants correct
✅ PASS: min_score_variants correct (includes zero scores)
✅ PASS: avg_fitness_variants correct
```

**Note:** Existing EvolutionTracker.json files won't have the new fields. The new fields will only appear in runs performed AFTER this fix.

## Clarification: Two Different "max_score" Metrics

### `max_score` (population-wide)
- Calculated from ENTIRE population after distribution
- Represents the best genome across all genomes (elites + non_elites)
- Used for overall evolution tracking
- Location: Calculated in `src/ea/run_evolution.py` (lines 387-410)

### `max_score_variants` (generation-specific)
- ✅ **NEW**: Calculated from temp.json BEFORE distribution
- Represents the best variant CREATED in this generation
- Used for variant performance tracking
- Location: Calculated in `src/main.py` (lines 591-616)

## Impact

### Before the Fix
- ❌ Missing data about variant performance
- ❌ Incomplete statistics (no max_score_variants)
- ❌ Inaccurate min/avg (zero scores filtered out)
- ❌ No way to track how well variants perform before distribution

### After the Fix
- ✅ Complete variant performance data
- ✅ All three statistics: max, min, avg
- ✅ Accurate calculations (includes all scores)
- ✅ Can track variant quality vs population quality
- ✅ Better insights into evolution effectiveness

## Backward Compatibility

- ✅ `min_score` field is kept for backward compatibility
- ✅ Existing code will continue to work
- ✅ New fields are additive (don't break existing logic)
- ✅ Default values (0.0001) used when no variants exist

## Future Improvements

Consider these enhancements:

1. **Variant Success Rate**: Track what percentage of variants become elites
2. **Operator Performance**: Break down variant statistics by operator type
3. **Score Distribution**: Track quartiles or percentiles of variant scores
4. **Improvement Tracking**: Compare variant scores to parent scores

## Questions?

If you see warnings like:
```
⚠️ WARNING: Using default values (0.0001) - likely no calculation performed
```

This is normal when:
- No variants were created in that generation (`variants_created: 0`)
- temp.json is empty
- The generation is generation 0 (no variants yet)

## Summary

This fix ensures **accurate and complete tracking** of variant performance throughout evolution. The system now properly calculates and stores max/min/avg statistics for variants from temp.json before they're distributed to elites/non_elites files.

**Key Change**: We now track BOTH population-wide performance (`max_score`) AND generation-specific variant performance (`max_score_variants`, `min_score_variants`, `avg_fitness_variants`).

