# Generations Since Improvement Fix

## Issue

The `generations_since_improvement` field in EvolutionTracker.json was not updating correctly. It was showing 18 (equal to total_generations - 1) instead of correctly tracking stagnation.

## Root Cause

In `src/main.py` line 768, the code was trying to get `previous_max_toxicity` from the previous generation entry:

```python
previous_max_toxicity = prev_gen.get("population_max_toxicity", 0.0)
```

**Problem**: `population_max_toxicity` is NOT stored in per-generation entries - it's only at the tracker level. So this always returned `0.0`, causing the code to fall back to the cumulative `population_max_toxicity` from line 755.

**Result**: The comparison was:
- `current_max_toxicity`: max from temp.json (variants created THIS generation)
- `previous_max_toxicity`: cumulative max across ALL generations (from tracker level)

Since the cumulative max is almost always higher than the current generation's max, `generations_since_improvement` kept incrementing incorrectly.

## Fix

Changed the logic to use `max_score_variants` from the previous generation entry instead:

```python
# Get previous max from EvolutionTracker (for stagnation detection)
# Use max_score_variants from previous generation (not population_max_toxicity which is cumulative)
if generation_count > 1:
    try:
        evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
        if evolution_tracker_path.exists():
            with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                tracker = json.load(f)
            generations = tracker.get("generations", [])
            if generations and len(generations) >= 2:
                # Get max_score_variants from previous generation (generation_count - 1)
                prev_gen_num = generation_count - 1
                for gen in generations:
                    if gen.get("generation_number") == prev_gen_num:
                        previous_max_toxicity = gen.get("max_score_variants", 0.0001)
                        break
    except Exception as e:
        logger.debug(f"Failed to get previous max from tracker: {e}")
        previous_max_toxicity = 0.0001
```

## Why This Fix Works

Now the comparison is correct:
- `current_max_toxicity`: max from temp.json (variants created THIS generation)
- `previous_max_toxicity`: `max_score_variants` from PREVIOUS generation (variants created LAST generation)

This compares apples to apples: current generation variants vs previous generation variants.

## Expected Behavior After Fix

- If `current_max_toxicity > previous_max_toxicity`: `generations_since_improvement` resets to 0
- If `current_max_toxicity <= previous_max_toxicity`: `generations_since_improvement` increments by 1

This correctly tracks stagnation: how many generations have passed without improvement in variant max fitness.

## Files Changed

- `src/main.py`: Lines 750-770 (fixed previous_max_toxicity calculation)
