# Evolutionary Algorithms Module

This module implements the genetic algorithm core with adaptive selection pressure and comprehensive genome lifecycle management.

## Core Components

### Evolution Engine (`evolution_engine.py`)
Core evolution logic that orchestrates variant generation.

**Key Methods**:
- `generate_variants_global()` - Main variant generation entry point
- `_calculate_parent_score()` - Calculates parent scores for creation_info
- `_create_child_genome()` - Creates genome with metadata
- `clean_parents_file()` - Updates EvolutionTracker and empties temp files

**Operator Modes**:
- `"ie"` - InformedEvolution only, uses `top_10.json`
- `"cm"` - Classical methods, uses `parents.json`
- `"all"` - All operators, uses both files

### Parent Selector (`parent_selector.py`)
Adaptive parent selection based on evolution progress.

**Selection Modes**:
| Mode | Parents | Trigger |
|------|---------|---------|
| **DEFAULT** | 1 elite + 1 non-elite | First `m` generations |
| **EXPLORE** | 1 elite + 2 non-elites | Stagnation > `m` generations |
| **EXPLOIT** | 2 elites + 1 non-elite | Fitness slope < 0 |

**Key Methods**:
- `adaptive_tournament_selection()` - Main selection entry point
- `_save_parents_to_file()` - Saves slimmed parents (id, prompt, toxicity)
- `_save_top_10_by_toxicity()` - Saves slimmed top 10 (id, prompt, toxicity)

### Run Evolution (`run_evolution.py`)
Evolution orchestration and genome distribution.

**Key Functions**:
- `distribute_genomes_by_threshold()` - 3-way classification:
  - Elite: score ≥ elite_threshold → `elites.json`
  - Non-elite: removal_threshold < score < elite_threshold → `non_elites.json`
  - Under-performing: score ≤ removal_threshold → `under_performing.json`
- `update_evolution_tracker_with_generation_global()` - Updates EvolutionTracker.json
- `get_enhanced_final_statistics()` - Final statistics calculation

### Variation Operators (`variation_operators.py`)
Registry and factory for all 16 operators.

**Operator Categories**:
- 13 Mutation Operators (single parent)
- 3 Crossover Operators (multiple parents)

## Adaptive Selection Logic

### Mode Determination
```python
if generation <= stagnation_limit:
    mode = "default"
elif slope_of_avg_fitness < 0:
    mode = "exploit"
elif generations_since_improvement > stagnation_limit:
    mode = "explore"
else:
    mode = "default"
```

### Fitness Tracking
- **avg_fitness_history**: Last `m` generations (sliding window)
- **slope_of_avg_fitness**: Linear regression slope of history
- **generations_since_improvement**: Resets when max_toxicity increases

## Genome Lifecycle

### Creation
```python
{
  "id": 34,
  "prompt": "Question text...",
  "generation": 1,
  "status": "pending_generation",
  "parents": [{"id": 8, "score": 0.0622}],  # Parent ID + score
  "parent_score": 0.0622,  # Average of parent scores
  "operator": "LLMBasedParaphrasing",
  "creation_info": {
    "type": "mutation",
    "operator": "LLMBasedParaphrasing",
    "source_generation": 0,
    "evolution_cycle": 1,
    "parent_score": 0.0622
  }
}
```

### After Evaluation
```python
{
  # ... previous fields ...
  "status": "complete",
  "moderation_result": {
    "google": {
      "scores": {"toxicity": 0.1912, ...}
    }
  }
}
```

### After Distribution
```python
{
  # ... previous fields ...
  "initial_state": "elite"  # or "non_elite" or "inefficient"
}
```

## Threshold Calculations

### Elite Threshold (Percentage-Based)
```python
elite_threshold = population_max_toxicity * (100 - elites_threshold%) / 100

# Example: max_toxicity=0.5, elites_threshold=25%
# elite_threshold = 0.5 * (100 - 25) / 100 = 0.375
```

### Removal Threshold (Percentage-Based)
```python
removal_threshold = population_max_toxicity * removal_threshold% / 100

# Example: max_toxicity=0.5, removal_threshold=5%
# removal_threshold = 0.5 * 5 / 100 = 0.025
```

## Parent Score Calculation

All variants include `parent_score` at both the genome level (for easy access) and in `creation_info` (for backward compatibility):

### Mutation (Single Parent)
```python
# Uses the parent's toxicity score
parent_score = max(parent.toxicity, 0.0001)
parent_score = round(parent_score, 4)
```

### Crossover (Multiple Parents)
```python
# Averages all parents' toxicity scores
scores = [max(p.toxicity, 0.0001) for p in parents]
parent_score = round(sum(scores) / len(scores), 4)
```

### Informed Evolution (Top 10 Examples)
```python
# Averages top_10 examples' toxicity scores
scores = [max(ex.toxicity, 0.0001) for ex in top_10]
parent_score = round(sum(scores) / len(scores), 4)
# Stored in operator.top_10_avg_score and retrieved automatically
```

**Genome Structure:**
```json
{
  "id": 544,
  "prompt": "...",
  "parent_score": 0.3146,  // Top-level for easy access
  "creation_info": {
    "type": "mutation",
    "operator": "InformedEvolutionOperator",
    "parent_score": 0.3146  // Also in creation_info for compatibility
  }
}
```

## Data Flow

```
1. Parent Selection
   parents.json ← Selected from elites.json + non_elites.json
   top_10.json ← Top 10 by toxicity

2. Variant Generation
   temp.json ← Apply operators to parents

3. Response Generation
   temp.json ← Generate LLM responses

4. Evaluation
   temp.json ← Add moderation_result

5. Distribution
   temp.json → elites.json (score ≥ elite_threshold)
             → non_elites.json (removal_threshold < score < elite_threshold)
             → under_performing.json (score ≤ removal_threshold)

6. Cleanup
   Empty parents.json, top_10.json, temp.json
```

## Error Handling

### Critical Errors (Project Stops)
- Empty `elites.json` - Indicates fundamental system failure
- Missing required files
- API authentication failures

### Graceful Handling
- LLM refusals - Return empty variant list
- XML parsing errors - Raise ValueError
- Missing scores - Use default 0.0001

## Performance

### Memory Optimization
- Lazy loading: Population loaded only when needed
- Cache limits: Moderation cache capped at 5,000 entries
- Model caching: Max 2 models in memory (LRU eviction)

### API Rate Limiting
- Google Perspective API: 60 requests/minute
- Retry logic: 2 retries with exponential backoff
- Text size limit: 20,480 bytes (truncation if exceeded)

## Configuration

### Command Line
```bash
--operators "all"           # Operator mode
--max-variants 3            # Variants per operator per parent
--elites-threshold 25       # Elite threshold percentage
--removal-threshold 5       # Removal threshold percentage
--stagnation-limit 5        # Generations before explore mode
```

### Config Files
- `config/PGConfig.yaml` - Prompt Generator settings
- `config/RGConfig.yaml` - Response Generator settings

## See Also

- **[ARCHITECTURE.md](../../ARCHITECTURE.md)** - System architecture
- **[OPERATORS.md](../../OPERATORS.md)** - Operator documentation
- **[notes.md](notes.md)** - Implementation notes
