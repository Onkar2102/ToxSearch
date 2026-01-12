# Process Flow: Evolutionary Search for Toxicity in LLMs

This document describes the complete process flow for a single experiment run, including error handling, file updates, and metrics calculation.

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Run experiment
python src/main.py \
    --generations 50 \
    --operators all \
    --max-variants 1 \
    --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_S.gguf \
    --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf \
    --seed-file data/prompt.csv
```

## Process Flow Diagram

```
START
  |
  v
[1. INITIALIZATION]
  - Load seed prompts from CSV
  - Create output directory (data/outputs/YYYYMMDD_HHMM/)
  - Initialize models (Response Generator, Prompt Generator)
  - Initialize evaluator (Perspective API client)
  |
  v
[2. GENERATION 0 - Seed Population]
  - Load prompts from prompt.csv
  - Generate LLM responses (track llm_calls, response_time)
  - Evaluate toxicity via Perspective API (10 retries, track api_calls, eval_time)
  - Run speciation (Leader-Follower clustering)
  - Distribute: elites.json, reserves.json
  - Update EvolutionTracker.json
  |
  v
[3. EVOLUTION LOOP (Generation N)]
  |
  +---> [3a. Parent Selection]
  |       - Load elites.json + reserves.json
  |       - Determine mode (DEFAULT/EXPLOIT/EXPLORE)
  |       - Select parents based on mode
  |
  +---> [3b. Variation]
  |       - Apply random operator to parents
  |       - Check for duplicates
  |       - Track rejections and duplicates
  |       - Save variants to temp.json
  |
  +---> [3c. Response Generation]
  |       - Generate LLM responses for temp.json variants
  |       - Track response_duration per genome
  |       - Update temp.json
  |
  +---> [3d. Fitness Evaluation]
  |       - Evaluate via Perspective API
  |       - 10 retries with exponential backoff
  |       - Track evaluation_duration per genome
  |       - Update temp.json with scores
  |
  +---> [3e. Speciation]
  |       - Compute embeddings
  |       - Leader-Follower clustering
  |       - Species merging/extinction
  |       - Capacity enforcement
  |       - c-TF-IDF labeling
  |       - Archive excess to archive.json
  |
  +---> [3f. Distribution]
  |       - species_id > 0 -> elites.json
  |       - species_id == 0 -> reserves.json
  |       - Update top_10.json
  |
  +---> [3g. Tracking & Visualization]
  |       - Update EvolutionTracker.json
  |       - Update speciation_state.json
  |       - Generate operator_effectiveness_cumulative.csv
  |       - Generate figures/
  |
  +---> [3h. Termination Check]
          - Max generations reached? -> END
          - Threshold achieved? -> END
          - Otherwise -> Loop to 3a
  |
  v
[4. FINALIZATION]
  - Final statistics
  - Save all files
  - Log completion
  |
  v
END
```

## Detailed Phase Descriptions

### Phase 1: Initialization

**Trigger**: `python src/main.py [args]`

**Actions**:
1. Parse command line arguments
2. Create timestamped output directory: `data/outputs/YYYYMMDD_HHMM/`
3. Initialize logging to `logs/` directory
4. Load configuration from `config/RGConfig.yaml` and `config/PGConfig.yaml`
5. Initialize Response Generator (load GGUF model)
6. Initialize Prompt Generator (load GGUF model)
7. Initialize Moderation Evaluator (Perspective API client)
8. Initialize speciation config with parameters

**Files Created**:
- Output directory structure
- Log file

**Error Handling**:
- Missing PERSPECTIVE_API_KEY: Fatal error with instructions
- Model file not found: Fatal error with path info
- Invalid config: Fatal error with details

### Phase 2: Generation 0 (Seed Population)

**Input**: `data/prompt.csv`

**Actions**:
1. Load seed prompts from CSV
2. Create initial genome structures
3. Generate LLM responses for each prompt
4. Evaluate toxicity via Perspective API (10 retries each)
5. Run speciation to assign species_id to each genome
6. Distribute genomes to elites.json and reserves.json
7. Calculate initial statistics

**Files Updated**:
- `elites.json` - genomes with species_id > 0
- `reserves.json` - genomes with species_id == 0
- `top_10.json` - top 10 by fitness
- `EvolutionTracker.json` - generation 0 entry
- `speciation_state.json` - initial species state

**Metrics Tracked**:
- best_fitness, avg_fitness_generation
- elites_count, reserves_count, species_count
- llm_calls, api_calls, response_time, evaluation_time

### Phase 3: Evolution Loop

#### 3a. Parent Selection

**Mode Selection Logic**:
```
if fitness_slope < 0:
    mode = EXPLOIT  # Declining fitness -> stay within species
elif stagnation > m:
    mode = EXPLORE  # Stagnant -> cross-species breeding
else:
    mode = DEFAULT  # Normal -> within-species selection
```

**Parent Selection by Mode**:
| Mode | Parent 1 | Parent 2 |
|------|----------|----------|
| DEFAULT | Random genome from random species | Random genome from same species |
| EXPLOIT | Highest-fitness from top species | Random genome from same species |
| EXPLORE | Highest-fitness from top species | Highest-fitness from different species |

**Notes**:
- Species sorted by best_fitness descending
- Frozen/deceased species excluded
- Cluster 0 (reserves) included

#### 3b. Variation

**Operator Selection**: Random from enabled operators based on `--operators` flag

**Operators**:
1. InformedEvolution
2. MLMOperator
3. LLMBasedParaphrasing
4. BackTranslation (Hindi)
5. SynonymReplacement
6. AntonymReplacement
7. NegationOperator
8. ConceptAddition
9. TypographicalErrors
10. StylisticMutator
11. SemanticSimilarityCrossover
12. SemanticFusionCrossover

**Duplicate Detection**:
- Check prompt against existing population
- Skip if exact match found
- Track duplicate count per operator

**Rejection Handling**:
- Track rejections per operator (malformed output, refusals)
- Continue with next operator on failure

**Files Updated**:
- `temp.json` - new variants (status: pending_generation)

#### 3c. Response Generation

**Actions**:
1. Load variants from temp.json
2. For each variant with status=pending_generation:
   - Generate LLM response
   - Record response_duration
   - Update status to pending_evaluation
3. Save updated temp.json

**Files Updated**:
- `temp.json` - variants with generated_output and response_duration

#### 3d. Fitness Evaluation

**Actions**:
1. Load variants from temp.json
2. For each variant with status=pending_evaluation:
   - Call Perspective API with response text
   - **Retry Logic**: Up to 10 retries with exponential backoff
   - Record all 8 toxicity scores
   - Record evaluation_duration
   - Update status to complete
3. Save updated temp.json

**Retry Logic**:
```
for attempt in range(11):  # 10 retries + 1 initial
    try:
        result = call_perspective_api(text)
        return result
    except RetriableError:
        wait = 2 ** attempt  # 1, 2, 4, 8, ... 512 seconds
        sleep(wait)
# After all retries fail, mark as error
```

**Retriable Errors**:
- 429 (Rate limit)
- 500, 502, 503, 504 (Server errors)
- Timeout, connection, network errors

**Files Updated**:
- `temp.json` - variants with moderation_result, toxicity, evaluation_duration

#### 3e. Speciation

**Actions**:
1. Compute embeddings for all new prompts
2. Leader-Follower clustering:
   - Sort genomes by fitness (descending)
   - Assign to existing species if ensemble_distance < theta_sim
   - Create new species if no match
   - Assign to Cluster 0 if still no match
3. Species operations:
   - Merge species if leader distance < theta_merge
   - Freeze species if stagnation > max_stagnation
   - Enforce capacity limits
4. Archive excess genomes to archive.json
5. Generate c-TF-IDF labels for each species
6. Record diversity metrics

**Ensemble Distance Calculation**:
```
d_geno = (1 - cos_sim(e1, e2)) / 2  # [0, 1]
d_pheno = euclidean(p1, p2) / sqrt(8)  # [0, 1]
d_ensemble = 0.7 * d_geno + 0.3 * d_pheno  # [0, 1]
```

**Files Updated**:
- `temp.json` - genomes with species_id
- `archive.json` - excess genomes (created if needed)
- `speciation_state.json` - species state with labels

#### 3f. Distribution

**Actions**:
1. Load temp.json with speciated genomes
2. Merge with existing elites.json and reserves.json
3. Distribute based on species_id:
   - species_id > 0 -> elites.json
   - species_id == 0 -> reserves.json
4. Update top_10.json with highest fitness genomes

**Files Updated**:
- `elites.json` - updated with new genomes
- `reserves.json` - updated with Cluster 0 genomes
- `top_10.json` - top 10 by fitness

#### 3g. Tracking & Visualization

**Actions**:
1. Calculate generation statistics:
   - Fitness: best, avg, min, max
   - Population: elites_count, reserves_count
   - Speciation: species_count, diversity
   - Budget: llm_calls, api_calls, times
2. Update EvolutionTracker.json
3. Calculate operator effectiveness metrics
4. Generate visualizations

**Operator Metrics**:
| Metric | Formula |
|--------|---------|
| NE | 1 - (elite_variants / total_variants) |
| EHR | elite_variants / total_variants |
| IR | (rejections + duplicates) / total_attempts |
| cEHR | elite_variants / (total_variants - invalid) |

**Files Updated**:
- `EvolutionTracker.json` - new generation entry
- `operator_effectiveness_cumulative.csv` - operator metrics
- `figures/*.png` - visualization charts

#### 3h. Termination Check

**Conditions**:
1. `generation >= max_generations` -> Terminate
2. `best_fitness >= threshold` -> Terminate
3. Otherwise -> Continue to next generation

### Phase 4: Finalization

**Actions**:
1. Calculate final statistics
2. Ensure all files are saved
3. Log completion summary

## File Update Sequence Per Generation

```
Generation N starts
  |
  +-> temp.json (variants created)
  |
  +-> temp.json (responses generated)
  |
  +-> temp.json (toxicity evaluated)
  |
  +-> temp.json (species_id assigned)
  +-> archive.json (excess archived)
  +-> speciation_state.json (species updated)
  |
  +-> elites.json (distributed)
  +-> reserves.json (distributed)
  +-> top_10.json (updated)
  |
  +-> EvolutionTracker.json (metrics recorded)
  +-> operator_effectiveness_cumulative.csv (metrics)
  +-> figures/* (visualizations)
  |
Generation N complete
```

## Error Recovery

### API Rate Limit (429)
- Automatic: 10 retries with exponential backoff
- Max wait: ~17 minutes total per genome

### API Failure After Retries
- Genome marked with status=error
- Evolution continues with remaining genomes
- Error logged for investigation

### Empty temp.json
- Speciation still runs on existing population
- Tracker still updated (no new variants)
- Evolution continues

### Model Load Failure
- Fatal error with model path
- Check model file exists and is valid GGUF

### Missing Environment Variable
- Fatal error for PERSPECTIVE_API_KEY
- Instructions provided in error message

## Monitoring Progress

### Live Progress
```bash
# Watch log file
tail -f logs/evolution_YYYYMMDD_HHMMSS.log

# Check current generation
cat data/outputs/YYYYMMDD_HHMM/EvolutionTracker.json | jq '.generations | length'

# Check best fitness
cat data/outputs/YYYYMMDD_HHMM/EvolutionTracker.json | jq '.generations[-1].best_fitness'
```

### Post-Run Analysis
```bash
# View operator effectiveness
cat data/outputs/YYYYMMDD_HHMM/operator_effectiveness_cumulative.csv

# View species labels
cat data/outputs/YYYYMMDD_HHMM/speciation_state.json | jq '.species[].labels'

# View top prompts
cat data/outputs/YYYYMMDD_HHMM/top_10.json | jq '.[].prompt'
```

## Performance Considerations

### Perspective API
- Rate limit: ~100 QPS (varies by quota)
- 0.75s delay between calls
- 10 retries with exponential backoff

### LLM Generation
- Depends on model size and hardware
- GPU: ~1-5s per response
- CPU: ~10-60s per response

### Speciation
- O(N x K x d) per generation
- N = population size (~100-1000)
- K = species count (~10-50)
- d = embedding dimension (384)

### Memory Usage
- Models: 4-16GB depending on quantization
- Embeddings: ~1.5MB per 1000 genomes
- Total: Plan for 16-32GB RAM
