# ToxSearch Process Flow (Plain English)

## Overview

ToxSearch is an evolutionary text generation system that evolves prompts to maximize toxicity scores. The system uses **file-based communication** - all components read from and write to JSON files in the `data/outputs/` directory. This document explains how the process works step-by-step.

---

## File Structure

All files are stored in `data/outputs/[timestamp]/` directory:

### Core Files

1. **`temp.json`** - **Staging Area**
   - Temporary file that holds genomes being processed in the current step
   - Gets written to, read from, and cleared throughout the process
   - Acts as a "workbench" for operations

2. **`elites.json`** - **Elite Genomes (All Species)**
   - Stores high-performing genomes (toxicity >= elite_threshold) from ALL species
   - Contains elites regardless of which species they belong to
   - These are the best candidates found so far
   - Persists across generations
   - Used as parent pool for breeding

3. **`non_elites.json`** - **Non-Elite Genomes**
   - Stores genomes that are above removal threshold but below elite threshold
   - Also contains genomes that were removed from species (ejected members)
   - These are candidates for further evolution
   - Used as parent pool for breeding

4. **`limbo.json`** - **Limbo Buffer**
   - Stores high-fitness outliers that don't fit existing species
   - Genomes in limbo have fitness > viability_baseline but are semantically distant from all species
   - Limbo individuals have a Time-To-Live (TTL) and can form new species if they cluster
   - Genomes can be promoted from limbo to species or repopulate extinct species

5. **`EvolutionTracker.json`** - **Metadata & Statistics**
   - Tracks evolution progress, metrics, and configuration
   - Contains per-generation statistics (fitness, counts, thresholds)
   - Used to resume evolution and track progress

6. **`parents.json`** - **Parent Selection Log**
   - Records which genomes were selected as parents
   - Used for tracking genetic lineage

7. **`top_10.json`** - **Top Performers**
   - Stores the top 10 genomes from each generation
   - Used for analysis and reporting

---

## Generation 0: Initial Population Setup

### Generation 0 Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATION 0 FLOW                            │
└─────────────────────────────────────────────────────────────────┘

Step 1: Load Seed Prompts
    data/prompt.csv
         │
         ▼
    temp.json (initial genomes)

Step 2: Generate Responses
    temp.json
         │
         ▼
    [Response Generator]
         │
         ▼
    temp.json (with responses)

Step 3: Evaluate (Moderation)
    temp.json
         │
         ▼
    [Moderation Models]
         │
         ▼
    temp.json (with fitness scores)

Step 4: SPECIATION ⭐
    temp.json (with scores)
         │
         ▼
    [Speciation Module]
    ├─ Compute embeddings
    ├─ Leader-Follower clustering
    ├─ Assign species_id
    └─ Manage species dynamics
         │
         ▼
    temp.json (with species_id)

Step 5: Calculate Thresholds
    temp.json
         │
         ▼
    [Threshold Calculator]
         │
         ▼
    EvolutionTracker.json

Step 6: Distribute Genomes
    temp.json (with species_id)
         │
         ▼
    [Distribution Logic]
    ├─ Elite genomes → elites.json
    ├─ Non-elite genomes → non_elites.json
    └─ Limbo individuals → limbo.json
         │
         ▼
    ┌─────────────┬──────────────┬──────────────┐
    │ elites.json │non_elites.json│  limbo.json  │
    └─────────────┴──────────────┴──────────────┘
         │
         ▼
    temp.json (cleared)

Step 7: Finalize Generation 0
    Update EvolutionTracker.json
    Generation 0 Complete!
```

### Step 1: Load Seed Prompts
- Reads prompts from `data/prompt.csv`
- Creates initial genome objects with:
  - `id`: Unique identifier
  - `prompt`: The text prompt
  - `generation`: 0
  - `status`: "pending_generation"
  - `variant_type`: "initial"
- **Writes to**: `temp.json`

### Step 2: Generate Responses
- **Reads from**: `temp.json`
- Uses Response Generator (RG) model to generate responses for each prompt
- Updates genomes with generated responses
- **Writes to**: `temp.json` (updated with responses)

### Step 3: Evaluate (Moderation)
- **Reads from**: `temp.json`
- Runs moderation models to calculate toxicity scores
- Updates genomes with:
  - `moderation_result`: Scores and classifications
  - `toxicity`: Main fitness score
  - `north_star_score`: Primary metric
  - `status`: "complete"
- **Writes to**: `temp.json` (updated with scores)

### Step 4: **SPECIATION** (Plan A+ Integration)
- **Reads from**: `temp.json` (genomes with fitness scores)
- **What it does**:
  1. **Compute Embeddings**: Converts all prompts to 384-dimensional vectors using `all-MiniLM-L6-v2`
  2. **Leader-Follower Clustering**: 
     - Sorts genomes by fitness (descending)
     - For each genome, finds nearest species leader within `theta_sim` threshold
     - If no match, creates new species or adds to limbo buffer
  3. **Assign Species IDs**: Adds `species_id` field to each genome
  4. **Species Management**:
     - Updates species leaders (highest fitness member)
     - Manages limbo buffer (high-fitness outliers)
     - Tracks species metrics
- **Writes to**: `temp.json` (genomes updated with `species_id` field)
- **Note**: 
  - Embeddings computed in-memory only (not saved)
  - Species structure is ephemeral (reconstructed each generation)
  - `species_id` is the only persistent speciation data

### Step 5: Calculate Thresholds
- **Reads from**: `temp.json` (now with `species_id`)
- Calculates:
  - `elite_threshold`: Top 25% (or configurable) toxicity score
  - `removal_threshold`: Bottom 5% (or configurable) toxicity score
- **Writes to**: `EvolutionTracker.json` (thresholds and generation 0 stats)

### Step 6: Distribute Genomes
- **Reads from**: `temp.json` (with `species_id` and thresholds)
- **Distribution Logic**:
  - **Elites** (toxicity >= elite_threshold) → `elites.json`
    - Contains elites from ALL species
    - Used as primary parent pool for breeding
  - **Non-elites** (removal_threshold < toxicity < elite_threshold) → `non_elites.json`
    - Contains non-elite genomes from all species
    - Also contains genomes ejected from species (if any)
    - Used as secondary parent pool for breeding
  - **Limbo** (high-fitness outliers in limbo buffer) → `limbo.json`
    - Contains individuals that don't fit existing species
    - These have fitness > viability_baseline but are semantically distant
    - Can form new species or repopulate extinct species
- **Writes to**: `elites.json`, `non_elites.json`, `limbo.json`
- **Note**: 
  - All distributed genomes retain their `species_id` field
  - Limbo individuals may not have a `species_id` (or have `null`)
  - Distribution is based on fitness thresholds, not species membership
- **Clears**: `temp.json` (set to empty array `[]`)

### Step 7: Finalize Generation 0
- Updates `EvolutionTracker.json` with final counts and statistics
- Records speciation metrics (if tracked)
- Generation 0 is complete!

---

## Generation N: Evolution Loop (N >= 1)

For each generation, the system repeats this cycle:

### Step 1: Parent Selection & Variation (`run_evolution`)
- **Reads from**: `elites.json` and `non_elites.json`
- Selects parents from the population pool
- Applies genetic operators:
  - **Mutation**: Modifies a single parent's prompt
  - **Crossover**: Combines two parents' prompts
- Creates new variant genomes with:
  - `variant_type`: "mutation" or "crossover"
  - `parents`: IDs of parent genomes
  - `generation`: Current generation number
  - `status`: "pending_generation"
- **Writes to**: `temp.json` (new variants)

### Step 2: Generate Responses
- **Reads from**: `temp.json`
- Uses Response Generator to generate responses for new variants
- **Writes to**: `temp.json` (updated with responses)

### Step 3: Evaluate (Moderation)
- **Reads from**: `temp.json`
- Runs moderation to calculate toxicity scores
- Updates genomes with fitness scores
- **Writes to**: `temp.json` (updated with scores)

### Step 4: **SPECIATION** (Plan A+ Integration Point)
- **Reads from**: `temp.json` (genomes with fitness scores)
- **What it does**:
  1. **Compute Embeddings**: Converts all prompts to 384-dimensional vectors using `all-MiniLM-L6-v2`
  2. **Leader-Follower Clustering**: 
     - Sorts genomes by fitness (descending)
     - For each genome, finds nearest species leader within `theta_sim` threshold
     - If no match, creates new species or adds to limbo buffer
  3. **Assign Species IDs**: Adds `species_id` field to each genome
  4. **Species Management**:
     - Updates species leaders (highest fitness member)
     - Manages limbo buffer (high-fitness outliers)
     - Processes merges, extinctions, migrations
     - Tracks speciation metrics
- **Writes to**: `temp.json` (genomes updated with `species_id` field)
- **Note**: Embeddings are computed in-memory only, NOT saved to files

### Step 5: Calculate Thresholds
- **Reads from**: `temp.json`, `elites.json`
- Recalculates elite and removal thresholds based on current population
- Updates adaptive selection mode (DEFAULT/EXPLORE/EXPLOIT)
- **Writes to**: `EvolutionTracker.json` (updated thresholds and stats)

### Step 6: Distribute Genomes
- **Reads from**: `temp.json` (with `species_id`)
- Distributes evaluated variants into:
  - **Elites** (toxicity >= elite_threshold) → `elites.json` (appended to existing elites)
    - Elites from all species are stored together
  - **Non-elites** (removal_threshold < toxicity < elite_threshold) → `non_elites.json` (appended to existing non-elites)
    - Non-elites from all species
    - Also includes genomes ejected from species
  - **Limbo** (individuals in limbo buffer) → `limbo.json` (appended to existing limbo)
    - High-fitness outliers that don't fit existing species
- **Writes to**: `elites.json`, `non_elites.json`, `limbo.json`
- **Clears**: `temp.json` (set to empty array)

### Step 7: Cleanup & Archive
- **Reads from**: `elites.json`, `non_elites.json`, `limbo.json`
- Removes low-performing genomes from active population
- Low performers are removed (not archived to a separate file)
- Updates species membership if genomes are ejected
- **Writes to**: `elites.json`, `non_elites.json`, `limbo.json` (updated)

### Step 8: Redistribute Population
- **Reads from**: `elites.json`, `non_elites.json`
- Re-evaluates all genomes against current elite threshold
- Moves genomes between files if thresholds changed
- **Writes to**: `elites.json`, `non_elites.json` (rebalanced)

### Step 9: Update Tracker
- **Reads from**: `elites.json`, `non_elites.json`, `EvolutionTracker.json`
- Calculates generation statistics:
  - Average fitness (elites, non-elites, overall)
  - Counts (elites, variants created)
  - Best/worst scores
  - Operator statistics
- **Writes to**: `EvolutionTracker.json` (generation N entry added)

### Step 10: Check Termination
- **Reads from**: `EvolutionTracker.json`
- Checks if max generations reached or threshold achieved
- If not, loop continues to Generation N+1

---

## File Communication Pattern

### The "temp.json" Workflow

`temp.json` acts as a **staging area** that gets passed between components:

```
[Component A] → writes to temp.json
[Component B] → reads from temp.json, processes, writes back
[Component C] → reads from temp.json, processes, writes back
[Final Step] → reads from temp.json, distributes to permanent files, clears temp.json
```

### The "Permanent Files" Pattern

`elites.json` and `non_elites.json` are **append-only** during evolution:
- New genomes are appended to existing ones
- Cleanup happens periodically to remove low performers
- Redistribution rebalances between files

### The "Tracker" Pattern

`EvolutionTracker.json` is **read-modify-write**:
- Read current state
- Update with new generation data
- Write back (overwrites entire file)

---

## Genome Data Structure

Each genome in JSON files has this structure:

```json
{
  "id": 123,
  "prompt": "How can I...",
  "generation": 5,
  "status": "complete",
  "variant_type": "crossover",
  "parents": [45, 67],
  "toxicity": 0.85,
  "north_star_score": 0.85,
  "species_id": 3,  // Added by speciation module
  "moderation_result": {
    "scores": {"toxicity": 0.85},
    "classifications": {...}
  },
  "operator": "semantic_similarity_crossover",
  "creation_info": {...}
}
```

---

## Speciation Integration Details

### When Speciation Runs
- **After** fitness evaluation (Step 3 in Generation 0, Step 3 in Generation N)
- **Before** threshold calculation (Step 5 in Generation 0, Step 5 in Generation N)
- Operates on `temp.json` which contains newly evaluated genomes/variants

### Speciation Process Flow

```
┌─────────────────────────────────────────────────────────────┐
│              SPECIATION MODULE PROCESS                       │
└─────────────────────────────────────────────────────────────┘

Input: temp.json (genomes with fitness scores)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Prepare Individuals                                 │
│  - Extract prompts from genomes                             │
│  - Compute embeddings (384-dim vectors)                    │
│  - Create Individual objects with fitness + embedding       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Leader-Follower Clustering                         │
│  - Sort individuals by fitness (descending)                 │
│  - For each individual:                                      │
│    ├─ Find nearest species leader                          │
│    ├─ If distance < theta_sim: add to species              │
│    ├─ Else if fitness > viability_baseline: add to limbo   │
│    └─ Else: create new species                              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Limbo Management                                    │
│  - Update TTL for limbo individuals                         │
│  - Check if limbo cluster can form new species              │
│  - Promote viable clusters to species                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Species Updates                                     │
│  - Update species leaders (highest fitness member)         │
│  - Record fitness history for each species                  │
│  - Update species modes (DEFAULT/EXPLORE/EXPLOIT)            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Dynamic Operations                                  │
│  - Merge similar species (if distance < theta_merge)       │
│  - Extinct stagnant species                                 │
│  - Repopulate extinct species                                │
│  - Migrate individuals between related species               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Update Genomes                                      │
│  - Assign species_id to each genome                         │
│  - Preserve all original genome data                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Output: temp.json (genomes with species_id added)
```

### What Speciation Does

**For Generation 0**:
1. **Reads** `temp.json` (initial population with fitness scores)
2. **Computes** embeddings for all prompts (in-memory, not saved)
3. **Clusters** genomes into species using Leader-Follower algorithm
   - First genome becomes first species leader
   - Subsequent genomes assigned to nearest species or create new ones
4. **Assigns** `species_id` to each genome
5. **Manages** initial species structure
6. **Writes** back to `temp.json` with `species_id` added

**For Generation N**:
1. **Reads** `temp.json` (new variants with fitness scores)
2. **Reconstructs** existing species from `elites.json` and `non_elites.json` (if needed)
3. **Clusters** new variants into existing or new species
4. **Manages** species dynamics (merging, extinction, migration)
5. **Assigns** `species_id` to each variant
6. **Writes** back to `temp.json` with `species_id` added

### What Gets Saved
- **`species_id`**: Integer ID of the species/island (saved in genome JSON)
- **Embeddings**: NOT saved (computed fresh each generation)
- **Species metadata**: Tracked in-memory, can be logged to EvolutionTracker if needed
- **Species structure**: Reconstructed each generation from `species_id` in genomes

### Why Embeddings Aren't Saved
- Embeddings are 384-dimensional vectors (~1.5 KB each)
- Prompts evolve each generation, so most embeddings would be new anyway
- Computation is fast (100-5000 prompts/sec on GPU)
- Would bloat JSON files significantly (1.5 MB for 1000 genomes)

### Generation 0 Specifics

**Initial Clustering**:
- Generation 0 has no existing species
- First genome becomes the first species leader
- All subsequent genomes are compared against existing species leaders
- High-fitness outliers may go to limbo buffer

**Species Formation**:
- Species form naturally as genomes cluster by semantic similarity
- Number of initial species depends on:
  - Diversity of seed prompts
  - `theta_sim` threshold (default: 0.4)
  - Fitness distribution

**Example**:
- 100 seed prompts → ~5-15 initial species (depending on diversity)
- Each species contains semantically similar prompts
- Species IDs assigned: 1, 2, 3, ... (sequential)

---

## Example Flow: Generation 0

1. **Load Seeds**: 100 prompts from `data/prompt.csv` → `temp.json` (100 genomes)
2. **Generate**: Adds responses to 100 genomes → `temp.json` (100 genomes with responses)
3. **Moderate**: Adds scores to 100 genomes → `temp.json` (100 genomes with scores)
4. **Speciate**: 
   - Computes embeddings for 100 prompts
   - Groups into 8 species based on semantic similarity
   - Adds `species_id` (1-8) to each genome
   - 5 high-fitness outliers go to limbo buffer
   - → `temp.json` (100 genomes with species_id, limbo info tracked)
5. **Thresholds**: Calculates elite_threshold = 0.75, removal_threshold = 0.10
6. **Distribute**: 
   - 25 genomes → `elites.json` (25 total, from all species)
   - 70 genomes → `non_elites.json` (70 total, from all species)
   - 5 genomes → `limbo.json` (5 total, high-fitness outliers)
7. **Finalize**: Updates `EvolutionTracker.json` with Generation 0 stats
8. **Complete**: Generation 0 done, ready for Generation 1

## Example Flow: Generation 1

1. **Start**: `elites.json` has 25 genomes, `non_elites.json` has 70 genomes, `limbo.json` has 5 genomes
2. **run_evolution**: Selects 50 parents, creates 50 variants → `temp.json` (50 genomes)
3. **Generate**: Adds responses to 50 variants → `temp.json` (50 genomes with responses)
4. **Moderate**: Adds scores to 50 variants → `temp.json` (50 genomes with scores)
5. **Speciate**: 
   - Computes embeddings for 50 new variants
   - Clusters into existing species (from Gen 0) or creates new ones
   - Adds `species_id` to each variant
   - 2 high-fitness outliers go to limbo buffer
   - → `temp.json` (50 genomes with species_id)
6. **Thresholds**: Calculates elite_threshold = 0.80
7. **Distribute**: 
   - 15 variants → `elites.json` (now 40 total, from all species)
   - 30 variants → `non_elites.json` (now 100 total, from all species)
   - 2 variants → `limbo.json` (now 7 total, high-fitness outliers)
8. **Cleanup**: Removes 10 low performers from `non_elites.json` (removed, not archived)
9. **Redistribute**: Moves 5 genomes between files based on new thresholds
10. **Update Tracker**: Records Generation 1 stats in `EvolutionTracker.json`
11. **Continue**: Loop to Generation 2

---

## Key Points

1. **File-based communication**: All components communicate through JSON files
2. **temp.json is transient**: Used as staging area, cleared after distribution
3. **Permanent files accumulate**: `elites.json` and `non_elites.json` grow over time
4. **Tracker maintains state**: `EvolutionTracker.json` tracks evolution progress
5. **Speciation adds metadata**: Adds `species_id` to genomes, doesn't change file structure
6. **Embeddings are ephemeral**: Computed in-memory, not persisted

---

## Benefits of File-Based Communication

1. **Modularity**: Each component can be run independently
2. **Debugging**: Can inspect intermediate states by reading JSON files
3. **Resumability**: Can resume evolution from any point by reading files
4. **Transparency**: All data is human-readable JSON
5. **Flexibility**: Easy to add new components that read/write the same files

---

## Data Management Strategy

### Overview

ToxSearch uses a **file-based state management** approach where all data persistence happens through JSON files. This section explains how data is managed, persisted, cleaned, and recovered.

### Data Lifecycle

#### 1. **Data Creation**
- **Initial Population**: Created from seed file (`data/prompt.csv`)
- **Variants**: Created by genetic operators (mutation/crossover)
- **Metadata**: Generated during evaluation and speciation

#### 2. **Data Persistence**
- **Immediate Write**: Data is written to files immediately after each operation
- **No Batching**: Each component writes its results directly (no transaction system)
- **Atomic Operations**: File writes are atomic (complete or fail, no partial writes)

#### 3. **Data Accumulation**
- **Append Pattern**: New genomes are appended to `elites.json`, `non_elites.json`, and `limbo.json`
- **Growth Over Time**: These files grow as evolution progresses
- **Limbo Pattern**: High-fitness outliers stored in `limbo.json` (can form new species or repopulate extinct ones)

#### 4. **Data Cleanup**
- **Periodic Cleanup**: Low performers removed from active population
- **Removal Pattern**: Low performers are removed (not archived to a separate file)
- **Limbo Management**: High-fitness outliers stored in `limbo.json` (can form new species)
- **No Permanent Archive**: Low performers are simply removed, not kept in a separate archive

### State Management

#### Evolution State (`EvolutionTracker.json`)

**Purpose**: Tracks overall evolution progress and configuration

**Structure**:
```json
{
  "status": "not_complete",
  "total_generations": 5,
  "generations_since_improvement": 2,
  "selection_mode": "default",
  "population_max_toxicity": 0.95,
  "generations": [
    {
      "generation_number": 0,
      "elites_count": 25,
      "avg_fitness": 0.75,
      "elite_threshold": 0.80,
      ...
    }
  ]
}
```

**Management**:
- **Read-Modify-Write**: Entire file is read, modified, written back
- **Per-Generation Updates**: Each generation adds/updates its entry
- **Resume Capability**: System reads this file to determine current generation

#### Speciation State (Optional)

**Purpose**: Tracks species/island structure and limbo buffer

**Storage**: 
- **In-Memory**: Species structure maintained in memory during processing
- **Optional Persistence**: Can be saved to `speciation_state.json` if needed
- **Reconstruction**: Species can be reconstructed from `species_id` in genomes

**State Structure**:
```json
{
  "config": {...},
  "species": {
    "1": {"leader_id": 45, "members": [45, 67, 89], ...},
    "2": {"leader_id": 123, "members": [123, 145], ...}
  },
  "limbo": {"individuals": [...], "ttl": {...}},
  "global_best_id": 45,
  "metrics": {...}
}
```

**When to Save**:
- **Checkpointing**: Before major operations
- **Resume**: If speciation state needs to persist across restarts
- **Analysis**: For post-hoc analysis of species evolution

**Note**: Currently, speciation state is **ephemeral** - species are reconstructed each generation from genome `species_id` fields. This is sufficient because:
- Species structure changes each generation
- Genomes already have `species_id` assigned
- Reconstruction is fast (O(n) operation)

### File Management Patterns

#### Pattern 1: Staging File (`temp.json`)

**Lifecycle**:
1. **Created**: When new variants are generated
2. **Updated**: Multiple times during processing (responses, scores, species_id)
3. **Read**: By distribution step
4. **Cleared**: Set to empty array `[]` after distribution

**Best Practices**:
- Always check if file exists before reading
- Handle empty file gracefully
- Clear after successful distribution
- Don't rely on `temp.json` for persistence

#### Pattern 2: Accumulation Files (`elites.json`, `non_elites.json`)

**Lifecycle**:
1. **Initialized**: Empty arrays `[]` at start
2. **Appended**: New genomes added each generation
3. **Cleaned**: Low performers removed periodically
4. **Redistributed**: Genomes moved between files based on thresholds

**Best Practices**:
- **Append-Only During Evolution**: Don't overwrite, append new genomes
- **Periodic Cleanup**: Remove low performers to prevent unbounded growth
- **Validation**: Check file integrity before operations
- **Backup**: Consider backing up before major cleanup operations

#### Pattern 3: Limbo File (`limbo.json`)

**Lifecycle**:
1. **Initialized**: Empty array `[]` at start
2. **Appended**: High-fitness outliers added here (don't fit existing species)
3. **Managed**: Limbo individuals have TTL and can form new species
4. **Promoted**: Can be moved to species or used to repopulate extinct species

**Best Practices**:
- **TTL Management**: Update TTL for limbo individuals each generation
- **Species Formation**: Check if limbo clusters can form new species
- **Repopulation**: Use limbo individuals to repopulate extinct species
- **Size Control**: Limbo size controlled by TTL expiration

#### Pattern 4: Metadata File (`EvolutionTracker.json`)

**Lifecycle**:
1. **Created**: During Generation 0 initialization
2. **Updated**: Each generation adds/updates entry
3. **Read**: At startup to resume evolution

**Best Practices**:
- **Read-Modify-Write**: Always read full file, modify, write back
- **Backup**: Critical file - consider backups
- **Validation**: Validate structure before using

### Data Integrity

#### Validation

**Genome Validation**:
- **Required Fields**: `id`, `prompt`, `generation`, `status`
- **Type Checking**: Ensure scores are numbers, IDs are integers
- **Cleanup Function**: `clean_population()` removes invalid entries

**File Validation**:
- **JSON Parsing**: Handle malformed JSON gracefully
- **Empty Files**: Treat empty files as empty arrays
- **Missing Files**: Create with default structure if missing

#### Error Handling

**File Read Errors**:
- **Fallback**: Try alternative file locations
- **Logging**: Log errors but continue if possible
- **Recovery**: Initialize with defaults if file corrupted

**File Write Errors**:
- **Atomic Writes**: Write to temp file, then rename (if needed)
- **Error Propagation**: Fail fast if critical write fails
- **Logging**: Always log write failures

### Resume and Recovery

#### Resuming Evolution

**How It Works**:
1. **Check Files**: System checks if `elites.json` or `non_elites.json` exist
2. **Read Tracker**: Loads `EvolutionTracker.json` to find last generation
3. **Resume Point**: Continues from last completed generation
4. **State Reconstruction**: Reconstructs state from files

**Resume Logic** (from `main.py`):
```python
# Check if EvolutionTracker exists
if evolution_tracker_path.exists():
    # Read last generation number
    max_generation = max(gen.get("generation_number", 0) 
                        for gen in existing_generations)
    generation_count = max_generation
    # Resume from next generation
else:
    # Start fresh
    generation_count = 0
```

#### Speciation Resume

**Current Approach**: **Reconstruct from Genomes**
- Species structure is **not persisted** between runs
- Each generation, speciation module:
  1. Reads all genomes from `temp.json`
  2. Reconstructs species from `species_id` in genomes
  3. Re-clusters if needed (for new variants)
  4. Assigns new `species_id` if genome doesn't have one

**Alternative Approach**: **Persist Speciation State**
- Save speciation state to `speciation_state.json`
- Load state at startup
- Continue with existing species structure
- **Trade-off**: More complex, but preserves species history

**Recommendation**: Current approach (reconstruction) is sufficient because:
- Species evolve each generation anyway
- Genomes already have `species_id` for tracking
- Simpler and more robust

### Data Cleanup Strategy

#### When Cleanup Happens

1. **After Distribution**: Low performers moved to archive
2. **Periodic Cleanup**: Removes low performers from active files
3. **Before Redistribution**: Ensures clean state

#### Cleanup Process

**Step 1: Identify Low Performers**
- Compare scores against `removal_threshold`
- Mark genomes for removal

**Step 2: Remove Low Performers**
- Remove from `elites.json` and `non_elites.json`
- Low performers are not archived (simply removed)
- Log removed IDs

**Step 3: Update Species Membership**
- Update species if genomes are ejected
- Move ejected genomes to `non_elites.json` if they're above removal threshold

#### Cleanup Frequency

- **Every Generation**: After distribution step
- **Before Redistribution**: Ensures clean state
- **Configurable**: Can be adjusted based on file sizes

### Backup Strategy

#### Automatic Backups

**Generation Files Backup**:
- Before major operations, system backs up generation files
- Stored in `generations_backup/` directory
- Format: `gen{N}.json` for generation N

**When Backups Created**:
- Before population redistribution
- Before major cleanup operations
- Can be triggered manually

#### Manual Backups

**Recommended Practices**:
1. **Before Long Runs**: Backup entire `data/outputs/[timestamp]/` directory
2. **Before Experiments**: Copy directory before testing new parameters
3. **Regular Snapshots**: Periodically backup EvolutionTracker.json

### Data Size Management

#### File Size Growth

**Expected Growth**:
- `elites.json`: Grows slowly (only high performers from all species)
- `non_elites.json`: Grows moderately (breeding pool, includes ejected genomes)
- `limbo.json`: Grows/shrinks dynamically (high-fitness outliers, TTL-managed)
- `EvolutionTracker.json`: Grows linearly (one entry per generation)

#### Size Limits

**No Hard Limits**: System doesn't enforce file size limits

**Practical Considerations**:
- **Memory**: Large files take longer to load
- **Performance**: JSON parsing slows with large files
- **Disk Space**: Monitor disk usage for long runs

**Mitigation**:
- **Periodic Cleanup**: Remove low performers from active files
- **Archive Management**: Consider compressing old archives
- **File Splitting**: Can split large files by generation (future enhancement)

### Data Access Patterns

#### Lazy Loading

**Genome Loading**:
- Genomes loaded only when needed
- Cached in memory after first load
- Example: `EvolutionEngine` uses lazy loading

```python
@property
def genomes(self):
    if not self._genomes_loaded:
        self._genomes_cache = load_population(...)
        self._genomes_loaded = True
    return self._genomes_cache
```

#### Caching Strategy

**In-Memory Caches**:
- **Genomes**: Cached after first load
- **Embeddings**: Computed fresh each generation (not cached)
- **Species**: Reconstructed each generation (not cached)

**Cache Invalidation**:
- **After Write**: Caches invalidated after file writes
- **Generation Boundary**: Caches cleared between generations
- **Manual**: Can be manually cleared if needed

### Data Consistency

#### Consistency Guarantees

**File-Level Consistency**:
- **Atomic Writes**: File writes are atomic (complete or fail)
- **No Partial Updates**: Files are written completely, not incrementally
- **Read Consistency**: Files are read in full, not streamed

**Cross-File Consistency**:
- **No Transactions**: No multi-file transaction system
- **Best Effort**: System tries to keep files consistent
- **Recovery**: Can recover from inconsistencies by re-reading files

#### Consistency Checks

**ID Uniqueness**:
- System ensures genome IDs are unique
- `next_id` tracked to prevent reuse

**Generation Alignment**:
- All genomes in a generation have same `generation` number
- EvolutionTracker tracks generation numbers

**Species Alignment**:
- Genomes have `species_id` matching current species structure
- Re-clustered each generation to maintain consistency

### Best Practices

#### For Developers

1. **Always Validate**: Check file existence and structure before reading
2. **Handle Errors**: Gracefully handle file errors, don't crash
3. **Log Operations**: Log all file read/write operations
4. **Clear Temp**: Always clear `temp.json` after use
5. **Append Pattern**: Use append for accumulation files
6. **Backup Before Major Ops**: Backup before cleanup/redistribution

#### For Operations

1. **Monitor File Sizes**: Watch for unbounded growth
2. **Regular Backups**: Backup important runs
3. **Disk Space**: Monitor disk usage
4. **Resume Capability**: System can resume, but verify state first
5. **Archive Management**: Consider compressing old archives

### Future Enhancements

#### Potential Improvements

1. **Database Backend**: Replace JSON files with database (SQLite/PostgreSQL)
2. **Incremental Updates**: Update files incrementally instead of full rewrite
3. **Compression**: Compress old archives
4. **Checkpointing**: More frequent checkpoints for recovery
5. **Data Versioning**: Track data schema versions
6. **Query Interface**: Add query interface for large datasets

