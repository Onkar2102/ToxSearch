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
   - Each species maintains up to 100 top genomes

3. **`reserves.json`** - **Cluster 0 (Holding Buffer)**
   - Stores genomes that don't fit existing species or were removed from species
   - Genomes below `removal_threshold` are removed and archived to `under_performing/`
   - Fixed capacity of 1000 genomes - excess removed based on fitness score
   - Cluster 0 individuals have a Time-To-Live (TTL) and can form new species if they cluster
   - Genomes can be promoted from Cluster 0 to species or repopulate extinct species
   - Has a fixed species_id of 0

4. **`EvolutionTracker.json`** - **Metadata & Statistics**
   - Tracks evolution progress, metrics, and configuration
   - Contains per-generation statistics (fitness, counts, thresholds)
   - Includes speciation summaries (species count, Cluster 0 size, etc.)
   - Used to resume evolution and track progress

5. **`speciation_state.json`** - **Speciation Module State**
   - Persists the full state of the speciation module across generations
   - Contains all species with their leaders, members, and metadata
   - Includes cluster origin tracking (merge, split, natural)
   - Contains Cluster 0 state (individuals, TTL, capacity)
   - Used internally by speciation module for state reconstruction

6. **`parents.json`** - **Parent Selection Log**
   - Records which genomes were selected as parents
   - Used for tracking genetic lineage

7. **`top_10.json`** - **Top Performers**
   - Stores the top 10 genomes from each generation
   - Used for analysis and reporting

### Archive Files

8. **`under_performing/`** - **Archive Directory**
   - Contains genomes removed from Cluster 0 due to low fitness
   - Genomes with fitness below `removal_threshold` are archived here
   - Also contains genomes removed when Cluster 0 exceeds capacity

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
    ├─ Elite genomes → elites.json (top 100 per species)
    └─ Non-elite & removed genomes → reserves.json (Cluster 0)
         │
         ▼
    ┌─────────────────┬─────────────────────┐
    │   elites.json   │  reserves.json (C0)    │
    └─────────────────┴─────────────────────┘
         │
         ▼
    [Cluster 0 Management]
    ├─ Filter by removal_threshold → under_performing/
    └─ Enforce capacity (max 1000)
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

### Step 4: **SPECIATION** (Dynamic Islands Integration)
- **Reads from**: `temp.json` (genomes with fitness scores)
- **What it does**:
  1. **Compute Embeddings**: Converts all prompts to 384-dimensional vectors using `all-MiniLM-L6-v2`
     - Adds `prompt_embedding` field (384-dim list) to each genome
    - L2-normalized embeddings live on the unit hypersphere; cosine distance yields **cone-shaped** clusters (θ_sim≈0.3–0.5 ≈ 45–70°)
    - Default path: direct 384D + Leader-Follower (fast, no training). Optional offline path: Parametric UMAP 384→16 + HDBSCAN/centroiding (better compression/visualization)
  2. **Leader-Follower Clustering**: 
     - Sorts genomes by fitness (descending)
     - For each genome, finds nearest species leader within constant `theta_sim` threshold
     - If no match, creates new species (cluster_origin="natural") or adds to Cluster 0
  3. **Assign Species IDs**: Adds `species_id` field to each genome (1+ for species, 0 for Cluster 0)
  4. **Mark Cluster 0 Members**: Adds `in_limbo` flag to genomes in Cluster 0
  5. **Species Management**:
     - Updates species leaders (highest fitness member)
     - Manages Cluster 0 (holding buffer for outliers)
     - All species have constant radius (`theta_sim`)
     - Tracks species metrics and cluster origins
- **Writes to**: `temp.json` (genomes updated with `prompt_embedding`, `species_id`, and `in_limbo` fields)
- **Note**: 
  - `prompt_embedding` field is stored as a list (JSON-compatible)
  - Species state persisted to `speciation_state.json`
  - Cluster origins tracked: "natural", "merge", or "split"
  - Persistent speciation data: `species_id`, `in_limbo`, cluster_origin, parent_ids

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
    - Each species maintains top 100 genomes by fitness
    - Used as primary parent pool for breeding
  - **Cluster 0** (non-elites, removed genomes, outliers) → `reserves.json`
    - Contains individuals below elite threshold
    - Contains genomes ejected from species due to capacity limits
    - Contains high-fitness outliers that don't fit existing species
    - Genomes below `removal_threshold` are archived to `under_performing/`
    - Maximum capacity of 1000 genomes (excess removed by fitness)
    - All Cluster 0 members have `species_id: 0`
- **Writes to**: `elites.json`, `reserves.json`
- **Note**: 
  - All distributed genomes retain their `species_id` field
  - Cluster 0 individuals have `species_id: 0` and `in_limbo: true`
  - Distribution is based on fitness thresholds and species membership
- **Clears**: `temp.json` (set to empty array `[]`)

### Step 7: Finalize Generation 0
- Updates `EvolutionTracker.json` with final counts and statistics
- Records speciation metrics (if tracked)
- Generation 0 is complete!

---

## Generation N: Evolution Loop (N >= 1)

For each generation, the system repeats this cycle:

### Step 1: Parent Selection & Variation (`run_evolution`)
- **Reads from**: `elites.json` and `reserves.json` (Cluster 0)
- Selects parents from the population pool
- Elite genomes are primary parent candidates
- Cluster 0 genomes can also serve as parent candidates
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

### Step 4: **SPECIATION** (Dynamic Islands Integration Point)
- **Reads from**: `temp.json` (genomes with fitness scores)
- **What it does**:
  1. **Compute Embeddings**: Converts all prompts to 384-dimensional vectors using `all-MiniLM-L6-v2`
     - Adds `prompt_embedding` field (384-dim list) to each genome
    - L2-normalized embeddings on the unit hypersphere; cosine distance creates **angular (cone)** clusters. θ_sim≈0.3–0.5 tunes cone width.
    - Optional: Pretrained Parametric UMAP 384→16 for offline analysis; default runtime uses direct 384D Leader-Follower for speed.
  2. **Leader-Follower Clustering**: 
     - Sorts genomes by fitness (descending)
     - For each genome, finds nearest species leader within constant `theta_sim` threshold
     - If no match, creates new species (cluster_origin="natural") or adds to Cluster 0
  3. **Assign Species IDs**: Adds `species_id` field to each genome (1+ for species, 0 for Cluster 0)
  4. **Mark Cluster 0 Members**: Adds `in_limbo` flag to genomes in Cluster 0
  5. **Species Management**:
     - Updates species leaders (highest fitness member)
     - Manages Cluster 0 (filters by removal_threshold, enforces capacity)
     - Processes merges (cluster_origin="merge", tracks parent_ids)
     - Processes splitting (cluster_origin="split")
     - Handles extinctions, migrations
     - All species maintain constant radius (`theta_sim`)
     - Tracks speciation metrics and cluster origins
- **Writes to**: `temp.json` (genomes updated with `prompt_embedding`, `species_id`, `in_limbo` fields)
- **Note**: Embeddings are added to genomes in `prompt_embedding` field (stored as JSON list)

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
    - Each species maintains top 100 genomes by fitness
  - **Cluster 0** (all non-elite genomes) → `reserves.json` (appended to existing Cluster 0)
    - All genomes below elite threshold
    - Genomes ejected from species due to capacity limits
    - Genomes that don't fit existing species
- **Post-Distribution Cleanup**:
  - Filter Cluster 0 by `removal_threshold` → archive removed to `under_performing/`
  - Enforce Cluster 0 capacity (max 1000) → archive excess to `under_performing/`
- **Writes to**: `elites.json`, `reserves.json`
- **Clears**: `temp.json` (set to empty array)

### Step 7: Cleanup & Archive
- **Reads from**: `elites.json`, `reserves.json`
- Removes low-performing genomes from active population
- Low performers (below `removal_threshold`) archived to `under_performing/`
- Excess Cluster 0 genomes (above 1000 capacity) archived to `under_performing/`
- Updates species membership if genomes are ejected
- **Writes to**: `elites.json`, `reserves.json` (updated)

### Step 8: Redistribute Population
- **Reads from**: `elites.json`, `reserves.json`
- Re-evaluates all genomes against current elite threshold
- Moves genomes between files if thresholds changed
- **Writes to**: `elites.json`, `reserves.json` (rebalanced)

### Step 9: Update Tracker
- **Reads from**: `elites.json`, `reserves.json`, `EvolutionTracker.json`
- Calculates generation statistics:
  - Average fitness (elites, Cluster 0, overall)
  - Counts (elites, Cluster 0 size, variants created)
  - Best/worst scores
  - Operator statistics
  - Speciation summary (species count, merges, splits, extinctions)
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

`elites.json` and `reserves.json` are **append-only** during evolution:
- New genomes are appended to existing ones
- Cleanup happens periodically to remove low performers
- Redistribution rebalances between files
- Cluster 0 (`reserves.json`) has capacity limit of 1000

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
  "prompt_embedding": [0.123, -0.456, ...],  // 384-dim array (added by speciation)
  "species_id": 3,  // Added by speciation module (0 = Cluster 0, 1+ = species)
  "in_limbo": false,  // Added by speciation (true if in Cluster 0)
  "moderation_result": {
    "scores": {"toxicity": 0.85},
    "classifications": {...}
  },
  "operator": "semantic_similarity_crossover",
  "creation_info": {...}
}
```

**Species ID Convention**:
- `species_id: 0` = Cluster 0 (holding buffer, non-elite genomes)
- `species_id: 1+` = Valid species/islands (elite clusters)

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
│ Step 1: Compute Embeddings                                  │
│  - Extract prompts from genomes in temp.json                │
│  - Compute L2-normalized embeddings (384-dim)               │
│  - Add "prompt_embedding" field to each genome              │
│  - Save updated genomes back to temp.json                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Leader-Follower Clustering                          │
│  - Sort individuals by fitness (descending)                 │
│  - For each individual:                                      │
│    ├─ Find nearest species leader                           │
│    ├─ If distance < theta_sim: add to species               │
│    ├─ Else if fitness > viability_baseline: add to Cluster 0│
│    └─ Else: create new species (cluster_origin="natural")   │
│  - All species have constant radius = theta_sim             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Cluster 0 Management                                │
│  - Update TTL for Cluster 0 individuals                     │
│  - Filter by removal_threshold (archive removed genomes)    │
│  - Enforce capacity (max 1000, archive excess)              │
│  - Check if clusters can form new species                   │
│  - Promote viable clusters to species (cluster_origin=      │
│    "natural")                                               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Species Updates                                     │
│  - Update species leaders (highest fitness member)          │
│  - Record fitness history for each species                  │
│  - Update species modes (DEFAULT/EXPLORE/EXPLOIT)           │
│  - Each species maintains top 100 members (excess to C0)    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Dynamic Operations                                  │
│  - Merge similar species (if distance < theta_merge)        │
│    └─ Creates new species with cluster_origin="merge"       │
│    └─ Records parent_ids=[id1, id2]                         │
│  - Split overcrowded species (cluster_origin="split")       │
│  - Extinct stagnant species (members → Cluster 0)           │
│  - Repopulate from Cluster 0 (cluster_origin="natural")     │
│  - Migrate individuals between related species              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Update Genomes                                      │
│  - Assign species_id to each genome (0 for Cluster 0)       │
│  - Mark in_limbo flag for Cluster 0 individuals             │
│  - Preserve all original genome data                        │
│  - Save state to speciation_state.json                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Output: temp.json (genomes with prompt_embedding, species_id, in_limbo added)
```

### What Speciation Does

**For Generation 0**:
1. **Reads** `temp.json` (initial population with fitness scores)
2. **Computes** embeddings for all prompts and saves to `temp.json`
3. **Clusters** genomes into species using Leader-Follower algorithm
   - First genome becomes first species leader (cluster_origin="natural")
   - Subsequent genomes assigned to nearest species or create new ones
   - All species have constant radius (`theta_sim`)
4. **Assigns** `species_id` to each genome (0 for Cluster 0, 1+ for species)
5. **Manages** initial species structure and Cluster 0
6. **Saves** speciation state to `speciation_state.json`
7. **Writes** back to `temp.json` with `species_id` added

**For Generation N**:
1. **Loads** existing speciation state from `speciation_state.json`
2. **Reads** `temp.json` (new variants with fitness scores)
3. **Clusters** new variants into existing or new species
4. **Manages** species dynamics:
   - Merging (cluster_origin="merge", records parent_ids)
   - Splitting (cluster_origin="split")
   - Extinction (members → Cluster 0)
   - Migration between species
5. **Manages** Cluster 0:
   - Filter by `removal_threshold` → archive to `under_performing/`
   - Enforce capacity (max 1000) → archive excess to `under_performing/`
   - Check for new species formation
6. **Assigns** `species_id` to each variant
7. **Saves** updated speciation state to `speciation_state.json`
8. **Writes** back to `temp.json` with `species_id` added

### What Gets Saved
- **`prompt_embedding`**: 384-dimensional L2-normalized embedding (saved in genome as list for JSON compatibility)
- **`species_id`**: Integer ID of the species/island (0 = Cluster 0, 1+ = species)
- **`in_limbo`**: Boolean flag indicating if genome is in Cluster 0 (saved in genome JSON)
- **Species metadata**: Saved to `speciation_state.json` including:
  - Leader embeddings, prompt, and fitness
  - Cluster origin ("natural", "merge", "split")
  - Parent IDs for merged clusters
  - Species mode (DEFAULT/EXPLORE/EXPLOIT)
- **Cluster 0 state**: Saved to `speciation_state.json` with ID 0

### Embedding Storage
- **Stored in**: `prompt_embedding` field in each genome (as list of floats, JSON-compatible)
- **Computation**: L2-normalized 384-dimensional vectors using `all-MiniLM-L6-v2` model
- **Persistence**: Embeddings are saved to `temp.json` and then to permanent files (`elites.json`, `reserves.json`)
- **Efficiency**: While embeddings add size (~1.5 KB per genome), they enable:
  - Fast re-clustering in subsequent generations
  - Offline analysis of species structure
  - Reduced computational overhead since speciation state is persisted
- **Processing**: When loading embeddings from JSON, convert list back to numpy array for distance computations

### Generation 0 Specifics

**Initial Clustering**:
- Generation 0 has no existing species
- First genome becomes the first species leader (cluster_origin="natural")
- All subsequent genomes are compared against existing species leaders
- High-fitness outliers may go to Cluster 0 (species_id=0)
- All species have constant radius = `theta_sim`

**Species Formation**:
- Species form naturally as genomes cluster by semantic similarity
- All new species have `cluster_origin: "natural"`
- Number of initial species depends on:
  - Diversity of seed prompts
  - `theta_sim` threshold (default: 0.4)
  - Fitness distribution

**Example**:
- 100 seed prompts → ~5-15 initial species (depending on diversity)
- Each species contains semantically similar prompts
- Species IDs assigned: 1, 2, 3, ... (sequential, 0 reserved for Cluster 0)
- Each species maintains up to 100 members (top by fitness)

---

## Example Flow: Generation 0

1. **Load Seeds**: 100 prompts from `data/prompt.csv` → `temp.json` (100 genomes)
2. **Generate**: Adds responses to 100 genomes → `temp.json` (100 genomes with responses)
3. **Moderate**: Adds scores to 100 genomes → `temp.json` (100 genomes with scores)
4. **Speciate**: 
   - Computes embeddings for 100 prompts and adds `prompt_embedding` field
   - Groups into 8 species based on semantic similarity (cluster_origin="natural")
   - Adds `species_id` (1-8) to each genome, 0 for Cluster 0
   - Marks 5 high-fitness outliers with `in_limbo: true` (species_id: 0)
   - Saves state to `speciation_state.json`
   - → `temp.json` (100 genomes with prompt_embedding, species_id, in_limbo)
5. **Thresholds**: Calculates elite_threshold = 0.75, removal_threshold = 0.10
6. **Distribute**: 
   - 25 genomes → `elites.json` (25 total, top 100 per species)
   - 75 genomes → `reserves.json` (Cluster 0, species_id: 0)
   - Cluster 0 cleanup: 10 below removal_threshold → `under_performing/`
7. **Finalize**: Updates `EvolutionTracker.json` with Generation 0 stats + speciation summary
8. **Complete**: Generation 0 done, ready for Generation 1

## Example Flow: Generation 1

1. **Start**: `elites.json` has 25 genomes, `reserves.json` has 65 genomes (Cluster 0)
2. **Load State**: Speciation module loads `speciation_state.json`
3. **run_evolution**: Selects 50 parents from elites + Cluster 0, creates 50 variants → `temp.json` (50 genomes)
4. **Generate**: Adds responses to 50 variants → `temp.json` (50 genomes with responses)
5. **Moderate**: Adds scores to 50 variants → `temp.json` (50 genomes with scores)
6. **Speciate**: 
   - Computes embeddings for 50 new variants and adds `prompt_embedding` field
   - Clusters into existing species (from Gen 0) or creates new ones
   - Adds `species_id` to each variant (0 for Cluster 0)
   - Marks Cluster 0 members with `in_limbo: true`
   - May merge species (cluster_origin="merge", records parent_ids)
   - May split overcrowded species (cluster_origin="split")
   - Saves updated state to `speciation_state.json`
   - → `temp.json` (50 genomes with prompt_embedding, species_id, in_limbo)
7. **Thresholds**: Calculates elite_threshold = 0.80
8. **Distribute**: 
   - 15 variants → `elites.json` (now 40 total, top 100 per species)
   - 35 variants → `reserves.json` (Cluster 0, species_id: 0)
9. **Cleanup**: 
   - Remove genomes below removal_threshold → `under_performing/`
   - Enforce Cluster 0 capacity (max 1000) → archive excess
10. **Redistribute**: Moves genomes between files based on new thresholds
11. **Update Tracker**: Records Generation 1 stats + speciation summary in `EvolutionTracker.json`
12. **Continue**: Loop to Generation 2

---

## Key Points

1. **File-based communication**: All components communicate through JSON files
2. **temp.json is transient**: Used as staging area, cleared after distribution
3. **Permanent files accumulate**: `elites.json` and `reserves.json` grow over time
4. **Cluster 0 has limits**: Maximum 1000 genomes, filtered by removal_threshold
5. **Tracker maintains state**: `EvolutionTracker.json` tracks evolution progress + speciation summary
6. **Speciation state persisted**: `speciation_state.json` stores full species structure
7. **Species ID convention**: 0 = Cluster 0, 1+ = valid species
8. **Constant radius**: All species have same radius threshold (`theta_sim`)
9. **Cluster origin tracking**: Species record how they formed (natural, merge, split)

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
- **Append Pattern**: New genomes are appended to `elites.json` and `reserves.json`
- **Growth Over Time**: These files grow as evolution progresses
- **Cluster 0 Pattern**: Non-elite genomes and outliers stored in `reserves.json` (Cluster 0)
- **Capacity Limits**: Cluster 0 has max capacity of 1000 genomes

#### 4. **Data Cleanup**
- **Periodic Cleanup**: Low performers removed from Cluster 0
- **Removal Threshold**: Genomes below `removal_threshold` are archived
- **Capacity Enforcement**: Excess Cluster 0 genomes (above 1000) are archived
- **Archive Location**: Removed genomes go to `under_performing/` directory
- **Cluster 0 Management**: Can form new species or repopulate extinct ones

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

#### Speciation State (`speciation_state.json`)

**Purpose**: Persists full species/island structure and Cluster 0 state

**Storage**: 
- **Persistent**: Full state saved to `speciation_state.json` each generation
- **Auto-Load**: Loaded at start of each generation's speciation process
- **Internal Use**: Used only within speciation module (not exposed externally)

**State Structure**:
```json
{
  "config": {...},
  "species": {
    "0": {
      "cluster_id": 0,
      "individuals": [...],
      "ttl": {...},
      "max_capacity": 1000,
      "removal_threshold": 0.1
    },
    "1": {
      "id": 1, "leader_id": 45, "members": [45, 67, 89],
      "cluster_origin": "natural", "parent_ids": null,
      "radius": 0.4, ...
    },
    "2": {
      "id": 2, "leader_id": 123, "members": [123, 145],
      "cluster_origin": "merge", "parent_ids": [3, 5],
      "radius": 0.4, ...
    }
  },
  "global_best_id": 45,
  "metrics": {...}
}
```

**Key Fields**:
- **`cluster_origin`**: How species was created ("natural", "merge", "split")
- **`parent_ids`**: For merged species, IDs of parent species [id1, id2]
- **`radius`**: Constant radius threshold (`theta_sim`) for all species
- **`max_capacity`**: Cluster 0 max size (1000)
- **`removal_threshold`**: Minimum fitness for Cluster 0 members

**When Saved**:
- **Every Generation**: Auto-saved after speciation processing
- **After Merges**: Records merged species with parent_ids
- **After Splits**: Records split species with cluster_origin="split"

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

#### Pattern 2: Accumulation File (`elites.json`)

**Lifecycle**:
1. **Initialized**: Empty array `[]` at start
2. **Appended**: Elite genomes added each generation
3. **Capacity Managed**: Each species maintains top 100 genomes by fitness
4. **Redistributed**: Genomes moved between elites/Cluster 0 based on thresholds

**Best Practices**:
- **Append During Evolution**: New elite genomes appended
- **Capacity Enforcement**: Excess genomes per species moved to Cluster 0
- **Validation**: Check file integrity before operations
- **Backup**: Consider backing up before major operations

#### Pattern 3: Cluster 0 File (`reserves.json`)

**Lifecycle**:
1. **Initialized**: Empty array `[]` at start
2. **Appended**: Non-elite genomes, ejected species members, outliers
3. **Filtered**: Genomes below `removal_threshold` archived to `under_performing/`
4. **Capacity Enforced**: Max 1000 genomes, excess archived by fitness
5. **Promoted**: Can form new species or repopulate extinct species

**Best Practices**:
- **TTL Management**: Update TTL for Cluster 0 individuals each generation
- **Threshold Filtering**: Remove genomes below `removal_threshold`
- **Capacity Enforcement**: Archive excess when above 1000
- **Species Formation**: Check if Cluster 0 clusters can form new species
- **Repopulation**: Use Cluster 0 individuals to repopulate extinct species

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
1. **Check Files**: System checks if `elites.json` or `reserves.json` exist
2. **Read Tracker**: Loads `EvolutionTracker.json` to find last generation
3. **Resume Point**: Continues from last completed generation
4. **State Reconstruction**: Loads speciation state from `speciation_state.json`

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

**Current Approach**: **Load from `speciation_state.json`**
- Full species structure is **persisted** to `speciation_state.json`
- Each generation, speciation module:
  1. Loads existing state from `speciation_state.json`
  2. Reads new genomes from `temp.json`
  3. Clusters new variants into existing or new species
  4. Updates Cluster 0 (filters, enforces capacity)
  5. Saves updated state back to `speciation_state.json`

**State Persistence Benefits**:
- Preserves species structure across generations
- Maintains cluster origin history (natural, merge, split)
- Tracks parent IDs for merged species
- Enables accurate speciation metrics over time
- Allows for resume without re-clustering entire population

### Data Cleanup Strategy

#### When Cleanup Happens

1. **After Distribution**: Low performers moved to `under_performing/` archive
2. **Cluster 0 Filtering**: Genomes below `removal_threshold` archived
3. **Capacity Enforcement**: Excess Cluster 0 genomes (above 1000) archived

#### Cleanup Process

**Step 1: Filter Cluster 0 by Removal Threshold**
- Compare Cluster 0 scores against `removal_threshold`
- Archive genomes below threshold to `under_performing/`
- Log removed IDs

**Step 2: Enforce Cluster 0 Capacity**
- Check if Cluster 0 size > 1000
- Archive excess genomes (lowest fitness first) to `under_performing/`
- Log removed IDs

**Step 3: Update Species Membership**
- Update species if genomes are ejected (capacity = 100 per species)
- Ejected genomes from species move to Cluster 0

#### Cleanup Frequency

- **Every Generation**: After distribution step
- **Cluster 0 Cleanup**: Each generation after clustering
- **Species Capacity**: Enforced during survivor selection

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
- `elites.json`: Grows slowly (top 100 per species × number of species)
- `reserves.json`: Bounded (max 1000 genomes in Cluster 0)
- `EvolutionTracker.json`: Grows linearly (one entry per generation)
- `speciation_state.json`: Grows slowly (species metadata only)

#### Size Limits

**Hard Limits**:
- **Species Capacity**: 100 genomes per species (enforced)
- **Cluster 0 Capacity**: 1000 genomes maximum (enforced)

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

