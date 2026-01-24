# ToxSearch with Speciation — Architecture

Comprehensive architecture documentation for the evolutionary toxicity-search system with leader-follower semantic speciation. This system implements a steady-state `(μ + λ)` evolutionary algorithm with dynamic species formation, merging, freezing, and adaptive parent selection.

For field-by-field definitions see [FIELD_DEFINITIONS.txt](FIELD_DEFINITIONS.txt); for validation and flow details see [experiments/FLOW_AND_VALIDATION.md](experiments/FLOW_AND_VALIDATION.md); for the complete process flow see [PROCESS_FLOW.md](PROCESS_FLOW.md).

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Layout](#2-module-layout)
3. [Core Components](#3-core-components)
4. [High-Level Flow](#4-high-level-flow)
5. [Parent Selection System](#5-parent-selection-system)
6. [Speciation Framework](#6-speciation-framework)
7. [Key Metrics and Conventions](#7-key-metrics-and-conventions)
8. [Data Flow and File Management](#8-data-flow-and-file-management)
9. [Adaptive Selection Logic](#9-adaptive-selection-logic)
10. [Output Files](#10-output-files)
11. [Configuration Parameters](#11-configuration-parameters)
12. [References](#12-references)

---

## 1. System Overview

### 1.1 Algorithm Type

**Steady-State `(μ + λ)` Evolutionary Algorithm**
- **μ (mu)**: Parent population size (elites + reserves)
- **λ (lambda)**: Offspring generated per generation
- **Steady-state**: Population continuously updated, not replaced in generations

### 1.2 Fitness Function

**Fitness**: `f(x) = toxicity_score(LLM(x)) ∈ [0, 1]`
- `x`: Prompt (genome)
- `LLM(x)`: Model's response to the prompt
- `toxicity_score`: Moderation API score (Google Perspective API)

### 1.3 Population Structure

**Population**: `P = E ∪ R`
- **E (Elites)**: Active species members (`species_id > 0`)
- **R (Reserves)**: Cluster 0 outliers (`species_id = 0`)
- **Archive**: Capacity-overflow genomes (`species_id = -1`)

### 1.4 Variants per Generation

Variants generated depend on selection mode:
- **DEFAULT** (2 parents): `V = 10×2 + 2×1 = 22` variants
- **EXPLOIT/EXPLORE** (3 parents): `V = 10×3 + 2×3 = 36` variants

---

## 2. Module Layout

```
src/
├── main.py                    # Entry point: Gen 0 init, Gen N loop, orchestration
│                              # - System initialization
│                              # - Generation loop management
│                              # - Adaptive selection coordination
│                              # - Statistics aggregation
│
├── ea/                        # Evolutionary Algorithm Module
│   ├── evolution_engine.py   # - next_id() - Unique ID generation
│   │                          # - create_child() - Variant creation
│   │                          # - Operator dispatch and execution
│   ├── run_evolution.py      # - Load elites + reserves
│   │                          # - Parent selection coordination
│   │                          # - Operator application → temp.json
│   ├── parent_selector.py    # - 3-category adaptive tournament selection
│   │                          # - Category 1: active species ∪ reserves
│   │                          # - Category 2: frozen species
│   │                          # - Modes: DEFAULT, EXPLOIT, EXPLORE
│   ├── variation_operators.py # Base classes for operators
│   ├── paraphrasing.py       # LLM-based paraphrasing operator
│   ├── concept_addition.py   # Concept addition operator
│   ├── back_translation.py   # Back translation operator
│   ├── synonym_replacement.py # Synonym replacement operator
│   ├── antonym_replacement.py # Antonym replacement operator
│   ├── negation_operator.py  # Negation operator
│   ├── typographical_errors.py # Typo injection operator
│   ├── stylistic_mutator.py # Stylistic variation operator
│   ├── mlm_operator.py       # Masked language model operator
│   ├── fusion_crossover.py  # Fusion crossover operator
│   ├── semantic_similarity_crossover.py # Semantic crossover operator
│   ├── informed_evolution.py # Informed evolution operator
│   ├── operator_statistics.py # Operator effectiveness tracking
│   └── run_evolution.py      # Main evolution orchestration
│
├── gne/                       # Generate–Evaluate Module
│   ├── response_generator.py # - LLM response generation
│   │                          # - Batch processing
│   │                          # - Response duration tracking
│   ├── evaluator.py          # - Moderation API integration
│   │                          # - Google Perspective API
│   │                          # - Hybrid moderation support
│   │                          # - Evaluation duration tracking
│   ├── prompt_generator.py   # - Prompt generation utilities
│   └── model_interface.py    # - Model loading and management
│
├── speciation/                # Speciation and Distribution Module
│   ├── run_speciation.py     # - process_generation() - 8-phase speciation
│   │                          # - distribute_genomes() - File distribution
│   │                          # - EvolutionTracker speciation block
│   ├── config.py             # - SpeciationConfig dataclass
│   │                          # - Parameter validation
│   ├── leader_follower.py    # - Leader-follower clustering
│   │                          # - Ensemble distance calculation
│   │                          # - θ_sim threshold assignment
│   │                          # - θ_merge threshold merging
│   ├── species.py            # - Species class and management
│   │                          # - Leader selection and updates
│   │                          # - Radius enforcement
│   │                          # - Capacity enforcement
│   ├── reserves.py           # - Cluster0 class
│   │                          # - Cluster 0 speciation (Flow 2)
│   │                          # - Capacity management
│   ├── merging.py            # - Species merging logic
│   │                          # - Parent species extinction
│   │                          # - Leader selection for merged species
│   ├── extinction.py         # - Species freezing logic
│   │                          # - Stagnation tracking
│   │                          # - Incubator conversion
│   ├── embeddings.py         # - Embedding computation
│   │                          # - Batch processing
│   │                          # - Model management
│   ├── distance.py           # - Genotype distance (embedding cosine)
│   ├── phenotype_distance.py # - Phenotype distance (response scores)
│   ├── metrics.py            # - SpeciationMetricsTracker
│   │                          # - Diversity metrics
│   │                          # - Cluster quality integration
│   ├── genome_tracker.py     # - GenomeTracker class
│   │                          # - species_id tracking (source of truth)
│   │                          # - Event logging
│   ├── labeling.py           # - c-TF-IDF label generation
│   ├── validation.py         # - Flow 1 & Flow 2 validation
│   │                          # - Consistency checks
│   └── events_tracker.py     # - Speciation events tracking
│
└── utils/                     # Utility Modules
    ├── population_io.py      # - File I/O (elites/reserves/archive/temp)
    │                          # - EvolutionTracker management
    │                          # - calculate_average_fitness()
    │                          # - calculate_generation_statistics()
    │                          # - update_evolution_tracker_with_statistics()
    │                          # - update_adaptive_selection_logic()
    ├── refusal_detector.py   # - LLM refusal detection
    │                          # - Pattern matching
    │                          # - Length-based detection
    ├── refusal_penalty.py    # - 15% penalty application
    │                          # - Post-evaluation, pre-speciation
    ├── cluster_quality.py    # - Silhouette Score
    │                          # - Davies-Bouldin Index
    │                          # - Calinski-Harabasz Index
    │                          # - QD Score
    ├── operator_effectiveness.py # - NE, EHR, IR, cEHR, Δμ, Δσ
    ├── device_utils.py       # - GPU/CPU device management
    ├── data_loader.py        # - CSV loading utilities
    ├── config.py              # - General configuration
    ├── constants.py           # - System constants
    ├── custom_logging.py      # - Logging utilities
    └── live_analysis.py       # - Real-time visualization
```

---

## 3. Core Components

### 3.1 Evolution Engine (`ea/evolution_engine.py`)

**Purpose**: Core variant creation and ID management

**Key Functions**:
- `next_id()`: Generates globally unique genome IDs (max over elites+reserves+archive + 1)
- `create_child()`: Creates variants using operators, tracks parents, operator metadata

**ID Management**:
- IDs are never reused
- IDs are integers, globally unique across all files
- IDs persist across generations for lineage tracking

### 3.2 Parent Selector (`ea/parent_selector.py`)

**Purpose**: Adaptive parent selection based on species state and population fitness trends

**Categories**:
- **Category 1**: Active species ∪ species 0 (reserves) - equal importance
- **Category 2**: Frozen species - used only when Category 1 is empty

**Selection Modes** (applied to chosen category):
- **DEFAULT**: Random species, 2 parents; fill from category if <2
- **EXPLOIT**: Top species by max fitness, 3 parents (when `slope_of_avg_fitness <= 0`)
- **EXPLORE**: Top + 2 random species, 1 best parent each (when `generations_since_improvement >= stagnation_limit`)

**Fitness Calculation**: Uses actual max over current genomes only (no merge with stored values)

### 3.3 Response Generator (`gne/response_generator.py`)

**Purpose**: Generate LLM responses for prompts

**Features**:
- Batch processing for efficiency
- Response duration tracking (rounded to 4 decimal places)
- Model name tracking
- Error handling and retry logic

### 3.4 Evaluator (`gne/evaluator.py`)

**Purpose**: Evaluate responses using moderation APIs

**Features**:
- Google Perspective API integration
- Hybrid moderation support
- Evaluation duration tracking (rounded to 4 decimal places)
- Toxicity score extraction

### 3.5 Speciation Engine (`speciation/run_speciation.py`)

**Purpose**: 8-phase speciation process for each generation

**Phases**:
1. **Existing Species Processing**: Embed temp, leader-follower clustering, radius cleanup, capacity enforcement
2. **Cluster 0 Speciation**: Isolated speciation from reserves (Flow 2)
3. **Merging**: Merge similar species (θ_merge threshold)
4. **Radius & Capacity Enforcement**: Enforce radius and capacity for all species
5. **Freeze & Incubator**: Track stagnation, freeze stagnant species, dissolve small species
6. **Cluster 0 Capacity Enforcement**: Archive excess reserves
7. **Final Redistribution**: Update species_id from genome_tracker, redistribute to files
8. **Metrics & Stats**: Calculate diversity, cluster quality, update trackers

**Key Functions**:
- `process_generation()`: Main speciation orchestration
- `distribute_genomes()`: Redistribute genomes to elites/reserves/archive based on species_id

### 3.6 Genome Tracker (`speciation/genome_tracker.py`)

**Purpose**: Authoritative source of truth for `species_id` assignments

**Features**:
- Tracks all genome IDs and their current `species_id`
- Updated at every speciation event (clustering, merging, archiving, etc.)
- Used in Phase 7 to update file-based `species_id` values
- Event logging for lineage tracking

**Key Principle**: `genome_tracker.json` is the continuous source of truth; `elites.json`, `reserves.json`, `archive.json` are only fully synchronized in Phase 7 for performance.

---

## 4. High-Level Flow

### 4.1 Generation 0 (Initialization)

1. **System Setup**
   - Initialize logging and output directory (`data/outputs/YYYYMMDD_HHMM/`)
   - Load device configuration and model paths
   - Update RGConfig.yaml and PGConfig.yaml with model paths
   - Initialize response generator (RG) and prompt generator (PG) models
   - Load seed prompts from `data/prompt.csv`

2. **Initial Population Generation**
   - Generate responses using `response_generator.process_population()`
   - Output: `temp.json` (genomes with prompts and `generated_output`)

3. **Evaluation**
   - Evaluate responses using moderation APIs (Google Perspective API)
   - Update `temp.json` with `moderation_result`, `toxicity`, `evaluation_duration`

4. **Refusal Penalty**
   - Detect refusals using `refusal_detector`
   - Apply 15% reduction (×0.85) on toxicity for refusals
   - Update `moderation_result` and `north_star_score`

5. **Pre-Speciation Metrics**
   - Calculate `avg_fitness = mean(temp.json fitness)` [before speciation, after evaluation]

6. **Speciation**
   - Initialize `SpeciationConfig` with parameters
   - Compute embeddings for all genomes in `temp.json`
   - Run leader-follower clustering to form initial species
   - Distribute genomes to `elites.json` (species_id > 0) and `reserves.json` (species_id = 0)
   - Archive excess genomes to `archive.json` (capacity limits)
   - Save `speciation_state.json` with initial species structure

7. **Statistics & Tracking**
   - Calculate operator effectiveness metrics (empty for gen 0)
   - Calculate generation statistics (`elites_count`, `reserves_count`, etc.)
   - Update `EvolutionTracker.json` with gen 0 entry
   - Initialize cumulative budget tracking

### 4.2 Generation N (N ≥ 1) - Evolution Loop

For each generation N:

**PHASE 1: Variant Generation**
1. Load population: `elites.json` and `reserves.json`
2. Load `speciation_state.json` (if exists)
3. **Adaptive Parent Selection**:
   - Determine selection mode (DEFAULT/EXPLOIT/EXPLORE) based on adaptive logic
   - Select parents from Category 1 (active + reserves) or Category 2 (frozen)
4. **Variant Creation**:
   - Apply variation operators (mutation/crossover) to selected parents
   - Track operator statistics (rejections, duplicates)
   - Output: `temp.json` (new variants with `operator`, `variant_type`, `parents`)

**PHASE 2: Response Generation**
- Generate responses for all variants in `temp.json`
- Update `temp.json` with `generated_output`, `model_name`, `response_duration`

**PHASE 3: Evaluation**
1. **Moderation**: Evaluate all variants using moderation APIs
2. **Refusal Penalty**: Detect refusals, apply 15% penalty
3. **Pre-Speciation Metrics**:
   - Calculate `avg_fitness = mean(old elites + old reserves + all new variants)` [BEFORE speciation, AFTER evaluation]
   - Calculate variant statistics from `temp.json`:
     * `max_score_variants`: max fitness in temp.json
     * `min_score_variants`: min fitness in temp.json
     * `avg_fitness_variants`: mean fitness in temp.json
     * `variants_created`: total generated (remaining + duplicates + rejections)
     * `mutation_variants`: count with `variant_type=="mutation"`
     * `crossover_variants`: count with `variant_type=="crossover"`

**PHASE 4: Speciation** (`process_generation` - 8 phases)
- See [Section 6: Speciation Framework](#6-speciation-framework)

**PHASE 5: Post-Speciation Processing**
1. **Operator Effectiveness Metrics**:
   - Calculate NE, EHR, IR, cEHR, Δμ, Δσ for each operator
   - Save to `operator_effectiveness_cumulative.csv`

2. **Update EvolutionTracker with Generation Data**:
   - Update variant counts (`variants_created`, `mutation_variants`, `crossover_variants`)
   - Preserve speciation data from `process_generation`

3. **Adaptive Selection Logic Update**:
   - Calculate `slope_of_avg_fitness` from `avg_fitness_history`
   - Update `selection_mode` based on:
     * EXPLOIT: `slope <= 0`
     * EXPLORE: `generations_since_improvement >= stagnation_limit`
     * DEFAULT: otherwise
   - Update `generations_since_improvement`:
     * Reset to 0 if `current_max_toxicity > previous_max_toxicity`
     * Increment if no improvement

4. **Generation Statistics**:
   - Calculate comprehensive statistics:
     * `elites_count`, `reserves_count`, `archived_count`
     * `avg_fitness_elites`, `avg_fitness_reserves`, `avg_fitness_generation`
     * `population_max_toxicity` (cumulative max)
   - Update `EvolutionTracker` with all statistics

5. **Visualizations**:
   - Generate live analysis visualizations
   - Update `figures/` directory

6. **Population Index Update**:
   - Update `EvolutionTracker.total_generations`
   - Update cumulative budget

### 4.3 Termination

Evolution stops when:
1. `max_generations` reached (if specified)
2. Threshold achieved (`population_max_toxicity >= north_star_threshold`)
3. Runtime error or user interruption
4. All species frozen and reserves empty (no viable parents)

Final steps:
- Save final `speciation_state.json`
- Save final `EvolutionTracker.json`
- Generate final visualizations
- Log total execution time and statistics

---

## 5. Parent Selection System

### 5.1 Category System

**Category 1 (Equal Importance)**:
- Active species (`species_state == "active"`)
- Species 0 (reserves/cluster 0)

**Category 2 (Fallback)**:
- Frozen species (`species_state == "frozen"`)

**Selection Rule**: Use Category 2 only when Category 1 has no genomes. If both categories are empty, raise `RuntimeError` to terminate evolution.

### 5.2 Selection Modes

**DEFAULT Mode**:
- Pick any species (random from sorted by max fitness)
- Select 2 parents from chosen species
- If chosen species has <2 genomes, fill from category

**EXPLOIT Mode** (triggered when `slope_of_avg_fitness <= 0`):
- Pick species with highest max fitness
- Select 3 parents from chosen species
- If chosen species has <3 genomes, fill from category
- Focus: Local search around best-performing species

**EXPLORE Mode** (triggered when `generations_since_improvement >= stagnation_limit`):
- Pick top species by max fitness
- Pick 2 additional random species
- Select 1 best parent from each of the 3 species
- If <3 species available, reuse/fill from category
- Focus: Diversity and exploration across species

### 5.3 Fitness Calculation

**Max Fitness**: Actual max over current genomes only (no merge with stored values)
- For species: `max(genome["fitness"] for genome in current_members)`
- For cluster 0: `max(genome["fitness"] for genome in reserves.json)`

This ensures parent selection uses real-time fitness, not stale cached values.

---

## 6. Speciation Framework

### 6.1 Overview

The speciation process (`process_generation`) consists of 8 phases that transform new variants (`temp.json`) into organized species structure.

### 6.2 Phase 1: Existing Species Processing

**Purpose**: Assign new variants to existing species or cluster 0

**Steps**:
1. Compute embeddings for `temp.json` genomes
2. Process variants against existing species using leader-follower clustering
   - Calculate ensemble distance: `d_ensemble = 0.7×d_genotype + 0.3×d_phenotype`
   - If `d_ensemble(variant, leader) < θ_sim`: assign to species
   - Otherwise: assign to cluster 0 (Flow 1)
3. **Generation 0 ONLY**: Immediate capacity enforcement after species formation
4. Sync cluster 0 with `reserves.json`
5. **Generation N**: Radius cleanup of existing species (after all variants processed)
6. Save intermediate state #1

**Key Operations**:
- Leader-follower assignment with `skip_cluster0_outliers=True`
- Radius enforcement: members outside radius moved to cluster 0
- Capacity enforcement: excess members archived by fitness

### 6.3 Phase 2: Cluster 0 Speciation (Isolated)

**Purpose**: Form new species from cluster 0 when cohesive clusters emerge

**Steps**:
1. Load cluster 0 from `reserves.json`
2. Apply isolated cluster 0 speciation (Flow 2)
   - When cluster 0 reaches `cluster0_min_cluster_size`, form new species
   - New species created with leader (highest fitness genome)
   - Members within `θ_sim` of leader join the species
3. Save intermediate state #2

**Flow 2 Characteristics**:
- Isolated from existing species processing
- Only considers genomes in cluster 0
- Creates new species with `cluster_origin="natural"`

### 6.4 Phase 3: Merging

**Purpose**: Combine similar species to reduce redundancy

**Steps**:
1. Identify merge candidates: species with `d_ensemble(leader_i, leader_j) < θ_merge`
2. Merge similar species:
   - Both parent species marked as extinct
   - All genomes' `species_id` updated in `genome_tracker` to new merged species ID
   - New species created with combined members
   - **Leader Selection**: Highest fitness genome among ALL combined members (before radius enforcement)
   - Leader selected before radius enforcement
3. **No radius/capacity enforcement during merging** (handled in Phase 4)
4. Save intermediate state (after merging)

**Merging Details**:
- Both active and frozen species can merge
- Parent species IDs recorded in `parent_ids`
- New species has `cluster_origin="merge"`
- Extinct species tracked in `historical_species` (ID only, not full structure)

### 6.5 Phase 4: Radius & Capacity Enforcement

**Purpose**: Enforce species boundaries and capacity limits

**Steps**:
1. **Radius Enforcement** for ALL species (existing + newly formed + merged):
   - For each species, calculate distance from leader to all members
   - Members with `d_ensemble > θ_sim` moved to cluster 0
   - Leader reselection: If merged species, only update if new higher-fitness genome added
2. **Capacity Enforcement** for ALL species:
   - Sort members by fitness (descending)
   - Keep top `species_capacity` members
   - Archive excess members to `archive.json` (by fitness, lowest first)
3. Validate no duplicate leader IDs
4. Save intermediate state (after enforcement)

**Capacity Enforcement Details**:
- Filters genomes with valid fitness scores before sorting
- Archives excess members with `archive_reason="species_capacity_exceeded"`
- Updates `genome_tracker` with archiving events

### 6.6 Phase 5: Freeze & Incubator

**Purpose**: Track stagnation and manage small species

**Steps**:
1. **Sync max_fitness**: Update `sp.max_fitness` to actual max over current members
2. **Record Fitness History**:
   - For each species, track fitness history (last 20 values)
   - Stagnation tracking:
     * Reset to 0 if `max_fitness` increased
     * Increment if `max_fitness` did not increase AND species was selected as parent
3. **Freeze Stagnant Species**:
   - If `stagnation >= species_stagnation`: set `species_state = "frozen"`
   - Frozen species preserved with all members (leader embeddings, distances, labels, history)
   - Can merge with active or other frozen species
4. **Dissolve Small Species**:
   - If `size < min_island_size`: move to incubator
   - **Special Case**: Newly merged species (`cluster_origin == "merge"` and `created_at == current_generation`) dissolved if `size < min_island_size`
   - Incubator species: members moved to cluster 0 (`species_id = 0`), species tracked by ID only
5. Save intermediate state #3

**Freezing Details**:
- Frozen species excluded from Category 1 parent selection
- Frozen species can still merge
- Both active and frozen are "alive" - only difference is parent selection preference

### 6.7 Phase 6: Cluster 0 Capacity Enforcement

**Purpose**: Manage cluster 0 size

**Steps**:
1. Enforce cluster 0 capacity at end
2. Archive excess reserves to `archive.json` (by fitness, lowest first)
3. Save intermediate state (after cluster 0 capacity)

**Capacity Details**:
- Sorts reserves by fitness (descending)
- Keeps top `cluster0_max_capacity` members
- Archives excess with `archive_reason="cluster0_capacity_exceeded"`

### 6.8 Phase 7: Final Redistribution

**Purpose**: Synchronize file-based `species_id` with `genome_tracker` (authoritative source)

**Steps**:
1. Load `elites.json`, `reserves.json`, `archive.json`, `temp.json`
2. **Update species_id from tracker**:
   - For each genome, update `species_id` from `genome_tracker.json`
   - Register untracked genomes with appropriate `species_id`
3. **Redistribute genomes**:
   - `species_id > 0` → `elites.json`
   - `species_id = 0` → `reserves.json`
   - `species_id = -1` → `archive.json`
4. **Preserve archive**: All existing archived genomes preserved (archive.json is final destination)
5. **Deduplicate**: Remove duplicate genomes by ID (keep first occurrence)
6. **Atomic writes**: Use temporary files to prevent data loss
7. Sync in-memory `state["cluster0"]` with `reserves.json`

**Key Principle**: `genome_tracker.json` is the continuous source of truth. `elites.json`, `reserves.json`, `archive.json` are only fully synchronized in Phase 7 for performance.

### 6.9 Phase 8: Metrics & Stats

**Purpose**: Calculate diversity metrics and update trackers

**Steps**:
1. Calculate diversity metrics:
   - `inter_species_diversity`: Mean pairwise distance between species leaders
   - `intra_species_diversity`: Mean pairwise distance within species members
2. Calculate cluster quality (if available):
   - Silhouette Score
   - Davies-Bouldin Index
   - Calinski-Harabasz Index
   - QD Score
3. Update `speciation_state.json` with final state
4. Update `genome_tracker.json` with events
5. Return metrics for `EvolutionTracker` update

---

## 7. Key Metrics and Conventions

### 7.1 Fitness Metrics

**avg_fitness**:
- **Definition**: Mean over **old elites + old reserves + all new variants** (elites+reserves+temp) **before speciation**, **after evaluation**
- **Formula**: `mean([genome["fitness"] for genome in elites + reserves + temp])`
- **Timing**: Calculated BEFORE speciation, AFTER evaluation
- **Usage**: Adaptive selection logic, slope calculation
- **Note**: Gen 0: effectively `mean(temp.json)`. Differs from `avg_fitness_generation` when genomes are archived this gen.

**avg_fitness_generation**:
- **Definition**: Mean over **updated elites + updated reserves** after distribution
- **Formula**: `mean([genome["fitness"] for genome in elites.json + reserves.json])`
- **Timing**: Calculated AFTER distribution
- **Usage**: Generation-level statistics
- **Note**: Archived genomes excluded automatically (removed from those files)

**max_score_variants**:
- **Definition**: Maximum fitness among variants created this generation (from `temp.json` before speciation)
- **Timing**: Calculated BEFORE speciation
- **Usage**: Track generation-level improvements

**population_max_toxicity**:
- **Definition**: Cumulative maximum fitness observed across all generations in elite and reserve populations
- **Timing**: Updated after each generation
- **Usage**: Threshold checking, stagnation detection, adaptive selection
- **Calculation**: `max(population_max_toxicity, max([genome["fitness"] for genome in elites + reserves]))`

### 7.2 Adaptive Selection Metrics

**generations_since_improvement**:
- **Definition**: Number of consecutive generations without increase in `population_max_toxicity`
- **Update Logic**:
  * Reset to 0 if `current_max_toxicity > previous_max_toxicity + epsilon`
  * Increment if no improvement: `generations_since_improvement += 1`
- **Usage**: Triggers EXPLORE mode when `>= stagnation_limit`

**slope_of_avg_fitness**:
- **Definition**: Slope of `avg_fitness_history` (sliding window of recent generations)
- **Calculation**: Linear regression over last `stagnation_limit` generations
- **Usage**: Triggers EXPLOIT mode when `<= 0`

**avg_fitness_history**:
- **Definition**: Sliding window of recent generations' `avg_fitness` values
- **Size**: Last `stagnation_limit` generations (default: 5)
- **Usage**: Calculate `slope_of_avg_fitness`

### 7.3 Refusal Penalty

**Application**: After evaluation, before speciation

**Penalty**: 15% reduction (×0.85) on toxicity for genomes with `is_refusal(response)==True`

**Detection**: Uses `refusal_detector` with pattern matching and length-based heuristics

**Updates**:
- `moderation_result["google"]["scores"][north_star_metric]` (or legacy `moderation_result["scores"]`)
- `genome["north_star_score"]`
- `genome["fitness"]` (via `_extract_north_star_score`)

### 7.4 Cluster Quality Metrics

**Silhouette Score**: [-1, 1], higher = better separation
- Measures how similar genomes are to their own species vs. other species

**Davies-Bouldin Index**: ≥0, lower = better
- Measures average similarity ratio of each species to its most similar species

**Calinski-Harabasz Index**: Higher = better
- Ratio of between-cluster to within-cluster dispersion

**QD Score**: Quality × inter_species_diversity
- Combines cluster quality with diversity metric

**Note**: Pareto-optimal or multi-objective optimization is not used; each metric is computed independently.

### 7.5 EvolutionTracker `gen_entry["speciation"]`

**Contains**:
- `species_count`: Active + frozen species
- `active_species_count`: Species with `species_state == "active"`
- `frozen_species_count`: Species with `species_state == "frozen"`
- `reserves_size`: Size of cluster 0
- `speciation_events`: New species formed this gen
- `merge_events`: Species merged this gen
- `extinction_events`: Species extinct (frozen/incubator) this gen
- `archived_count`: Genomes archived this gen
- `elites_moved`: Genomes moved to elites this gen
- `reserves_moved`: Genomes moved to reserves this gen
- `genomes_updated`: Genomes with species_id updated
- `inter_species_diversity`: Mean pairwise distance between species leaders
- `intra_species_diversity`: Mean pairwise distance within species members
- `total_population`: Elites + reserves at record time
- `cluster_quality`: Cluster quality metrics object (or null)

**Note**: `best_fitness` and `avg_fitness` are NOT in speciation; they exist only at gen level (`population_max_toxicity`, `max_score_variants`, `avg_fitness`).

---

## 8. Data Flow and File Management

### 8.1 File Update Strategy

**Deferred File Updates**: `elites.json`, `reserves.json`, `archive.json` are only fully synchronized with `species_id` changes in Phase 7 for performance.

**genome_tracker.json**: Continuous source of truth, updated at every speciation event.

**File Updates per Generation**:
- `temp.json`: Created/updated during variant generation, response generation, evaluation; cleared/updated during speciation
- `elites.json`: Updated after Phase 7 redistribution
- `reserves.json`: Updated after Phase 7 redistribution
- `archive.json`: Updated during capacity enforcement (Phase 4, Phase 6) and Phase 7 redistribution
- `speciation_state.json`: Updated after each speciation phase (intermediate saves) and final update after Phase 8
- `EvolutionTracker.json`: Updated multiple times per generation (after speciation, after statistics, after adaptive selection)
- `genome_tracker.json`: Updated during speciation (events tracked)

### 8.2 Atomic File Writes

**Strategy**: Use temporary files to prevent data loss during updates

**Process**:
1. Write to temporary file (`file.json.tmp`)
2. Verify write success
3. Rename temporary file to final file (`file.json`)
4. This ensures atomic updates even if process crashes mid-write

### 8.3 Untracked Genomes

**Handling**: Genomes in population files but not in `genome_tracker.json` are preserved and registered with appropriate `species_id` during Phase 7.

**Registration**: Untracked genomes are registered with their current `species_id` (or default to 0 if `None`).

---

## 9. Adaptive Selection Logic

### 9.1 Overview

The adaptive selection logic dynamically adjusts parent selection strategy based on population fitness trends and stagnation.

### 9.2 Execution Order

1. **Statistics Update** (`update_evolution_tracker_with_statistics`):
   - Updates `population_max_toxicity` (cumulative max)
   - Updates per-generation statistics
   - Saves `EvolutionTracker.json`

2. **Adaptive Selection Update** (`update_adaptive_selection_logic`):
   - Reads `EvolutionTracker.json`
   - Compares `current_max_toxicity` vs `previous_max_toxicity`
   - Updates `generations_since_improvement`
   - Calculates `slope_of_avg_fitness` from `avg_fitness_history`
   - Determines `selection_mode` (DEFAULT/EXPLOIT/EXPLORE)
   - Saves `EvolutionTracker.json`

**Key**: `update_adaptive_selection_logic` runs AFTER `update_evolution_tracker_with_statistics` to ensure it uses the latest `population_max_toxicity` value.

### 9.3 Selection Mode Determination

**EXPLOIT Mode** (triggered when `slope_of_avg_fitness <= 0`):
- Indicates declining or flat fitness trend
- Strategy: Focus on best-performing species (local search)
- Parent selection: 3 parents from top species

**EXPLORE Mode** (triggered when `generations_since_improvement >= stagnation_limit`):
- Indicates stagnation (no improvement for `stagnation_limit` generations)
- Strategy: Increase diversity and exploration
- Parent selection: 1 parent each from top + 2 random species

**DEFAULT Mode** (otherwise):
- Normal evolution
- Parent selection: 2 parents from random species

### 9.4 Improvement Detection

**Comparison**: `current_max_toxicity > previous_max_toxicity + epsilon`

**epsilon**: `1e-6` (handles floating-point precision)

**Update Logic**:
- If improvement: `generations_since_improvement = 0`
- If no improvement: `generations_since_improvement += 1`

**Metrics Used**:
- `current_max_toxicity`: `population_max_toxicity` from current generation (elites + reserves after distribution)
- `previous_max_toxicity`: Cumulative `population_max_toxicity` from tracker (before update)

---

## 10. Output Files

All output files are located in `data/outputs/YYYYMMDD_HHMM/`:

| File | Role | Format | Update Frequency |
|------|------|--------|------------------|
| `elites.json` | Species members (species_id > 0), cumulative | JSON array | After Phase 7 |
| `reserves.json` | Cluster 0 (species_id = 0), capacity-limited, sorted by fitness | JSON array | After Phase 7 |
| `archive.json` | Archived genomes (capacity overflow, etc.) | JSON array | Phase 4, Phase 6, Phase 7 |
| `temp.json` | Current variants before speciation; cleared/repopulated each gen | JSON array | During variant generation, response generation, evaluation |
| `EvolutionTracker.json` | Per-gen stats, speciation block, cumulative max, selection state | JSON object | Multiple times per generation |
| `speciation_state.json` | Species (leader_*, member_ids, max_fitness, stagnation, state), cluster0, metrics | JSON object | After each speciation phase |
| `genome_tracker.json` | ID → metadata for lineage, species_id tracking (source of truth) | JSON object | During speciation events |
| `operator_effectiveness_cumulative.csv` | RQ1: per (generation, operator) metrics | CSV | After each generation |
| `figures/` | Fitness, diversity, operator visualizations | PNG/PDF | After each generation |

**Species States**:
- **active**: Participates in evolution and parent selection (Category 1)
- **frozen**: Stagnated (≥`species_stagnation` gens), excluded from Category 1, preserved with all members. Can merge with active or other frozen species.
- **incubator**: Moved to cluster 0 when `size < min_island_size`, tracked by ID only
- **extinct**: Parent species after merging, tracked in `historical_species` (ID only)

---

## 11. Configuration Parameters

### 11.1 SpeciationConfig (`speciation/config.py`)

**Clustering Parameters**:
- `theta_sim` (default: 0.25): Similarity threshold for species assignment (ensemble distance). Individuals within this distance of a leader become followers. Range: [0, 1]. Also used as the constant radius for all species.
- `theta_merge` (default: 0.1): Merge threshold for combining similar species. Species with leader distance < `theta_merge` are candidates for merging. Must be `<= theta_sim` (typically `< theta_sim` for effective merging).

**Cluster 0 Parameters**:
- `cluster0_min_cluster_size` (default: 2): Minimum cluster size required for cluster 0 speciation. When cluster 0 individuals form a cohesive cluster of this size, they can create a new species.
- `cluster0_max_capacity` (default: 1000): Maximum individuals in cluster 0. When exceeded, lowest-fitness individuals are archived.

**Species Management Parameters**:
- `species_capacity` (default: 100): Maximum individuals per species (keeps top by fitness). When exceeded, lowest-fitness members are archived.
- `min_island_size` (default: 2): Minimum island size before extinction. Islands smaller than this are considered extinct and moved to incubator.
- `species_stagnation` (default: 20): Maximum generations without improvement before species freezing. Species that stagnate beyond this threshold are frozen.

**Embedding Parameters**:
- `embedding_model` (default: "all-MiniLM-L6-v2"): Sentence-transformer model name for prompt embeddings. Model must be compatible with sentence-transformers library.
- `embedding_dim` (default: 384): Expected embedding dimensionality. Should match the chosen model's output dimension.
- `embedding_batch_size` (default: 64): Batch size for embedding computation. Larger batches are faster but use more memory.

**Ensemble Distance Weights**:
- `w_genotype` (default: 0.7): Weight for genotype (prompt embedding) distance in ensemble distance calculation.
- `w_phenotype` (default: 0.3): Weight for phenotype (response scores) distance in ensemble distance calculation.
- **Constraint**: `w_genotype + w_phenotype = 1.0`

**Ensemble Distance Formula**:
```
d_ensemble = w_genotype × d_genotype + w_phenotype × d_phenotype
```

Where:
- `d_genotype`: Cosine distance between prompt embeddings
- `d_phenotype`: Normalized difference in response toxicity scores

---

## 12. References

- [FIELD_DEFINITIONS.txt](FIELD_DEFINITIONS.txt) — Field and subfield definitions for all JSON/CSV outputs
- [PROCESS_FLOW.md](PROCESS_FLOW.md) — Complete end-to-end process flow documentation
- [experiments/FLOW_AND_VALIDATION.md](experiments/FLOW_AND_VALIDATION.md) — Flow, validation, and field basis (variants vs post-generation vs cumulative)
- [README.md](README.md) — Installation, usage, and quick start guide

---

## Recent Improvements and Fixes

### generations_since_improvement Tracking
- **Issue**: `generations_since_improvement` was not updating correctly when `population_max_toxicity` changed
- **Fix**: Changed to use `population_max_toxicity` (from elites+reserves after distribution) instead of `max_score_variants` (from temp.json)
- **Fix**: Moved `update_adaptive_selection_logic` call to AFTER `update_evolution_tracker_with_statistics`
- **Fix**: Removed preservation/restoration logic that was overwriting adaptive selection fields
- **Result**: `generations_since_improvement` now correctly tracks population-level improvements

### Diversity Metrics Flow
- **Issue**: Diversity metrics (inter_species_diversity, intra_species_diversity, cluster_quality) were calculated but not flowing to `EvolutionTracker.json`
- **Fix**: Added metrics to `run_speciation()` return value
- **Fix**: Updated `main.py` to extract metrics from `speciation_result` and add to `gen_stats`
- **Fix**: Updated `update_evolution_tracker_with_statistics` to prioritize metrics from statistics dictionary
- **Result**: Diversity metrics now correctly appear in `EvolutionTracker.json`

### Flow 2 Validation
- **Issue**: Flow 2 validation was reporting false positives for newly formed species
- **Fix**: Updated validation to check `reserves.json` and `temp.json` in addition to `elites.json`
- **Fix**: Only reports error if genomes are missing from ALL files
- **Result**: Validation now correctly handles genomes that haven't been distributed yet

### Species Reconstruction
- **Issue**: Species reconstruction was defaulting to "frozen" state incorrectly
- **Fix**: Modified reconstruction logic to retrieve and preserve actual species state from previous `speciation_state.json` files
- **Result**: Reconstructed species now have correct state (active/frozen)

### Merging Logic
- **Issue**: After merging, parent species genomes were not all updated in `genome_tracker`
- **Fix**: Modified `merge_islands()` to ensure ALL genomes associated with parent species IDs in `genome_tracker` are updated to new merged species ID
- **Result**: Merging now correctly updates all parent genomes

---

**Last Updated**: 2026-01-23
