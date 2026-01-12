# System Architecture: Evolutionary Search for Toxicity in LLMs

## Overview

This system implements a genetic algorithm framework with **semantic speciation** (Leader-Follower clustering) for evolving text prompts to elicit toxic responses from large language models, enabling comprehensive AI safety evaluation through adversarial prompt generation.

## Core Algorithm

### Evolutionary Process
The system uses a steady-state genetic algorithm with **semantic speciation** (Leader-Follower clustering) with the following components:

1. **Population Initialization**: Load initial prompts from CSV
2. **Parent Selection**: Adaptive tournament selection (species-aware)
3. **Variation**: Apply 12 different operators to generate offspring
4. **Response Generation**: LLM generates responses to evolved prompts
5. **Fitness Evaluation**: Score responses using Google Perspective API (toxicity)
6. **Speciation**: Semantic clustering into species using Leader-Follower algorithm
7. **Distribution**: Distribute genomes into elites.json (species_id > 0) and reserves.json (species_id == 0)
8. **Termination**: Stop when convergence criteria are met

### Mathematical Framework

**Fitness Function**:
```
f(x) = toxicity_score(response to prompt x) in [0, 1]
```

The fitness function evaluates toxicity in the LLM's response, not the prompt itself.

**Population Structure**:
```
Active Population = elites.json + reserves.json
- elites.json: Genomes with species_id > 0 (assigned to species)
- reserves.json: Cluster 0 outliers (species_id == 0, don't fit existing species)
- archive.json: Archived genomes (NOT part of population)
```

## Component Architecture

### Evolution Engine
Core evolution logic implementing the genetic algorithm. Manages the evolutionary cycle, coordinates operator application, and maintains population state throughout the evolution process.

### Parent Selector
Adaptive parent selection mechanism that adjusts selection strategy based on evolution progress and fitness landscape. Species are sorted by best_fitness in descending order.

**Selection Modes**:
| Mode | Parent 1 | Parent 2 | Trigger |
|------|----------|----------|---------|
| **DEFAULT** | Random genome from random species | Random genome from same species | First m generations |
| **EXPLORE** | Highest-fitness genome from top species | Highest-fitness genome from different random species | Stagnation > m generations |
| **EXPLOIT** | Highest-fitness genome from top species | Random genome from same species | Fitness slope < 0 (declining) |

**Notes**:
- Frozen/deceased species are excluded from selection
- Cluster 0 (reserves) is included in the selection pool
- Species are sorted by best_fitness before selection

### Variation Operators (12 Total)

#### Mutation Operators (10)
1. **Informed Evolution**: LLM-guided evolution using top performers
2. **Masked Language Model**: Contextual word substitution
3. **Paraphrasing**: Semantic-preserving text transformation
4. **Back Translation**: Hindi roundtrip translation (other languages disabled for performance)
5. **Synonym Replacement**: Lexical substitution with POS awareness
6. **Antonym Replacement**: Lexical substitution with POS awareness
7. **Negation**: Logical operator insertion
8. **Concept Addition**: Semantic concept injection
9. **Typographical Errors**: Character-level noise injection
10. **Stylistic Mutation**: Writing style transformation

#### Crossover Operators (2)
1. **Semantic Similarity**: Crossbreeding based on semantic distance
2. **Semantic Fusion**: Hybrid prompt generation

### Response Generation
Generates responses from target LLMs using the evolved prompts. Supports multiple model architectures through a unified interface using llama-cpp-python.

### Moderation Evaluation
Evaluates generated responses for toxicity using Google Perspective API. Provides comprehensive toxicity scoring across 8 dimensions:
- TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT, PROFANITY, THREAT, SEXUALLY_EXPLICIT, FLIRTATION

**Error Handling**:
- **10 retries** (11 total attempts) with exponential backoff for rate limits
- Wait times: 1s, 2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s, 512s between retries
- Automatic handling of 429 (rate limit) and 5xx (server) errors
- 0.75s delay between evaluations to stay within quota

### Population Management
Manages population state, handles I/O operations, and maintains population statistics. Supports both monolithic and split file formats for scalability.

### Speciation Module
Implements Leader-Follower clustering with semantic embeddings to maintain diverse species that evolve independently.

**Key Components**:

1. **Embedding Computation**: L2-normalized 384-dim embeddings using all-MiniLM-L6-v2
2. **Leader-Follower Clustering**: Fitness-sorted assignment to species based on ensemble distance (theta_sim threshold)
   - **Leader Definition**: Best-fitness member in each species
   - **Incremental Updates**: Only species that receive new members are updated (capacity enforcement, fitness recording)
3. **Reserves (Cluster 0)**: Holding area for high-fitness outliers that don't fit existing species
   - Part of active population (population = elites + reserves)
   - Fixed capacity (cluster0_max_capacity), excess archived
   - New species can be created from reserves via agglomerative clustering
4. **Species Operations**:
   - **Merging**: Combine similar species when leaders are close (theta_merge threshold)
   - **Extinction**: Freeze stagnant species (stagnation > max_stagnation)
   - **Capacity Enforcement**: Remove excess genomes from species/reserves when capacity exceeded
5. **c-TF-IDF Labeling**: Each species gets 10 keyword labels based on member prompts
6. **Metrics Tracking**: Species count, diversity, merge/extinction events, budget

**Integration Point**: After fitness evaluation (moderation), before distribution. Each genome receives a species_id field for species-aware operations.

**Mathematical Framework**:

**Genotype Distance (Semantic)**:
```
d_genotype(u, v) = 1 - (e_u . e_v) in [0, 2]
```
where e_u, e_v are L2-normalized embeddings.

Normalized to [0, 1]:
```
d_genotype_norm(u, v) = (1 - (e_u . e_v)) / 2 in [0, 1]
```

**Phenotype Distance (Toxicity)**:
```
d_phenotype(u, v) = ||p_u - p_v||_2 / sqrt(8) in [0, 1]
```
where p_u, p_v are 8-dimensional toxicity score vectors (all 8 Perspective API attributes).

**Ensemble Distance**:
```
d_ensemble(u, v) = 0.7 x d_genotype_norm(u, v) + 0.3 x d_phenotype(u, v) in [0, 1]
```

**Geometry Note**:
- All embeddings live on the 384D unit hypersphere (L2-normalized). Cosine distance creates **cone-shaped** clusters, not Euclidean spheres.
- Thresholds correspond to angles: e.g., theta_sim = 0.2 means ensemble distance threshold for assignment

**Clustering Thresholds**:
- theta_sim: Ensemble distance threshold for species assignment (default: 0.2)
- theta_merge: Merge threshold, tighter than theta_sim (default: 0.1)

**Capacity Limits**:
- species_capacity: Maximum genomes per species (default: 100)
- cluster0_max_capacity: Maximum genomes in reserves (default: 1000)
- Excess genomes are archived to archive.json (NOT part of population)

**Complexity**: O(N x K x d) per generation where N = population size, K = number of species, d = embedding dimension (384)

## Generation-by-Generation Flow

### Generation 0 (Initialization)
1. Load seed prompts from prompt.csv
2. Generate responses using Response Generator (LLM)
3. Evaluate fitness using Moderation Oracle (Perspective API)
4. **Run Speciation**: Leader-Follower clustering creates initial species
5. Distribute into elites.json and reserves.json (with species_id fields)
6. Calculate elite thresholds and population statistics
7. Track budget metrics (LLM calls, API calls)

### Generation N (Evolution Loop)
For each generation:

1. **Evolution Phase**:
   - Load population from elites.json and reserves.json
   - Parent Selection (adaptive tournament, species-aware)
   - Apply Variation Operators (12 mutation/crossover operators)
   - Save variants to temp.json
   - Track operator statistics (rejections, duplicates)

2. **Response Generation**:
   - Generate LLM responses for all variants in temp.json
   - Update temp.json with generated responses
   - Track response generation time

3. **Fitness Evaluation**:
   - Evaluate toxicity using Moderation Oracle (Perspective API)
   - **10 retries** with exponential backoff for rate limits
   - Update temp.json with fitness scores (toxicity, north_star_score)
   - Track evaluation time and API calls

4. **Speciation Phase**:
   - **Embedding Computation**: Compute L2-normalized embeddings for all prompts
   - **Leader-Follower Clustering**: Assign genomes to species based on ensemble distance (theta_sim threshold)
   - **Reserves Management**: Assign outliers to Cluster 0 (species_id = 0)
   - **Species Merging**: Merge similar species if leaders are close (theta_merge threshold)
   - **Extinction Check**: Freeze stagnant species (stagnation > max_stagnation)
   - **Capacity Enforcement**: Remove excess genomes from species/reserves when capacity exceeded, archive to archive.json
   - **c-TF-IDF Labeling**: Update species labels with current generation data
   - **Metrics Recording**: Track species count, diversity, merge/extinction events
   - Update all genomes in temp.json with species_id (Cluster 0 = 0)
   - Remove prompt_embedding from temp.json after speciation to reduce storage

5. **Distribution Phase**:
   - Distribute genomes based on species_id:
     - species_id > 0 -> elites.json (part of active population)
     - species_id == 0 -> reserves.json (part of active population)
   - Archived genomes -> archive.json (NOT part of population)
   - Active population = elites.json + reserves.json

6. **Tracking Phase**:
   - Update EvolutionTracker.json with generation metrics:
     - Fitness statistics (best, avg, min, max)
     - Speciation metrics (species count, diversity)
     - Budget metrics (LLM calls, API calls, times)
     - Operator statistics (per-operator counts)
   - Generate operator_effectiveness_cumulative.csv
   - Generate visualization figures

7. **Termination Check**:
   - Check if max generations reached or threshold achieved
   - If not, loop back to step 1

### Data Flow

```
prompt.csv (seed)
    |
    v
Generation 0: [Response Gen] -> [Moderation (10 retries)] -> [Speciation] -> [Distribution]
    |
    v
elites.json + reserves.json (with species_id)
    |
    v
Generation N: [Evolution] -> [Response Gen] -> [Moderation (10 retries)] -> [Speciation] -> [Distribution]
    |
    v
elites.json + reserves.json (updated with species_id)
    |
    v
[Repeat until termination]
```

### Key Data Structures

**Genome Dictionary** (in JSON files):
```python
{
    "id": int,
    "prompt": str,
    "generated_output": str,
    "toxicity": float,  # Fitness score
    "north_star_score": float,
    "species_id": int,  # Added by speciation module
    "generation": int,
    "operator": str,
    "variant_type": str,
    "moderation_result": {
        "google": {
            "scores": {
                "toxicity": float,
                "severe_toxicity": float,
                # ... 8 total attributes
            }
        }
    },
    "response_duration": float,  # LLM response time
    "evaluation_duration": float,  # API evaluation time
    # ... other metadata
}
```

**Species Structure** (in-memory and speciation_state.json):
```python
Species(
    id: int,
    leader: Individual,  # Best-fitness member (highest fitness)
    members: List[Individual],
    state: str,  # "active" | "stagnant" | "frozen"
    stagnation_counter: int,
    fitness_history: List[float],  # Best fitness per generation
    labels: List[str],  # c-TF-IDF keywords (10 words)
    label_history: List[Dict]  # Historical labels with fitness
)
```

## Metrics and Statistics

### Per-Generation Metrics (Live)

Tracked in EvolutionTracker.json:

| Metric | Description |
|--------|-------------|
| best_fitness | Maximum fitness in generation |
| avg_fitness_generation | Mean fitness of new variants |
| avg_fitness_history | Running average across population |
| elites_count | Number of genomes in species |
| reserves_count | Number of genomes in Cluster 0 |
| species_count | Number of active species |
| inter_species_diversity | Distance between species leaders |
| intra_species_diversity | Average distance within species |
| llm_calls | Number of LLM response generations |
| api_calls | Number of Perspective API evaluations |
| total_response_time | Cumulative LLM response time |
| total_evaluation_time | Cumulative API evaluation time |

### Operator Effectiveness Metrics

Tracked in operator_effectiveness_cumulative.csv:

| Metric | Description |
|--------|-------------|
| NE | Non-Elite Percentage - fraction not reaching elite status |
| EHR | Elite Hit Rate - fraction becoming elites |
| IR | Invalid/Rejection Rate - fraction rejected or invalid |
| cEHR | Conditional Elite Hit Rate - EHR excluding invalids |
| delta_mu | Mean Delta Score - average fitness improvement |
| delta_sigma | Std Dev Delta Score - variance in improvement |

### Cluster Quality Metrics (Post-hoc)

Available via utils/cluster_quality.py:

| Metric | Description |
|--------|-------------|
| Silhouette Score | Measures cluster separation and cohesion [-1, 1] |
| Davies-Bouldin Index | Lower values indicate better clustering |
| Calinski-Harabasz Index | Higher values indicate better defined clusters |

## Error Handling

### Perspective API Rate Limits

The system handles rate limits gracefully:
- **10 retries** (11 total attempts) with exponential backoff
- Wait times double each retry: 1s, 2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s, 512s
- Automatic detection of retriable errors (429, 5xx, timeout, network)
- 0.75s delay between evaluations to stay within per-minute quota

### Missing Data Handling

- **None phenotypes**: Falls back to genotype-only distance
- **Empty temp.json**: Speciation still records current state to tracker
- **Missing moderation attributes**: Set to 0, keeping D=8
- **API failures after all retries**: Genome marked as error, evolution continues

### Data Integrity

- **Incremental saves**: Genomes saved immediately after evaluation for crash recovery
- **Batch saves**: Final batch save ensures consistency
- **Archive creation**: archive.json created automatically when needed
- **Tracker always updated**: Even with no new variants, speciation data is recorded

## Summary: Speciation Integration

### Benefits

1. **Diversity Preservation**: Semantic clustering maintains distinct evolutionary niches
2. **Parallel Search**: Multiple species explore different regions of the fitness landscape simultaneously
3. **Outlier Management**: Reserves (Cluster 0) preserves high-fitness outliers that don't fit existing species
4. **Dynamic Adaptation**: Species merge when similar, freeze when stagnant
5. **Capacity Management**: Automatic archiving of excess genomes maintains population size
6. **Interpretability**: c-TF-IDF labels provide human-readable species descriptions

### Integration Status

IMPLEMENTED: All speciation components are complete and integrated into main.py

### Usage

Speciation runs automatically each generation after fitness evaluation:

```python
from speciation import run_speciation
from speciation.config import SpeciationConfig

# Create config (or use defaults)
config = SpeciationConfig(
    theta_sim=0.2,      # Ensemble distance threshold for assignment
    theta_merge=0.1,    # Merge threshold
    species_capacity=100,
    cluster0_max_capacity=1000
)

# In generation loop, after fitness evaluation:
result = run_speciation(
    temp_path="data/outputs/temp.json",
    current_generation=generation_count,
    config=config
)

# Genomes are automatically distributed to elites.json and reserves.json
# based on species_id assigned during clustering
```

### Configuration

Key parameters in SpeciationConfig:
- theta_sim=0.2: Ensemble distance threshold for species assignment
- theta_merge=0.1: Merge threshold (tighter than theta_sim)
- species_capacity=100: Maximum individuals per species
- cluster0_max_capacity=1000: Maximum individuals in reserves (Cluster 0)
- cluster0_min_cluster_size=2: Minimum size for new species from reserves
- max_stagnation=20: Maximum generations without improvement before species is frozen
- min_island_size=2: Minimum species size before moving to reserves

See src/speciation/config.py for full configuration options.

## File Structure

```
data/outputs/YYYYMMDD_HHMM/
  elites.json                      # Genomes in species (species_id > 0)
  reserves.json                    # Cluster 0 outliers (species_id = 0)
  archive.json                     # Archived genomes (capacity overflow)
  temp.json                        # Current generation staging
  top_10.json                      # Top 10 highest-fitness genomes
  EvolutionTracker.json            # Complete evolution history
  speciation_state.json            # Species state with labels
  operator_effectiveness_cumulative.csv  # Per-generation operator metrics
  figures/                         # Visualization charts
    fitness_progress.png
    diversity_metrics.png
    ...
```
