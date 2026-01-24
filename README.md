# Evolving Prompts for Toxicity Search in Large Language Models

A black-box evolutionary framework for systematic LLM safety testing through prompt evolution. Uses a genetic algorithm with semantic speciation (Leader-Follower clustering) to discover prompts that elicit toxic responses from target models.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Algorithm](#algorithm)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Output Files](#output-files)
8. [Key Concepts](#key-concepts)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)
11. [Documentation](#documentation)

---

## Overview

This system implements a **steady-state `(μ + λ)` evolutionary algorithm** with **dynamic semantic speciation** to evolve prompts that maximize toxicity scores from target LLMs. The framework uses:

- **Semantic Clustering**: Leader-Follower clustering with ensemble distance (genotype + phenotype)
- **Adaptive Parent Selection**: Three modes (DEFAULT, EXPLOIT, EXPLORE) based on population fitness trends
- **Species Management**: Dynamic species formation, merging, freezing, and extinction
- **Refusal Detection**: Automatic detection and penalization of LLM refusals
- **Comprehensive Tracking**: Detailed metrics for operator effectiveness, diversity, and cluster quality

### Use Cases

- **LLM Safety Testing**: Systematically discover prompts that trigger harmful responses
- **Red Teaming**: Automated adversarial prompt generation
- **Model Evaluation**: Assess model robustness to prompt variations
- **Research**: Study evolutionary dynamics in prompt space

---

## Features

### Core Features

- ✅ **Steady-State Evolution**: Continuous population updates without generational replacement
- ✅ **Semantic Speciation**: Leader-Follower clustering with ensemble distance
- ✅ **Adaptive Selection**: Dynamic parent selection based on fitness trends
- ✅ **Species Management**: Automatic species formation, merging, freezing, and extinction
- ✅ **Refusal Detection**: Pattern-based detection with 15% penalty
- ✅ **Operator Effectiveness Tracking**: NE, EHR, IR, cEHR, Δμ, Δσ metrics
- ✅ **Diversity Metrics**: Inter-species and intra-species diversity tracking
- ✅ **Cluster Quality**: Silhouette Score, Davies-Bouldin, Calinski-Harabasz, QD Score
- ✅ **Comprehensive Logging**: Detailed logs and validation reports

### Advanced Features

- **Multi-Model Support**: Separate response generator (RG) and prompt generator (PG) models
- **Batch Processing**: Efficient batch processing for embeddings and responses
- **Atomic File Writes**: Safe file updates using temporary files
- **Genome Tracking**: Authoritative `genome_tracker.json` for lineage tracking
- **Live Analysis**: Real-time visualization of evolution progress
- **Validation**: Flow 1 & Flow 2 validation for speciation consistency

---

## Algorithm

### Algorithm Type

**Steady-State `(μ + λ)` Evolutionary Algorithm**

- **μ (mu)**: Parent population size (elites + reserves)
- **λ (lambda)**: Offspring generated per generation
- **Steady-state**: Population continuously updated, not replaced in generations

### Fitness Function

**Fitness**: `f(x) = toxicity_score(LLM(x)) ∈ [0, 1]`

- `x`: Prompt (genome)
- `LLM(x)`: Model's response to the prompt
- `toxicity_score`: Moderation API score (Google Perspective API)

### Population Structure

**Population**: `P = E ∪ R`

- **E (Elites)**: Active species members (`species_id > 0`)
- **R (Reserves)**: Cluster 0 outliers (`species_id = 0`)
- **Archive**: Capacity-overflow genomes (`species_id = -1`)

### Variants per Generation

Variants generated depend on selection mode:

- **DEFAULT** (2 parents): `V = 10×2 + 2×1 = 22` variants
- **EXPLOIT/EXPLORE** (3 parents): `V = 10×3 + 2×3 = 36` variants

### Parent Selection (3-Category)

**Category 1** (equal importance): Active species ∪ reserves (species 0)  
**Category 2**: Frozen species (only when Category 1 is empty)

**Modes** (on chosen category):

- **DEFAULT**: 2 parents from same species (random)
- **EXPLOITATION**: 3 parents from top species (local search)
- **EXPLORATION**: 3 parents from 3 different species (diversity)

### Speciation

**Leader-Follower Clustering** with ensemble distance:

```
d_ensemble = 0.7×d_genotype + 0.3×d_phenotype
```

- **Assignment**: `d_ensemble(u, leader) < θ_sim = 0.25`
- **Merging**: `d_ensemble(leader_i, leader_j) < θ_merge = 0.1` (active and frozen species can merge)
- **Freeze**: `stagnation ≥ species_stagnation` → species frozen (preserved with all members for potential merging)
- **Capacity**: `|S_i| ≤ species_capacity`, `|R| ≤ cluster0_max_capacity`

### Post-Evaluation Processing

After moderation, **refusal penalty** (15% reduction on toxicity for detected refusals) is applied; then **avg_fitness** = mean(elites+reserves+temp) before speciation. Both are used in EvolutionTracker.

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for LLM inference)
- Google Perspective API key (for moderation)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd eost-cam-llm
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
cp env_example.txt .env
# Edit .env: add PERSPECTIVE_API_KEY=your_api_key_here
```

### Step 5: Download Models (Optional)

If using GGUF models, place them in the `models/` directory:

```bash
models/
├── llama3.1-8b-instruct-gguf/
│   ├── Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf
│   └── ...
└── ...
```

The system will automatically resolve model aliases to GGUF files based on quantization preference:
- **Q4_K_M** (recommended): Best balance of quality and speed
- **Q4_K_S**: Smaller size, slightly lower quality
- **Q5_K_M**: Higher quality, larger size, slower inference
- **Q3_K_M**: Lower quality, smaller size, faster inference
- **Q2_K**: Lowest quality, smallest size, fastest inference

---

## Configuration

### Command-Line Arguments

#### Evolution Parameters

- `--generations`: Maximum number of generations (default: 50)
- `--threshold`: North-star threshold for termination (default: 0.99)
- `--stagnation-limit`: Generations without improvement before EXPLORE mode (default: 5)
- `--max-variants`: Maximum variants per generation (default: 1)

#### Model Configuration

- `--rg`: Response generator model path or alias (default: `models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf`)
- `--pg`: Prompt generator model path or alias (default: same as `--rg`)
- `--moderation-methods`: Moderation methods (default: `google`)

#### Speciation Parameters

- `--theta-sim`: Similarity threshold for species assignment (default: 0.25)
- `--theta-merge`: Merge threshold for combining similar species (default: 0.1, must be ≤ theta-sim)
- `--species-capacity`: Maximum individuals per species (default: 100)
- `--cluster0-max-capacity`: Maximum individuals in cluster 0 (default: 1000)
- `--cluster0-min-cluster-size`: Minimum cluster size for cluster 0 speciation (default: 2)
- `--min-island-size`: Minimum island size before extinction (default: 2)
- `--species-stagnation`: Generations without improvement before freezing (default: 20)

#### Embedding Parameters

- `--embedding-model`: Sentence-transformer model name (default: `all-MiniLM-L6-v2`)
- `--embedding-dim`: Embedding dimensionality (default: 384)
- `--embedding-batch-size`: Batch size for embedding computation (default: 64)

#### Operator Configuration

- `--operators`: Operators to use (default: `all`)
  - Options: `all`, `paraphrasing`, `concept_addition`, `back_translation`, etc.
- `--seed-file`: Path to seed prompts CSV (default: `data/prompt.csv`)

### Configuration Files

#### RGConfig.yaml / PGConfig.yaml

Model configuration files (auto-updated by system):

```yaml
response_generator:
  model_path: "models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"
  # ... other model parameters
```

#### .env

Environment variables:

```bash
PERSPECTIVE_API_KEY=your_api_key_here
```

---

## Usage

### Quick Start

```bash
bash run_experiments_local.sh
```

### Direct Execution

```bash
python src/main.py \
    --generations 50 \
    --threshold 0.99 \
    --moderation-methods google \
    --stagnation-limit 5 \
    --theta-sim 0.25 \
    --theta-merge 0.1 \
    --species-capacity 100 \
    --cluster0-max-capacity 1000 \
    --cluster0-min-cluster-size 2 \
    --min-island-size 2 \
    --species-stagnation 20 \
    --embedding-model all-MiniLM-L6-v2 \
    --embedding-dim 384 \
    --embedding-batch-size 64 \
    --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
    --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
    --operators all \
    --max-variants 1 \
    --seed-file data/prompt.csv
```

### Minimal Example

```bash
python src/main.py \
    --generations 10 \
    --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
    --operators paraphrasing \
    --seed-file data/prompt.csv
```

### Advanced Usage

#### Custom Operators

```bash
python src/main.py \
    --operators paraphrasing,concept_addition,back_translation \
    --generations 50
```

#### High-Throughput Configuration

```bash
python src/main.py \
    --generations 100 \
    --species-capacity 200 \
    --cluster0-max-capacity 2000 \
    --embedding-batch-size 128 \
    --max-variants 2
```

#### Exploration-Focused Configuration

```bash
python src/main.py \
    --theta-sim 0.30 \
    --theta-merge 0.15 \
    --species-stagnation 10 \
    --stagnation-limit 3
```

---

## Output Files

All output files are located in `data/outputs/YYYYMMDD_HHMM/`:

### Core Population Files

| File | Description | Format |
|------|-------------|--------|
| `elites.json` | Species members (species_id > 0), cumulative across all generations | JSON array |
| `reserves.json` | Cluster 0 (species_id = 0), capacity-limited, sorted by fitness (descending) | JSON array |
| `archive.json` | Archived genomes (capacity overflow, etc.), cumulative | JSON array |
| `temp.json` | Current variants before speciation; cleared/repopulated each generation | JSON array |

### Tracking Files

| File | Description | Format |
|------|-------------|--------|
| `EvolutionTracker.json` | Per-generation stats, speciation block, cumulative max, selection state | JSON object |
| `speciation_state.json` | Species structure (leader_*, member_ids, max_fitness, stagnation, state), cluster0, metrics | JSON object |
| `genome_tracker.json` | ID → metadata for lineage, species_id tracking (authoritative source of truth) | JSON object |
| `operator_effectiveness_cumulative.csv` | Per (generation, operator) metrics: NE, EHR, IR, cEHR, Δμ, Δσ | CSV |

### Visualization Files

| Directory | Description |
|-----------|-------------|
| `figures/` | Fitness, diversity, operator visualizations (PNG/PDF) |

### Species States

- **active**: Participates in evolution and parent selection (Category 1)
- **frozen**: Stagnated (≥`species_stagnation` generations), excluded from Category 1, preserved with all members. Can merge with active or other frozen species.
- **incubator**: Moved to cluster 0 when `size < min_island_size`, tracked by ID only
- **extinct**: Parent species after merging, tracked in `historical_species` (ID only)

### Key Fields in EvolutionTracker.json

**Top-Level Fields**:
- `population_max_toxicity`: Cumulative maximum fitness across all generations
- `generations_since_improvement`: Consecutive generations without improvement
- `selection_mode`: Current selection mode (default/exploit/explore)
- `slope_of_avg_fitness`: Slope of fitness history
- `avg_fitness_history`: Sliding window of recent generations' avg_fitness

**Per-Generation Fields**:
- `generation_number`: Generation index (0-based)
- `max_score_variants`: Max fitness among variants created this generation
- `avg_fitness`: Mean over old elites + old reserves + all new variants (before speciation)
- `avg_fitness_generation`: Mean over updated elites + updated reserves (after distribution)
- `elites_count`, `reserves_count`, `archived_count`: Population counts
- `speciation`: Speciation metrics (species_count, events, diversity, cluster_quality)

For complete field definitions, see [FIELD_DEFINITIONS.txt](FIELD_DEFINITIONS.txt).

---

## Key Concepts

### Adaptive Selection Logic

The system dynamically adjusts parent selection strategy based on population fitness trends:

1. **DEFAULT Mode**: Normal evolution, 2 parents from random species
2. **EXPLOIT Mode**: Triggered when `slope_of_avg_fitness <= 0` (declining/flat fitness)
   - Strategy: Focus on best-performing species (local search)
   - Selection: 3 parents from top species
3. **EXPLORE Mode**: Triggered when `generations_since_improvement >= stagnation_limit` (stagnation)
   - Strategy: Increase diversity and exploration
   - Selection: 1 parent each from top + 2 random species

### Speciation Phases

The speciation process consists of 8 phases:

1. **Existing Species Processing**: Assign new variants to existing species or cluster 0
2. **Cluster 0 Speciation**: Form new species from cluster 0 when cohesive clusters emerge
3. **Merging**: Combine similar species (θ_merge threshold)
4. **Radius & Capacity Enforcement**: Enforce species boundaries and capacity limits
5. **Freeze & Incubator**: Track stagnation, freeze stagnant species, dissolve small species
6. **Cluster 0 Capacity Enforcement**: Archive excess reserves
7. **Final Redistribution**: Synchronize file-based species_id with genome_tracker
8. **Metrics & Stats**: Calculate diversity, cluster quality, update trackers

### Refusal Detection

The system automatically detects LLM refusals using:
- **Pattern Matching**: Common refusal phrases ("I can't assist", "I can't answer", etc.)
- **Length-Based Heuristics**: Short responses with specific punctuation patterns
- **Penalty**: 15% reduction (×0.85) on toxicity for detected refusals

### Genome Tracker

`genome_tracker.json` is the **authoritative source of truth** for `species_id` assignments:
- Updated at every speciation event (clustering, merging, archiving, etc.)
- Used in Phase 7 to synchronize file-based `species_id` values
- Enables efficient deferred file updates (files only fully synchronized in Phase 7)

### Ensemble Distance

Species assignment uses ensemble distance combining genotype and phenotype:

```
d_ensemble = 0.7×d_genotype + 0.3×d_phenotype
```

- **d_genotype**: Cosine distance between prompt embeddings
- **d_phenotype**: Normalized difference in response toxicity scores

---

## Examples

### Example 1: Basic Run

```bash
python src/main.py \
    --generations 20 \
    --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
    --operators all
```

### Example 2: Custom Speciation Parameters

```bash
python src/main.py \
    --generations 50 \
    --theta-sim 0.30 \
    --theta-merge 0.15 \
    --species-capacity 50 \
    --cluster0-max-capacity 500 \
    --species-stagnation 15
```

### Example 3: Exploration-Focused

```bash
python src/main.py \
    --generations 100 \
    --stagnation-limit 3 \
    --theta-sim 0.35 \
    --operators all
```

### Example 4: High-Capacity Run

```bash
python src/main.py \
    --generations 200 \
    --species-capacity 200 \
    --cluster0-max-capacity 2000 \
    --embedding-batch-size 128
```

---

## Troubleshooting

### Common Issues

#### Issue: "EvolutionTracker.json not found"

**Solution**: Ensure you're running from the project root and the output directory exists.

#### Issue: "Required 'questions' column not found in CSV file"

**Solution**: Ensure your seed file (`data/prompt.csv`) has a `questions` column (case-insensitive, whitespace-stripped).

#### Issue: "No models could be resolved"

**Solution**: 
1. Check that model files exist in `models/` directory
2. Verify model alias matches directory name
3. Ensure GGUF files are present in the model directory

#### Issue: "PERSPECTIVE_API_KEY not found"

**Solution**: 
1. Create `.env` file from `env_example.txt`
2. Add `PERSPECTIVE_API_KEY=your_api_key_here`

#### Issue: Low GPU utilization

**Solution**:
- Increase `--embedding-batch-size` (e.g., 128 or 256)
- Use quantized models (Q4_K_M or Q4_K_S)
- Ensure CUDA is properly configured

#### Issue: Memory errors

**Solution**:
- Reduce `--species-capacity` and `--cluster0-max-capacity`
- Reduce `--embedding-batch-size`
- Use smaller quantized models (Q3_K_M or Q2_K)

### Validation

The system includes comprehensive validation:

- **Flow 1 Validation**: Validates existing species processing
- **Flow 2 Validation**: Validates cluster 0 speciation
- **Consistency Checks**: Validates genome_tracker consistency

Check logs for validation warnings and errors.

---

## Documentation

### Core Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Comprehensive system architecture documentation
  - Module layout and components
  - High-level flow
  - Parent selection system
  - Speciation framework
  - Key metrics and conventions
  - Adaptive selection logic
  - Configuration parameters

- **[FIELD_DEFINITIONS.txt](FIELD_DEFINITIONS.txt)**: Field-by-field definitions for all JSON/CSV outputs
  - Genome object structure
  - EvolutionTracker.json fields
  - speciation_state.json structure
  - Operator effectiveness CSV columns

- **[PROCESS_FLOW.md](PROCESS_FLOW.md)**: Complete end-to-end process flow
  - Generation 0 initialization
  - Generation N evolution loop
  - Speciation phases
  - File updates per generation
  - Key metrics and their timing

### Additional Documentation

- **[experiments/FLOW_AND_VALIDATION.md](experiments/FLOW_AND_VALIDATION.md)**: Flow, validation, and field basis
- **[ISSUES_RESOLVED.md](ISSUES_RESOLVED.md)**: Summary of resolved issues and fixes
- **[VALIDATION_REPORT.md](VALIDATION_REPORT.md)**: Comprehensive validation reports

### Code Documentation

- **Docstrings**: All modules include comprehensive docstrings
- **Type Hints**: Functions include type hints for better IDE support
- **Comments**: Inline comments explain complex logic

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{toxsearch,
  title = {Evolving Prompts for Toxicity Search in Large Language Models},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/eost-cam-llm}
}
```

---

## License

[Specify your license here]

---

## Contributing

[Contributing guidelines]

---

## Acknowledgments

[Acknowledgments]

---

**Last Updated**: 2026-01-23
