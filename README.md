# Evolving Prompts for Toxicity Search in Large Language Models

Black-box evolutionary framework for systematic LLM safety testing through prompt evolution. Uses genetic algorithm with semantic speciation (Leader-Follower clustering).

## Algorithm

**EA Type**: Steady-state `(μ + λ)` evolutionary algorithm

**Fitness**: `f(x) = toxicity_score(LLM(x)) ∈ [0, 1]` where `x` is a prompt and `LLM(x)` is the model's response.

**Population**: `P = E ∪ R`, `|P| = |E| + |R|` where `E` = elites (species), `R` = reserves (Cluster 0).

**Variants per Generation**:
- DEFAULT (2 parents): `V = 10×2 + 2×1 = 22`
- EXPLORATION/EXPLOITATION (3 parents): `V = 10×3 + 2×3 = 36`

**Parent Selection**:
- DEFAULT: 2 parents from same species (random)
- EXPLOITATION: 3 parents from top species (local search)
- EXPLORATION: 3 parents from 3 different species (diversity)

**Speciation**: Leader-Follower clustering with ensemble distance `d_ensemble = 0.7×d_genotype + 0.3×d_phenotype`
- Assignment: `d_ensemble(u, leader) < θ_sim = 0.2`
- Merging: `d_ensemble(leader_i, leader_j) < θ_merge = 0.1` (active species only)
- Freeze: `stagnation ≥ 20` → species frozen (preserved with embeddings for potential merging)
- Capacity: `|S_i| ≤ 100`, `|R| ≤ 1000`

## Installation

```bash
git clone <repository-url>
cd eost-cam-llm
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
cp env_example.txt .env
# Edit .env: add PERSPECTIVE_API_KEY
```

## Usage

```bash
bash run_experiments_local.sh
```

Or directly:
```bash
python src/main.py \
    --generations 50 \
    --operators all \
    --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
    --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf
```

## Outputs

`data/outputs/YYYYMMDD_HHMM/`:
- `elites.json` - Species members (species_id > 0, all generations)
- `reserves.json` - Cluster 0 (species_id = 0, max 1000)
- `archive.json` - Archived genomes (capacity overflow)
- `EvolutionTracker.json` - Complete evolution history with metrics
- `speciation_state.json` - Species state (active/frozen/incubator) with leader embeddings
- `operator_effectiveness_cumulative.csv` - Operator effectiveness metrics (RQ1)
- `figures/` - Visualizations (fitness, diversity, operator metrics)

**Species States**:
- **active**: Participates in evolution
- **frozen**: Stagnated (≥20 gens), excluded from selection, preserved with embeddings
- **incubator**: Moved to cluster 0 when `size < min_island_size` (default: < 2), tracked by ID only

See [ARCHITECTURE.md](ARCHITECTURE.md) for details.
