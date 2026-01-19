# Evolving Prompts for Toxicity Search in Large Language Models

Black-box evolutionary framework for systematic LLM safety testing through prompt evolution. Uses genetic algorithm with semantic speciation (Leader-Follower clustering).

## Algorithm

**EA Type**: Steady-state `(μ + λ)` evolutionary algorithm

**Fitness**: `f(x) = toxicity_score(LLM(x)) ∈ [0, 1]` where `x` is a prompt and `LLM(x)` is the model's response.

**Population**: `P = E ∪ R`, `|P| = |E| + |R|` where `E` = elites (species), `R` = reserves (Cluster 0).

**Variants per Generation**:
- DEFAULT (2 parents): `V = 10×2 + 2×1 = 22`
- EXPLORATION/EXPLOITATION (3 parents): `V = 10×3 + 2×3 = 36`

**Parent Selection (3-category)**: Category 1 = active species ∪ reserves (species 0); Category 2 = frozen. Use Category 2 only when Category 1 is empty. Modes on the chosen category:
- DEFAULT: 2 parents from same species (random)
- EXPLOITATION: 3 parents from top species (local search)
- EXPLORATION: 3 parents from 3 different species (diversity)

**Post-evaluation**: After moderation, **refusal penalty** (15% reduction on toxicity for detected refusals) is applied; then **avg_fitness** = mean(elites+reserves+temp) before speciation. Both are used in EvolutionTracker.

**Speciation**: Leader-Follower clustering with ensemble distance `d_ensemble = 0.7×d_genotype + 0.3×d_phenotype`
- Assignment: `d_ensemble(u, leader) < θ_sim = 0.2`
- Merging: `d_ensemble(leader_i, leader_j) < θ_merge = 0.1` (active and frozen species can merge)
- Freeze: `stagnation ≥ 20` → species frozen (preserved with all members for potential merging)
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
- `temp.json` - Variants before speciation (per generation; cleared/repopulated each gen)
- `EvolutionTracker.json` - Per-generation stats (avg_fitness, variant stats, speciation block), cumulative max, selection state. best_fitness/avg_fitness at gen level only; speciation block has species_count, diversity, cluster_quality, etc.
- `speciation_state.json` - Species (active/frozen/incubator), leader_*, member_ids, max_fitness, stagnation; cluster0; metrics
- `genome_tracker.json` - ID → metadata for lineage
- `operator_effectiveness_cumulative.csv` - Operator effectiveness (RQ1)
- `figures/` - Fitness, diversity, operator visualizations

**Species States**:
- **active**: Participates in evolution and parent selection
- **frozen**: Stagnated (≥20 gens), excluded from parent selection, preserved with all members (leader embeddings, distances, labels, history). Can merge with active or other frozen species. Both active and frozen are "alive" - only difference is parent selection preference.
- **incubator**: Moved to cluster 0 when `size < min_island_size` (default: < 2), tracked by ID only

**Docs**:
- [ARCHITECTURE.md](ARCHITECTURE.md) — Modules, flow, parent selection, speciation, metrics
- [FIELD_DEFINITIONS.txt](FIELD_DEFINITIONS.txt) — Field definitions for all JSON/CSV outputs
- [experiments/FLOW_AND_VALIDATION.md](experiments/FLOW_AND_VALIDATION.md) — Flow, validation, and field basis
