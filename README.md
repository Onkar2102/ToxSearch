# Evolving Prompts for Toxicity Search in Large Language Models

A black-box evolutionary framework for systematically testing LLM safety through prompt evolution to elicit toxic responses. The system uses a genetic algorithm with semantic speciation (Leader-Follower clustering) to evolve prompts and evaluate model vulnerabilities.

For detailed algorithmic design, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Algorithmic Scope

This system implements a steady-state evolutionary algorithm for black-box behavioral testing of large language models. It is not a supervised learning framework, nor does it perform gradient-based optimization or model fine-tuning.

The evolutionary loop, parent selection, variation operators, and speciation mechanisms are implemented explicitly and are configurable via command-line parameters.

## Evolutionary Model

- **EA Type**: Steady-state (μ + λ) evolutionary algorithm
- **Population Structure**: μ genomes maintained across generations (elites + reserves)
- **Offspring Generation**: λ variants generated per cycle (configurable via `--max-variants`)
- **Fitness Function**: 
  ```
  f(x) = toxicity_score(LLM(x)) ∈ [0, 1]
  ```
  where `x` is a prompt and `LLM(x)` is the model's response. The fitness evaluates toxicity in the response, not the prompt itself.
- **Speciation**: Semantic + behavioral clustering using Leader-Follower algorithm

## Speciation

The system uses semantic speciation to partition the population into dynamically evolving species (islands). Speciation is used to preserve diversity, prevent premature convergence, and support parallel exploration of distinct prompt strategies.

**Distance Metrics**:
- **Genotype Distance**: `d_genotype(u,v) = 1 - (e_u · e_v)` where `e_u, e_v` are L2-normalized 384D embeddings
- **Phenotype Distance**: `d_phenotype(u,v) = ||p_u - p_v||₂ / √8` where `p_u, p_v` are 8D toxicity vectors
- **Ensemble Distance**: `d_ensemble(u,v) = 0.7 · d_genotype_norm(u,v) + 0.3 · d_phenotype(u,v)`

**Speciation Thresholds**:
- Species assignment: `d_ensemble(u, leader) < θ_sim` (default: `θ_sim = 0.2`)
- Species merging: `d_ensemble(leader_i, leader_j) < θ_merge` (default: `θ_merge = 0.1`)

A reserve population (cluster 0) stores outliers and enables the emergence of new species. Species merge when similar, freeze when stagnant, and maintain capacity limits:
- Species capacity: `|S_i| ≤ C_species` (default: `C_species = 100`)
- Reserves capacity: `|Cluster_0| ≤ C_reserves` (default: `C_reserves = 1000`)

## Installation

### Prerequisites
- **Python**: 3.8 or higher
- **Google Perspective API Key**: Required for toxicity evaluation ([Get it here](https://developers.perspectiveapi.com/))
- **Hardware**: CUDA/GPU, MPS (Apple Silicon), or CPU support
- **Disk Space**: ~10GB+ for GGUF models

### Setup Instructions

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd eost-cam-llm
```

#### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Set Up Environment Variables
```bash
# Create .env from the example environment file
cp env_example.txt .env

# Edit .env and add your API keys
nano .env  # or use your preferred text editor
```

Add your API keys to `.env`:
```env
# REQUIRED: Google Perspective API Key
PERSPECTIVE_API_KEY=your_google_perspective_api_key_here

# OPTIONAL: HuggingFace Hub Token (only needed for downloading gated/private models)
# HF_TOKEN=your_huggingface_token_here
# HUGGINGFACE_HUB_TOKEN=your_huggingface_token_here
```

**Get your API keys:**
- **Google Perspective API Key**: [Get it here](https://developers.perspectiveapi.com/)
- **HuggingFace Token**: [Get it here](https://huggingface.co/settings/tokens) (optional, only for gated models)

#### 5. Dataset Setup

Download the initial prompt dataset from HuggingFace:
```bash
# Download harmful prompt datasets from HuggingFace
# This will create data/prompt.csv and data/prompt_extended.csv
python src/utils/data_loader.py
```

This creates:
- `data/prompt.csv` - 100 prompts (seed population)
- `data/prompt_extended.csv` - Full dataset

**Note**: For custom experiments, create your own `data/prompt.csv` with a 'questions' column.

#### 6. Download Models

Download GGUF models to the `models/` directory. You can either:

1. **Manual download** from HuggingFace (recommended)
2. **Use download script**: `python src/utils/download_models.py` (requires HF token in `.env`)

Place models in `models/<model-name>/` directories. Example structure:
```
models/
├── llama3.2-3b-instruct-gguf/
│   └── Llama-3.2-3B-Instruct-Q4_K_M.gguf
└── llama3.1-8b-instruct-gguf/
    └── Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf
```

## Running Experiments

### Quick Start

Edit `run_experiments_local.sh` to configure your experiments, then run:

```bash
bash run_experiments_local.sh
```

### Direct Python Execution

```bash
python src/main.py \
    --generations 50 \
    --operators all \
    --max-variants 1 \
    --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
    --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
    --seed-file data/prompt.csv
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--generations` | None | Maximum evolution generations (runs indefinitely if not set) |
| `--operators` | `all` | Operator mode: `ie`, `cm`, or `all` |
| `--max-variants` | 1 | Maximum variants per evolution cycle |
| `--rg` | - | Response generator model path (.gguf file) |
| `--pg` | - | Prompt generator model path (.gguf file) |
| `--seed-file` | `data/prompt.csv` | Seed prompt file |
| `--theta-sim` | 0.2 | Species assignment threshold |
| `--theta-merge` | 0.1 | Species merge threshold |
| `--species-capacity` | 100 | Maximum genomes per species |
| `--cluster0-max-capacity` | 1000 | Maximum genomes in reserves |

Run `python src/main.py --help` for complete argument list.

## Outputs

Each experiment creates a timestamped directory `data/outputs/YYYYMMDD_HHMM/` containing:

- `elites.json` - Genomes in species (species_id > 0)
- `reserves.json` - Cluster 0 outliers (species_id = 0)
- `archive.json` - Archived genomes (capacity overflow)
- `EvolutionTracker.json` - Complete evolution history and per-generation metrics
- `speciation_state.json` - Species state with labels
- `operator_effectiveness_cumulative.csv` - Per-generation operator metrics
- `figures/` - Visualization charts

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed metrics and statistics documentation.

## Reproducibility Notes

- All experiments are stochastic; results depend on random seeds.
- Exact reproducibility requires fixing Python, NumPy, and model seeds.
- Perspective API scores may vary slightly due to backend updates.
- Logs and intermediate states are stored to enable post-hoc analysis.

## Troubleshooting

### Common Issues

**`PERSPECTIVE_API_KEY not set`**
- Ensure `.env` file exists with `PERSPECTIVE_API_KEY=your_key_here`

**`Model file not found`**
- Verify model files exist in `models/<model-name>/` directory
- Check model path in command line arguments

**`CUDA out of memory`**
- Use lower quantization (Q4_K_M instead of Q8)
- Reduce `--gpu-layers` or run on CPU

**Rate limit errors**
- System automatically retries with exponential backoff
- Check API quota at [Google Cloud Console](https://console.cloud.google.com)
- Reduce `--max-variants` to lower API call frequency

**No variants generated**
- Check logs in `logs/` directory
- Verify population has valid genomes
- May indicate population convergence

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed error handling documentation.

## Project Structure

```
eost-cam-llm/
├── src/
│   ├── main.py              # Entry point
│   ├── ea/                  # Evolutionary algorithm
│   ├── gne/                 # Generation and evaluation
│   ├── speciation/          # Semantic speciation module
│   └── utils/               # Utilities
├── config/                  # Model configurations
├── data/                    # Prompts and outputs
├── models/                  # GGUF model files
├── ARCHITECTURE.md          # Detailed architecture
└── requirements.txt
```

## License

MIT License - See LICENSE file for details.
