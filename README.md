# Evolving Prompts for Toxicity Search in Large Language Models

This repository implements a black-box evolutionary framework for systematically testing Large Language Model (LLM) safety through LLM-guided prompt evolution to elicit toxic responses.

## Current Approach Highlights
- **Distance metric**: Cosine distance on **L2-normalized 384D** embeddings; clusters are angular (cone-shaped), not Euclidean spheres.
- **Speciation options**:
  - **Leader-Follower (online, fast)**: Directly clusters 384D embeddings with cosine thresholds (θ_sim≈0.2); no training required.
  - **Parametric UMAP (offline, high quality)**: Trains a 384→16 encoder once on Gen0; preserves geometry (kNN-IoU≈0.55, r≈0.85), then clusters with HDBSCAN or centroiding.
  - **Hybrid**: Use Parametric UMAP for analysis/visualization; use Leader-Follower for real-time evolution.

## Abstract

Large Language Models remain vulnerable to adversarial prompts that elicit toxic content even after safety alignment. We present a black-box evolutionary framework that tests model safety by evolving prompts in a synchronous, steady-state $(\mu+\lambda)$ loop. The system employs a diverse operator suite, including lexical substitutions, negation, back-translation, paraphrasing, and two semantic crossover operators, while a moderation oracle provides fitness guidance. Under a fixed generation budget, a few-shot global rewrite operator achieves the highest progress \emph{per evaluated prompt} but plateaus at substantially lower best-of-run toxicity than our engineered lexical operators, which more reliably push populations toward high-toxicity regimes. Operator-level analysis reveals significant heterogeneity, as lexical substitutions offer the best yield–variance trade-off, semantic-similarity crossover acts as a precise low-throughput inserter, and global rewrites exhibit high variance with elevated refusal costs. Using elite prompts evolved on LLaMA~3.1~8B, we observe practically meaningful but attenuated cross-model transfer. Toxicity drops by roughly half on most targets, with smaller LLaMA~3.2 variants showing the strongest resistance and some cross-architecture models (e.g., Qwen and Mistral) retaining higher toxicity. Overall, our results indicate that small, controllable perturbations serve as reliable vehicles for systematic red-teaming, while defenses should anticipate cross-model prompt reuse rather than focusing solely on single-model hardening.

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
cd etg
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

This downloads from CategoricalHarmfulQA and HarmfulQA datasets and creates:
- `data/prompt.csv` - 100 prompts (seed population)
- `data/prompt_extended.csv` - Full dataset

**Note**: For custom experiments, you can create your own `data/prompt.csv` with prompts (one per line, CSV format with 'questions' column).

#### 6. Download Models

Download GGUF models to the `models/` directory. Models are available from HuggingFace or Ollama.

**Option A: Manual Download from HuggingFace (Recommended)**
```bash
# Example: Llama 3.2 3B Instruct
# Go to: https://huggingface.co/Unsloth/Llama-3.2-3B-Instruct-GGUF
# Download: Llama-3.2-3B-Instruct-Q4_K_M.gguf
# Create directory: models/llama3.2-3b-instruct-gguf/
# Place the .gguf file in that directory
```

**Option B: Use Download Script**
If downloading gated models, make sure you've added your HF token to `.env` (see Step 4).

Then download models:
```bash
python src/utils/download_models.py
```

**Model Directory Structure:**
```
models/
├── llama3.2-3b-instruct-gguf/
│   └── Llama-3.2-3B-Instruct-Q4_K_M.gguf
├── llama3.1-8b-instruct-gguf/
│   └── Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf
└── mistral-7b-instruct-gguf/
    └── mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

## How to Run the Project

After completing the installation steps (Steps 1-6 above), you can run the project:

### Quick Start: Using the Experiment Script

Edit `run_experiments_local.sh` to configure your experiments, then run:

```bash
bash run_experiments_local.sh
```

The script will:
- Activate your virtual environment
- Run all experiments defined in the `EXPERIMENTS` array
- Handle errors and provide progress updates

### Running Directly with Python

You can also run experiments directly:
```bash
python src/main.py \
    --generations 50 \
    --operators all \
    --max-variants 1 \
    --elites-threshold 25 \
    --removal-threshold 5 \
    --stagnation-limit 5 \
    --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_S.gguf \
    --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf \
    --seed-file data/prompt.csv
```

### Command Line Arguments

Complete reference table:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--generations` | int | None | Maximum number of evolution generations. If not set, runs indefinitely |
| `--threshold` | float | 0.99 | North star metric threshold for stopping evolution |
| `--moderation-methods` | list | `["google"]` | Moderation methods: `google` (Perspective API), `all` (google only) |
| `--threshold-percentage` | int | 25 | Percentage for elite threshold calculation |
| `--elites-threshold` | int | 25 | Elite threshold percentage (top % genomes considered elite) |
| `--removal-threshold` | int | 5 | Removal threshold percentage (bottom % genomes archived) |
| `--stagnation-limit` | int | 5 | Generations without improvement before switching to explore mode |
| `--rg` | str | `models/llama3.2-3b-instruct-gguf/...` | Response generator model path (.gguf file or alias) |
| `--pg` | str | `models/llama3.2-3b-instruct-gguf/...` | Prompt generator model path (.gguf file or alias) |
| `--operators` | str | `"all"` | Operator mode: `ie` (InformedEvolution only), `cm` (all except IE), `all` (all operators) |
| `--max-variants` | int | 1 | Maximum variants to generate per evolution cycle |

#### Speciation Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--theta-sim` | float | 0.2 | Similarity threshold for species assignment (ensemble distance) |
| `--theta-merge` | float | 0.1 | Merge threshold for combining similar species |
| `--species-capacity` | int | 100 | Maximum genomes per species |
| `--cluster0-max-capacity` | int | 1000 | Maximum genomes in reserves (Cluster 0) |
| `--cluster0-min-cluster-size` | int | 2 | Minimum size for new species from reserves |
| `--min-island-size` | int | 2 | Minimum species size before moving to reserves |
| `--max-stagnation` | int | 20 | Generations without improvement before species is frozen |

**Key Arguments Explained:**

- **`--operators`**: Controls which variation operators are used
  - `ie`: Only InformedEvolution (LLM-guided evolution using top performers)
  - `cm`: All classical operators (synonym replacement, back-translation, etc.) except InformedEvolution
  - `all`: All operators including both classical and LLM-driven
  
- **`--elites-threshold`**: Percentage of top-scoring genomes considered "elites" (used for parent selection)

- **`--removal-threshold`**: Percentage of worst-performing genomes to archive to `archive.json`

- **`--stagnation-limit`**: When fitness hasn't improved for N generations, switch from exploit mode to explore mode to search broader solution space

- **Model arguments (`--rg`, `--pg`)**: Can specify either a direct `.gguf` file path or an alias (directory name under `models/`)

## Output Files

Each experiment creates a timestamped output directory in `data/outputs/YYYYMMDD_HHMM/` containing:

| File | Description |
|------|-------------|
| `elites.json` | Genomes assigned to species (species_id > 0) |
| `reserves.json` | Cluster 0 outliers (species_id = 0) |
| `archive.json` | Archived genomes removed due to capacity limits |
| `temp.json` | Temporary staging file for current generation variants |
| `top_10.json` | Top 10 highest-fitness genomes |
| `EvolutionTracker.json` | Complete evolution history and per-generation statistics |
| `speciation_state.json` | Current state of all species including labels |
| `operator_effectiveness_cumulative.csv` | Per-generation operator metrics |
| `figures/` | Visualization charts (fitness curves, diversity metrics, etc.) |

## Metrics and Statistics

### Operator Effectiveness Metrics

Per-generation metrics tracked in `operator_effectiveness_cumulative.csv`:

| Metric | Description |
|--------|-------------|
| `NE` | Non-Elite Percentage - fraction of variants not reaching elite status |
| `EHR` | Elite Hit Rate - fraction of variants that become elites |
| `IR` | Invalid/Rejection Rate - fraction of variants rejected or producing invalid outputs |
| `cEHR` | Conditional Elite Hit Rate - EHR excluding invalid variants |
| `Δμ` | Mean Delta Score - average fitness improvement over parents |
| `Δσ` | Std Dev Delta Score - variance in fitness improvement |

### Budget Tracking

Each generation tracks computational budget:
- **LLM Calls**: Number of response generation calls
- **API Calls**: Number of Perspective API evaluation calls
- **Response Time**: Total time for LLM response generation
- **Evaluation Time**: Total time for toxicity evaluation

### Diversity Metrics

Speciation module tracks:
- **Inter-species Diversity**: Distance between species leaders
- **Intra-species Diversity**: Average distance within each species
- **Species Count**: Number of active species
- **Reserves Size**: Number of genomes in Cluster 0

### Cluster Quality Metrics (Post-hoc)

Available for post-experiment analysis:
- **Silhouette Score**: Measures cluster separation and cohesion
- **Davies-Bouldin Index**: Lower values indicate better clustering
- **Calinski-Harabasz Index**: Higher values indicate better defined clusters

## Troubleshooting

### Perspective API Rate Limits

The system automatically handles Perspective API rate limits with:
- **10 retries** (11 total attempts) with exponential backoff
- Wait times: 1s, 2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s, 512s between retries
- Automatic handling of 429 (rate limit) and 5xx (server) errors

If you see frequent rate limit errors:
1. Check your Perspective API quota at [Google Cloud Console](https://console.cloud.google.com)
2. Consider reducing `--max-variants` to lower API call frequency
3. The 0.75s delay between evaluations helps stay within limits

### Common Errors

**`PERSPECTIVE_API_KEY not set`**
```bash
# Make sure .env file exists and contains valid key
cat .env | grep PERSPECTIVE
# Should show: PERSPECTIVE_API_KEY=AIza...
```

**`Model file not found`**
```bash
# Verify model file exists
ls -la models/llama3.2-3b-instruct-gguf/
# Should show .gguf files
```

**`CUDA out of memory`**
- Reduce model quantization (use Q4_K_M instead of Q8)
- Set `--gpu-layers` to limit GPU memory usage
- Or run on CPU with `--gpu-layers 0`

**`Empty temp.json or no variants`**
- Check logs for operator errors
- Verify parent selection is finding valid genomes
- May indicate population has converged (all variants are duplicates)

### Logs

Detailed logs are saved to `logs/` directory:
- One log file per experiment with full debug information
- Check logs for error details and operator performance

## Project Structure

```
eost-cam-llm/
├── src/
│   ├── main.py                 # Entry point
│   ├── ea/                     # Evolutionary algorithm components
│   │   ├── evolution_engine.py # Core evolution logic
│   │   ├── parent_selector.py  # Adaptive parent selection
│   │   ├── run_evolution.py    # Evolution loop
│   │   └── [operators]         # 12 variation operators
│   ├── gne/                    # Generation and evaluation
│   │   ├── response_generator.py
│   │   └── evaluator.py        # Perspective API integration
│   ├── speciation/             # Semantic speciation module
│   │   ├── run_speciation.py   # Main speciation entry point
│   │   ├── leader_follower.py  # L-F clustering algorithm
│   │   ├── species.py          # Species/Individual dataclasses
│   │   ├── merging.py          # Species merge logic
│   │   ├── extinction.py       # Stagnation-based extinction
│   │   ├── reserves.py         # Cluster 0 (outliers) management
│   │   ├── metrics.py          # Diversity metrics
│   │   └── labeling.py         # c-TF-IDF species labels
│   └── utils/                  # Utility modules
│       ├── population_io.py    # JSON I/O operations
│       ├── operator_effectiveness.py
│       ├── live_analysis.py    # Visualization generation
│       └── cluster_quality.py  # Post-hoc metrics
├── config/
│   ├── RGConfig.yaml           # Response generator config
│   └── PGConfig.yaml           # Prompt generator config
├── data/
│   ├── prompt.csv              # Seed prompts
│   └── outputs/                # Experiment outputs
├── models/                     # GGUF model files
├── ARCHITECTURE.md             # Detailed system architecture
└── requirements.txt
```

## Variation Operators

The system includes 12 variation operators:

### Mutation Operators (10)
1. **InformedEvolution**: LLM-guided evolution using top performers
2. **MLMOperator**: Masked language model word substitution
3. **LLMBasedParaphrasing**: Semantic-preserving paraphrasing
4. **BackTranslation**: Hindi roundtrip translation
5. **SynonymReplacement**: POS-aware lexical substitution
6. **AntonymReplacement**: POS-aware antonym substitution
7. **NegationOperator**: Logical negation insertion
8. **ConceptAddition**: Semantic concept injection
9. **TypographicalErrors**: Character-level noise
10. **StylisticMutator**: Writing style transformation

### Crossover Operators (2)
1. **SemanticSimilarityCrossover**: Crossbreeding based on semantic distance
2. **SemanticFusionCrossover**: Hybrid prompt generation

<!-- ## Citation

If you use this framework in your research, please cite:

```bibtex
@article{evolving_prompts_toxicity_2025,
  title={Evolving Prompts for Toxicity Search in Large Language Models},
  author={...},
  journal={...},
  year={2025}
}
``` -->

## License

MIT License - See LICENSE file for details.
