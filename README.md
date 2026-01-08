# Evolving Prompts for Toxicity Search in Large Language Models

This repository implements a black-box evolutionary framework for systematically testing Large Language Model (LLM) safety through LLM-guided prompt evolution to elicit toxic responses.

## Current Approach Highlights
- **Distance metric**: Cosine distance on **L2-normalized 384D** embeddings; clusters are angular (cone-shaped), not Euclidean spheres.
- **Speciation options**:
  - **Leader-Follower (online, fast)**: Directly clusters 384D embeddings with cosine thresholds (θ_sim≈0.3–0.5); no training required.
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
| `--rg` | str | `models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf` | Response generator model path (.gguf file or alias) |
| `--pg` | str | `models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf` | Prompt generator model path (.gguf file or alias) |
| `--operators` | str | `"all"` | Operator mode: `ie` (InformedEvolution only), `cm` (all except IE), `all` (all operators) |
| `--max-variants` | int | 1 | Maximum variants to generate per evolution cycle |

**Key Arguments Explained:**

- **`--operators`**: Controls which variation operators are used
  - `ie`: Only InformedEvolution (LLM-guided evolution using top performers)
  - `cm`: All classical operators (synonym replacement, back-translation, etc.) except InformedEvolution
  - `all`: All operators including both classical and LLM-driven
  
- **`--elites-threshold`**: Percentage of top-scoring genomes considered "elites" (used for parent selection)

- **`--removal-threshold`**: Percentage of worst-performing genomes to archive to `under_performing.json`

- **`--stagnation-limit`**: When fitness hasn't improved for N generations, switch from exploit mode to explore mode to search broader solution space

- **Model arguments (`--rg`, `--pg`)**: Can specify either a direct `.gguf` file path or an alias (directory name under `models/`)


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