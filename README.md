# Evolving Prompts for Toxicity Search in Large Language Models

This repository implements a black-box evolutionary framework for systematically testing Large Language Model (LLM) safety through LLM-guided prompt evolution to elicit toxic responses.

## Abstract

Large Language Models (LLMs) have demonstrated impressive capabilities, but they remain vulnerable to adversarial prompts that elicit toxic or harmful outputs. This paper presents a black-box evolutionary framework for systematically testing LLM safety by LLM-guided prompt evolution to elicit toxic responses from target LLM. Using a synchronous, steady-state evolutionary loop, we generate candidate prompts through a diverse set of mutation and crossover operators, including POS-aware phrase-level substitutions, informed paraphrasing, back-translation, concept injection, semantic similarity and fusion crossover. A moderation oracle serves as the fitness function guiding the search. Our results show that maintaining a diverse population of prompts uncovers a broader range of toxic behaviors, with LLM-driven operators achieving the highest toxicity and lexical diversity. These findings underscore the importance of adversarial evolution in evaluating and hardening LLM safety, and highlight transferable risks across models with different architectures.

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
# create .env from the example environment file

# Edit .env and add your Google Perspective API key
nano .env  # or use your preferred text editor
```

Add your API key to `.env`:
```env
PERSPECTIVE_API_KEY=your_actual_api_key_here
```

#### 5. Dataset Setup

**Option A: Use Included Dataset (Recommended for Quick Start)**
The project includes `data/prompt.csv` with 101 initial prompts as seed population. This file is automatically loaded during first run.

**Option B: Download Fresh Dataset from HuggingFace**
```bash
# Download harmful prompt datasets from HuggingFace
# This will create/update data/prompt.csv and data/prompt_extended.csv
python src/utils/data_loader.py
```
**Note**: For custom experiments, you can replace `data/prompt.csv` with your own prompts (one per line, CSV format with 'questions' column).

#### 6. Download Models

**Option A: Use Existing Models (Recommended)**
```bash
# Clone or copy GGUF models to models/ directory
# Models are available from HuggingFace or Ollama
# Example: Llama 3.2 3B Instruct
# Create directory: models/llama3.2-3b-instruct-gguf/
# Download .gguf files into that directory
```

**Option B: Download from HuggingFace**
If downloading gated models, add your HF token to `.env`:
```env
HF_TOKEN=your_huggingface_token_here
```

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

### Pre-Run Checklist

Before running, ensure you have completed these steps **in order**:

**Step 1: Dataset Setup**
```bash
# Option A: Use included dataset (already in data/prompt.csv)
# Option B: Download fresh dataset from HuggingFace:
python src/utils/data_loader.py

# This downloads from CategoricalHarmfulQA and HarmfulQA datasets
# Creates data/prompt.csv (100 prompts) and data/prompt_extended.csv (full dataset)
```

**Step 2: Download Models**
```bash
# Download GGUF models to models/ directory
# Recommended: Llama-3.2-3B-Instruct (Q4_K_M quantization, ~2GB)

# Option A: Manual download from HuggingFace
# Go to: https://huggingface.co/Unsloth/Llama-3.2-3B-Instruct-GGUF
# Download: Llama-3.2-3B-Instruct.Q4_K_M.gguf
# Place in: models/llama3.2-3b-instruct-gguf/

# Option B: Use download script (if available)
python src/utils/download_models.py
```

**Step 3: Environment Variables**
```bash
# Already configured in .env file
# Contains: PERSPECTIVE_API_KEY
```

### Quick Start Execution

Once all prerequisites are ready:

```bash
# Run with default settings (uses Llama 3.2 3B, all operators)
python src/main.py
```

### Custom Configuration
```bash
# Run with custom models and parameters
python src/main.py \
    --generations 50 \
    --operators all \
    --max-variants 1 \
    --elites-threshold 25 \
    --removal-threshold 5 \
    --stagnation-limit 5 \
    --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_S.gguf \
    --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf
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

### Results Output

Results will be saved in `data/outputs/[timestamp]/` with:
- `EvolutionTracker.json` - Generation-by-generation progress
- `elites.json` - High-performing prompts
- `non_elites.json` - Mid-performing prompts
- `under_performing.json` - Archived prompts

## System Overview

## Project Structure

```
eost-cam-llm/
├── src/
│   ├── main.py                    # Entry point
│   ├── ea/                        # Evolutionary algorithms
│   │   ├── evolution_engine.py    # Core evolution logic
│   │   ├── parent_selector.py     # Adaptive parent selection
│   │   └── [12 operator files]    # Variation operators
│   ├── gne/                       # Generation & evaluation
│   │   ├── prompt_generator.py    # Prompt generation
│   │   ├── response_generator.py  # Response generation
│   │   └── evaluator.py           # Moderation API calls
│   └── utils/
│       └── population_io.py       # Population I/O & metrics
├── config/                        # Model configurations
├── data/                          # Input data and results
└── experiments/                   # Analysis notebooks
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{evolving_prompts_toxicity_2025,
  title={Evolving Prompts for Toxicity Search in Large Language Models},
  author={...},
  journal={...},
  year={2025}
}
```

## Contributing

This is a research framework. Contributions should focus on:
- Improving operator diversity and effectiveness
- Enhancing population management strategies
- Extending to additional LLM architectures
- Advancing safety evaluation metrics

## License

MIT License - See LICENSE file for details.