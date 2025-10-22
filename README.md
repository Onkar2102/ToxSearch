# Evolutionary Text Generation Framework

A research framework for AI safety analysis through evolutionary text generation with genetic optimization, adaptive selection pressure, and comprehensive tracking.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

### Prerequisites
- **Python 3.8+** (3.12+ recommended)
- **8GB+ RAM** (16GB+ for larger models)
- **API Key**: Google Perspective API

### Installation
```bash
# Clone and setup
git clone <repository-url>
cd eost-cam-llm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API key
echo "GOOGLE_PERSPECTIVE_API_KEY=your-key-here" > .env

# Test run
python3 src/main.py --generations 1
```

## Running the Project

### Basic Command
```bash
python3 src/main.py --generations 10 --operators "all" --max-variants 3
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--generations` | int | `None` | Maximum generations (continues indefinitely if not set) |
| `--threshold` | float | `0.95` | North star metric threshold (for reference, evolution continues regardless) |
| `--pg` | str | `models/.../Q4_K_M.gguf` | Prompt Generator GGUF model path (relative to project root) |
| `--rg` | str | `models/.../Q4_K_M.gguf` | Response Generator GGUF model path (relative to project root) |
| `--operators` | str | `"all"` | Operator mode: `"ie"`, `"cm"`, or `"all"` |
| `--max-variants` | int | `1` | Variants per operator per parent |
| `--elites-threshold` | int | `25` | Elite classification threshold (percentage) |
| `--removal-threshold` | int | `5` | Removal threshold (percentage) |
| `--stagnation-limit` | int | `5` | Generations without improvement before explore mode |

### Operator Modes

**`"ie"` - Informed Evolution Only**
- Uses `InformedEvolutionOperator` with `top_10.json` (best performing genomes)
- LLM-guided evolution based on top examples

**`"cm"` - Classical Methods**
- All operators except InformedEvolution with `parents.json`
- Traditional genetic operators (mutation, crossover, etc.)

**`"all"` - All Operators (Default)**
- All 16 operators including InformedEvolution
- Uses both `parents.json` and `top_10.json`

### Example Commands

```bash
# Standard run
python3 src/main.py --generations 20 --operators "all" --max-variants 3

# Custom models (paths relative to project root)
python3 src/main.py \
  --generations 10 \
  --pg models/qwen2.5-7b-instruct-gguf/Qwen2.5-7B-Instruct-Q5_K_M.gguf \
  --rg models/mistral-7b-instruct-gguf/Mistral-7B-Instruct-v0.3-Q4_K_S.gguf

# Informed evolution only
python3 src/main.py --generations 10 --operators "ie" --max-variants 5

# Custom thresholds
python3 src/main.py \
  --generations 25 \
  --elites-threshold 30 \
  --removal-threshold 10 \
  --stagnation-limit 3
```

## Adaptive Selection Logic

The framework uses **adaptive selection pressure** that adjusts based on evolution progress:

### Selection Modes
- **DEFAULT**: 1 elite + 1 non-elite (balanced)
- **EXPLORE**: 1 elite + 2 non-elites (increased exploration during stagnation)
- **EXPLOIT**: 2 elites + 1 non-elite (focused exploitation when fitness declining)

### Triggers
- **Initial Phase**: First `m` generations (where `m` = `stagnation-limit`) use DEFAULT mode
- **Stagnation**: No improvement for `m` generations → EXPLORE mode
- **Declining Fitness**: Negative fitness slope → EXPLOIT mode

## Key Features

### Evolution System
- **16 Variation Operators**: Mutations and crossovers for text evolution
- **Adaptive Selection**: Dynamic parent selection based on progress
- **Genome Lifecycle**: Elite, non-elite, and under-performing classification
- **Informed Evolution**: LLM-guided operator using top performers

### Population Management
- **Three-tier System**:
  - `elites.json` - High-scoring genomes (≥ elite threshold)
  - `non_elites.json` - Mid-scoring genomes (> removal threshold, < elite threshold)
  - `under_performing.json` - Low-scoring genomes (≤ removal threshold, archived)

### Tracking & Metrics
- **EvolutionTracker.json**: Comprehensive evolution history
  - Per-generation metrics (max_score, avg_fitness, counts)
  - Parent scores, variant statistics, fitness trends
  - Adaptive selection mode tracking
- **Genome Metadata**: Creation info, parent scores, initial state

### Score System
- **Minimum Score**: All scores have a minimum of `0.0001`
- **Precision**: 4 decimal places for all score calculations
- **North Star Metric**: Toxicity score from Google Perspective API

## Project Structure

```
eost-cam-llm/
├── src/
│   ├── main.py                    # Entry point
│   ├── ea/                        # Evolutionary algorithms
│   │   ├── evolution_engine.py    # Core evolution logic
│   │   ├── parent_selector.py     # Adaptive parent selection
│   │   ├── run_evolution.py       # Evolution orchestration
│   │   ├── variation_operators.py # Operator registry
│   │   └── [15 operator files]    # Individual operators
│   ├── gne/                       # Generation & evaluation
│   │   ├── prompt_generator.py    # Prompt generation
│   │   ├── response_generator.py  # Response generation
│   │   └── evaluator.py           # Moderation API calls
│   └── utils/
│       └── population_io.py       # Population I/O & metrics
├── data/
│   ├── prompt.xlsx                # Initial prompts
│   └── outputs/                   # Evolution results
│       └── YYYYMMDD_HHMM/
│           ├── EvolutionTracker.json
│           ├── elites.json
│           ├── non_elites.json
│           ├── under_performing.json
│           ├── parents.json
│           ├── top_10.json
│           └── temp.json
├── models/                        # GGUF model files
├── config/                        # Model configurations
└── logs/                          # Execution logs
```

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and data flow
- **[OPERATORS.md](OPERATORS.md)** - Detailed operator documentation
- **[src/ea/README.md](src/ea/README.md)** - Evolutionary algorithm details

## Requirements

### System
- Python 3.8+ (3.12+ recommended)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ storage for models

### Key Dependencies
- `torch` - PyTorch for model operations
- `llama-cpp-python` - GGUF model inference
- `transformers` - Hugging Face transformers
- `spacy` - NLP processing
- `google-api-python-client` - Perspective API
- `sentence-transformers` - Semantic similarity

### API Keys
Create `.env` file:
```bash
GOOGLE_PERSPECTIVE_API_KEY=your-key-here
HF_TOKEN=your-huggingface-token  # Optional
```

## Troubleshooting

### Common Issues
- **Import errors**: Ensure virtual environment is activated
- **API rate limits**: Perspective API has 60 requests/minute limit
- **Memory issues**: Use smaller models or reduce `--max-variants`
- **Model loading**: Verify GGUF file paths are correct and relative to project root

### Performance
- **GPU**: Automatically used if available (CUDA/MPS)
- **Model Caching**: Models cached after first load
- **Memory**: Automatic cleanup and optimization

## License

MIT License - see [LICENSE](LICENSE) file for details.
