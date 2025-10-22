# Evolutionary Text Generation Framework

A research framework for AI safety analysis through evolutionary text generation, moderation evaluation, and genetic optimization with **automatic process monitoring and recovery**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Quick Setup](#quick-setup)
- [Installation](#installation)
- [API Keys Configuration](#api-keys-configuration)
- [Running the Project](#running-the-project)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Documentation Index](#documentation-index)
- [License](#license)

## Quick Setup

### Prerequisites
- **Python 3.8+** (Python 3.12+ recommended)
- **macOS/Linux** (Windows support via WSL)
- **8GB+ RAM** (16GB+ recommended for larger models)
- **API Keys**: Google Perspective API key

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd eost-cam-llm
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create environment file**
   Create a `.env` file in the project root:
   ```bash
   # Required API keys
   GOOGLE_PERSPECTIVE_API_KEY=your-google-perspective-api-key-here
   
   # Optional (for Hugging Face models)
   HF_TOKEN=your-huggingface-token-here
   ```

5. **Device Setup (Optional)**
   The framework supports multiple device types:
   
   **For Apple Silicon (MPS):**
   ```bash
   # PyTorch with MPS support is automatically detected
   # No additional setup required
   ```
   
   **For NVIDIA GPU (CUDA):**
   ```bash
   # Install CUDA-enabled PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   
   **For CPU-only:**
   ```bash
   # CPU-only PyTorch (default)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

6. **Verify installation**
   ```bash
   # Test with a single generation
   python3 src/main.py --generations 1 --threshold 0.5
   ```

## Installation

### Automated Installation
Use your preferred environment manager, then install from `requirements.txt`.
```bash
pip install -r requirements.txt
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Create .env file manually
echo "GOOGLE_PERSPECTIVE_API_KEY=your-api-key-here" > .env
```

## API Keys Configuration

### Required API Keys

1. **Google Perspective API Key**
   - Get from: https://developers.perspectiveapi.com/
   - Used for: Content moderation and toxicity evaluation

### Environment Variables
Create a `.env` file in the project root:
```bash
# Required API keys
GOOGLE_PERSPECTIVE_API_KEY=your-google-perspective-api-key-here

# Optional (for Hugging Face models)
HF_TOKEN=your-huggingface-token-here
```

## Running the Project

### Basic Usage
```bash
# Run the pipeline
python3 src/main.py --generations 25
```

### Command Line Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--generations` | int | `None` | Maximum number of evolution generations. If not set, runs until north star metric is achieved |
| `--threshold` | float | `0.95` | North star metric threshold for stopping evolution |
| `--pg` | str | `models/<dir>/<file>.gguf` | Prompt Generator model. Pass the relative path to the `.gguf` file under the `models/` directory |
| `--rg` | str | `models/<dir>/<file>.gguf` | Response Generator model. Pass the relative path to the `.gguf` file under the `models/` directory |
| `--operators` | str | `"all"` | Operator configuration mode: `ie` (InformedEvolution only), `cm` (all except InformedEvolution), `all` (all operators) |
| `--max-variants` | int | `3` | Maximum number of variants to generate per operator |
| `--elites-threshold` | int | `25` | Elite threshold percentage for classifying genomes as elites |
| `--removal-threshold` | int | `5` | Removal threshold percentage for worst performing genomes |
| `--stagnation-limit` | int | `5` | Number of generations without improvement before switching to explore mode |

### Example Commands
```bash
# Quick test run
python3 src/main.py --generations 1 --threshold 0.5

# Full evolution run
python3 src/main.py --generations 50 --threshold 0.99

# Run with specific operators
python3 src/main.py --generations 10 --operators "ie" --max-variants 5

# Run with all operators except InformedEvolution
python3 src/main.py --generations 10 --operators "cm" --max-variants 3

# Run with all operators (default)
python3 src/main.py --generations 10 --operators "all" --max-variants 3

# Run with custom thresholds
python3 src/main.py --generations 20 --elites-threshold 30 --removal-threshold 10

# Run with adaptive selection (stagnation limit)
python3 src/main.py --generations 25 --stagnation-limit 3

# Run with direct GGUF files (always pass relative path under `models/`)
python3 src/main.py \
   --pg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
   --rg models/qwen2.5-7b-instruct-gguf/Qwen2.5-7B-Instruct-Q4_K_S.gguf

# Advanced run with all custom parameters (pass model paths under `models/`)
python3 src/main.py \
   --generations 30 \
   --operators "all" \
   --max-variants 4 \
   --elites-threshold 20 \
   --removal-threshold 8 \
   --stagnation-limit 4 \
   --pg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
   --rg models/qwen2.5-7b-instruct-gguf/Qwen2.5-7B-Instruct-Q4_K_S.gguf

### Model path policy
Always pass the relative path to the `.gguf` model file under the `models/` directory for both the prompt generator (`--pg`) and response generator (`--rg`).

Examples (correct):

```bash
# Prompt Generator
--pg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Response Generator
--rg models/qwen2.5-7b-instruct-gguf/Qwen2.5-7B-Instruct-Q4_K_S.gguf
```

The code no longer expects or resolves model "aliases"; providing a direct relative path is required and avoids ambiguous variant selection logic.

### Operator Modes

The `--operators` parameter controls which variation operators are used:

#### **`ie` (Informed Evolution Only)**
- **Operators**: Only `InformedEvolutionOperator`
- **Data Source**: Uses `top_10.json` (top performing genomes)
- **Purpose**: Focus on LLM-guided evolution using best examples
- **Use Case**: When you want to leverage the best performing genomes to guide evolution

#### **`cm` (Classical Methods)**
- **Operators**: All operators except `InformedEvolutionOperator`
- **Data Source**: Uses `parents.json` (selected parents)
- **Purpose**: Traditional genetic algorithm operators (mutation, crossover)
- **Use Case**: When you want to avoid LLM-guided evolution and use classical methods

#### **`all` (All Operators)**
- **Operators**: All 16 variation operators including `InformedEvolutionOperator`
- **Data Source**: Uses both `parents.json` and `top_10.json`
- **Purpose**: Maximum diversity and exploration
- **Use Case**: Default mode for comprehensive evolution

### Adaptive Selection Logic

The framework now includes **adaptive selection pressure** that dynamically adjusts parent selection based on evolution progress:

#### **Selection Modes:**
- **DEFAULT**: 1 elite + 1 non-elite (balanced exploration/exploitation)
- **EXPLORE**: 1 elite + 2 non-elites (increased exploration when stuck)
- **EXPLOIT**: 2 elites + 1 non-elite (focused exploitation when fitness declining)

#### **Adaptive Triggers:**
- **Initial Generations**: First `m` generations (where `m` = `--stagnation-limit`) always use DEFAULT mode
- **Stagnation**: After `m` generations without improvement → EXPLORE mode
- **Declining Fitness**: When fitness slope < 0 → EXPLOIT mode

#### **Configuration:**
```bash
# Customize stagnation limit (default: 5 generations)
python3 src/main.py --stagnation-limit 3

# Customize elite threshold (default: 25%)
python3 src/main.py --elites-threshold 30

# Customize removal threshold (default: 5%)
python3 src/main.py --removal-threshold 10
```

## Requirements

### System Requirements
- **Python**: 3.8+ (3.12+ recommended)
- **RAM**: 8GB+ (16GB+ recommended for larger models)
- **Storage**: 10GB+ for models and data
- **OS**: macOS/Linux (Windows via WSL)

### Python Dependencies
See `requirements.txt` for complete list. Key dependencies:
- `torch` - PyTorch for model operations
- `transformers` - Hugging Face transformers
- `spacy` - Natural language processing
- `google-api-python-client` - Google Perspective API
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `sentence-transformers` - Semantic similarity
- `nltk` - Natural language toolkit

### Model Requirements
- **Prompt Generator**: Qwen2.5-7B-Instruct (default)
- **Response Generator**: Llama3.2-3B-Instruct (default)
- **spaCy Model**: `en_core_web_sm`
- **NLTK Data**: `punkt` tokenizer

## Troubleshooting

### Common Issues

**Import errors**
- Ensure virtual environment is activated: `source venv/bin/activate`
- Check Python version: `python3 --version`

**API rate limits**
- Google Perspective API: 60 requests/minute limit
- Added 0.75-second delay between evaluations to prevent rate limiting

**Memory issues**
- Use smaller models or reduce batch sizes
- Monitor memory usage in logs
- Enable garbage collection

**Model loading errors**
- Check model files exist in `models/` directory
- Verify model paths in config files
- Ensure sufficient disk space

**GPU usage**
- Models run on GPU by default (`n_gpu_layers: -1`)
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Getting Help
- Check logs in `logs/` directory
- Review [Architecture Overview](ARCHITECTURE.md)
- See [Evolutionary Algorithms Guide](src/ea/README.md)
- Check [Tests README](tests/README.md) for testing

### Performance Optimization
- **GPU Acceleration**: Enabled by default for both PG and RG models
- **Memory Management**: Automatic cleanup and garbage collection
- **Model Caching**: Efficient reuse of loaded models
- **Parallel Processing**: Available for operator execution

## Documentation Index

### 📚 **Core Documentation**
- **[Architecture Overview](ARCHITECTURE.md)** - Complete system architecture and component interactions
- **[Evolutionary Algorithms Guide](src/ea/README.md)** - Genetic algorithms, variation operators, and evolution strategies
- **[EA Notes](src/ea/notes.md)** - Detailed implementation notes and data flow

### 📖 **Additional Documentation**
- **[LLM POS-Aware Synonym Replacement](docs/LLM_POSAwareSynonymReplacement.md)** - Detailed guide for POS-aware operations
- **[vLLM Migration Guide](docs/vLLM_Migration_Guide.md)** - Migration guide for vLLM integration
- **[LLM POS Test Updates](docs/README_llm_pos_test_updates.md)** - Updates and testing information

### 🧪 **Testing Documentation**
- **[Tests README](tests/README.md)** - Testing framework and test execution guide

### 🔧 **Configuration Files**
- **[RGConfig.yaml](config/RGConfig.yaml)** - Response Generator configuration
- **[PGConfig.yaml](config/PGConfig.yaml)** - Prompt Generator configuration
- **[requirements.txt](requirements.txt)** - Python dependencies

### 📊 **Data Files**
- **[prompt.xlsx](data/prompt.xlsx)** - Input prompts for evolution
- **[outputs/](data/outputs/)** - Evolution results and tracking data
- **[models/](models/)** - Local model files and configurations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.