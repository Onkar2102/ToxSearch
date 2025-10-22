# Evolutionary Text Generation Framework

A research framework for AI safety analysis through evolutionary text generation with genetic optimization, adaptive selection pressure, and comprehensive tracking.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

### Prerequisites

**Required:**
- **Python 3.8+** (3.12+ recommended)
- **8GB+ RAM** (16GB+ recommended for larger models)
- **10GB+ Disk Space** (for models)
- **API Key**: Google Perspective API (required)

**Optional (for GPU acceleration):**
- **NVIDIA GPU** with 6GB+ VRAM (8GB+ recommended)
- **CUDA Toolkit 11.8+** (12.1+ recommended)
- **Apple Silicon** (M1/M2/M3) with macOS 12.3+

### Installation

#### CPU Installation (Default)
```bash
# Clone and setup
git clone <repository-url>
cd eost-cam-llm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API key
echo "PERSPECTIVE_API_KEY=your-key-here" > .env

# Test run
python3 src/main.py --generations 1
```

#### GPU Installation (NVIDIA CUDA)
For 10-50x faster inference on NVIDIA GPUs:

```bash
# 1. Prerequisites: Install NVIDIA CUDA Toolkit
# Download from: https://developer.nvidia.com/cuda-downloads
# Verify installation: nvcc --version

# 2. Clone and setup
git clone <repository-url>
cd eost-cam-llm
python3 -m venv venv
source venv/bin/activate

# 3. Install CPU dependencies first
pip install -r requirements.txt

# 4. Uninstall CPU versions of torch and llama-cpp-python
pip uninstall torch torchvision torchaudio llama-cpp-python -y

# 5. Install PyTorch with CUDA support (choose your CUDA version)
# For CUDA 12.1:
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu121
# For CUDA 11.8:
# pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu118

# 6. Install llama-cpp-python with CUDA support (takes 5-10 minutes)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python==0.3.16 --force-reinstall --no-cache-dir

# 7. Configure API key
echo "PERSPECTIVE_API_KEY=your-key-here" > .env

# 8. Verify GPU detection
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 9. Test run - will automatically use GPU
python3 src/main.py --generations 1
```

**GPU Installation Script** (automated):
```bash
# Use the included setup_gpu.sh script
bash setup_gpu.sh

# This script will:
# - Detect your CUDA version
# - Uninstall CPU packages
# - Install PyTorch with CUDA
# - Compile llama-cpp-python with CUDA
# - Verify GPU detection
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
│   ├── prompt.csv                 # Initial prompts (CSV format)
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

### System Requirements

**Minimum (CPU):**
- Python 3.8+ (3.12+ recommended)
- 8GB RAM
- 10GB disk space
- CPU with AVX2 support

**Recommended (GPU):**
- Python 3.12+
- 16GB+ RAM
- 20GB+ disk space (for multiple models)
- NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- CUDA 12.1+ installed
- Or Apple Silicon M1/M2/M3 with 16GB+ unified memory

**Tested Configurations:**
| Hardware | Model Size | Performance | Notes |
|----------|-----------|-------------|-------|
| CPU (Intel i7) | 3B Q4 | ~5 tok/s | Usable for testing |
| RTX 3090 | 7B Q5 | ~70 tok/s | Recommended |
| RTX 4090 | 7B Q6 | ~120 tok/s | Excellent |
| M1 Max | 7B Q4 | ~20 tok/s | Good |
| M2 Ultra | 7B Q6 | ~50 tok/s | Very good |

### Key Dependencies
- `torch` - PyTorch for model operations (CPU or CUDA versions)
- `llama-cpp-python` - GGUF model inference (CPU or CUDA versions)
- `transformers` - Hugging Face transformers
- `spacy` - NLP processing (includes en_core_web_sm model)
- `google-api-python-client` - Google Perspective API client
- `sentence-transformers` - Semantic similarity and embeddings
- `pandas` - Data processing and analysis
- `pytest` - Testing framework

See `requirements.txt` for complete dependency list (organized by usage).

### API Keys
Create `.env` file in project root:
```bash
# Required: Google Perspective API for toxicity evaluation
PERSPECTIVE_API_KEY=your-perspective-api-key-here

# Optional: Hugging Face token for downloading gated models
HF_TOKEN=your-huggingface-token

# Optional: Control logging verbosity (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
```

**Get API Keys:**
- **Perspective API**: [Get API Key](https://developers.perspectiveapi.com/s/docs-get-started)
- **Hugging Face**: [Generate Token](https://huggingface.co/settings/tokens)

## GPU Configuration

### Automatic GPU Detection
The project automatically detects and uses available GPUs:
- **NVIDIA GPUs**: CUDA support (best performance)
- **Apple Silicon**: MPS (Metal Performance Shaders) support
- **CPU Fallback**: Automatically falls back if GPU unavailable

### Configuration Files
GPU settings are in `config/PGConfig.yaml` and `config/RGConfig.yaml`:

```yaml
device_config:
  auto_detect: true              # Automatically detect best device
  preferred_device: null         # Force device: 'cuda', 'mps', or 'cpu'
  fallback_to_cpu: true          # Fallback to CPU if GPU fails
  
  cuda:
    enable_cudnn_benchmark: true # Optimize for NVIDIA GPUs
    enable_tf32: true            # Use TensorFloat-32 (Ampere+ GPUs)
    memory_fraction: 0.8         # Use 80% of GPU memory
  
  mps:
    enable_metal_performance_shaders: true
    memory_pressure_relief: true
```

### GPU Performance Tips

**Model Selection:**
- Use larger quantizations (Q4, Q5, Q6) on GPU for best speed/quality
- Q2/Q3 quantizations are better for CPU
- GPU can handle 7B Q5 models efficiently

**Memory Management:**
- Adjust `memory_fraction` in config if you get OOM errors
- Monitor GPU memory: `nvidia-smi` (CUDA) or Activity Monitor (macOS)
- Reduce `max_memory_usage_gb` in config files if needed

**Optimization:**
- `enable_tf32: true` - Faster on RTX 30xx/40xx (Ampere+)
- `enable_cudnn_benchmark: true` - Better for fixed input sizes
- Models are cached after first load

### Verify GPU Usage
```bash
# Check GPU detection
python3 -c "from utils.device_utils import get_device_info; import json; print(json.dumps(get_device_info(), indent=2))"

# Monitor GPU during execution
# NVIDIA:
watch -n 1 nvidia-smi

# macOS:
sudo powermetrics --samplers gpu_power -i 1000
```

### Expected Performance
| Device | Model Size | Inference Speed | Memory |
|--------|-----------|----------------|---------|
| **CPU** | 7B Q4 | 2-5 tokens/s | 6-8 GB |
| **RTX 3090** | 7B Q4 | 50-80 tokens/s | 8-10 GB |
| **RTX 4090** | 7B Q6 | 100-150 tokens/s | 10-12 GB |
| **M1 Max** | 7B Q4 | 15-25 tokens/s | 8-10 GB |
| **M2 Ultra** | 7B Q6 | 40-60 tokens/s | 10-12 GB |

## Troubleshooting

### Common Issues

**Import errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
# Verify Python version
python3 --version  # Should be 3.8+
```

**API rate limits**
- Perspective API: 60 requests/minute limit
- Solution: Code includes automatic retry with exponential backoff
- Monitor: Check logs for rate limit warnings

**Memory issues**
```bash
# Use smaller models
--pg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Reduce variants per generation
--max-variants 1

# Reduce memory limits in config files
max_memory_usage_gb: 8.0
```

**Model loading errors**
```bash
# Verify model paths are relative to project root
ls -lh models/llama3.1-8b-instruct-gguf/

# Check model file exists
python3 -c "import os; print(os.path.exists('models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'))"
```

**GPU not detected**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify PyTorch CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check llama-cpp-python CUDA
python3 -c "from llama_cpp import Llama; print('llama-cpp-python supports GPU')"

# If not working, reinstall with CUDA support
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python==0.3.16 --force-reinstall --no-cache-dir
```

**Out of Memory (OOM) errors**
```bash
# GPU OOM: Reduce memory_fraction in config
# Edit config/RGConfig.yaml:
cuda:
  memory_fraction: 0.6  # Reduce from 0.8

# Or switch to smaller model
--rg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

### Performance Optimization

**GPU Optimization:**
- Use CUDA 12.1+ for best performance
- Enable TF32 for Ampere+ GPUs (RTX 30xx/40xx)
- Monitor GPU utilization with `nvidia-smi`
- Batch processing automatically optimized

**Model Caching:**
- Models cached after first load (check logs)
- Cache persists across runs
- Clear cache: restart Python process

**Memory Management:**
- Automatic cleanup enabled by default
- Monitor with: `ps aux | grep python`
- Manual cleanup: `del model; import gc; gc.collect()`

## License

MIT License - see [LICENSE](LICENSE) file for details.
