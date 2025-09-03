# Evolutionary Text Generation Framework

A research framework for AI safety analysis through evolutionary text generation, moderation evaluation, and genetic optimization with **automatic process monitoring and recovery**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Quick Start](#quick-start)
- [Recent Updates](#recent-updates)
- [app.py Command Line Arguments](#appy-command-line-arguments)
- [Safety Features](#safety-features)
- [Documentation](#documentation)
  - [Architecture Overview](ARCHITECTURE.md)
  - [Design Document](design_document.md)
  - [Evolutionary Algorithms](src/ea/README.md)
  - [Generation & Evaluation](#generation--evaluation)
  - [Utilities](#utilities)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Monitoring & Recovery](#monitoring--recovery)
- [Output Structure](#output-structure)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

```bash
# Run full environment setup (RECOMMENDED)
python app.py --setup

# Run with interactive setup and monitoring (RECOMMENDED)
python app.py --interactive

# Run directly with process monitoring
python app.py --generations 25

# Run core pipeline directly
python src/main.py --generations 25
```

### **Monitoring Features**
- **Timeout Protection**: 5-minute timeout for model loading
- **CPU Fallback**: Automatic fallback to CPU if MPS hangs
- **Memory Management**: Real-time memory monitoring and cleanup
- **Stuck Process Detection**: Identifies and restarts stuck processes
- **Graceful Recovery**: Saves state before restarting

## Recent Updates

### **Latest Fixes (Current Architecture)**
- **Single Population File**: Consolidated all generations into a single `outputs/Population.json` (deprecated split file mode)
- **Accurate `max_score`**: Tracker now records the actual best score per generation, not the parent's score
- **Hybrid Moderation Cache Fix**: Separated Google/OpenAI caches to prevent cross-API result contamination
- **Variant Count Tuning**: Reduced operator `max_variants` from 10 to 5 to control population growth
- **Memory & Imports**: Lazy imports (torch/pandas/psutil) and safer path resolution

### **Key Improvements**
- **Population Management**: Single `Population.json` with `population_index.json` for fast lookups
- **Evolution Tracking**: `EvolutionTracker.json` records per-generation best and variant counts
- **Hybrid Moderation**: Google Perspective API + OpenAI moderation with normalized scores
- **M3 Mac Optimization**: Specialized optimizations for Apple Silicon performance

## app.py Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--generations` | int | `None` | Maximum number of evolution generations. If not set, runs until north star metric is achieved |
| `--threshold` | float | `0.95` | North star metric threshold for stopping evolution |
| `model_names` | list | `[]` | Model names to use (currently not used) |
| `--check-interval` | int | `1800` | Health check interval in seconds (30 minutes) |
| `--stuck-threshold` | int | `7200` | Stuck detection threshold in seconds (2 hours) |
| `--memory-threshold` | float | `20.0` | Memory threshold in GB |
| `--max-restarts` | int | `5` | Maximum restart attempts |
| `--interactive` | flag | `False` | Run in interactive mode with setup and monitoring |
| `--setup` | flag | `False` | Run full environment setup (install requirements, optimize config) |
| `--no-monitor` | flag | `False` | Run without process monitoring |

## Safety Features

- **Automatic Restart**: Detects and recovers from stuck processes
- **Memory Protection**: Prevents out-of-memory crashes
- **Timeout Protection**: Prevents infinite hanging
- **State Preservation**: Saves progress before restarting
- **Comprehensive Logging**: Detailed monitoring and debugging information
- **Hybrid Moderation**: Dual API approach for robust safety evaluation

## Documentation

### **[Architecture Overview](ARCHITECTURE.md)**
Comprehensive system architecture, component interactions, and data flow diagrams.

### **[Design Document](design_document.md)**
Detailed, professional design specification: goals, data models, algorithms, operations.

### **[Evolutionary Algorithms](src/ea/README.md)**
Complete guide to genetic algorithms, variation operators, and evolution strategies.

### **Generation & Evaluation** (`src/gne/`)
- `LLaMaTextGenerator.py` - LLaMA model integration with memory management
- `hybrid_moderation.py` - Hybrid moderation using Google Perspective API + OpenAI

### **Utilities** (`src/utils/`)
- `population_io.py` - Single-file population management (`Population.json`) and `EvolutionTracker.json`
- `custom_logging.py` - Performance and memory logging
- `m3_optimizer.py` - M3 Mac optimization utilities
- `config.py` - Configuration management

## Usage Examples

### **Basic Evolution Run**
```bash
# Run evolution until threshold is reached
python src/main.py --threshold 0.99

# Run for specific number of generations
python src/main.py --generations 10
```

### **Population Management**
```bash
# Initialize population from prompt.xlsx
python -c "from src.utils.population_io import load_and_initialize_population; load_and_initialize_population('data/prompt.xlsx', 'outputs')"

# Load specific generation
python -c "from src.utils.population_io import load_population_generation; genomes = load_population_generation(1, 'outputs')"
```

## Configuration

The system uses `config/modelConfig.yaml` for:
- Model parameters (LLaMA 3.2 3B Instruct)
- Memory optimization settings
- Batch processing configuration
- API keys and endpoints

## Monitoring & Recovery

- **Real-time Monitoring**: Memory usage, process health, execution time
- **Automatic Recovery**: Detects stuck processes and restarts automatically
- **Progress Tracking**: EvolutionTracker.json for comprehensive progress monitoring
- **Performance Logging**: Detailed timing and resource usage metrics

## Output Structure

```
outputs/YYYY-MM-DD/
├── Population.json        # All genomes across all generations (single file)
├── population_index.json  # Lightweight index/metadata for Population.json
├── EvolutionTracker.json  # Evolution progress tracking (global, per generation)
└── final_statistics.json  # Final analysis results (optional)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.