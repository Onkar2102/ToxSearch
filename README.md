# Evolutionary Text Generation Framework

A research framework for AI safety analysis through evolutionary text generation, moderation evaluation, and genetic optimization with **automatic process monitoring and recovery**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Quick Start](#quick-start)
- [app.py Command Line Arguments](#appy-command-line-arguments)
- [Safety Features](#safety-features)
- [Documentation](#documentation)
  - [Architecture Overview](ARCHITECTURE.md)
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

## Documentation

### **[Architecture Overview](ARCHITECTURE.md)**
Comprehensive system architecture, component interactions, and data flow diagrams.

### **[Evolutionary Algorithms](src/ea/README.md)**
Complete guide to genetic algorithms, variation operators, and evolution strategies.

### **Generation & Evaluation** (`src/gne/`)
- `LLaMaTextGenerator.py` - LLaMA model integration with memory management
- `openai_moderation.py` - OpenAI moderation API for safety evaluation

### **Utilities** (`src/utils/`)
- `population_io.py` - Population data management with EvolutionTracker
- `custom_logging.py` - Performance and memory logging
- `m3_optimizer.py` - M3 Mac optimization utilities
- `config.py` - Configuration management


## License

MIT License - see LICENSE file for details.