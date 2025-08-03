# Evolutionary Text Generation Framework

A research framework for AI safety analysis through evolutionary text generation, moderation evaluation, and genetic optimization with **automatic process monitoring and recovery**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## Process Monitoring & Recovery

The framework includes **automatic process monitoring** to handle common issues:

### **Built-in Monitoring** (in main.py)
- **Runtime monitoring**: Maximum 30 minutes per generation
- **Memory monitoring**: Automatic cleanup when usage exceeds 20GB
- **CPU monitoring**: Detects stuck processes (< 1% CPU for 5+ minutes)
- **Automatic restart**: Self-healing with up to 5 restart attempts

### **External Process Monitor** (app.py)
```bash
# Basic usage
python app.py --generations 25

# Custom monitoring parameters
python app.py \
  --generations 10 \
  --max-runtime 900 \
  --check-interval 30 \
  --memory-threshold 15.0 \
  --max-restarts 3
```

### **Monitoring Features**
- **Timeout Protection**: 5-minute timeout for model loading
- **CPU Fallback**: Automatic fallback to CPU if MPS hangs
- **Memory Management**: Real-time memory monitoring and cleanup
- **Stuck Process Detection**: Identifies and restarts stuck processes
- **Graceful Recovery**: Saves state before restarting

## Configuration

### Model Configuration (`config/modelConfig.yaml`)
```yaml
llama:
  name: meta-llama/Llama-3.2-3B-instruct
  # Alternative smaller model for faster testing
  # name: microsoft/DialoGPT-medium
  
  # Memory management settings
  enable_memory_cleanup: true
  max_memory_usage_gb: 12.0
  adaptive_batch_sizing: true
  
  # Model loading settings
  model_loading_timeout: 300  # 5 minutes timeout
  enable_cpu_fallback: true   # Fallback to CPU if GPU fails
  enable_model_caching: true  # Cache loaded models
```

### Process Monitor Configuration
```bash
# Monitor with custom settings
python app.py \
  --generations 25 \
  --max-runtime 1800 \
  --check-interval 60 \
  --memory-threshold 20.0 \
  --max-restarts 5
```

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


## License

MIT License - see LICENSE file for details.