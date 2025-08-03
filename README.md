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

## Project Architecture

### Core Pipeline
```
Input Prompts → Text Generation → Safety Evaluation → Evolution → New Variants
```

### System Components

**Main Entry Point**
- `app.py` - Unified entry point with setup, monitoring, and execution

**Generation (`src/gne/`)**
- `LLaMaTextGenerator.py` - LLaMA model integration with enhanced memory management and timeout protection
- `openai_moderation.py` - OpenAI moderation API for safety evaluation

**Evolution (`src/ea/`)**
- `EvolutionEngine.py` - Genetic algorithm orchestration
- `TextVariationOperators.py` - Mutation and crossover operators with timeout protection
- `ParentSelector.py` - Fitness-based parent selection
- `RunEvolution.py` - Evolution pipeline execution

**Utilities (`src/utils/`)**
- `population_io.py` - Population data management with EvolutionTracker initialization
- `custom_logging.py` - Performance and memory logging
- `m3_optimizer.py` - M3 Mac optimization utilities

**Process Monitoring**
- `app.py` - External process monitor with automatic restart capabilities
- Built-in health monitoring in `main.py`

**Configuration**
- `config/modelConfig.yaml` - Model and memory settings with timeout configurations
- `data/prompt.xlsx` - Input prompts for evolution

### Memory Management
- **Adaptive batch sizing** based on available memory
- **Automatic memory cleanup** after each generation
- **Real-time monitoring** with configurable thresholds
- **Out-of-memory error handling** with graceful recovery
- **Timeout protection** for model loading operations

### Data Flow
1. **Setup & Initialization** - Environment setup, population creation, EvolutionTracker initialization
2. **Population Initialization** - Load prompts from Excel, create Generation 0
3. **Text Generation** - LLaMA model with memory optimization and timeout protection
4. **Safety Evaluation** - OpenAI moderation scoring
5. **Evolution** - Genetic algorithms create new variants
6. **Iteration** - Repeat until stopping conditions

### Output Structure
- `outputs/Population.json` - Main population data
- `outputs/EvolutionTracker.json` - Evolution tracking and metadata
- `outputs/population_index.json` - Population file metadata
- `outputs/gen*.json` - Generation files
- `logs/` - Detailed execution logs with health monitoring

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

## Troubleshooting

### Common Issues & Solutions

**1. Process Stuck at "Device set to use mps:0"**
- **Fixed**: Added 5-minute timeout for model loading
- **Fixed**: Automatic CPU fallback if MPS hangs
- **Fixed**: Enhanced error handling and logging

**2. Memory Issues**
- **Fixed**: Real-time memory monitoring
- **Fixed**: Automatic memory cleanup
- **Fixed**: Configurable memory thresholds

**3. Process Hanging**
- **Fixed**: Built-in health monitoring
- **Fixed**: External process monitor (app.py)
- **Fixed**: Automatic restart with state preservation

### Monitoring Commands

```bash
# Check current process status
ps aux | grep python | grep -v grep

# Monitor logs in real-time
tail -f logs/$(ls -t logs/ | head -1)

# Check memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

## Performance Optimization

### M3 Mac Optimizations
- **MPS Backend**: Uses Metal Performance Shaders for GPU acceleration
- **Memory Management**: Adaptive batch sizing based on available memory
- **Model Caching**: Prevents repeated model loading
- **Timeout Protection**: Prevents infinite hanging during model operations

### Memory Usage Monitoring
The system provides real-time memory monitoring:
```
[INFO] Starting batch generation. Initial memory: 6.04GB
[INFO] Good memory availability: 50.3% used
[INFO] Memory cleanup completed. Current usage: 6.07GB
[INFO] Memory reduction: 0.00GB
```

## Usage Examples

### Environment Setup
```bash
python app.py --setup
```

### Interactive Mode (RECOMMENDED)
```bash
python app.py --interactive
```

### Basic Evolution Run
```bash
python app.py --generations 25
```

### Custom Monitoring
```bash
python app.py \
  --generations 10 \
  --check-interval 30 \
  --memory-threshold 15.0
```

### Direct Pipeline Execution
```bash
python src/main.py --generations 25
```

### Smaller Model for Testing
Edit `config/modelConfig.yaml`:
```yaml
name: microsoft/DialoGPT-medium  # Smaller model for faster testing
```

## Safety Features

- **Automatic Restart**: Detects and recovers from stuck processes
- **Memory Protection**: Prevents out-of-memory crashes
- **Timeout Protection**: Prevents infinite hanging
- **State Preservation**: Saves progress before restarting
- **Comprehensive Logging**: Detailed monitoring and debugging information

## Monitoring Dashboard

The system provides comprehensive monitoring:
- **Real-time memory usage** with cleanup statistics
- **CPU usage monitoring** to detect stuck processes
- **Runtime tracking** with automatic timeout enforcement
- **Generation progress** with detailed logging
- **Error recovery** with automatic restart capabilities

## Recent Improvements

### **Unified Entry Point**
- **Integrated Setup**: All setup functionality moved to `app.py`
- **Removed Redundancy**: Eliminated `setup.py` and `memory_monitor.py`
- **Cleaner Interface**: Single entry point for all operations

### **Enhanced Initialization**
- **EvolutionTracker**: Automatic initialization during population setup
- **Population Index**: File metadata tracking for better organization
- **Generation Files**: Proper generation file management (gen0.json, etc.)

### **Improved Process Monitoring**
- **Built-in Health Checks**: Automatic restart capabilities in `main.py`
- **External Monitor**: Process monitoring in `app.py`
- **Memory Management**: Real-time memory tracking and cleanup

### **Code Cleanup**
- **Removed Emojis**: Professional text-based status messages
- **Simplified Structure**: Fewer files, better organization
- **Better Documentation**: Updated architecture and usage examples

## License

MIT License - see LICENSE file for details.