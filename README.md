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

### **Latest Features (Current Architecture)**
- **16 Text Variation Operators**: Complete suite of mutation and crossover operators
- **Multi-Language Back Translation**: 5 languages (Hindi, French, German, Japanese, Chinese) with both model-based and LLM-based approaches
- **Steady-State Population Management**: Elite preservation with continuous evolution
- **Hybrid Translation Approaches**: Helsinki-NLP models + LLaMA-based translation
- **Enhanced LLM Integration**: Task-specific templates and generation parameters
- **Comprehensive Operator Suite**: 13 mutation + 3 crossover operators

### **Key Improvements**
- **Dual Translation Methods**: Model-based (Helsinki-NLP) and LLM-based (LLaMA) back translation
- **Language Coverage**: Full support for Hindi, French, German, Japanese, Chinese
- **Steady-State Evolution**: Elite preservation with continuous population management
- **Task-Specific Templates**: Configurable prompts for different tasks (translation, paraphrasing)
- **Memory Optimization**: Efficient model loading and caching
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
- `LLaMaTextGenerator.py` - LLaMA model integration with memory management and task-specific templates
- `hybrid_moderation.py` - Hybrid moderation using Google Perspective API + OpenAI

### **Utilities** (`src/utils/`)
- `population_io.py` - Steady-state population management (`elites.json`) and `EvolutionTracker.json`
- `custom_logging.py` - Performance and memory logging
- `m3_optimizer.py` - M3 Mac optimization utilities
- `config.py` - Configuration management
- `constants.py` - System constants and configuration
- `download_models.py` - Model download utilities

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

# Load elites for analysis
python -c "from src.utils.population_io import load_elites; elites = load_elites('outputs/elites.json')"
```

### **Operator Testing**
```bash
# Test all operators
python tests/test_operators_demo.py

# Test specific back translation
python -c "from src.ea.TextVariationOperators import LLMBackTranslationHIOperator; op = LLMBackTranslationHIOperator(); print(op.apply('Hello world'))"
```

## Configuration

The system uses `config/modelConfig.yaml` for:
- Model parameters (LLaMA 3.2 3B Instruct)
- Task-specific templates (translation, paraphrasing)
- Generation parameters per task
- Memory optimization settings
- Batch processing configuration
- API keys and endpoints

### **Task-Specific Configuration**
```yaml
task_templates:
  translation:
    en_to_target: "Translate the following text from English to {target_language}..."
    target_to_en: "Translate the following text from {source_language} to English..."

task_generation_args:
  translation:
    temperature: 0.8
    top_p: 0.9
    top_k: 40
    max_new_tokens: 2048
```

## Monitoring & Recovery

- **Real-time Monitoring**: Memory usage, process health, execution time
- **Automatic Recovery**: Detects stuck processes and restarts automatically
- **Progress Tracking**: EvolutionTracker.json for comprehensive progress monitoring
- **Performance Logging**: Detailed timing and resource usage metrics

## Output Structure

```
outputs/
├── elites.json              # Steady-state elite population
├── Population.json          # Full population (if needed)
├── population_index.json    # Population metadata
├── EvolutionTracker.json    # Evolution progress tracking
└── final_statistics.json   # Final analysis results (optional)
```

## Text Variation Operators

### **Mutation Operators (13)**
- `LLM_POSAwareSynonymReplacement` - LLaMA-based synonym replacement
- `BertMLMOperator` - BERT masked language model
- `LLMBasedParaphrasingOperator` - OpenAI GPT-4 paraphrasing
- `BackTranslationHIOperator` - Hindi back-translation (Helsinki-NLP)
- `BackTranslationFROperator` - French back-translation (Helsinki-NLP)
- `BackTranslationDEOperator` - German back-translation (Helsinki-NLP)
- `BackTranslationJAOperator` - Japanese back-translation (Helsinki-NLP)
- `BackTranslationZHOperator` - Chinese back-translation (Helsinki-NLP)
- `LLMBackTranslationHIOperator` - Hindi back-translation (LLaMA)
- `LLMBackTranslationFROperator` - French back-translation (LLaMA)
- `LLMBackTranslationDEOperator` - German back-translation (LLaMA)
- `LLMBackTranslationJAOperator` - Japanese back-translation (LLaMA)
- `LLMBackTranslationZHOperator` - Chinese back-translation (LLaMA)

### **Crossover Operators (3)**
- `OnePointCrossover` - Single-point crossover
- `SemanticSimilarityCrossover` - Semantic similarity-based crossover
- `InstructionPreservingCrossover` - Instruction structure preservation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.