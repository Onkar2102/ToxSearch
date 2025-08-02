# Evolutionary Text Generation Framework

A research framework for AI safety analysis through evolutionary text generation, moderation evaluation, and genetic optimization.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Architecture

### Core Pipeline
```
Input Prompts → Text Generation → Safety Evaluation → Evolution → New Variants
```

### System Components

**Generation (`src/gne/`)**
- `LLaMaTextGenerator.py` - LLaMA model integration with enhanced memory management
- `openai_moderation.py` - OpenAI moderation API for safety evaluation

**Evolution (`src/ea/`)**
- `EvolutionEngine.py` - Genetic algorithm orchestration
- `TextVariationOperators.py` - Mutation and crossover operators
- `ParentSelector.py` - Fitness-based parent selection
- `RunEvolution.py` - Evolution pipeline execution

**Utilities (`src/utils/`)**
- `population_io.py` - Population data management
- `custom_logging.py` - Performance and memory logging
- `m3_optimizer.py` - Memory optimization utilities

**Configuration**
- `config/modelConfig.yaml` - Model and memory settings
- `data/prompt.xlsx` - Input prompts for evolution

### Memory Management
- **Adaptive batch sizing** based on available memory
- **Automatic memory cleanup** after each generation
- **Real-time monitoring** with configurable thresholds
- **Out-of-memory error handling** with graceful recovery

### Data Flow
1. **Population Initialization** - Load prompts from Excel
2. **Text Generation** - LLaMA model with memory optimization
3. **Safety Evaluation** - OpenAI moderation scoring
4. **Evolution** - Genetic algorithms create new variants
5. **Iteration** - Repeat until stopping conditions

### Output Structure
- `outputs/Population.json` - Main population data
- `outputs/EvolutionStatus.json` - Current evolution status
- `logs/` - Detailed execution logs

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Run the evolution pipeline
python src/main.py
```

## Configuration

Edit `config/modelConfig.yaml` to adjust:
- Model parameters (temperature, tokens, batch size)
- Memory management settings
- Generation parameters

## License

MIT License - see LICENSE file for details.