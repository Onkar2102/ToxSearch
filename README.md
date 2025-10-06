# Evolutionary Text Generation Framework

A research framework for AI safety analysis through evolutionary text generation, moderation evaluation, and genetic optimization with **automatic process monitoring and recovery**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Quick Start](#quick-start)
- [app.py Command Line Arguments](#appy-command-line-arguments)
- [Documentation](#documentation)
  - [Architecture Overview](ARCHITECTURE.md)
  - [Design Document](design_document.md)
  - [Evolutionary Algorithms](src/ea/README.md)
  - [Generation & Evaluation](#generation--evaluation)
  - [Utilities](#utilities)
- [Usage Examples](#usage-examples)
- [Output Structure](#output-structure)
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

 

## Architecture at a Glance

```mermaid
flowchart TD
  A[Input Prompts: data/prompt.xlsx] --> B[Initialize Population → outputs/elites.json]
  B --> C[Steady-State Evolution Loop]
  
  subgraph "Evolution Loop"
    C --> D[Parent Selection: Top Elite + Random]
    D --> E[Text Generation: LLaMaTextGenerator]
    E --> F[Safety Evaluation: Hybrid Moderation]
    F --> G[Evolution: 16 Variation Operators]
    G --> H[Update Elites: outputs/elites.json]
    H --> I{Threshold Reached?}
    I -->|No| D
    I -->|Yes| J[Complete]
  end
  
  F --> K[EvolutionTracker.json]
  H --> L[Population.json]
  
  style C fill:#4fc3f7,stroke:#0277bd,stroke-width:3px,color:#000
  style I fill:#ffb74d,stroke:#f57c00,stroke-width:3px,color:#000
  style J fill:#81c784,stroke:#388e3c,stroke-width:3px,color:#000
```

## System Components Overview

```mermaid
graph TB
  subgraph "Input Layer"
    A1[data/prompt.xlsx]
  end
  
  subgraph "Core Pipeline"
    B1[Population Initialization]
    B2[Text Generation - LLaMA]
    B3[Safety Evaluation - Hybrid API]
    B4[Evolution Engine - 16 Operators]
  end
  
  subgraph "Storage Layer"
    C1[outputs/elites.json<br/>Steady-State Population]
    C2[outputs/EvolutionTracker.json<br/>Progress Tracking]
    C3[outputs/Population.json<br/>Full Population]
  end
  
  subgraph "Text Variation Operators"
    D1[Mutation Operators<br/>10 Total]
    D2[Crossover Operators<br/>2 Total]
    D3[Multi-Language Support<br/>5 Languages]
  end
  
  A1 --> B1
  B1 --> B2
  B2 --> B3
  B3 --> B4
  B4 --> B1
  
  B1 --> C1
  B3 --> C2
  B4 --> C3
  
  B4 --> D1
  B4 --> D2
  B4 --> D3
  
  style B1 fill:#64b5f6,stroke:#1976d2,stroke-width:2px,color:#000
  style B2 fill:#ba68c8,stroke:#7b1fa2,stroke-width:2px,color:#000
  style B3 fill:#4caf50,stroke:#2e7d32,stroke-width:2px,color:#000
  style B4 fill:#ff9800,stroke:#ef6c00,stroke-width:2px,color:#000
```

## Memory Management Architecture

```mermaid
flowchart LR
  A[Memory Monitor] --> B[Adaptive Batch Sizing]
  B --> C[Model Caching]
  C --> D[Lazy Loading]
  D --> E[Memory Cleanup]
  E --> A
  
  subgraph "Memory Optimization"
    F[Real-time Tracking]
    G[Threshold Alerts]
    H[PyTorch Cache Clear]
    I[Garbage Collection]
  end
  
  A --> F
  A --> G
  E --> H
  E --> I
  
  style A fill:#f48fb1,stroke:#c2185b,stroke-width:3px,color:#000
  style E fill:#66bb6a,stroke:#388e3c,stroke-width:3px,color:#000
```

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

### Current Active Operators (12 Total)

#### **Mutation Operators (10)**
1. **LLM_POSAwareSynonymReplacement** - LLaMA-based synonym replacement using POS tagging
2. **MLMOperator** - BERT masked language model for word replacement
3. **LLMBasedParaphrasingOperator** - LLaMA-based paraphrasing with optimization
4. **LLM_POSAwareAntonymReplacement** - LLaMA-based antonym replacement
5. **StylisticMutator** - Stylistic text mutations
6. **LLMBackTranslationHIOperator** - Hindi back-translation (LLaMA)
7. **LLMBackTranslationFROperator** - French back-translation (LLaMA)
8. **LLMBackTranslationDEOperator** - German back-translation (LLaMA)
9. **LLMBackTranslationJAOperator** - Japanese back-translation (LLaMA)
10. **LLMBackTranslationZHOperator** - Chinese back-translation (LLaMA)

#### **Crossover Operators (2)**
1. **SemanticSimilarityCrossover** - Semantic similarity-based crossover
2. **InstructionPreservingCrossover** - LLM-based instruction structure preservation

### Deprecated Operators
- **POSAwareSynonymReplacement** - Replaced by LLM version
- **PointCrossover** - Deprecated and commented out
- **Classic Back-translation operators** - Replaced by LLM versions

```mermaid
graph TB
  subgraph "Active Mutation Operators (10)"
    A1[LLM_POSAwareSynonymReplacement]
    A2[MLMOperator]
    A3[LLMBasedParaphrasingOperator]
    A4[LLM_POSAwareAntonymReplacement]
    A5[StylisticMutator]
    A6[LLM Back-Translation: 5 Languages]
      B2 --> C2
      B2 --> C3
      B2 --> C4
      B2 --> C5
    end
  end
  
  subgraph "Crossover Operators (3)"
    D1[PointCrossover<br/>Single-point crossover]
    D2[SemanticSimilarityCrossover<br/>Semantic similarity-based]
    D3[InstructionPreservingCrossover<br/>Instruction structure preservation]
  end
  
  style A1 fill:#42a5f5,stroke:#1565c0,stroke-width:2px,color:#000
  style A2 fill:#42a5f5,stroke:#1565c0,stroke-width:2px,color:#000
  style A3 fill:#42a5f5,stroke:#1565c0,stroke-width:2px,color:#000
  style B1 fill:#ab47bc,stroke:#6a1b9a,stroke-width:2px,color:#000
  style B2 fill:#ab47bc,stroke:#6a1b9a,stroke-width:2px,color:#000
  style D1 fill:#66bb6a,stroke:#2e7d32,stroke-width:2px,color:#000
  style D2 fill:#66bb6a,stroke:#2e7d32,stroke-width:2px,color:#000
  style D3 fill:#66bb6a,stroke:#2e7d32,stroke-width:2px,color:#000
```

### **Operator Selection Logic**

```mermaid
flowchart TD
  A[Parent Selection] --> B{Number of Parents?}
  B -->|1 Parent| C[Mutation Operators<br/>10 Total]
  B -->|2+ Parents| D[Crossover Operators<br/>2 Total]
  
  C --> E[Apply Selected Operator]
  D --> E
  E --> F[Generate Variants<br/>Max 5 per operator]
  F --> G[Deduplication]
  G --> H[Add to Population]
  
  style C fill:#42a5f5,stroke:#1565c0,stroke-width:2px,color:#000
  style D fill:#66bb6a,stroke:#2e7d32,stroke-width:2px,color:#000
  style F fill:#ffb74d,stroke:#f57c00,stroke-width:2px,color:#000
```

## Recent Changes & Updates

### Import System Refactoring (October 2025)
- ✅ **Eliminated all try-except import patterns** for cleaner, faster imports
- ✅ **Standardized import conventions** throughout the project
- ✅ **Fixed relative/absolute import inconsistencies**
- ✅ **Improved error messages** when dependencies are missing

### Operator Consolidation
- ✅ **Removed classic POS-aware synonym replacement** - now using LLM version only
- ✅ **Deprecated point crossover operator** - commented out but retained for reference
- ✅ **Deprecated classic back-translation** - now using LLM-based versions only
- ✅ **Updated to 12 active operators** (10 mutation + 2 crossover)

### Configuration Improvements
- ✅ **Config-driven prompt templates** for instruction-preserving crossover
- ✅ **Centralized system instructions** in modelConfig.yaml
- ✅ **Single-variant behavior** for consistency across operators

### Current Active Operators
**Mutation Operators (10):**
1. LLM_POSAwareSynonymReplacement
2. MLMOperator
3. LLMBasedParaphrasingOperator
4. LLM_POSAwareAntonymReplacement
5. StylisticMutator
6. LLMBackTranslationHIOperator
7. LLMBackTranslationFROperator
8. LLMBackTranslationDEOperator
9. LLMBackTranslationJAOperator
10. LLMBackTranslationZHOperator

**Crossover Operators (2):**
1. SemanticSimilarityCrossover
2. InstructionPreservingCrossover

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

 

## License

MIT License - see LICENSE file for details.