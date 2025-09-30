# Project Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVOLUTIONARY TEXT GENERATION FRAMEWORK                  │
│                    (Steady-State Population & 16 Operators)                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Pipeline Flow

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│   INPUT     │───▶│  GENERATION  │───▶│  EVALUATION  │───▶│  EVOLUTION  │
│  PROMPTS    │    │   (LLaMA)    │    │   (Hybrid)   │    │ (Genetic)   │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│data/prompt. │    │Generated     │    │Moderation    │    │New Variants │
│xlsx         │    │Responses     │    │Scores        │    │Created      │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
```

## Recent Architecture Improvements

### **Steady-State Population Management**
- **Elite Preservation**: Top performers maintained in `elites.json`
- **Continuous Evolution**: Population evolves continuously without generation boundaries
- **Dynamic Redistribution**: Elites redistributed to population when thresholds exceeded
- **Memory Efficiency**: Single-file population with lazy loading

### **16 Text Variation Operators**
- **13 Mutation Operators**: Including 10 back-translation operators (5 languages × 2 methods)
- **3 Crossover Operators**: One-point, semantic similarity, instruction-preserving
- **Dual Translation Approaches**: Helsinki-NLP models + LLaMA-based translation
- **Multi-Language Support**: Hindi, French, German, Japanese, Chinese

### **Enhanced Evolution Tracking**
- `EvolutionTracker.json`: Comprehensive generation performance tracking
- `population_index.json`: Fast population metadata and counts
- Accurate max_score: Represents actual generation performance, not parent scores

### **Memory Optimization**
- Lazy Imports: Prevents circular dependencies and premature model loading
- Targeted Loading: Filter in memory by generation/prompt as needed
- Absolute Paths: Robust cross-platform file handling
- Model Caching: Efficient reuse of loaded models across operators

## Detailed Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MAIN PIPELINE                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │   Population    │  │   Text          │  │   Safety        │           │
│  │  Initialization │  │  Generation     │  │  Evaluation     │           │
│  │                 │  │                 │  │                 │           │
│  │ • Load prompts  │  │ • LLaMA Model   │  │ • Hybrid API    │           │
│  │ • Create        │  │ • Task Templates│  │ • Google +      │           │
│  │   genomes       │  │ • Memory Mgmt   │  │   OpenAI        │           │
│  │ • Set status    │  │ • Batch Proc    │  │ • Toxicity      │           │
│  │ • Steady state  │  │ • Error Handle  │  │   Scoring       │           │
│  │ • Elite mgmt    │  │ • Lazy loading  │  │ • Multi-metric  │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│           │                   │                   │                       │
│           ▼                   ▼                   ▼                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │   Evolution     │  │   Population    │  │   Analysis      │           │
│  │   Engine        │  │   Management    │  │   & Logging     │           │
│  │                 │  │                 │  │                 │           │
│  │ • Genetic Algo  │  │ • Steady State  │  │ • Performance   │           │
│  │ • 16 Operators  │  │ • Elite Track   │  │   Monitoring    │           │
│  │ • Mutation      │  │ • Status Track  │  │ • Memory Stats  │           │
│  │ • Crossover     │  │ • Lineage       │  │ • Error Logs    │           │
│  │ • Selection     │  │ • Deduplication │  │ • Evolution     │           │
│  │ • Accurate      │  │ • Index Mgmt    │  │   Tracking      │           │
│  │   Tracking      │  │ • Lazy Loading  │  │ • Operator      │           │
│  │ • Multi-lang    │  │ • Redistribution│  │   Analytics     │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Text Variation Operators Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TEXT VARIATION OPERATORS                           │
│                              (16 Total)                                   │
└─────────────────────────────────────────────────────────────────────────────┘

MUTATION OPERATORS (13)
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Core LLM      │    │   BERT-based    │    │   OpenAI        │
│   Operators     │    │   Operators     │    │   Operators     │
│                 │    │                 │    │                 │
│ • POS-Aware     │    │ • BERT MLM      │    │ • Paraphrasing  │
│   Synonym       │    │   Replacement   │    │   (GPT-4)       │
│   Replacement   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BACK-TRANSLATION OPERATORS                         │
│                    (5 Languages × 2 Methods = 10 Operators)              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model-Based   │    │   LLM-Based     │    │   Languages     │
│   (Helsinki-NLP)│    │   (LLaMA)       │    │   Supported     │
│                 │    │                 │    │                 │
│ • BackTrans_HI  │    │ • LLMBackTrans_HI│   │ • Hindi (HI)    │
│ • BackTrans_FR  │    │ • LLMBackTrans_FR│   │ • French (FR)   │
│ • BackTrans_DE  │    │ • LLMBackTrans_DE│   │ • German (DE)   │
│ • BackTrans_JA  │    │ • LLMBackTrans_JA│   │ • Japanese (JA) │
│ • BackTrans_ZH  │    │ • LLMBackTrans_ZH│   │ • Chinese (ZH)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘

CROSSOVER OPERATORS (3)
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   One-Point     │    │   Semantic      │    │   Instruction   │
│   Crossover     │    │   Similarity    │    │   Preserving    │
│                 │    │   Crossover     │    │   Crossover     │
│ • Single split  │    │ • Embedding-    │    │ • Structure-    │
│ • Random point  │    │   based         │    │   aware         │
│ • Preserve      │    │ • Similarity    │    │ • Maintain      │
│   order         │    │   threshold     │    │   instructions  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Memory Management Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY MANAGEMENT SYSTEM                           │
│                    (Steady-State Population Management)                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│   Memory    │───▶│  Adaptive    │───▶│  Batch       │───▶│  Cleanup    │
│  Monitor    │    │  Batch Size  │    │  Processing  │    │  System     │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│• Real-time  │    │• Calculate   │    │• Process     │    │• Clear      │
│  tracking   │    │  optimal     │    │  in smaller  │    │  PyTorch    │
│• Threshold  │    │  batch size  │    │  batches     │    │  cache      │
│  alerts     │    │• Memory      │    │• Memory      │    │• Force GC   │
│• System     │    │  available   │    │  efficient   │    │• Model      │
│  memory     │    │• Lazy loading│    │• Targeted    │    │  cache      │
│• Elite mgmt │    │• Elite       │    │  loading     │    │  mgmt       │
│• Steady     │    │  preservation│    │• Operator    │    │• Operator   │
│  state      │    │              │    │  caching     │    │  cleanup    │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
```

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                    │
│                            (Steady-State)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT LAYER
┌─────────────┐
│data/prompt. │ ──▶ Population Initialization
│xlsx         │
└─────────────┘

PROCESSING LAYER
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│Generation   │───▶│Evaluation    │───▶│Evolution     │
│Phase        │    │Phase         │    │Phase         │
└─────────────┘    └──────────────┘    └──────────────┘

STORAGE LAYER (Steady-State)
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│outputs/     │    │outputs/      │    │outputs/      │
│elites.json  │    │Evolution     │    │population_   │
│(Elites)     │    │Tracker.json  │    │index.json    │
│             │    │(Progress)    │    │(Index/meta)  │
└─────────────┘    └──────────────┘    └──────────────┘
```

## Evolution Tracking Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVOLUTION TRACKING                                 │
│                    (Steady-State Population Management)                    │
└─────────────────────────────────────────────────────────────────────────────┘

TRACKER STRUCTURE
┌─────────────────────────────────────────────────────────────────────────────┐
│EvolutionTracker.json                                                      │
├── scope: "global"                                                        │
├── status: "not_complete"                                                 │
├── total_generations: 5                                                   │
├── population_max_toxicity: 0.3775                                       │
├── population_best_genome_id: "40"                                        │
└── generations: [                                                         │
    ├── generation_number: 0                                               │
    │   ├── genome_id: "1"                                                 │
    │   └── max_score: 0.0                                                 │
    ├── generation_number: 2                                               │
    │   ├── genome_id: "122"                                               │
    │   ├── max_score: 0.361                                                │
    │   ├── variants_created: 23                                           │
    │   ├── mutation_variants: 21                                          │
    │   ├── crossover_variants: 2                                          │
    │   └── parents: {                                                     │
    │       ├── mutation_parent: {...}                                     │
    │       └── crossover_parents: [...]                                   │
    │   }                                                                   │
    └── ...                                                                │
]                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Structure Architecture

```
EOST-CAM-LLM/
├── app.py                        # Main entry point with setup and monitoring
├── config/
│   └── modelConfig.yaml          # Model, task templates, and memory settings
├── data/
│   └── prompt.xlsx               # Input prompts
├── src/
│   ├── main.py                   # Core evolution pipeline
│   ├── gne/                      # Generation & Evaluation
│   │   ├── LLaMaTextGenerator.py # LLaMA integration with task-specific templates
│   │   ├── hybrid_moderation.py  # Hybrid safety evaluation (Google + OpenAI)
│   │   └── __init__.py           # Lazy import functions
│   ├── ea/                       # Evolutionary Algorithms
│   │   ├── EvolutionEngine.py    # Genetic algorithm core (steady-state)
│   │   ├── TextVariationOperators.py # 16 mutation/crossover operators
│   │   ├── ParentSelector.py     # Selection strategies (steady-state)
│   │   ├── RunEvolution.py       # Evolution pipeline
│   │   └── __init__.py           # Package exports
│   └── utils/                    # Utilities
│       ├── population_io.py      # Steady-state population management
│       ├── custom_logging.py     # Performance tracking
│       ├── m3_optimizer.py       # M3 Mac optimization
│       ├── config.py             # Configuration utilities
│       ├── constants.py          # System constants
│       ├── download_models.py    # Model download utilities
│       └── __init__.py           # Lazy import functions
├── outputs/
│   ├── elites.json              # Steady-state elite population
│   ├── Population.json          # Full population (if needed)
│   ├── population_index.json    # Population metadata/index
│   ├── EvolutionTracker.json    # Evolution progress tracking
│   └── final_statistics.json   # Final analysis results (optional)
├── logs/                         # Log files
├── tests/
│   └── test_operators_demo.py   # Operator testing and demonstration
├── design_document.md           # Formal design specification
└── ARCHITECTURE.md              # Architecture document
```

## Performance Characteristics

### **Memory Usage**
- **Elites Population**: ~500KB (124 genomes)
- **Full Population**: ~2.8MB (2885 genomes)
- **Model Loading**: Efficient caching and reuse across operators
- **Steady-State**: Continuous memory management with elite preservation

### **Scalability**
- **Population Growth**: Controlled through elite redistribution
- **Operator Efficiency**: Lazy loading and model reuse
- **Memory Management**: Real-time monitoring and adaptive batch sizing

### **Optimization Features**
- **Lazy Imports**: Prevents circular dependencies
- **Model Caching**: Reuse loaded models across operators
- **Memory Monitoring**: Real-time memory usage tracking
- **Automatic Cleanup**: PyTorch cache and garbage collection
- **Steady-State Management**: Efficient elite preservation

## Recent Architecture Improvements

### **1. Steady-State Population Management**
- Problem: Generation-based evolution had artificial boundaries
- Solution: Continuous evolution with elite preservation
- Benefits: More natural evolution, better performance tracking, memory efficiency

### **2. 16 Text Variation Operators**
- Problem: Limited text variation capabilities
- Solution: Comprehensive operator suite with dual translation approaches
- Benefits: Rich text variation, multi-language support, diverse evolution strategies

### **3. Dual Translation Approaches**
- Problem: Single translation method limited diversity
- Solution: Helsinki-NLP models + LLaMA-based translation
- Benefits: Complementary approaches, better coverage, robust translation

### **4. Task-Specific Templates**
- Problem: Generic prompts for all tasks
- Solution: Configurable templates per task type
- Benefits: Better task performance, precise control, improved results

### **5. Enhanced Memory Management**
- Problem: Memory pressure with multiple operators
- Solution: Model caching, lazy loading, adaptive batch sizing
- Benefits: Lower memory usage, faster execution, better scalability

This enhanced architecture provides a robust, scalable, and memory-efficient framework for evolutionary text generation with comprehensive operator support, steady-state population management, and multi-language capabilities.