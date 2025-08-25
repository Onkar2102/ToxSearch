# Project Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVOLUTIONARY TEXT GENERATION FRAMEWORK                  │
│                       (Single Population.json & Accurate Tracking)         │
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

### **Single-File Population**
- Before: Split generation files (`gen0.json`, `gen1.json`, ...)
- After: Consolidated into a single `outputs/Population.json`
- Benefits: Simpler I/O, fewer file handles, consistent view across generations

### **Enhanced Evolution Tracking**
- `EvolutionTracker.json`: Comprehensive generation performance tracking
- `population_index.json`: Fast population metadata and counts
- Accurate max_score: Represents actual generation performance, not parent scores

### **Memory Optimization**
- Lazy Imports: Prevents circular dependencies and premature model loading
- Targeted Loading: Filter in memory by generation/prompt as needed
- Absolute Paths: Robust cross-platform file handling

## Detailed Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MAIN PIPELINE                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │   Population    │  │   Text          │  │   Safety        │           │
│  │  Initialization │  │  Generation     │  │  Evaluation     │           │
│  │                 │  │                 │  │                 │           │
│  │ • Load prompts  │  │ • LLaMA Model   │  │ • Hybrid API    │           │
│  │ • Create        │  │ • Memory Mgmt   │  │ • Google +      │           │
│  │   genomes       │  │ • Batch Proc    │  │   OpenAI        │           │
│  │ • Set status    │  │ • Error Handle  │  │ • Toxicity      │           │
│  │ • Single file   │  │ • Lazy loading  │  │   Scoring       │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│           │                   │                   │                       │
│           ▼                   ▼                   ▼                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │   Evolution     │  │   Population    │  │   Analysis      │           │
│  │   Engine        │  │   Management    │  │   & Logging     │           │
│  │                 │  │                 │  │                 │           │
│  │ • Genetic Algo  │  │ • Single File   │  │ • Performance   │           │
│  │ • Mutation      │  │ • Status Track  │  │   Monitoring    │           │
│  │ • Crossover     │  │ • Lineage       │  │ • Memory Stats  │           │
│  │ • Selection     │  │ • Deduplication │  │ • Error Logs    │           │
│  │ • Accurate      │  │ • Index Mgmt    │  │ • Evolution     │           │
│  │   Tracking      │  │ • Lazy Loading  │  │   Tracking      │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Memory Management Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY MANAGEMENT SYSTEM                           │
│                        (Applies to Single-File Population)                │
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
│• Single file│    │• Generation  │    │  loading     │    │  mgmt       │
│  loading    │    │              │    │              │    │             │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
```

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                    │
│                                (Single File)                              │
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

STORAGE LAYER (Single File)
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│outputs/     │    │outputs/      │    │outputs/      │
│Population.  │    │Evolution     │    │population_   │
│json         │    │Tracker.json  │    │index.json    │
│(All gens)   │    │(Progress)    │    │(Index/meta)  │
└─────────────┘    └──────────────┘    └──────────────┘
```

## Evolution Tracking Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVOLUTION TRACKING                                 │
│                           (Enhanced max_score)                            │
└─────────────────────────────────────────────────────────────────────────────┘

TRACKER STRUCTURE
┌─────────────────────────────────────────────────────────────────────────────┐
│EvolutionTracker.json                                                      │
├── prompt_id: 0                                                           │
│   ├── status: "not_complete"                                            │
│   ├── total_generations: 2                                              │
│   └── generations: [                                                     │
│       ├── generation_number: 0                                          │
│       │   ├── genome_id: "1"                                            │
│       │   └── max_score: 0.0395                                         │
│       └── generation_number: 1                                          │
│           ├── genome_id: "15"                                           │
│           ├── max_score: 0.0450                                         │
│           ├── variants_created: 20                                      │
│           ├── mutation_variants: 20                                     │
│           └── crossover_variants: 0                                     │
│   ]                                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Structure Architecture

```
EOST-CAM-LLM/
├── app.py                        # Main entry point with setup and monitoring
├── config/
│   └── modelConfig.yaml          # Model and memory settings
├── data/
│   └── prompt.xlsx               # Input prompts
├── src/
│   ├── main.py                   # Core evolution pipeline
│   ├── gne/                      # Generation & Evaluation
│   │   ├── LLaMaTextGenerator.py # LLaMA integration with memory management
│   │   ├── hybrid_moderation.py  # Hybrid safety evaluation (Google + OpenAI)
│   │   └── __init__.py           # Lazy import functions
│   ├── ea/                       # Evolutionary Algorithms
│   │   ├── EvolutionEngine.py    # Genetic algorithm core
│   │   ├── TextVariationOperators.py # Mutation/crossover (lazy loading)
│   │   ├── ParentSelector.py     # Selection strategies
│   │   ├── RunEvolution.py       # Evolution pipeline
│   │   └── __init__.py           # Package exports
│   └── utils/                    # Utilities
│       ├── population_io.py      # Single-file population management
│       ├── custom_logging.py     # Performance tracking
│       ├── m3_optimizer.py       # M3 Mac optimization
│       ├── config.py             # Configuration utilities
│       └── __init__.py           # Lazy import functions
├── outputs/
│   ├── Population.json           # Unified population store
│   ├── population_index.json     # Population metadata/index
│   ├── EvolutionTracker.json     # Evolution progress tracking
│   └── final_statistics.json     # Final analysis results (optional)
├── logs/                         # Log files
├── GENOME_STRUCTURE.md           # Genome data structure documentation
└── ARCHITECTURE.md               # This architecture document
```

## Performance Characteristics

### **Memory Usage**
- **Generation 0**: ~5.5KB (2 genomes)
- **Generation 1**: ~41KB (28 genomes)
- **Generation 2**: ~50KB (34 genomes)
- **Total**: Efficient split-file storage with lazy loading

### **Scalability**
- **Population Growth**: Linear memory usage with generations
- **File Management**: Automatic generation file creation and management
- **Indexing**: Fast population file discovery and access

### **Optimization Features**
- **Lazy Imports**: Prevents circular dependencies
- **Targeted Loading**: Load only needed generations
- **Memory Monitoring**: Real-time memory usage tracking
- **Automatic Cleanup**: PyTorch cache and garbage collection

## Recent Architecture Improvements

### **1. Split File Architecture**
- **Problem**: Single large Population.json file caused memory issues
- **Solution**: Split into generation-specific files (gen0.json, gen1.json, etc.)
- **Benefits**: Memory-efficient loading, better scalability, faster access

### **2. Enhanced Evolution Tracking**
- **Problem**: max_score was incorrectly set to parent's score
- **Solution**: Load generation files to calculate actual best scores
- **Benefits**: Accurate performance tracking, better evolution insights

### **3. Memory Optimization**
- **Problem**: Loading entire population for simple operations
- **Solution**: Targeted loading functions and lazy imports
- **Benefits**: Reduced memory usage, faster operations, better scalability

### **4. Path Resolution**
- **Problem**: Relative path issues causing file not found errors
- **Solution**: Absolute path resolution with robust file handling
- **Benefits**: Cross-platform compatibility, reliable file access

This enhanced architecture provides a robust, scalable, and memory-efficient framework for evolutionary text generation with comprehensive tracking and optimization features. 