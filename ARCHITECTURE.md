# Project Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVOLUTIONARY TEXT GENERATION FRAMEWORK                  │
│                           (Enhanced with Split Files & Accurate Tracking)  │
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

### **Split File Architecture**
- **Before**: Single `Population.json` file (memory-intensive)
- **After**: Split generation files (`gen0.json`, `gen1.json`, etc.)
- **Benefits**: Memory-efficient loading, better scalability, faster access

### **Enhanced Evolution Tracking**
- **EvolutionTracker.json**: Comprehensive generation performance tracking
- **population_index.json**: Fast population file discovery
- **Accurate max_score**: Represents actual generation performance, not parent scores

### **Memory Optimization**
- **Lazy Imports**: Prevents circular dependencies and premature model loading
- **Targeted Loading**: Load only needed generations instead of entire population
- **Absolute Paths**: Robust cross-platform file handling

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
│  │ • Split files   │  │ • Lazy loading  │  │   Scoring       │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│           │                   │                   │                       │
│           ▼                   ▼                   ▼                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │   Evolution     │  │   Population    │  │   Analysis      │           │
│  │   Engine        │  │   Management    │  │   & Logging     │           │
│  │                 │  │                 │  │                 │           │
│  │ • Genetic Algo  │  │ • Split Files   │  │ • Performance   │           │
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
│                           (Enhanced with Split Files)                     │
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
│  memory     │    │• Conservative │    │• Error       │    │  cache      │
│• Split file │    │• Lazy loading│    │• Targeted    │    │• Generation │
│  loading    │    │• Generation  │    │  loading     │    │  file mgmt  │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
```

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                    │
│                           (Enhanced with Split Files)                     │
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

STORAGE LAYER (Split File Architecture)
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│outputs/     │    │outputs/      │    │outputs/      │    │logs/        │
│gen0.json    │    │gen1.json     │    │gen2.json     │    │Status.json  │
│(Initial)    │    │(Variants)    │    │(Variants)    │    │              │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│outputs/     │    │outputs/      │    │outputs/      │    │outputs/     │
│population_  │    │Evolution     │    │final_        │    │*.log        │
│index.json   │    │Tracker.json  │    │statistics.   │    │files        │
│(File Index) │    │(Progress)    │    │json          │    │(Logs)       │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
```

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPONENT INTERACTIONS                             │
│                           (Enhanced with Lazy Imports)                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   main.py   │◄────────┤  LLaMaText   │◄────────┤  Evolution  │
│             │         │  Generator   │         │  Engine     │
│• Orchestrates│         │              │         │              │
│• Controls   │         │• Model Load  │         │• Genetic    │
│  flow       │         │• Generation  │         │  Algorithms │
│• Error      │         │• Memory Mgmt │         │• Selection  │
│  handling   │         │• Batch Proc  │         │• Variation  │
│• Absolute   │         │• Lazy loading│         │• Accurate   │
│  paths      │         │• M3 optimize │         │  tracking   │
└─────────────┘         └──────────────┘         └─────────────┘
       │                       │                       │
       ▼                       ▼                       ▼
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  Hybrid     │         │  Population  │         │  Text       │
│  Moderation │         │  I/O         │         │  Variation  │
│             │         │              │         │  Operators  │
│• Google API │         │• Split Files │         │• Mutation   │
│• OpenAI API │         │• JSON Load   │         │• Crossover  │
│• Safety     │         │• JSON Save   │         │• Selection  │
│  Scoring    │         │• Status Mgmt │         │• Diversity  │
│• Dual API   │         │• Deduplication│        │• Lazy init  │
│• Fallback   │         │• Index Mgmt  │         │• Memory opt │
└─────────────┘         └──────────────┘         └─────────────┘
```

## Memory Management Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY MANAGEMENT FLOW                             │
│                           (Enhanced with Split Files)                     │
└─────────────────────────────────────────────────────────────────────────────┘

START GENERATION
       │
       ▼
┌─────────────┐
│ Check       │ ──▶ Memory Usage Below Threshold?
│ Memory      │
└─────────────┘
       │
       ▼
┌─────────────┐         ┌──────────────┐
│ Calculate   │◄────────┤  Adaptive    │
│ Batch Size  │         │  Algorithm   │
└─────────────┘         └──────────────┘
       │
       ▼
┌─────────────┐
│ Process     │ ──▶ Batch Complete?
│ Batch       │
└─────────────┘
       │
       ▼
┌─────────────┐
│ Cleanup     │ ──▶ Memory Still High?
│ Memory      │
└─────────────┘
       │
       ▼
┌─────────────┐
│ Continue    │ ──▶ More Batches?
│ or Stop     │
└─────────────┘
```

## Split File Architecture Details

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SPLIT FILE ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────────────┘

GENERATION FILES
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│gen0.json    │    │gen1.json     │    │gen2.json     │
│(Initial)    │    │(Variants)    │    │(Variants)    │
│• 2 genomes  │    │• 28 genomes  │    │• 34 genomes  │
│• 5.5KB      │    │• 41KB        │    │• 50KB        │
└─────────────┘    └──────────────┘    └──────────────┘

INDEX FILES
┌─────────────┐    ┌──────────────┐
│population_  │    │Evolution     │
│index.json   │    │Tracker.json  │
│• File list  │    │• Progress    │
│• Counts     │    │• Scores      │
│• Metadata   │    │• Lineage     │
└─────────────┘    └──────────────┘

LOADING STRATEGIES
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│load_population│  │load_population│  │load_population│
│(All gens)   │  │_generation()  │  │_lazy()       │
│• Full pop   │  │• Single gen   │  │• One at time │
│• Memory     │  │• Memory       │  │• Minimal     │
│  intensive  │  │  efficient    │  │  memory      │
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
│       │   └── max_score: 0.0395  ← Original genome score                │
│       └── generation_number: 1                                          │
│           ├── genome_id: "15"                                           │
│           ├── max_score: 0.0450  ← Best score in generation 1           │
│           ├── variants_created: 28                                      │
│           ├── mutation_variants: 28                                     │
│           └── crossover_variants: 0                                     │
│   ]                                                                      │
└─────────────────────────────────────────────────────────────────────────────┘

SCORE CALCULATION
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│Load Gen     │───▶│Extract       │───▶│Find Best     │
│File         │    │Scores        │    │Score         │
│• genX.json  │    │• Google API  │    │• Max score   │
│• Filter by  │    │• OpenAI API  │    │• Genome ID   │
│  prompt_id  │    │• Fallback    │    │• Update      │
└─────────────┘    └──────────────┘    └──────────────┘
```

## File Structure Architecture

```
eost-cam-llm/
├── app.py                        # Main entry point with setup and monitoring
├── config/
│   └── modelConfig.yaml          # Model and memory settings
├── data/
│   └── prompt.xlsx               # Input prompts
├── src/
│   ├── main.py                   # Core evolution pipeline (enhanced)
│   ├── gne/                      # Generation & Evaluation
│   │   ├── LLaMaTextGenerator.py # LLaMA integration with memory management
│   │   ├── hybrid_moderation.py  # Hybrid safety evaluation (Google + OpenAI)
│   │   └── __init__.py           # Lazy import functions
│   ├── ea/                       # Evolutionary Algorithms
│   │   ├── EvolutionEngine.py    # Genetic algorithm core
│   │   ├── TextVariationOperators.py # Mutation/crossover (lazy loading)
│   │   ├── ParentSelector.py     # Selection strategies
│   │   ├── RunEvolution.py       # Evolution pipeline (enhanced max_score)
│   │   ├── VariationOperators.py # Base variation operators
│   │   └── __init__.py           # Package exports
│   └── utils/                    # Utilities
│       ├── population_io.py      # Data management with split files
│       ├── custom_logging.py     # Performance tracking
│       ├── m3_optimizer.py       # M3 Mac optimization
│       ├── config.py             # Configuration utilities
│       └── __init__.py           # Lazy import functions
├── outputs/                      # Generated data (split file architecture)
│   ├── gen0.json                # Generation 0 (initial population)
│   ├── gen1.json                # Generation 1 variants
│   ├── gen2.json                # Generation 2 variants
│   ├── population_index.json    # Population file index
│   ├── EvolutionTracker.json    # Evolution progress tracking
│   └── final_statistics.json    # Final analysis results
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