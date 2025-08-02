# Project Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVOLUTIONARY TEXT GENERATION FRAMEWORK                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Pipeline Flow

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│   INPUT     │───▶│  GENERATION  │───▶│  EVALUATION  │───▶│  EVOLUTION  │
│  PROMPTS    │    │   (LLaMA)    │    │   (OpenAI)   │    │ (Genetic)   │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│data/prompt. │    │Generated     │    │Moderation    │    │New Variants │
│xlsx         │    │Responses     │    │Scores        │    │Created      │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
```

## Detailed Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MAIN PIPELINE                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │   Population    │  │   Text          │  │   Safety        │           │
│  │  Initialization │  │  Generation     │  │  Evaluation     │           │
│  │                 │  │                 │  │                 │           │
│  │ • Load prompts  │  │ • LLaMA Model   │  │ • OpenAI API    │           │
│  │ • Create        │  │ • Memory Mgmt   │  │ • Toxicity      │           │
│  │   genomes       │  │ • Batch Proc    │  │   Scoring       │           │
│  │ • Set status    │  │ • Error Handle  │  │ • Multi-cat     │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│           │                   │                   │                       │
│           ▼                   ▼                   ▼                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │   Evolution     │  │   Population    │  │   Analysis      │           │
│  │   Engine        │  │   Management    │  │   & Logging     │           │
│  │                 │  │                 │  │                 │           │
│  │ • Genetic Algo  │  │ • JSON Storage  │  │ • Performance   │           │
│  │ • Mutation      │  │ • Status Track  │  │   Monitoring    │           │
│  │ • Crossover     │  │ • Lineage       │  │ • Memory Stats  │           │
│  │ • Selection     │  │ • Deduplication │  │ • Error Logs    │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Memory Management Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY MANAGEMENT SYSTEM                           │
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
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
```

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                    │
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

STORAGE LAYER
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│outputs/     │    │outputs/      │    │logs/         │
│Population.  │    │Evolution     │    │Status.json   │
│json         │    │              │    │              │
└─────────────┘    └──────────────┘    └──────────────┘
```

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPONENT INTERACTIONS                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   main.py   │◄────────┤  LLaMaText   │◄────────┤  Evolution  │
│             │         │  Generator   │         │  Engine     │
│• Orchestrates│         │              │         │              │
│• Controls   │         │• Model Load  │         │• Genetic    │
│  flow       │         │• Generation  │         │  Algorithms │
│• Error      │         │• Memory Mgmt │         │• Selection  │
│  handling   │         │• Batch Proc  │         │• Variation  │
└─────────────┘         └──────────────┘         └─────────────┘
       │                       │                       │
       ▼                       ▼                       ▼
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  OpenAI     │         │  Population  │         │  Text       │
│  Moderation │         │  I/O         │         │  Variation  │
│             │         │              │         │  Operators  │
│• Safety     │         │• JSON Load   │         │• Mutation   │
│  Scoring    │         │• JSON Save   │         │• Crossover  │
│• API Calls  │         │• Status Mgmt │         │• Selection  │
│• Rate Limit │         │• Deduplication│        │• Diversity  │
└─────────────┘         └──────────────┘         └─────────────┘
```

## Memory Management Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY MANAGEMENT FLOW                             │
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

## File Structure Architecture

```
eost-cam-llm/
├── config/
│   └── modelConfig.yaml          # Model and memory settings
├── data/
│   └── prompt.xlsx               # Input prompts
├── src/
│   ├── main.py                   # Main orchestration
│   ├── gne/                      # Generation & Evaluation
│   │   ├── LLaMaTextGenerator.py # LLaMA integration
│   │   └── openai_moderation.py  # Safety evaluation
│   ├── ea/                       # Evolutionary Algorithms
│   │   ├── EvolutionEngine.py    # Genetic algorithm core
│   │   ├── TextVariationOperators.py # Mutation/crossover
│   │   ├── ParentSelector.py     # Selection strategies
│   │   └── RunEvolution.py       # Evolution pipeline
│   └── utils/                    # Utilities
│       ├── population_io.py      # Data management
│       ├── custom_logging.py     # Performance tracking
│       └── m3_optimizer.py       # Memory optimization
├── outputs/                      # Generated data
│   ├── Population.json           # Main population
│   └── EvolutionStatus.json      # Current status
└── logs/                         # Execution logs
```

## Key Architectural Principles

### 1. **Modular Design**
- Each component has a single responsibility
- Clear interfaces between components
- Easy to test and maintain

### 2. **Memory Optimization**
- Adaptive batch sizing based on available memory
- Automatic cleanup after each operation
- Real-time monitoring and alerts

### 3. **Error Resilience**
- Graceful handling of out-of-memory errors
- Comprehensive logging for debugging
- Automatic recovery mechanisms

### 4. **Scalable Architecture**
- Configurable batch sizes and memory limits
- Support for different model sizes
- Extensible evolution operators

### 5. **Data Flow**
- Clear separation of input, processing, and output
- JSON-based data storage for flexibility
- Status tracking throughout the pipeline

This architecture ensures the system is robust, memory-efficient, and maintainable while providing the core evolutionary text generation capabilities. 