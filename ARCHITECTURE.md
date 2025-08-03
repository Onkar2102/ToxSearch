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
├── app.py                        # Main entry point with setup and monitoring
├── config/
│   └── modelConfig.yaml          # Model and memory settings
├── data/
│   └── prompt.xlsx               # Input prompts
├── src/
│   ├── main.py                   # Core evolution pipeline
│   ├── gne/                      # Generation & Evaluation
│   │   ├── LLaMaTextGenerator.py # LLaMA integration with memory management
│   │   ├── openai_moderation.py  # Safety evaluation
│   │   └── README.md             # Generation module documentation
│   ├── ea/                       # Evolutionary Algorithms
│   │   ├── EvolutionEngine.py    # Genetic algorithm core
│   │   ├── TextVariationOperators.py # Mutation/crossover
│   │   ├── ParentSelector.py     # Selection strategies
│   │   ├── RunEvolution.py       # Evolution pipeline
│   │   └── VariationOperators.py # Base variation operators
│   └── utils/                    # Utilities
│       ├── population_io.py      # Data management with EvolutionTracker
│       ├── custom_logging.py     # Performance tracking
│       ├── m3_optimizer.py       # M3 Mac optimization
│       ├── config.py             # Configuration utilities
│       └── README.md             # Utils module documentation
├── outputs/                      # Generated data
│   ├── Population.json           # Main population
│   ├── EvolutionTracker.json     # Evolution tracking and metadata
│   ├── population_index.json     # Population file metadata
│   └── gen*.json                # Generation files
├── logs/                         # Execution logs
├── experiments/                  # Analysis and experiments
│   ├── *.ipynb                  # Jupyter notebooks
│   ├── *.md                     # Analysis reports
│   └── *.png                    # Generated visualizations
└── outputs1/                     # Backup/experimental outputs
```

## Entry Points and Setup

### **Primary Entry Points**
1. **`app.py --setup`** - Full environment setup and configuration
2. **`app.py --interactive`** - Interactive mode with setup checks and monitoring
3. **`app.py --generations N`** - Direct execution with process monitoring
4. **`src/main.py --generations N`** - Core pipeline execution

### **Setup and Initialization**
- **Environment Setup**: Python version, virtual environment, requirements installation
- **Configuration**: M3 optimization, model settings, memory limits
- **Data Initialization**: Population creation, EvolutionTracker setup, population_index creation
- **File Structure**: Automatic creation of required directories and files

## Key Architectural Principles

### 1. **Unified Entry Point**
- Single `app.py` handles all functionality (setup, monitoring, execution)
- Integrated setup and runtime checks
- Consistent user experience

### 2. **Modular Design**
- Each component has a single responsibility
- Clear interfaces between components
- Easy to test and maintain

### 3. **Memory Optimization**
- Adaptive batch sizing based on available memory
- Automatic cleanup after each operation
- Real-time monitoring and alerts

### 4. **Error Resilience**
- Graceful handling of out-of-memory errors
- Comprehensive logging for debugging
- Automatic recovery mechanisms

### 5. **Scalable Architecture**
- Configurable batch sizes and memory limits
- Support for different model sizes
- Extensible evolution operators

### 6. **Data Flow**
- Clear separation of input, processing, and output
- JSON-based data storage for flexibility
- Status tracking throughout the pipeline

### 7. **Process Monitoring**
- Built-in health monitoring in main.py
- External process monitor in app.py
- Automatic restart capabilities

This architecture ensures the system is robust, memory-efficient, and maintainable while providing the core evolutionary text generation capabilities. 