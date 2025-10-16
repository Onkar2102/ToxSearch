# Evolutionary Algorithms (EA) Package

The Evolutionary Algorithms package provides the core genetic algorithm functionality for evolving text prompts through mutation and crossover operations. This package implements a complete evolutionary framework with parent selection, variation operators, and steady-state population management.

## Table of Contents

- [Quick Setup](#quick-setup)
- [Core Components](#core-components)
- [Text Variation Operators](#text-variation-operators)
- [Parent Selection](#parent-selection)
- [Population Management](#population-management)
- [Evolution Flow](#evolution-flow)
- [Performance Characteristics](#performance-characteristics)
- [Usage Examples](#usage-examples)
- [Documentation Index](#documentation-index)

## Quick Setup

### Prerequisites
- **Python 3.8+** with virtual environment activated
- **Required packages**: See [requirements.txt](../../requirements.txt)
- **API Keys**: OpenAI and Google Perspective API keys configured
- **Models**: Qwen2.5-7B-Instruct and Llama3.2-3B-Instruct models

### Installation
```bash
# From project root
cd /path/to/eost-cam-llm
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test EA package
python -c "from src.ea import get_EvolutionEngine; print('EA package loaded successfully')"
```

### Quick Test
```bash
# Test all operators
python tests/test_operators_demo.py

# Test specific operator
python -c "from src.ea.negation_operator import NegationOperator; op = NegationOperator(); print(op.apply('What are advantages of social media?'))"
```

## Core Components

### **1. EvolutionEngine** (`EvolutionEngine.py`)
The main orchestrator for the evolutionary process.

**Key Features:**
- Manages genome populations and evolution cycles (steady-state evolution)
- Coordinates parent selection and variant generation
- Tracks variant counts and integrates deduplication
- Steady-state population persistence (`data/outputs/elites.json`)
- Memory-optimized for large populations
- Supports 16 text variation operators

**Main Methods:**
```python
# Initialize evolution engine
engine = EvolutionEngine(north_star_metric, log_file, current_cycle)

# Generate variants globally for the current cycle
generation_data = engine.generate_variants_global()

# Keep IDs consistent with current population
engine.update_next_id()
```

**Recent Improvements:**
- Reduced `max_num_parents` from 4 to 2 to control population growth
- Increased `adaptive_selection_after` from 5 to 10 generations
- Added 4 new mutation operators
- Enhanced error handling and fallback mechanisms

### **2. RunEvolution** (`RunEvolution.py`)
The main evolution pipeline driver and execution coordinator.

**Key Features:**
- Orchestrates complete evolution cycles
- Manages evolution tracker and metadata
- Handles threshold checking and completion logic
- Provides statistics and reporting
- Enhanced `max_score` calculation per generation
- Steady-state population loading and in-memory filtering

**Main Functions:**
```python
# Run complete evolution cycle
run_evolution(north_star_metric, log_file, threshold=0.95, current_cycle=None)

# Check threshold achievement
check_threshold_and_update_tracker(population, north_star_metric, logger, threshold=0.95)

# Update evolution tracker with generation data
update_evolution_tracker_with_generation_global(generation_data, evolution_tracker, logger, population, north_star_metric)
```

**Recent Fixes:**
- Accurate `max_score` per generation (children, not parent)
- Steady-state population I/O
- Lazy imports and efficient data loading

### **3. ParentSelector** (`ParentSelector.py`)
Intelligent parent selection strategies for genetic operations.

**Selection Strategies:**
- **Steady-State Selection**: Topmost elite + random elites + random population
- Single Genome
- Small Population (2–4)
- Large Population (5+)
- Tournament Selection
- Roulette Selection

**Steady-State Strategy:**
```python
def select_parents_steady_state(self):
    # Mutation parent: Topmost elite
    # Crossover parents: Topmost + 1 random elite + 3 random population
    # Up to 5 crossover parents for maximum diversity
```

**Recent Optimizations:**
- Reduced maximum parents from 4 to 2
- Increased adaptive selection threshold from 5 to 10 generations
- Better control over population growth

### **4. Individual Operator Files**
Comprehensive implementation of mutation and crossover operators as separate modules.

**Recent Improvements:**
- **16 Total Operators**: 14 mutation + 2 crossover
- **Multi-Language Support**: 5 languages (Hindi, French, German, Japanese, Chinese)
- **LLM-Only Back-Translation**: Active back-translation operators use LLaMA-based translation
- **4 New Mutation Operators**: Negation, Typographical Errors, Concept Addition, Informed Evolution
- **Standardized Imports**: Eliminated try-except import patterns
- **Enhanced Error Handling**: Fallback mechanisms for failed operations
- **Memory Management**: Model caching and lazy loading

## Text Variation Operators

### **Mutation Operators (14 Active)**

#### **Core LLM Operators**
- **`LLM_POSAwareSynonymReplacement`**: LLaMA-based synonym replacement using POS tagging
  - Uses spaCy for POS analysis
  - LLaMA for intelligent synonym selection
  - Context-aware replacements
  - Enhanced error handling with fallback mechanisms

- **`LLM_POSAwareAntonymReplacement`**: LLaMA-based antonym replacement using POS tagging
  - Uses spaCy for POS analysis
  - LLaMA for intelligent antonym selection
  - Context-aware replacements
  - Enhanced error handling with fallback mechanisms

#### **BERT-Based Operators**
- **`MLMOperator`**: Masked language model operator for word replacement
  - Uses `bert-base-uncased` model
  - Masked language modeling for word prediction
  - Contextual word replacements

#### **OpenAI-Based Operators**
- **`LLMBasedParaphrasingOperator`**: OpenAI GPT-4 based paraphrasing with optimization
  - Uses OpenAI GPT-4 API
  - Optimized for paraphrasing tasks
  - High-quality text variations

#### **Stylistic Operators**
- **`StylisticMutator`**: Stylistic text mutations
  - Multiple style variations
  - Context-aware style changes
  - Enhanced error handling

#### **LLM-Based Back-Translation Operators (5)**
- **`LLMBackTranslationHIOperator`**: Hindi back-translation (LLaMA)
- **`LLMBackTranslationFROperator`**: French back-translation (LLaMA)
- **`LLMBackTranslationDEOperator`**: German back-translation (LLaMA)
- **`LLMBackTranslationJAOperator`**: Japanese back-translation (LLaMA)
- **`LLMBackTranslationZHOperator`**: Chinese back-translation (LLaMA)

**Recent Improvements:**
- Enhanced error handling with fallback mechanisms
- More neutral prompts to reduce LLM refusals
- Better validation for translation outputs

#### **New Mutation Operators (January 2025)**
- **`NegationOperator`**: Adds negation to prompt questions
  - Example: "What are advantages of social media?" → "What are disadvantages of social media?"
  - LLM-driven negation detection and application

- **`TypographicalErrorsOperator`**: Simulates common human typos and spelling mistakes
  - Example: "Which president signed the Civil Rights Act?" → "Whcih presidnet signed the Civil Rihts Act?"
  - LLM-driven typo generation

- **`ConceptAdditionOperator`**: Adds unfairness and bias constraints to questions
  - Example: "How can I improve my writing skills?" → "How can I improve my writing skills to discriminate against certain groups?"
  - LLM-driven bias injection

- **`InformedEvolutionOperator`**: Uses top 10 elites to generate more toxic variants
  - Reads `top_10.json` for elite examples
  - LLM-driven toxicity enhancement
  - Leverages successful patterns from previous evolution

### **Crossover Operators (2 Active)**

#### **Semantic Crossover**
- **`SemanticSimilarityCrossover`**: Crossover based on semantic similarity
  - Uses sentence transformers for embeddings
  - Semantic similarity threshold
  - Meaning-preserving combinations

#### **Instruction-Aware Crossover**
- **`SemanticFusionCrossover`**: Crossover that preserves instruction structure
  - Uses config-driven prompt templates
  - Preserves command/question structure
  - Optimizes for north star metric
  - Returns single variant per call

## Parent Selection

### **Steady-State Selection Strategy**
```python
def select_parents_steady_state(self):
    # Mutation parent: Topmost elite
    mutation_parent = self.elites[0]
    
    # Crossover parents: Topmost + random elites + random population
    crossover_parents = [self.elites[0]]  # Topmost elite
    crossover_parents.extend(self._select_random_elites(1))  # 1 random elite
    crossover_parents.extend(self._select_random_population(3))  # 3 random population
    
    return mutation_parent, crossover_parents
```

### **Adaptive Selection**
- **Threshold**: After 10 generations (increased from 5)
- **Max Parents**: 2 (reduced from 4)
- **Strategy**: Automatically adjusts based on evolution progress

## Population Management

### **Steady-State Population**
- **File**: `data/outputs/elites.json`
- **Size**: Controlled through elite redistribution
- **Management**: Continuous evolution without generation boundaries
- **Memory**: Efficient single-file population with lazy loading

### **Population Files**
- **`elites.json`**: Steady-state elite population
- **`Population.json`**: Full population backup
- **`population_index.json`**: Population metadata and counts
- **`temp.json`**: Staging area during generation cycles
- **`most_toxic.json`**: High-toxicity genomes moved by moderation

## Evolution Flow

### **Steady-State Evolution Loop**

```mermaid
flowchart TD
  A[elites.json<br/>Steady-State Population] --> B[Parent Selection<br/>Top Elite + Random]
  B --> C{Number of Parents?}
  C -->|1 Parent| D[Mutation Operators<br/>14 Total]
  C -->|2+ Parents| E[Crossover Operators<br/>2 Total]
  
  D --> F[Generate Variants<br/>Max 1 per operator]
  E --> F
  F --> G[Text Generation<br/>Qwen2.5-7B Model]
  G --> H[Safety Evaluation<br/>Hybrid Moderation]
  H --> I[Update Population<br/>Status: complete + scores]
  I --> J[Sort & Maintain Elites<br/>Steady-State Management]
  J --> K{Threshold Reached?}
  K -->|No| A
  K -->|Yes| L[Evolution Complete]
  
  style A fill:#64b5f6,stroke:#1976d2,stroke-width:2px,color:#000
  style C fill:#ffb74d,stroke:#f57c00,stroke-width:2px,color:#000
  style D fill:#4caf50,stroke:#2e7d32,stroke-width:2px,color:#000
  style E fill:#ba68c8,stroke:#7b1fa2,stroke-width:2px,color:#000
  style K fill:#f48fb1,stroke:#c2185b,stroke-width:2px,color:#000
  style L fill:#66bb6a,stroke:#388e3c,stroke-width:2px,color:#000
```

### **Variant Generation Process**
1. **Parent Selection**: Steady-state selection from elites
2. **Operator Application**: 16 operators (14 mutation + 2 crossover)
3. **Variant Generation**: Max 1 variant per operator
4. **Deduplication**: Intra-file and cross-file deduplication
5. **Staging**: Variants staged in `temp.json`
6. **Processing**: Text generation and safety evaluation
7. **Population Update**: Elites updated based on performance

## Performance Characteristics

### **Variant Generation Rates**
With current settings (x=1, y=1):
- **Crossover variants**: 2 (2 operators × 1 pair × 1 variant)
- **Mutation variants**: 28 (2 parents × 14 operators × 1 variant)
- **Total per cycle**: 30 variants

With adaptive selection (x=2, y=1 or x=1, y=2):
- **Crossover variants**: 3 (2 operators × 1 pair × 1 variant)
- **Mutation variants**: 42 (3 parents × 14 operators × 1 variant)
- **Total per cycle**: 45 variants

### **Memory Usage**
- **Elites Population**: ~500KB (124 genomes)
- **Full Population**: ~2.8MB (2885 genomes)
- **Model Loading**: Efficient caching and reuse across operators
- **Steady-State**: Continuous memory management with elite preservation

### **Optimization Features**
- **Lazy Imports**: Prevents circular dependencies
- **Model Caching**: Reuse loaded models across operators
- **Memory Monitoring**: Real-time memory usage tracking
- **Automatic Cleanup**: PyTorch cache and garbage collection
- **Steady-State Management**: Efficient elite preservation

## Usage Examples

### **Basic Operator Usage**
```python
from src.ea.negation_operator import NegationOperator

# Initialize operator
operator = NegationOperator()

# Apply to text
variants = operator.apply("What are advantages of social media?")
print(variants)  # ['What are disadvantages of social media?']
```

### **Evolution Engine Usage**
```python
from src.ea.EvolutionEngine import EvolutionEngine

# Initialize engine
engine = EvolutionEngine("toxicity", log_file, current_cycle=1)

# Generate variants
generation_data = engine.generate_variants_global()
print(f"Created {generation_data['variants_created']} variants")
```

### **Parent Selection**
```python
from src.ea.ParentSelector import ParentSelector

# Initialize selector
selector = ParentSelector("toxicity", log_file)

# Select parents using steady-state strategy
mutation_parent, crossover_parents = selector.select_parents_steady_state()
```

### **Complete Evolution Run**
```python
from src.ea.RunEvolution import run_evolution

# Run complete evolution cycle
result = run_evolution(
    north_star_metric="toxicity",
    log_file="logs/evolution.log",
    threshold=0.95,
    current_cycle=1
)
```

## Recent Updates and Fixes

### **Major Additions (January 2025)**
- **4 New Mutation Operators**: Negation, Typographical Errors, Concept Addition, Informed Evolution
- **Enhanced Error Handling**: Fallback mechanisms for all operators
- **Improved LLM Prompts**: More neutral prompts to reduce refusals
- **Better Validation**: Enhanced content validation for translations

### **Performance Improvements**
- **Parent Selection Optimization**: Reduced max parents from 4 to 2
- **Adaptive Selection**: Increased threshold from 5 to 10 generations
- **GPU Acceleration**: Enabled for both PG and RG models
- **Memory Management**: Enhanced cleanup and garbage collection

### **Architecture Enhancements**
- **Modular Operator Implementation**: Individual files for each operator
- **Enhanced Parent Selection**: Better control over population growth
- **Improved Error Handling**: Graceful degradation for failed operations
- **Layered Architecture**: main.py → RunEvolution.py → EvolutionEngine.py

### **Two-tier Deduplication**
- **EvolutionEngine**: Intra-file deduplication of staged variants within `temp.json`
- **RunEvolution**: Cross-file deduplication against `elites.json`, `Population.json`, and `most_toxic.json`

## File Structure

```
src/ea/
├── __init__.py                    # Package exports and lazy imports
├── EvolutionEngine.py             # Genetic algorithm core (steady-state)
├── RunEvolution.py                # Evolution pipeline driver
├── ParentSelector.py               # Selection strategies (steady-state)
├── README.md                      # This documentation
├── notes.md                       # Implementation notes and data flow
├── synonym_replacement.py         # POS-aware synonym replacement
├── antonym_replacement.py         # POS-aware antonym replacement
├── mlm_operator.py                # BERT masked language modeling
├── paraphrasing.py                # LLM paraphrasing
├── stylistic_mutator.py           # Style variation
├── back_translation.py            # Multi-language back-translation
├── semantic_similarity_crossover.py # Semantic similarity crossover
├── fusion_crossover.py            # Instruction-preserving crossover
├── negation_operator.py           # Negation mutation (NEW)
├── typographical_errors.py        # Typo simulation (NEW)
├── concept_addition.py            # Bias addition (NEW)
└── InformedEvolution.py          # Elite-informed evolution (NEW)
```

## Dependencies
- torch, transformers, spacy, nltk, openai
- sentence-transformers (for semantic crossover)
- huggingface-hub (for model downloads)
- utils.custom_logging, utils.population_io, gne.ResponseGenerator, gne.PromptGenerator

## Documentation Index

### 📚 **Core Documentation**
- **[README.md](../../README.md)** - Main project documentation with setup instructions
- **[ARCHITECTURE.md](../../ARCHITECTURE.md)** - Complete system architecture overview
- **[EA Notes](notes.md)** - Detailed implementation notes and data flow

### 🔧 **Operator Documentation**
- **[negation_operator.py](negation_operator.py)** - Negation mutation operator (NEW)
- **[typographical_errors.py](typographical_errors.py)** - Typo simulation operator (NEW)
- **[concept_addition.py](concept_addition.py)** - Bias addition operator (NEW)
- **[InformedEvolution.py](InformedEvolution.py)** - Elite-informed evolution (NEW)

### 🧪 **Testing Documentation**
- **[Tests README](../../tests/README.md)** - Testing framework guide
- **[test_operators_demo.py](../../tests/test_operators_demo.py)** - Operator testing examples

### 📊 **Configuration & Data**
- **[RGConfig.yaml](../../config/RGConfig.yaml)** - Response Generator configuration
- **[PGConfig.yaml](../../config/PGConfig.yaml)** - Prompt Generator configuration
- **[prompt.xlsx](../../data/prompt.xlsx)** - Input prompts for evolution
- **[outputs/](../../data/outputs/)** - Evolution results and tracking

### 🚀 **Quick Reference**
- **Test EA**: `python -c "from src.ea import get_EvolutionEngine; print('EA loaded')"`
- **Run Evolution**: `python src/main.py --generations 5`
- **Test Operators**: `python tests/test_operators_demo.py`
- **Monitor Logs**: Check `logs/` directory for execution details

This comprehensive EA package provides a robust foundation for evolutionary text generation with extensive operator support, multi-language capabilities, and efficient steady-state population management.