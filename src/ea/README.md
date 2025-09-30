# Evolutionary Algorithms (EA) Package

The Evolutionary Algorithms package provides the core genetic algorithm functionality for evolving text prompts through mutation and crossover operations. This package implements a complete evolutionary framework with parent selection, variation operators, and steady-state population management.

## 🧬 **Core Components**

### **1. EvolutionEngine** (`EvolutionEngine.py`)
The main orchestrator for the evolutionary process.

**Key Features:**
- Manages genome populations and evolution cycles (steady-state evolution)
- Coordinates parent selection and variant generation
- Tracks variant counts and integrates deduplication
- Steady-state population persistence (`outputs/elites.json`)
- Memory-optimized for large populations
- Supports 16 text variation operators

**Main Methods (excerpt):**
```python
# Initialize evolution engine
engine = EvolutionEngine(north_star_metric, log_file, current_cycle)

# Generate variants globally for the current cycle
generation_data = engine.generate_variants_global()

# Keep IDs consistent with current population
engine.update_next_id()
```

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

### **4. TextVariationOperators** (`TextVariationOperators.py`)
Comprehensive implementation of mutation and crossover operators.

**Recent Improvements:**
- **16 Total Operators**: 13 mutation + 3 crossover
- **Multi-Language Support**: 5 languages (Hindi, French, German, Japanese, Chinese)
- **Dual Translation Approaches**: Helsinki-NLP models + LLaMA-based translation
- Lazy Initialization
- Memory Management
- Error Handling
- Variant cap tuned to reduce growth

## 🔄 **Variation Operators**

### **Mutation Operators (13)**

#### **Core LLM Operators**
- **`LLM_POSAwareSynonymReplacement`**: LLaMA-based synonym replacement using POS tagging
  - Uses spaCy for POS analysis
  - LLaMA for intelligent synonym selection
  - Context-aware replacements

#### **BERT-Based Operators**
- **`BertMLMOperator`**: BERT masked language model for word replacement
  - Uses `bert-base-uncased` model
  - Masked language modeling for word prediction
  - Contextual word replacements

#### **OpenAI-Based Operators**
- **`LLMBasedParaphrasingOperator`**: OpenAI GPT-4 based paraphrasing with optimization
  - Uses OpenAI GPT-4 API
  - Optimized for paraphrasing tasks
  - High-quality text variations

#### **Model-Based Back-Translation Operators (5)**
- **`BackTranslationHIOperator`**: Hindi back-translation (Helsinki-NLP)
- **`BackTranslationFROperator`**: French back-translation (Helsinki-NLP)
- **`BackTranslationDEOperator`**: German back-translation (Helsinki-NLP)
- **`BackTranslationJAOperator`**: Japanese back-translation (Helsinki-NLP)
- **`BackTranslationZHOperator`**: Chinese back-translation (Helsinki-NLP)

#### **LLM-Based Back-Translation Operators (5)**
- **`LLMBackTranslationHIOperator`**: Hindi back-translation (LLaMA)
- **`LLMBackTranslationFROperator`**: French back-translation (LLaMA)
- **`LLMBackTranslationDEOperator`**: German back-translation (LLaMA)
- **`LLMBackTranslationJAOperator`**: Japanese back-translation (LLaMA)
- **`LLMBackTranslationZHOperator`**: Chinese back-translation (LLaMA)

### **Crossover Operators (3)**

#### **Structural Crossover**
- **`OnePointCrossover`**: Single-point crossover between two prompts
  - Random split point selection
  - Preserves prompt structure
  - Simple but effective

#### **Semantic Crossover**
- **`SemanticSimilarityCrossover`**: Crossover based on semantic similarity
  - Uses sentence transformers for embeddings
  - Semantic similarity threshold
  - Meaning-preserving combinations

#### **Instruction-Aware Crossover**
- **`InstructionPreservingCrossover`**: Crossover that preserves instruction structure
  - Identifies instruction patterns
  - Preserves command/question structure
  - Maintains semantic coherence

## 📊 **Operator Selection Logic**

```python
def get_applicable_operators(num_parents: int, north_star_metric, log_file=None):
    if num_parents == 1:
        return get_single_parent_operators(north_star_metric, log_file)  # 13 mutation operators
    return get_multi_parent_operators(log_file)  # 3 crossover operators
```

## 🔧 **Configuration**

```python
# Maximum variants per operator (tuned)
max_variants = 5  # Reduced from 10 to control growth

# Back-translation configuration
target_languages = ['Hindi', 'French', 'German', 'Japanese', 'Chinese']
translation_methods = ['model_based', 'llm_based']  # Dual approaches
```

## 📈 **Evolution Flow**

1) **Population Initialization** → `elites.json`
2) **Parent Selection** → Steady-state selection from elites
3) **Variant Generation** → 16 operators, capped variants, deduplicated
4) **Tracker Update** → Per-generation best score, counts, parent information

### Steady-State Loop (Architecture)

```mermaid
flowchart LR
  A[elites.json (working set)] --> B[Parent selection]
  B --> C[Mutation (13) / Crossover (3)]
  C --> D[Variants (pending_generation)]
  D --> E[Generation (LLaMA)]
  E --> F[Evaluation (Hybrid Moderation)]
  F --> G[Status: complete + scores]
  G --> H[Sort & maintain elites]
  H --> A
```

## 🌍 **Multi-Language Support**

### **Supported Languages**
- **Hindi (HI)**: Devanagari script support
- **French (FR)**: Romance language variations
- **German (DE)**: Germanic language structure
- **Japanese (JA)**: Complex script handling
- **Chinese (ZH)**: Character-based translation

### **Translation Approaches**
1. **Model-Based**: Helsinki-NLP translation models
   - Specialized translation models
   - Fast and efficient
   - Reliable for common language pairs

2. **LLM-Based**: LLaMA model translation
   - Context-aware translation
   - Better handling of complex sentences
   - More natural output

## 📁 **File Structure**

```
src/ea/
├── __init__.py
├── EvolutionEngine.py
├── RunEvolution.py
├── ParentSelector.py
├── TextVariationOperators.py
├── VariationOperators.py
└── README.md
```

## 🔗 **Dependencies**
- torch, transformers, spacy, nltk, openai
- sentence-transformers (for semantic crossover)
- huggingface-hub (for model downloads)
- utils.custom_logging, utils.population_io, gne.LLaMaTextGenerator

## 📝 **Recent Updates and Fixes**

### **Major Additions**
- **16 Text Variation Operators**: Complete operator suite
- **Multi-Language Back-Translation**: 5 languages with dual approaches
- **Steady-State Population Management**: Elite preservation and continuous evolution
- **Task-Specific Templates**: Configurable prompts for different tasks
- **Enhanced Memory Management**: Model caching and lazy loading

### **Performance Improvements**
- `max_score` correctness per generation
- Steady-state population with indexing
- Lazy imports; robust path handling
- Model reuse across operators
- Adaptive batch sizing

### **Architecture Enhancements**
- Dual translation approaches for better coverage
- Comprehensive operator selection logic
- Enhanced parent selection strategies
- Improved error handling and recovery

## 🚀 **Usage Examples**

### **Basic Operator Usage**
```python
from src.ea.TextVariationOperators import LLMBackTranslationHIOperator

# Initialize operator
operator = LLMBackTranslationHIOperator()

# Apply to text
variants = operator.apply("Hello world")
print(variants)  # ['नमस्ते दुनिया'] (Hindi translation)
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

This comprehensive EA package provides a robust foundation for evolutionary text generation with extensive operator support, multi-language capabilities, and efficient steady-state population management.