# Evolutionary Algorithms (EA) Package

The Evolutionary Algorithms package provides the core genetic algorithm functionality for evolving text prompts through mutation and crossover operations. This package implements a complete evolutionary framework with parent selection, variation operators, and generation management.

## 🧬 **Core Components**

### **1. EvolutionEngine** (`EvolutionEngine.py`)
The main orchestrator for the evolutionary process.

**Key Features:**
- Manages genome populations and generation cycles
- Coordinates parent selection and variant generation
- Handles evolution tracking and metadata
- Single-file population persistence (`outputs/Population.json`)
- Memory-optimized for large populations

**Main Methods:**
```python
# Initialize evolution engine
engine = EvolutionEngine(north_star_metric, log_file, current_cycle)

# Generate variants for a specific prompt
variants = engine.generate_variants(prompt_id)

# Update genome IDs
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
- Single-file population loading and in-memory filtering

**Main Functions:**
```python
# Run complete evolution cycle
run_evolution(north_star_metric, log_file, threshold=0.95, current_cycle=None)

# Check threshold achievement
check_threshold_and_update_tracker(population, north_star_metric, logger, threshold=0.95)

# Get pending prompt IDs
pending_ids = get_pending_prompt_ids(evolution_tracker, logger)

# Update evolution tracker with generation data
update_evolution_tracker_with_generation(prompt_id, generation_data, evolution_tracker, logger)
```

**Recent Fixes:**
- Accurate `max_score` per generation (children, not parent)
- Single-file population I/O
- Lazy imports and efficient data loading

### **3. ParentSelector** (`ParentSelector.py`)
Intelligent parent selection strategies for genetic operations.

**Selection Strategies:**
- Single Genome
- Small Population (2–4)
- Large Population (5+)
- Tournament Selection
- Roulette Selection

### **4. TextVariationOperators** (`TextVariationOperators.py`)
Concrete implementation of mutation and crossover operators.

**Recent Improvements:**
- Lazy Initialization
- Memory Management
- Error Handling
- Variant cap tuned to reduce growth

## 🔄 **Variation Operators**

### **Mutation Operators** (Single Parent)
- POSAwareSynonymReplacement (spaCy + BERT MLM)
- BertMLMOperator (BERT MLM)
- LLMBasedParaphrasingOperator (LLM paraphrasing)
- BackTranslationOperator (EN ↔ X ↔ EN)

### **Crossover Operators** (Multiple Parents)
- OnePointCrossover
- SemanticSimilarityCrossover
- InstructionPreservingCrossover

## 📊 **Operator Selection Logic**

```python
def get_applicable_operators(num_parents: int, north_star_metric, log_file=None):
    if num_parents == 1:
        return get_single_parent_operators(north_star_metric, log_file)
    return get_multi_parent_operators(log_file)
```

## 🔧 **Configuration**

```python
# Maximum variants per operator (tuned)
max_variants = 5  # Reduced from 10 to control growth
```

## 📈 **Evolution Flow**

1) Population Initialization → `Population.json`
2) Parent Selection → per-prompt parents
3) Variant Generation → capped variants, deduplicated
4) Tracker Update → per-generation best score, counts

## 📁 **File Structure**

```
src/ea/
├── __init__.py
├── EvolutionEngine.py
├── RunEvolution.py
├── ParentSelector.py
├── TextVariationOperators.py
└── VariationOperators.py
```

## 🔗 **Dependencies**
- torch, transformers, spacy, nltk, openai
- utils.custom_logging, utils.population_io, gne.LLaMaTextGenerator

## 📝 **Recent Updates and Fixes**
- `max_score` correctness per generation
- Single-file population with indexing
- Lazy imports; robust path handling 