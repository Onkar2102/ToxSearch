# Evolutionary Algorithms (EA) Package

The Evolutionary Algorithms package provides the core genetic algorithm functionality for evolving text prompts through mutation and crossover operations. This package implements a complete evolutionary framework with parent selection, variation operators, and generation management.

## 🧬 **Core Components**

### **1. EvolutionEngine** (`EvolutionEngine.py`)
The main orchestrator for the evolutionary process.

**Key Features:**
- Manages genome populations and generation cycles
- Coordinates parent selection and variant generation
- Handles evolution tracking and metadata
- Provides generation file management

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

**Main Functions:**
```python
# Run complete evolution cycle
run_evolution(north_star_metric, log_file, threshold=0.95, current_cycle=None)

# Check threshold achievement
check_threshold_and_update_tracker(population, north_star_metric, logger, threshold=0.95)

# Get pending prompt IDs
pending_ids = get_pending_prompt_ids(evolution_tracker, logger)
```

### **3. ParentSelector** (`ParentSelector.py`)
Intelligent parent selection strategies for genetic operations.

**Selection Strategies:**
- **Single Genome**: When only one genome available
- **Small Population**: 2-4 genomes (uses top 3 for crossover)
- **Large Population**: 5+ genomes (uses top 3 for crossover)
- **Tournament Selection**: Random tournament-based selection
- **Roulette Selection**: Fitness-proportional selection

**Usage:**
```python
selector = ParentSelector(north_star_metric, log_file)

# Select parents for genetic operations
mutation_parent, crossover_parents = selector.select_parents(prompt_genomes, prompt_id)

# Tournament selection
mutation_parent, crossover_parents = selector.select_tournament_parents(prompt_genomes, tournament_size=3)
```

### **4. TextVariationOperators** (`TextVariationOperators.py`)
Concrete implementation of mutation and crossover operators.

## 🔄 **Variation Operators**

### **Mutation Operators** (Single Parent)

#### **1. POSAwareSynonymReplacement**
- **Type**: BERT-based synonym replacement
- **Strategy**: Uses spaCy POS tagging + BERT MLM
- **Target**: Adjectives, Verbs, Nouns, Adverbs
- **Process**: Masks words and predicts replacements using BERT

#### **2. BertMLMOperator**
- **Type**: BERT Masked Language Model
- **Strategy**: Random word masking and prediction
- **Process**: Masks random words and uses BERT to predict alternatives

#### **3. LLMBasedParaphrasingOperator**
- **Type**: LLM-based paraphrasing
- **Strategy**: Uses LLaMA model for intelligent paraphrasing
- **Process**: Generates multiple paraphrased versions of input text

#### **4. BackTranslationOperator**
- **Type**: Back-translation through multiple languages
- **Strategy**: English → Foreign → English translation chain
- **Languages**: English, French, German, Spanish, Chinese
- **Process**: Translates to intermediate language and back to English

### **Crossover Operators** (Multiple Parents)

#### **1. OnePointCrossover**
- **Type**: Traditional genetic algorithm crossover
- **Strategy**: Single crossover point between two parents
- **Process**: Splits parents at random point and combines segments

#### **2. SemanticSimilarityCrossover**
- **Type**: Semantics-aware crossover
- **Strategy**: Uses sentence embeddings for semantic similarity
- **Process**: Combines semantically similar segments from parents

#### **3. InstructionPreservingCrossover**
- **Type**: Instruction-aware crossover
- **Strategy**: Preserves instruction structure while varying content
- **Process**: Maintains instruction format while combining content

## 📊 **Operator Selection Logic**

### **Single Parent Operators** (Mutation)
```python
def get_single_parent_operators(north_star_metric, log_file=None):
    return [
        POSAwareSynonymReplacement(log_file),
        BertMLMOperator(log_file),
        LLMBasedParaphrasingOperator(north_star_metric, log_file),
        BackTranslationOperator(log_file)
    ]
```

### **Multi-Parent Operators** (Crossover)
```python
def get_multi_parent_operators(log_file=None):
    return [
        OnePointCrossover(log_file),
        SemanticSimilarityCrossover(log_file),
        InstructionPreservingCrossover(log_file)
    ]
```

### **Dynamic Operator Selection**
```python
def get_applicable_operators(num_parents: int, north_star_metric, log_file=None):
    if num_parents == 1:
        return get_single_parent_operators(north_star_metric, log_file)
    elif num_parents >= 2:
        return get_multi_parent_operators(log_file)
    else:
        return []
```

## 🔧 **Configuration**

### **Evolution Parameters**
```python
# North star metric for fitness evaluation
north_star_metric = "toxicity"  # or "safety", "harmfulness", etc.

# Threshold for stopping evolution
threshold = 0.95  # Stop when score >= 0.95

# Generation cycle number
current_cycle = 1  # Current evolution generation
```

### **Operator Configuration**
```python
# Maximum variants per operator
max_variants = 10  # Limit variants to prevent explosion

# Logging configuration
log_file = "logs/evolution.log"

# Performance monitoring
with PerformanceLogger(logger, "Operator Application"):
    variants = operator.apply(text)
```

## 📈 **Evolution Flow**

### **1. Population Initialization**
```python
# Load initial population
population = load_and_initialize_population("data/prompt.xlsx", "outputs/Population.json")
```

### **2. Parent Selection**
```python
# Select parents for each prompt
for prompt_id in pending_prompt_ids:
    mutation_parent, crossover_parents = parent_selector.select_parents(prompt_genomes, prompt_id)
```

### **3. Variant Generation**
```python
# Generate variants using selected operators
operators = get_applicable_operators(len(parents), north_star_metric)
for operator in operators:
    variants = operator.apply(parent_text)
    new_genomes.extend(create_genomes_from_variants(variants))
```

### **4. Evolution Tracking**
```python
# Update evolution tracker
update_evolution_tracker_with_generation(prompt_id, generation_data, evolution_tracker, logger)

# Check completion criteria
check_threshold_and_update_tracker(population, north_star_metric, logger, threshold)
```

## 📊 **Performance Monitoring**

### **Memory Management**
- **Adaptive batch sizing** based on available memory
- **Automatic cleanup** after operator application
- **Real-time monitoring** with configurable thresholds

### **Timeout Protection**
- **5-minute timeout** for model loading operations
- **CPU fallback** if GPU operations hang
- **Graceful error handling** with detailed logging

### **Progress Tracking**
```python
# Generation statistics
generation_data = {
    "generation_number": current_cycle,
    "parents": parent_info,
    "variants_created": total_variants,
    "mutation_variants": mutation_count,
    "crossover_variants": crossover_count
}
```

## 🔍 **Debugging and Logging**

### **Comprehensive Logging**
```python
logger = get_logger("EvolutionEngine", log_file)
logger.info(f"Generating variants for prompt_id={prompt_id}")
logger.debug(f"Selected mutation parent: {mutation_parent['id']}")
logger.warning(f"No crossover parents selected for prompt_id={prompt_id}")
```

### **Performance Profiling**
```python
with PerformanceLogger(logger, "Variant Generation"):
    variants = operator.apply(text)
```

## 🚀 **Usage Examples**

### **Basic Evolution Cycle**
```python
from ea.EvolutionEngine import EvolutionEngine
from ea.RunEvolution import run_evolution

# Run complete evolution
run_evolution("toxicity", "logs/evolution.log", threshold=0.95, current_cycle=1)
```

### **Custom Parent Selection**
```python
from ea.ParentSelector import ParentSelector

selector = ParentSelector("toxicity", "logs/selection.log")
mutation_parent, crossover_parents = selector.select_parents(genomes, prompt_id)
```

### **Operator Application**
```python
from ea.TextVariationOperators import POSAwareSynonymReplacement

operator = POSAwareSynonymReplacement("logs/operators.log")
variants = operator.apply("Your input text here")
```

## 📁 **File Structure**

```
src/ea/
├── __init__.py                 # Package initialization and exports
├── EvolutionEngine.py          # Core evolution orchestrator
├── RunEvolution.py             # Evolution pipeline driver
├── ParentSelector.py           # Parent selection strategies
├── TextVariationOperators.py   # Concrete variation operators
└── VariationOperators.py       # Base operator classes
```

## 🔗 **Dependencies**

### **Core Dependencies**
- `torch` - PyTorch for BERT models
- `transformers` - Hugging Face transformers
- `spacy` - Natural language processing
- `nltk` - WordNet for synonyms
- `openai` - OpenAI API for LLM operations

### **Internal Dependencies**
- `utils.custom_logging` - Logging utilities
- `utils.population_io` - Population management
- `gne.LLaMaTextGenerator` - LLaMA model integration

## 📝 **Notes**

- **Thread Safety**: Operators are designed to be thread-safe
- **Memory Efficiency**: Automatic cleanup prevents memory leaks
- **Error Recovery**: Graceful handling of operator failures
- **Extensibility**: Easy to add new variation operators
- **Monitoring**: Comprehensive logging and performance tracking

This package provides a complete evolutionary framework for text generation with robust error handling, performance monitoring, and extensible operator system. 