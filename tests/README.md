# Test Suite for Separated Operator Architecture

This directory contains comprehensive test suites for the newly refactored operator architecture. The monolithic `TextVariationOperators.py` file has been broken down into individual operator files for better maintainability and testing.

## Test Files Overview

### 🧬 Mutation Operator Tests
- **`test_mutation_operators.py`** - Tests individual mutation operators
  - POS-aware synonym replacement
  - BERT MLM word replacement
  - LLM paraphrasing
  - Back-translation operators
  - Error handling and edge cases

### 🔄 Crossover Operator Tests
- **`test_crossover_operators.py`** - Tests individual crossover operators
  - One-point crossover
  - Semantic similarity crossover
  - Instruction-preserving crossover
  - Error handling for edge cases

### 🌍 Back-Translation Tests
- **`test_back_translation_operators.py`** - Tests back-translation operators
  - Helsinki-NLP model-based operators (5 languages)
  - LLaMA-based operators (5 languages)
  - Inheritance testing for base classes

### 🛠️ Helper Function Tests
- **`test_operator_helpers.py`** - Tests utility functions
  - `get_single_parent_operators()`
  - `get_multi_parent_operators()`
  - `get_applicable_operators()`
  - `limit_variants()`
  - `get_generator()`

### 📊 Comprehensive Demo
- **`test_all_operators_demo.py`** - Complete demonstration
  - Shows all operators working with new architecture
  - Compares old vs new approaches
  - Performance and functionality verification

### 🔧 Updated Demo
- **`test_operators_demo.py`** - Updated original demo file
  - Uses new modular imports
  - Demonstrates functionality improvements
  - Shows architecture benefits

### 🧪 Pytest Suite
- **`pytest_operators.py`** - Comprehensive pytest test suite
  - Structured test classes
  - Fixtures for test data
  - Comprehensive assertions
  - Parametrized tests

## Running Tests

### Individual Test Files
```bash
# Run mutation operator tests
python tests/test_mutation_operators.py

# Run crossover operator tests
python tests/test_crossover_operators.py

# Run comprehensive demo
python tests/test_all_operators_demo.py
```

### Pytest Suite
```bash
# Run all pytest tests
python -m pytest tests/pytest_operators.py

# Run with verbose output
python -m pytest tests/pytest_operators.py -v

# Run specific test class
python -m pytest tests/pytest_operators.py::TestMutationOperatorsSeparated -v
```

### Quick Demo
```bash
# Run the updated demo
python tests/test_operators_demo.py
```

## New Architecture Benefits

### ✅ Separation of Concerns
- Each operator in its own file (`~100-200 lines`)
- Clear inheritance hierarchies
- Modular imports

### ✅ Improved Testability
- Individual operator testing
- Mock-friendly architecture
- Isolated test cases

### ✅ Better Maintainability
- Easy to locate and modify specific operators
- Reduced coupling between components
- Clear file organization

### ✅ Enhanced Performance
- Selective imports (only load what you need)
- Reduced memory footprint
- Faster startup times

## File Structure

```
tests/
├── README.md                        # This documentation
├── test_mutation_operators.py       # Mutation operator tests
├── test_crossover_operators.py      # Crossover operator tests
├── test_back_translation_operators.py # Back-translation tests
├── test_operator_helpers.py        # Helper function tests
├── test_all_operators_demo.py      # Comprehensive demo
├── test_operators_demo.py          # Updated original demo
└── pytest_operators.py              # Pytest test suite
```

## Operator Files Structure

```
src/ea/
├── operator_helpers.py              # Helper functions
├── base_operators.py               # Base classes
├── pos_aware_synonym_replacement.py # POS-aware operator
├── bert_mlm_operator.py            # BERT MLM operator
├── llm_paraphrasing_operator.py    # LLM paraphrasing
├── back_translation_french.py      # French back-translation
├── back_translation_german.py      # German back-translation
├── back_translation_japanese.py    # Japanese back-translation
├── back_translation_chinese.py      # Chinese back-translation
├── back_translation_hindi.py        # Hindi back-translation
├── llm_back_translation_french.py  # LLaMA French back-translation
├── llm_back_translation_german.py  # LLaMA German back-translation
├── llm_back_translation_japanese.py # LLaMA Japanese back-translation
├── llm_back_translation_chinese.py  # LLaMA Chinese back-translation
├── llm_back_translation_hindi.py    # LLaMA Hindi back-translation
├── one_point_crossover.py          # One-point crossover
├── semantic_similarity_crossover.py # Semantic similarity crossover
└—— instruction_preserving_crossover.py # Instruction-preserving crossover
```

## Test Coverage

- ✅ **Import Testing** - All operators can be imported from their files
- ✅ **Instantiation Testing** - All operators instantiate correctly
- ✅ **Functionality Testing** - Operators produce expected outputs
- ✅ **Error Handling** - Edge cases and error conditions handled
- ✅ **Inheritance Testing** - Proper inheritance from base classes
- ✅ **Helper Function Testing** - Utility functions work correctly
- ✅ **Integration Testing** - All components work together

## Dependencies

The tests assume the following are available:
- Required Python packages (torch, transformers, etc.)
- LLaMA model in `models/` directory
- Helsinki-NLP models (downloaded automatically)
- OpenAI API key (for LLM operators)
- NLTK punkt tokenizer (downloaded automatically)

## Notes

- Some tests may show warnings about model weights - this is expected
- Model loading may take time on first run
- OpenAI API key required for paraphrasing and instruction-preserving crossover tests
- Back-translation models download automatically on first use

## Contributing

When adding new operators:
1. Create new operator file following naming convention
2. Add corresponding test file
3. Update `operator_helpers.py` with new operators
4. Add tests to appropriate test suite
5. Update documentation
