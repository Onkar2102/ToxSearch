# Test Suite for Evolutionary Text Generation Framework

This directory contains comprehensive test suites for the evolutionary text generation framework, including tests for all 16 text variation operators, evolution engine components, and integration tests.

## Table of Contents

- [Quick Setup](#quick-setup)
- [Test Files Overview](#test-files-overview)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [New Operators Testing](#new-operators-testing)
- [Performance Testing](#performance-testing)
- [Documentation Index](#documentation-index)

## Quick Setup

### Prerequisites
- **Python 3.8+** with virtual environment activated
- **Required packages**: See [requirements.txt](../requirements.txt)
- **API Keys**: OpenAI and Google Perspective API keys configured
- **Models**: Qwen2.5-7B-Instruct and Llama3.2-3B-Instruct models

### Installation
```bash
# From project root
cd /path/to/eost-cam-llm
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```

### Quick Test
```bash
# Test all operators
python tests/test_operators_demo.py

# Test new mutation operators
python tests/test_new_mutation_operators.py

# Run pytest suite
python -m pytest tests/ -v
```

## Test Files Overview

### 🧬 **Core Operator Tests**
- **`test_mutation_operators.py`** - Tests individual mutation operators
  - POS-aware synonym replacement
  - POS-aware antonym replacement
  - BERT MLM word replacement
  - LLM paraphrasing
  - Stylistic mutator
  - Back-translation operators (5 languages)
  - Error handling and edge cases

- **`test_crossover_operators.py`** - Tests individual crossover operators
  - Semantic similarity crossover
  - Semantic fusion crossover
  - Error handling for edge cases

### 🌍 **Back-Translation Tests**
- **`test_back_translation_operators.py`** - Tests back-translation operators
  - LLM-based operators (5 languages)
  - Inheritance testing for base classes
  - Translation quality validation

### 🆕 **New Operators Tests**
- **`test_new_mutation_operators.py`** - Tests the 4 new mutation operators
  - NegationOperator
  - TypographicalErrorsOperator
  - ConceptAdditionOperator
  - InformedEvolutionOperator

### 🛠️ **Helper Function Tests**
- **`test_operator_helpers.py`** - Tests utility functions
  - `get_single_parent_operators()`
  - `get_multi_parent_operators()`
  - `get_applicable_operators()`
  - `limit_variants()`
  - `get_generator()`

### 📊 **Comprehensive Demo**
- **`test_all_operators_demo.py`** - Complete demonstration
  - Shows all 16 operators working
  - Compares old vs new approaches
  - Performance and functionality verification

### 🔧 **Updated Demo**
- **`test_operators_demo.py`** - Updated original demo file
  - Uses new modular imports
  - Demonstrates functionality improvements
  - Shows architecture benefits

### 🧪 **Pytest Suite**
- **`pytest_operators.py`** - Comprehensive pytest test suite
  - Structured test classes
  - Fixtures for test data
  - Comprehensive assertions
  - Parametrized tests

### 🔍 **Import Tests**
- **`test_imports_only.py`** - Tests import functionality
  - Verifies all operators can be imported
  - Tests lazy import system
  - Validates package structure

### 🔢 **Token Verification**
- **`token_id_verification.py`** - Tests token ID consistency
  - Verifies tokenization consistency
  - Tests model compatibility
  - Validates text processing

## Running Tests

### Individual Test Files
```bash
# Run mutation operator tests
python tests/test_mutation_operators.py

# Run crossover operator tests
python tests/test_crossover_operators.py

# Run comprehensive demo
python tests/test_all_operators_demo.py

# Test new operators
python tests/test_new_mutation_operators.py
```

### Pytest Suite
```bash
# Run all pytest tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/pytest_operators.py::TestMutationOperatorsSeparated -v

# Run specific test file
python -m pytest tests/test_mutation_operators.py -v
```

### Quick Demo
```bash
# Run the updated demo
python tests/test_operators_demo.py

# Test imports only
python tests/test_imports_only.py
```

## Test Coverage

### ✅ **Operator Testing**
- **Import Testing** - All 16 operators can be imported from their files
- **Instantiation Testing** - All operators instantiate correctly
- **Functionality Testing** - Operators produce expected outputs
- **Error Handling** - Edge cases and error conditions handled
- **Inheritance Testing** - Proper inheritance from base classes

### ✅ **Integration Testing**
- **Helper Function Testing** - Utility functions work correctly
- **Integration Testing** - All components work together
- **Performance Testing** - Operators perform within expected time limits
- **Memory Testing** - Memory usage is within acceptable limits

### ✅ **New Operators Testing**
- **NegationOperator** - Tests negation detection and application
- **TypographicalErrorsOperator** - Tests typo generation
- **ConceptAdditionOperator** - Tests bias injection
- **InformedEvolutionOperator** - Tests elite-informed evolution

## New Operators Testing

### **NegationOperator Tests**
```python
def test_negation_operator():
    operator = NegationOperator()
    result = operator.apply("What are advantages of social media?")
    assert "disadvantages" in result[0].lower()
    assert "advantages" not in result[0].lower()
```

### **TypographicalErrorsOperator Tests**
```python
def test_typographical_errors_operator():
    operator = TypographicalErrorsOperator()
    result = operator.apply("Which president signed the Civil Rights Act?")
    # Should contain typos but maintain meaning
    assert len(result[0]) > 0
    assert "president" in result[0].lower() or "presidnet" in result[0].lower()
```

### **ConceptAdditionOperator Tests**
```python
def test_concept_addition_operator():
    operator = ConceptAdditionOperator()
    result = operator.apply("How can I improve my writing skills?")
    # Should add bias/unfairness concepts
    assert len(result[0]) > 0
    assert any(word in result[0].lower() for word in ["discriminate", "bias", "unfair"])
```

### **InformedEvolutionOperator Tests**
```python
def test_informed_evolution_operator():
    operator = InformedEvolutionOperator()
    result = operator.apply("What are the benefits of exercise?")
    # Should generate more toxic variants based on elite examples
    assert len(result[0]) > 0
    assert result[0] != "What are the benefits of exercise?"
```

## Performance Testing

### **Operator Performance**
- **Response Time** - Each operator should complete within reasonable time
- **Memory Usage** - Memory consumption should be within acceptable limits
- **Error Rate** - Error rate should be minimal (< 5%)

### **Integration Performance**
- **End-to-End Testing** - Complete evolution cycle performance
- **Memory Management** - Memory usage during long runs
- **GPU Utilization** - GPU usage for model operations

### **Load Testing**
- **Concurrent Operations** - Multiple operators running simultaneously
- **Large Population** - Performance with large population sizes
- **Long Runs** - Performance over extended periods

## Dependencies

The tests assume the following are available:
- Required Python packages (torch, transformers, etc.)
- Qwen2.5-7B-Instruct model in `models/` directory
- Llama3.2-3B-Instruct model in `models/` directory
- OpenAI API key (for LLM operators)
- Google Perspective API key (for evaluation)
- NLTK punkt tokenizer (downloaded automatically)
- spaCy en_core_web_sm model (downloaded automatically)

## Test Data

### **Sample Inputs**
- **Questions**: "What are advantages of social media?"
- **Statements**: "Which president signed the Civil Rights Act?"
- **Instructions**: "How can I improve my writing skills?"
- **Complex Text**: Multi-sentence prompts with various structures

### **Expected Outputs**
- **Variants**: Modified versions of input text
- **Quality**: Maintained meaning with appropriate changes
- **Format**: Consistent output format across operators

## Notes

- Some tests may show warnings about model weights - this is expected
- Model loading may take time on first run
- OpenAI API key required for paraphrasing and instruction-preserving crossover tests
- Google Perspective API key required for evaluation tests
- Back-translation models download automatically on first use
- GPU acceleration is enabled by default for better performance

## Contributing

When adding new operators:
1. Create new operator file following naming convention
2. Add corresponding test file
3. Update `operator_helpers.py` with new operators
4. Add tests to appropriate test suite
5. Update documentation
6. Ensure all tests pass

## File Structure

```
tests/
├── README.md                        # This documentation
├── conftest.py                      # Pytest configuration
├── test_mutation_operators.py       # Mutation operator tests
├── test_crossover_operators.py      # Crossover operator tests
├── test_back_translation_operators.py # Back-translation tests
├── test_operator_helpers.py        # Helper function tests
├── test_all_operators_demo.py      # Comprehensive demo
├── test_operators_demo.py          # Updated original demo
├── test_new_mutation_operators.py  # New operators testing
├── test_imports_only.py            # Import testing
├── token_id_verification.py        # Token verification
├── pytest_operators.py             # Pytest test suite
├── data/                           # Test data files
├── output/                         # Test output files
└── outputs/                        # Test result files
```

## Documentation Index

### 📚 **Core Documentation**
- **[README.md](../README.md)** - Main project documentation with setup instructions
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - Complete system architecture overview
- **[EA README](../src/ea/README.md)** - Evolutionary algorithms guide

### 🧪 **Test Documentation**
- **[test_operators_demo.py](test_operators_demo.py)** - Main operator testing demo
- **[test_new_mutation_operators.py](test_new_mutation_operators.py)** - New operators testing
- **[test_all_operators_demo.py](test_all_operators_demo.py)** - Comprehensive operator demo
- **[pytest_operators.py](pytest_operators.py)** - Pytest test suite

### 🔧 **Operator Documentation**
- **[negation_operator.py](../src/ea/negation_operator.py)** - Negation mutation operator (NEW)
- **[typographical_errors.py](../src/ea/typographical_errors.py)** - Typo simulation operator (NEW)
- **[concept_addition.py](../src/ea/concept_addition.py)** - Bias addition operator (NEW)
- **[InformedEvolution.py](../src/ea/InformedEvolution.py)** - Elite-informed evolution (NEW)

### 📊 **Configuration & Data**
- **[RGConfig.yaml](../config/RGConfig.yaml)** - Response Generator configuration
- **[PGConfig.yaml](../config/PGConfig.yaml)** - Prompt Generator configuration
- **[prompt.xlsx](../data/prompt.xlsx)** - Input prompts for evolution
- **[outputs/](../data/outputs/)** - Evolution results and tracking

### 🚀 **Quick Reference**
- **Test All**: `python tests/test_operators_demo.py`
- **Test New**: `python tests/test_new_mutation_operators.py`
- **Pytest**: `python -m pytest tests/ -v`
- **Run Evolution**: `python src/main.py --generations 1`
- **Monitor Logs**: Check `logs/` directory for execution details

This comprehensive test suite ensures the reliability and functionality of all 16 text variation operators and the overall evolutionary framework.