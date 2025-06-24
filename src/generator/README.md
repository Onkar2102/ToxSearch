# Generator Package

> **📋 For comprehensive project overview, installation, and configuration details, see the [main README.md](../../README.md)**

The `generator` package provides text generation capabilities for the evolutionary text generation framework. This package implements LLaMA model integration with configurable generation parameters, batch processing, and performance optimization.

## Package Overview

### Core Responsibilities
- **Text Generation**: LLaMA model integration for controlled text generation
- **Batch Processing**: Efficient population-wide generation with configurable batch sizes
- **Performance Optimization**: Hardware-specific optimization and memory management
- **Factory Pattern**: Modular generator architecture for extensibility
- **Prompt Templating**: Role-based prompt formatting with user/assistant prefixes

### Architecture
```
generator/
├── LLaMaTextGenerator.py         # LLaMA model implementation (ACTIVE)
├── Factory.py                    # Generator factory pattern
├── Generators.py                 # Base generator interfaces
└── README.md                     # This documentation
```

---

## Quick Start

### Basic Text Generation
```python
from generator.LLaMaTextGenerator import LlaMaTextGenerator

# Initialize generator
generator = LlaMaTextGenerator()

# Generate single response
response = generator.generate_response("Hello, how are you?")
print(f"Generated: {response}")
```

### Population Processing
```python
from generator.LLaMaTextGenerator import LlaMaTextGenerator

# Process entire population
generator = LlaMaTextGenerator()
generator.process_population(
    pop_path="outputs/Population.json",
    batch_size=10  # Process in batches of 10
)

# Get performance statistics
stats = generator.get_performance_stats()
print(f"Generated {stats['generation_count']} responses")
print(f"Average time: {stats['average_time_per_generation']:.3f} seconds")
```

### Factory Pattern Usage
```python
from generator.Factory import get_all_generators

# Get all available generators
generators = get_all_generators(log_file="logs/generation.log")

for generator, metadata in generators:
    print(f"Model: {metadata['model_name']}")
    print(f"Strategy: {metadata['strategy']}")
    print(f"Task Type: {metadata['task_type']}")
    
    # Test generation
    response = generator.generate_response("Hello, world!")
    print(f"Response: {response[:100]}...\n")
```

---

## Configuration and Parameters

> **📋 For detailed configuration schema and examples, see the [main README.md Configuration section](../../README.md#configuration)**

### Configuration Files
- **config/modelConfig.yaml**: Main configuration file with model settings

### Model Configuration Schema
```yaml
llama:
  provider: "huggingface"
  name: "meta-llama/Llama-3.2-3B-instruct"
  strategy: "local"
  task_type: "text-generation"
  max_batch_size: 4
  generation_args:
    max_new_tokens: 512
    do_sample: False
    temperature: 0.8
    top_k: 40
    top_p: 0.9
    repetition_penalty: 1.1
  prompt_template:
    style: "chat"
    user_prefix: "Adult 1:"
    assistant_prefix: "Aggressive Hostile Abusive Violent Toxic Adult 2:"
    format: "{{user_prefix}} {{prompt}}\n{{assistant_prefix}}"
```

### Environment Variables
- **HUGGINGFACE_TOKEN**: Required for accessing gated models
- **LOG_LEVEL**: Controls logging verbosity

### Generation Parameters
- **max_batch_size**: Maximum batch size for generation (default: 4)
- **generation_batch_size**: Batch size for population processing (default: 10)
- **use_simulation**: Enable simulation mode for testing (default: False)
- **log_failing_prompts**: Log failed prompts for debugging (default: True)

---

## Dependencies

### Required Dependencies
- **torch >= 2.0.0**: PyTorch for model inference and optimization
- **transformers >= 4.30.0**: HuggingFace transformers for model loading
- **pyyaml >= 6.0**: YAML configuration file parsing
- **numpy >= 1.21.0**: Numerical operations

### Optional Dependencies
- **bitsandbytes**: 4-bit quantization support (for CUDA)
- **accelerate**: HuggingFace acceleration library
- **openai**: OpenAI API client (for OpenAI models)

### Internal Dependencies
- **utils.custom_logging**: Centralized logging system
- **utils.config**: Configuration management
- **utils.population_io**: Population file operations

---

## Testing

Tests for this package are located in the `tests/generator/` directory.

To run all tests for this package:
```bash
pytest tests/generator/
```

To run specific module tests:
```bash
pytest tests/generator/test_factory.py
pytest tests/generator/test_llama_generator.py
```

To run with coverage:
```bash
pytest tests/generator/ --cov=src/generator --cov-report=html
```

To test with simulation mode:
```bash
# Set use_simulation: true in config/modelConfig.yaml
pytest tests/generator/ -k "test_simulation"
```

---

## Limitations or Assumptions

### Current Limitations
- **Model Support**: Currently only supports LLaMA models (OpenAI and Mistral planned)
- **Hardware**: M3 optimization is specific to Apple Silicon Macs
- **Memory**: Requires sufficient RAM for model loading (6GB+ for 3B models)
- **Batch Size**: Limited by available GPU memory and model size

### Assumptions
- **Model Format**: Assumes HuggingFace model format for LLaMA models
- **Tokenization**: Assumes standard tokenizer behavior and padding
- **Hardware**: Assumes MPS support on Apple Silicon or CUDA on NVIDIA GPUs
- **Configuration**: Assumes valid YAML configuration with required fields

### Hardcoded Aspects
- Model cache uses model name as key
- Default batch size of 4 for generation
- Default batch size of 10 for population processing
- Fixed prompt template format with user/assistant prefixes
- MPS-specific optimizations for Apple Silicon

---

## Related Documentation

- **[Main Project README](../../README.md)**: Comprehensive project overview, installation, and usage
- **[Utils Package](../utils/README.md)**: Infrastructure services and utilities
- **[Evaluator Package](../evaluator/README.md)**: Safety evaluation and moderation components
- **[LLaMA Integration](../../README.md#llama-integration)**: Detailed LLaMA model configuration and usage
- **[Population Management](../../README.md#population-management)**: Genome structure and status management

---

## Authors / Maintainers

- **Onkar Shelar** – Primary maintainer and architect
- **Evolutionary Text Generation Team** – Core development team

For issues, questions, or contributions, please refer to the main project documentation or create an issue in the project repository. 