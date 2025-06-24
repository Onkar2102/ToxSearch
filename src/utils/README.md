# Utils Package

> **📋 For comprehensive project overview, installation, and configuration details, see the [main README.md](../../README.md)**

The `utils` package provides essential utility functions and infrastructure for the evolutionary text generation framework. This package handles configuration management, logging, population I/O operations, and performance optimization.

## Package Overview

### Core Responsibilities
- **Configuration Management**: YAML-based configuration loading, validation, and access
- **Logging Infrastructure**: Centralized logging with performance monitoring and file rotation
- **Population I/O**: JSON-based population file operations with validation
- **Evolution Tracking**: Parent selection and generation tracking utilities
- **Performance Optimization**: Hardware-specific optimization for Apple M3 Macs

### Architecture
```
utils/
├── config.py                    # Configuration loading and validation
├── custom_logging.py            # Logging infrastructure and performance monitoring
├── population_io.py             # Population file operations
├── evolution_utils.py           # Evolution tracking utilities
├── m3_optimizer.py              # M3-specific performance optimization
└── README.md                    # This documentation
```

---

## Quick Start

### Basic Configuration Access
```python
from utils.config import load_config, get_config_value

# Load configuration
config = load_config("config/modelConfig.yaml")

# Access nested configuration values
batch_size = get_config_value(config, "llama.max_batch_size", default=4)
model_name = get_config_value(config, "llama.name")
```

### Logging Setup and Performance Monitoring
```python
from utils.custom_logging import get_logger, PerformanceLogger

# Create logger with automatic file rotation
logger = get_logger("my_module")

# Monitor operation performance
with PerformanceLogger(logger, "Data Processing"):
    # Your operation here
    result = process_data()
    logger.info("Processed %d items", len(result))
```

### Population Management
```python
from utils.population_io import load_population, save_population, load_and_initialize_population

# Load existing population
population = load_population("outputs/Population.json", logger=logger)

# Initialize from Excel file
load_and_initialize_population("data/prompts.xlsx", "outputs/Population.json")

# Save with validation
save_population(population, "outputs/Population_new.json", logger=logger)
```

### Evolution Tracking
```python
from utils.evolution_utils import append_parents_by_generation_entry

# Track parent selection for evolution
append_parents_by_generation_entry(
    prompt_id=1,
    generation=2,
    parent_ids=["genome_1", "genome_3"],
    parent_type="crossover",
    logger=logger
)
```

### Performance Optimization
```python
from utils.m3_optimizer import get_system_info, optimize_config_for_m3

# Get system information
system_info = get_system_info()
print(f"Available memory: {system_info['memory_available_gb']:.1f} GB")

# Generate optimized configuration
optimized_config = optimize_config_for_m3()
```

---

## Configuration and Parameters

> **📋 For detailed configuration schema and examples, see the [main README.md Configuration section](../../README.md#configuration)**

### Configuration Files
- **config/modelConfig.yaml**: Main configuration file with model, evolution, and evaluation settings

### Environment Variables
- **LOG_LEVEL**: Controls logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Configuration Schema
```yaml
model:
  name: "meta-llama/Llama-3.2-3B-instruct"
  type: "llama"
  parameters:
    max_batch_size: 4
    temperature: 0.8

evolution:
  population_size: 100
  mutation_rate: 0.1
  crossover_rate: 0.2

evaluation:
  metrics: ["violence", "toxicity"]
  api:
    provider: "openai"
```

---

## Dependencies

### Required Dependencies
- **pyyaml >= 6.0**: YAML configuration file parsing
- **pandas >= 1.5.0**: Excel file reading for population initialization
- **psutil >= 5.9.0**: System information gathering (M3 optimizer)
- **torch >= 2.0.0**: PyTorch for M3 optimization and hardware detection

### Optional Dependencies
- **openpyxl**: Excel file support (for population initialization)
- **xlsxwriter**: Excel file writing (if needed)

### Internal Dependencies
- **utils.custom_logging**: Used by all other utility modules
- **utils.config**: Configuration validation and access patterns
- **utils.population_io**: Population file operations

---

## Testing

Tests for this package are located in the `tests/utils/` directory.

To run all tests for this package:
```bash
pytest tests/utils/
```

To run specific module tests:
```bash
pytest tests/utils/test_config.py
pytest tests/utils/test_custom_logging.py
pytest tests/utils/test_population_io.py
```

To run with coverage:
```bash
pytest tests/utils/ --cov=src/utils --cov-report=html
```

---

## Limitations or Assumptions

### Current Limitations
- **M3 Optimization**: Hardware optimization is specific to Apple M3 Macs
- **Configuration Validation**: Limited to predefined schema validation
- **Log File Rotation**: Fixed 10MB file size limit with 10 backup files
- **Population Format**: Assumes specific JSON structure for population files

### Assumptions
- **File System**: Assumes write access to `logs/`, `outputs/`, and `config/` directories
- **Hardware**: M3 optimizer assumes macOS with `system_profiler` command available
- **Memory**: Performance optimization assumes sufficient RAM for model loading
- **Encoding**: All text files are assumed to be UTF-8 encoded

### Hardcoded Aspects
- Log file naming convention includes timestamp, run ID, and system info
- PerformanceLogger uses millisecond precision timing
- Population validation enforces specific required fields
- M3 optimizer uses conservative memory estimates for batch sizing

---

## Related Documentation

- **[Main Project README](../../README.md)**: Comprehensive project overview, installation, and usage
- **[Generator Package](../generator/README.md)**: Text generation components and interfaces
- **[Evaluator Package](../evaluator/README.md)**: Safety evaluation and moderation components
- **[Project Structure](../../README.md#project-structure)**: Complete directory layout and file descriptions

---

## Authors / Maintainers

- **Onkar Shelar** – Primary maintainer and architect
- **Evolutionary Text Generation Team** – Core development team

For issues, questions, or contributions, please refer to the main project documentation or create an issue in the project repository. 