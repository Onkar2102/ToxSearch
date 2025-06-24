# Evaluator Package

> **📋 For comprehensive project overview, installation, and configuration details, see the [main README.md](../../README.md)**

The `evaluator` package provides safety evaluation and moderation capabilities for the evolutionary text generation framework. This package implements OpenAI moderation API integration with comprehensive toxicity analysis, batch processing, and performance optimization.

## Package Overview

### Core Responsibilities
- **Safety Evaluation**: OpenAI moderation API integration for toxicity scoring
- **Batch Processing**: Efficient population-wide evaluation with configurable batch sizes
- **Performance Optimization**: Async processing, caching, and rate limiting
- **Score Normalization**: Standardized scoring across multiple toxicity categories
- **Population Integration**: Seamless integration with population management system

### Architecture
```
evaluator/
├── openai_moderation.py          # OpenAI moderation API implementation (ACTIVE)
├── test.py                       # Evaluation testing utilities
└── README.md                     # This documentation
```

---

## Quick Start

### Basic Evaluation
```python
from evaluator.openai_moderation import run_moderation_on_population

# Evaluate entire population
run_moderation_on_population(
    pop_path="outputs/Population.json",
    north_star_metric="violence"
)
```

### Individual Text Evaluation
```python
from evaluator.openai_moderation import evaluate_single_text

# Evaluate single text
result = evaluate_single_text("This is a test message")
print(f"Flagged: {result['flagged']}")
print(f"Violence score: {result['scores']['violence']}")
```

### Async Batch Evaluation
```python
import asyncio
from evaluator.openai_moderation import AsyncOpenAIModerationEvaluator

async def custom_evaluation():
    evaluator = AsyncOpenAIModerationEvaluator()
    
    # Evaluate population asynchronously
    await evaluator.evaluate_population(
        pop_path="outputs/Population.json",
        north_star_metric="violence"
    )
    
    # Get performance statistics
    stats = evaluator.get_performance_stats()
    print(f"Evaluated {stats['evaluation_count']} texts")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"Average time: {stats['average_evaluation_time']:.3f} seconds")

# Run custom evaluation
asyncio.run(custom_evaluation())
```

### Score Normalization
```python
from evaluator.openai_moderation import normalize_moderation_scores

# Normalize raw scores
raw_scores = {
    "violence": 0.123456,
    "toxicity": 0.987654,
    "harassment": 0.000001
}

normalized_scores = normalize_moderation_scores(raw_scores)
print(normalized_scores)
# Output: {'violence': 0.1235, 'toxicity': 1.0000, 'harassment': 0.0001}
```

---

## Configuration and Parameters

> **📋 For detailed configuration schema and examples, see the [main README.md Configuration section](../../README.md#configuration)**

### Configuration Files
- **config/modelConfig.yaml**: Model configuration for batch size settings

### Environment Variables
- **OPENAI_API_KEY**: Required OpenAI API key for moderation API access
- **OPENAI_ORG_ID**: Optional OpenAI organization ID
- **OPENAI_PROJECT_ID**: Optional OpenAI project ID
- **LOG_LEVEL**: Controls logging verbosity

### Evaluation Parameters
- **evaluation_batch_size**: Batch size for population processing (default: 10)
- **north_star_metric**: Primary metric for evolution (default: "violence")
- **concurrency_limit**: Maximum concurrent API requests (default: 10)
- **timeout**: API request timeout in seconds (default: 30)

### API Configuration
```python
# OpenAI API settings
api_key = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")
project_id = os.getenv("OPENAI_PROJECT_ID")

# API endpoints
base_url = "https://api.openai.com/v1"
moderation_url = f"{base_url}/moderations"
model = "text-moderation-latest"  # or "omni-moderation-latest"
```

---

## Dependencies

### Required Dependencies
- **openai >= 1.0.0**: OpenAI API client for moderation API
- **aiohttp >= 3.8.0**: Async HTTP client for API requests
- **python-dotenv >= 0.19.0**: Environment variable loading
- **pyyaml >= 6.0**: YAML configuration file parsing

### Optional Dependencies
- **asyncio**: Built-in Python async support
- **threading**: Built-in Python threading for cache safety
- **hashlib**: Built-in Python hashing for cache keys

### Internal Dependencies
- **utils.custom_logging**: Centralized logging system
- **utils.population_io**: Population file operations

---

## Testing

Tests for this package are located in the `tests/evaluator/` directory.

To run all tests for this package:
```bash
pytest tests/evaluator/
```

To run specific module tests:
```bash
pytest tests/evaluator/test_openai_moderation.py
pytest tests/evaluator/test_caching.py
```

To run with coverage:
```bash
pytest tests/evaluator/ --cov=src/evaluator --cov-report=html
```

To test with mock API responses:
```bash
# Set up mock responses
pytest tests/evaluator/ -k "test_mock_api"
```

To test async functionality:
```bash
pytest tests/evaluator/ -k "test_async" -v
```

---

## Limitations or Assumptions

### Current Limitations
- **API Dependency**: Requires OpenAI API access and valid API key
- **Rate Limits**: Subject to OpenAI API rate limits and quotas
- **Model Support**: Only supports OpenAI moderation models
- **Caching**: In-memory cache only (not persistent across restarts)
- **Batch Size**: Limited to 100 texts per API request (OpenAI limit)

### Assumptions
- **API Availability**: Assumes OpenAI API is accessible and responsive
- **Text Format**: Assumes UTF-8 encoded text input
- **Response Format**: Assumes standard OpenAI moderation API response format
- **Network**: Assumes stable internet connection for API calls
- **Memory**: Assumes sufficient memory for caching and batch processing

### Hardcoded Aspects
- OpenAI API endpoint URLs
- Default batch size of 10 for population processing
- Default concurrency limit of 10 requests
- Default timeout of 30 seconds for API requests
- Score normalization range (0.0001 to 1.0000)
- Cache key generation using MD5 hash

---

## Related Documentation

- **[Main Project README](../../README.md)**: Comprehensive project overview, installation, and usage
- **[Utils Package](../utils/README.md)**: Infrastructure services and utilities
- **[Generator Package](../generator/README.md)**: Text generation components and interfaces
- **[OpenAI Moderation](../../README.md#openai-moderation)**: Detailed moderation API integration
- **[Evaluation System](../../README.md#evaluation-system)**: Comprehensive toxicity analysis and scoring
- **[Population Management](../../README.md#population-management)**: Genome structure and status management

---

## Authors / Maintainers

- **Onkar Shelar** – Primary maintainer and architect
- **Evolutionary Text Generation Team** – Core development team

For issues, questions, or contributions, please refer to the main project documentation or create an issue in the project repository. 