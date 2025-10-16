# Evolutionary Text Generation Framework

A research framework for AI safety analysis through evolutionary text generation, moderation evaluation, and genetic optimization with **automatic process monitoring and recovery**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Quick Setup](#quick-setup)
- [Installation](#installation)
- [API Keys Configuration](#api-keys-configuration)
- [Running the Project](#running-the-project)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Documentation Index](#documentation-index)
- [License](#license)

## Quick Setup

### Prerequisites
- **Python 3.8+** (Python 3.12+ recommended)
- **macOS/Linux** (Windows support via WSL)
- **8GB+ RAM** (16GB+ recommended for larger models)
- **API Keys**: OpenAI API key and Google Perspective API key

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd eost-cam-llm
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Run automated setup (RECOMMENDED)**
   ```bash
   python3 app.py --setup
   ```
   This will:
   - Install all required dependencies
   - Create `.env` template file
   - Optimize configuration for your system
   - Verify all required files are present

4. **Configure API keys**
   Edit the `.env` file and add your API keys:
   ```bash
   # Required API keys
   OPENAI_API_KEY=your-openai-api-key-here
   PERSPECTIVE_API_KEY=your-google-perspective-api-key-here
   
   # Optional (for Hugging Face models)
   HF_TOKEN=your-huggingface-token-here
   ```

5. **Verify installation**
   ```bash
   # Test with a single generation
   python3 src/main.py --generations 1 --threshold 0.5
   ```

## Installation

### Automated Installation (Recommended)
```bash
# Run full environment setup
python3 app.py --setup
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Create .env file manually
cp .env.example .env  # Edit with your API keys
```

## API Keys Configuration

### Required API Keys

1. **OpenAI API Key**
   - Get from: https://platform.openai.com/api-keys
   - Used for: LLM-based text generation and paraphrasing

2. **Google Perspective API Key**
   - Get from: https://developers.perspectiveapi.com/
   - Used for: Content moderation and toxicity evaluation

### Environment Variables
Create a `.env` file in the project root:
```bash
# Required API keys
OPENAI_API_KEY=your-openai-api-key-here
PERSPECTIVE_API_KEY=your-google-perspective-api-key-here

# Optional (for Hugging Face models)
HF_TOKEN=your-huggingface-token-here
```

## Running the Project

### Basic Usage
```bash
# Run with interactive setup and monitoring (RECOMMENDED)
python3 app.py --interactive

# Run directly with process monitoring
python3 app.py --generations 25

# Run core pipeline directly
python3 src/main.py --generations 25
```

### Command Line Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--generations` | int | `None` | Maximum number of evolution generations. If not set, runs until north star metric is achieved |
| `--threshold` | float | `0.95` | North star metric threshold for stopping evolution |
| `--pg` | str | `qwen2.5-7b-instruct-gguf` | Prompt Generator model name |
| `--rg` | str | `llama3.2-3b-instruct-gguf` | Response Generator model name |
| `--check-interval` | int | `1800` | Health check interval in seconds (30 minutes) |
| `--stuck-threshold` | int | `7200` | Stuck detection threshold in seconds (2 hours) |
| `--memory-threshold` | float | `20.0` | Memory threshold in GB |
| `--max-restarts` | int | `5` | Maximum restart attempts |
| `--interactive` | flag | `False` | Run in interactive mode with setup and monitoring |
| `--setup` | flag | `False` | Run full environment setup (install requirements, optimize config) |
| `--no-monitor` | flag | `False` | Run without process monitoring |

### Example Commands
```bash
# Quick test run
python3 src/main.py --generations 1 --threshold 0.5

# Full evolution run
python3 src/main.py --generations 50 --threshold 0.99

# Run with specific models
python3 src/main.py --generations 10 --pg qwen2.5-7b-instruct-gguf --rg llama3.2-3b-instruct-gguf

# Interactive mode with monitoring
python3 app.py --interactive
```

## Requirements

### System Requirements
- **Python**: 3.8+ (3.12+ recommended)
- **RAM**: 8GB+ (16GB+ recommended for larger models)
- **Storage**: 10GB+ for models and data
- **OS**: macOS/Linux (Windows via WSL)

### Python Dependencies
See `requirements.txt` for complete list. Key dependencies:
- `torch` - PyTorch for model operations
- `transformers` - Hugging Face transformers
- `spacy` - Natural language processing
- `openai` - OpenAI API client
- `google-api-python-client` - Google Perspective API
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `sentence-transformers` - Semantic similarity
- `nltk` - Natural language toolkit

### Model Requirements
- **Prompt Generator**: Qwen2.5-7B-Instruct (default)
- **Response Generator**: Llama3.2-3B-Instruct (default)
- **spaCy Model**: `en_core_web_sm`
- **NLTK Data**: `punkt` tokenizer

## Troubleshooting

### Common Issues

**Import errors**
- Ensure virtual environment is activated: `source venv/bin/activate`
- Check Python version: `python3 --version`

**API rate limits**
- Google Perspective API: 60 requests/minute limit
- OpenAI API: Check your usage limits
- Added 0.5-second delay between evaluations to prevent rate limiting

**Memory issues**
- Use smaller models or reduce batch sizes
- Monitor memory usage in logs
- Enable garbage collection: `--memory-threshold 20.0`

**Model loading errors**
- Check model files exist in `models/` directory
- Verify model paths in config files
- Ensure sufficient disk space

**GPU usage**
- Models run on GPU by default (`n_gpu_layers: -1`)
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Getting Help
- Check logs in `logs/` directory
- Review [Architecture Overview](ARCHITECTURE.md)
- See [Evolutionary Algorithms Guide](src/ea/README.md)
- Check [Tests README](tests/README.md) for testing

### Performance Optimization
- **GPU Acceleration**: Enabled by default for both PG and RG models
- **Memory Management**: Automatic cleanup and garbage collection
- **Model Caching**: Efficient reuse of loaded models
- **Parallel Processing**: Available for operator execution

## Documentation Index

### 📚 **Core Documentation**
- **[Architecture Overview](ARCHITECTURE.md)** - Complete system architecture and component interactions
- **[Evolutionary Algorithms Guide](src/ea/README.md)** - Genetic algorithms, variation operators, and evolution strategies
- **[EA Notes](src/ea/notes.md)** - Detailed implementation notes and data flow

### 📖 **Additional Documentation**
- **[LLM POS-Aware Synonym Replacement](docs/LLM_POSAwareSynonymReplacement.md)** - Detailed guide for POS-aware operations
- **[vLLM Migration Guide](docs/vLLM_Migration_Guide.md)** - Migration guide for vLLM integration
- **[LLM POS Test Updates](docs/README_llm_pos_test_updates.md)** - Updates and testing information

### 🧪 **Testing Documentation**
- **[Tests README](tests/README.md)** - Testing framework and test execution guide

### 🔧 **Configuration Files**
- **[RGConfig.yaml](config/RGConfig.yaml)** - Response Generator configuration
- **[PGConfig.yaml](config/PGConfig.yaml)** - Prompt Generator configuration
- **[requirements.txt](requirements.txt)** - Python dependencies

### 📊 **Data Files**
- **[prompt.xlsx](data/prompt.xlsx)** - Input prompts for evolution
- **[outputs/](data/outputs/)** - Evolution results and tracking data
- **[models/](models/)** - Local model files and configurations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.