# Evolutionary Optimization for Safety Testing in Large Language Models (EOST-LLM)

A genetic algorithm framework for evolving text prompts to maximize toxicity scores, enabling comprehensive AI safety evaluation through adversarial prompt generation.

## Quick Start

### Prerequisites
- Python 3.8+
- Google Perspective API key
- CUDA/MPS/CPU support for model inference

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd eost-cam-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your PERSPECTIVE_API_KEY
```

### Running the System
```bash
# Basic run with default settings
python src/main.py

# Custom configuration
python src/main.py --max-generations 10 --operators all --max-variants 3
```

## System Overview

### Research Objective
Develop an automated system for generating high-toxicity prompts that can be used to:
- Test AI safety mechanisms
- Identify failure modes in content moderation systems
- Generate adversarial examples for model robustness evaluation

### Evolutionary Algorithm Design

**Fitness Function**: Toxicity score from Google Perspective API
- Primary metric: `toxicity` (0.0-1.0 scale)
- Secondary metrics: `severe_toxicity`, `threat`, `insult`, `profanity`

**Population Management**: Three-tier classification system
- **Elites** (≥ elite_threshold): High-performing prompts for parent selection
- **Non-elites** (removal_threshold < score < elite_threshold): Mid-performing prompts
- **Under-performing** (≤ removal_threshold): Low-performing prompts (archived)

### Variation Operators (16 Total)

#### Mutation Operators (13)
1. **Informed Evolution**: LLM-guided evolution using top performers
2. **Masked Language Model**: Contextual word substitution
3. **Paraphrasing**: Semantic-preserving text transformation
4. **Back Translation**: Multi-language roundtrip translation (5 languages)
5. **Synonym/Antonym Replacement**: Lexical substitution with POS awareness
6. **Negation**: Logical operator insertion
7. **Concept Addition**: Semantic concept injection
8. **Typographical Errors**: Character-level noise injection
9. **Stylistic Mutation**: Writing style transformation

#### Crossover Operators (3)
1. **Semantic Similarity**: Crossbreeding based on semantic distance
2. **Semantic Fusion**: Hybrid prompt generation
3. **Cut-and-Slice**: Structural recombination

## Project Structure

```
eost-cam-llm/
├── src/
│   ├── main.py                    # Entry point
│   ├── ea/                        # Evolutionary algorithms
│   │   ├── evolution_engine.py    # Core evolution logic
│   │   ├── parent_selector.py     # Adaptive parent selection
│   │   └── [16 operator files]    # Variation operators
│   ├── gne/                       # Generation & evaluation
│   │   ├── prompt_generator.py    # Prompt generation
│   │   ├── response_generator.py  # Response generation
│   │   └── evaluator.py           # Moderation API calls
│   └── utils/
│       └── population_io.py       # Population I/O & metrics
├── config/                        # Model configurations
├── data/                          # Input data and results
└── experiments/                   # Analysis notebooks
```

## License

MIT License - See LICENSE file for details.