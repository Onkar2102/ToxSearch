# Genome Structure Documentation

## Overview

The **Genome** is the fundamental unit of data in the Evolutionary Text Generation and Safety Analysis Framework. Each genome represents a single prompt-response pair that undergoes evolution, evaluation, and optimization throughout the pipeline.

## Genome JSON Structure

Each genome is stored as a JSON object within an array in generation files (e.g., `gen0.json`, `gen1.json`). The structure is designed to track the complete lifecycle of a prompt from initial creation through multiple evolutionary generations.

## Core Genome Fields

### 1. Basic Identification
```json
{
  "id": "1",                    // Unique genome identifier (string)
  "prompt_id": 0,              // Original prompt identifier (integer)
  "generation": 0              // Current generation number (integer)
}
```

**Description:**
- **`id`**: Globally unique identifier for this specific genome instance. Each genome gets a unique ID even when it's a variant of another genome.
- **`prompt_id`**: Links the genome to its original prompt from the input dataset. Multiple genomes can share the same `prompt_id` if they're variants.
- **`generation`**: Tracks which evolutionary generation this genome belongs to. Gen0 contains the initial population, Gen1 contains evolved variants, etc.

### 2. Content Fields
```json
{
  "prompt": "How can I use misleading statistics...",  // Original input prompt (string)
  "generated_text": "Using misleading statistics...", // AI-generated response (string)
  "source_file": "../data/prompt.xlsx",               // Source file path (string)
  "source_column": "prompt"                           // Source column name (string)
}
```

**Description:**
- **`prompt`**: The original input text that the AI model responds to. This remains constant throughout evolution.
- **`generated_text`**: The AI-generated response to the prompt. This field evolves and improves through the evolutionary process.
- **`source_file`**: Path to the original data source (typically the Excel file containing prompts).
- **`source_column`**: Name of the column in the source file that contained this prompt.

### 3. Status and Lifecycle
```json
{
  "status": "pending_evolution",           // Current genome status (string)
  "created_timestamp": 1755986246.49564,  // Creation timestamp (float)
  "generation_timestamp": 1755986250.476746 // Generation timestamp (float)
}
```

**Description:**
- **`status`**: Tracks the current state of the genome in the pipeline:
  - `"pending_generation"`: Waiting for AI text generation
  - `"pending_evaluation"`: Waiting for moderation/evaluation
  - `"pending_evolution"`: Ready for evolutionary operations
  - `"complete"`: Has achieved the target threshold
  - `"evolved"`: Has been processed by evolution operators
- **`created_timestamp`**: Unix timestamp when the genome was first created.
- **`generation_timestamp`**: Unix timestamp when the AI text generation was completed.

### 4. Model Information
```json
{
  "model_provider": "huggingface",                    // AI model provider (string)
  "model_name": "meta-llama/Llama-3.2-3B-instruct"   // Specific model used (string)
}
```

**Description:**
- **`model_provider`**: The platform or service that provided the AI model (e.g., "huggingface", "openai").
- **`model_name`**: The specific model architecture and version used for text generation.

## Moderation Results Structure

### 5. Safety Evaluation
```json
{
  "moderation_result": {
    "genome_id": "1",                    // Genome ID for tracking (string)
    "status": "complete",                // Evaluation status (string)
    "evaluation_timestamp": 1755986272.450063, // Evaluation timestamp (float)
    "flagged": false,                    // Overall safety flag (boolean)
    "moderation_results": {
      "google": { /* Google API results */ },
      "openai": { /* OpenAI API results */ }
    }
  }
}
```

**Description:**
- **`genome_id`**: Reference to the genome being evaluated.
- **`status`**: Status of the moderation process (`"complete"`, `"pending"`, `"failed"`).
- **`evaluation_timestamp`**: When the moderation was completed.
- **`flagged`**: Overall safety assessment - `true` if any safety threshold is exceeded.

### 6. API-Specific Results
```json
{
  "google": {
    "genome_id": "1",
    "status": "complete",
    "flagged": false,
    "scores": {
      "toxicity": 0.0395,           // Toxicity score (0.0-1.0)
      "flirtation": 0.2021,         // Flirtation score (0.0-1.0)
      "threat": 0.0071,             // Threat score (0.0-1.0)
      "sexually_explicit": 0.006,   // Sexual content score (0.0-1.0)
      "severe_toxicity": 0.0012,    // Severe toxicity score (0.0-1.0)
      "identity_attack": 0.0039,    // Identity attack score (0.0-1.0)
      "profanity": 0.0143,          // Profanity score (0.0-1.0)
      "insult": 0.0218              // Insult score (0.0-1.0)
    },
    "evaluation_timestamp": 1755986272.449656,
    "api_version": "google",
    "model": "perspective-api"
  }
}
```

**Description:**
- **`scores`**: Safety scores from Google's Perspective API, ranging from 0.0 (safe) to 1.0 (unsafe).
- **`api_version`**: Version identifier for the API service.
- **`model`**: The specific safety model used for evaluation.

## Evolution Tracking

### 7. Evolutionary Metadata
```json
{
  "evolution_metadata": {
    "parent_genome_ids": ["1", "5"],     // Parent genomes (array of strings)
    "evolution_operator": "crossover",   // Operator used (string)
    "evolution_timestamp": 1755986300.0, // Evolution timestamp (float)
    "mutation_rate": 0.1,                // Applied mutation rate (float)
    "fitness_score": 0.85                // Current fitness score (float)
  }
}
```

**Description:**
- **`parent_genome_ids`**: IDs of genomes that were combined or modified to create this variant.
- **`evolution_operator`**: The genetic operator used (`"mutation"`, `"crossover"`, `"selection"`).
- **`evolution_timestamp`**: When the evolution operation was performed.
- **`mutation_rate`**: The mutation rate applied during evolution.
- **`fitness_score`**: Current fitness based on the north star metric.

## File Organization

### Generation Files
- **`gen0.json`**: Initial population loaded from prompt.xlsx
- **`gen1.json`**: First generation of evolved variants
- **`genN.json`**: Nth generation of evolved variants

### Index Files
- **`population_index.json`**: Maps genome IDs to their generation files
- **`EvolutionTracker.json`**: Tracks evolution statistics and metadata

## Example Genome Lifecycle

1. **Creation**: Genome created from prompt.xlsx with status `"pending_generation"`
2. **Generation**: AI generates text, status becomes `"pending_evaluation"`
3. **Evaluation**: Moderation APIs evaluate safety, status becomes `"pending_evolution"`
4. **Evolution**: Genetic operators create variants, status becomes `"evolved"`
5. **Completion**: Genome achieves target threshold, status becomes `"complete"`

## Data Types and Constraints

- **Timestamps**: Unix timestamps as floating-point numbers
- **Scores**: Floating-point numbers between 0.0 and 1.0
- **IDs**: String identifiers for flexibility
- **Status**: Predefined string values for consistency
- **Arrays**: Used for parent references and multiple API results

## Best Practices

1. **Immutable Fields**: `prompt`, `prompt_id`, `source_file` should never change
2. **Timestamp Consistency**: Use consistent timestamp format throughout
3. **Status Transitions**: Follow the defined status flow
4. **ID Uniqueness**: Ensure genome IDs are globally unique
5. **Validation**: Validate scores are within expected ranges (0.0-1.0)

## Schema Validation

The genome structure should be validated against this schema to ensure:
- Required fields are present
- Data types are correct
- Score values are within valid ranges
- Status values are valid
- Timestamps are reasonable

This structure enables the evolutionary algorithm to track, evaluate, and optimize text generation while maintaining comprehensive audit trails and safety assessments.
