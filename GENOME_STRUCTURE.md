# Genome Structure Documentation

## Overview

The **Genome** is the fundamental unit of data in the Evolutionary Text Generation and Safety Analysis Framework. Each genome represents a single prompt-response pair that undergoes evolution, evaluation, and optimization throughout the pipeline.

## Genome JSON Structure

Each genome is stored as a JSON object within a single array in `outputs/Population.json`. The structure is designed to track the complete lifecycle of a prompt from initial creation through multiple evolutionary generations.

## Core Genome Fields

### 1. Basic Identification
```json
{
  "id": "1",
  "prompt_id": 0,
  "generation": 0
}
```

### 2. Content Fields
```json
{
  "prompt": "How can I use misleading statistics...",
  "generated_text": "Using misleading statistics...",
  "source_file": "../data/prompt.xlsx",
  "source_column": "prompt"
}
```

### 3. Status and Lifecycle
```json
{
  "status": "pending_evolution",
  "created_timestamp": 1755986246.49564,
  "generation_timestamp": 1755986250.476746
}
```

Status values:
- "pending_generation"
- "pending_evaluation"
- "pending_evolution"
- "complete"
- "evolved"
- "error" (on failure; see `error` field)

### 4. Model Information
```json
{
  "model_provider": "huggingface",
  "model_name": "meta-llama/Llama-3.2-3B-instruct"
}
```

## Moderation Results Structure

### 5. Safety Evaluation
```json
{
  "moderation_result": {
    "genome_id": "1",
    "status": "complete",
    "evaluation_timestamp": 1755986272.450063,
    "flagged": false,
    "moderation_results": {
      "google": { /* Google API results */ },
      "openai": { /* OpenAI API results */ }
    }
  }
}
```

## Evolution Tracking

### 6. Evolutionary Metadata (example)
```json
{
  "parents": ["5", "7"],
  "operator": "POSAwareSynonymReplacement"
}
```

## File Organization

### Population File
- `outputs/Population.json`: Single file containing all genomes across generations

### Index & Tracker Files
- `outputs/population_index.json`: Metadata and quick stats for `Population.json`
- `outputs/EvolutionTracker.json`: Tracks evolution statistics and metadata

## Example Genome Lifecycle

1. Creation → `pending_generation`
2. Generation → `pending_evaluation`
3. Evaluation → `pending_evolution`
4. Evolution → variants created
5. Completion → `complete`
6. Error (optional) → `status: "error"` with `error` object

### Error Field (optional)
```json
{
  "status": "error",
  "error": {
    "type": "generation_failed|evaluation_failed|evolution_failed",
    "message": "Detailed message",
    "stage": "text_generation|moderation|evolution",
    "timestamp": 1756125000.0
  }
}
```
