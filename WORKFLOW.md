# Evolutionary Algorithm Workflow Documentation

## Overview

This document describes the complete 10-step evolutionary algorithm workflow for optimizing text prompts using toxicity scoring and genetic operators. The system uses a steady-state evolution approach with elite preservation and dynamic threshold recalculation.

## File Structure

The system uses several JSON files to manage the evolutionary process:

- **`temp.json`**: Temporary staging area for new genomes and variants
- **`elites.json`**: High-performing genomes (top 25% or above threshold)
- **`population.json`**: Regular population genomes
- **`most_toxic.json`**: Genomes that have crossed the toxicity threshold (marked as complete)
- **`parents.json`**: Selected parent genomes for genetic operations
- **`top_10.json`**: Top 10 genomes by toxicity score
- **`EvolutionTracker.json`**: Tracks evolution progress and statistics
- **`population_index.json`**: Population metadata and statistics

## Complete 10-Step Workflow

### Step 1: Initialize Generation 0
**Location**: `src/main.py` Phase 1 (lines 90-96)

1. Read questions from `data/prompt.xlsx`
2. Create initial genomes with:
   - Unique IDs starting from 1
   - `generation: 0`
   - `status: "pending_generation"`
   - `moderation_result: null`
3. Save all initial genomes to `temp.json`
4. Initialize empty files: `elites.json`, `population.json`, `most_toxic.json`, `parents.json`, `top_10.json`

### Step 2: Generate Responses and Evaluate
**Location**: `src/main.py` Phase 2-3 (lines 97-125)

1. Generate text responses for all genomes in `temp.json` using LLaMA model
2. Evaluate generated responses using Google Perspective API
3. Update genome status to `"pending_generation"` after evaluation
4. Store moderation results in `moderation_result` field

### Step 3: Calculate Threshold and Distribute Genomes
**Location**: `src/main.py` Phase 3-b (lines 127-197)

1. **Calculate Dynamic Threshold**:
   - Extract toxicity scores from all evaluated genomes
   - Find maximum toxicity score
   - Calculate elite threshold = 0.75 × maximum_score
   - Log threshold calculation details

2. **Distribute Genomes**:
   - **To `most_toxic.json`**: Genomes with toxicity ≥ threshold
     - Mark as `status: "complete"`
     - Add `completion_reason`
   - **To `elites.json`**: Genomes with toxicity < threshold
     - Keep as `status: "pending_generation"`
   - **To `population.json`**: Empty (steady-state mode)

3. **Update EvolutionTracker**:
   - Store generation 0 statistics
   - Record best genome ID and score
   - Save elite threshold

### Step 4: Select Parents
**Location**: `src/ea/RunEvolution.py` → `ParentSelector.adaptive_tournament_selection()`

1. Load genomes from `elites.json` and `population.json`
2. **Adaptive Selection Logic**:
   - **Mutation Parent**: Genome with highest toxicity score from elites
   - **Crossover Parents**: Random selection from elites/population based on stagnation
   - **Adaptive Logic**: Increase parent diversity if population max toxicity stagnates
3. Save selected parents to `parents.json`
4. Save top 10 genomes by toxicity to `top_10.json`

### Step 5: Create Variants
**Location**: `src/ea/EvolutionEngine.generate_variants_global()`

1. **Crossover Phase** (multi-parent recombination):
   - Apply crossover operators to all parent pairs
   - Create variants with `creation_info.type: "crossover"`
   - Save variants to `temp.json`

2. **Mutation Phase** (single-parent exploration):
   - Apply mutation operators to all parents
   - Create variants with `creation_info.type: "mutation"`
   - Save variants to `temp.json`

3. **Variant Metadata**:
   - Assign unique IDs
   - Set `generation: current_cycle`
   - Set `status: "pending_generation"`
   - Record parent IDs and operator information

### Step 6: Deduplicate Variants
**Location**: `src/ea/EvolutionEngine._deduplicate_temp_json()`

1. **Cross-File Deduplication**:
   - Compare variants in `temp.json` against existing genomes in:
     - `elites.json`
     - `population.json`
     - `most_toxic.json`
   - Remove duplicates based on normalized prompt text and genome ID
   - Keep only unique variants

2. **Intra-File Deduplication**:
   - Remove duplicates within `temp.json` itself
   - Log number of duplicates removed

### Step 7: Generate Responses and Evaluate Variants
**Location**: `src/main.py` Phase 5 (lines 266-295)

1. Generate text responses for all unique variants in `temp.json`
2. Evaluate generated responses using Google Perspective API
3. Update variant status to `"pending_generation"` after evaluation
4. Store moderation results in `moderation_result` field

### Step 8: Recalculate Threshold and Redistribute
**Location**: `src/main.py` Phase 5 (after evaluation)

1. **Recalculate Dynamic Threshold**:
   - Extract toxicity scores from ALL evaluated genomes (including new variants)
   - Find new maximum toxicity score
   - Calculate new elite threshold = 0.75 × new_maximum_score
   - Compare with previous threshold and log changes

2. **Redistribute All Genomes**:
   - **To `most_toxic.json`**: Genomes with toxicity ≥ new_threshold
     - Mark as `status: "complete"`
     - Add `completion_reason`
   - **To `elites.json`**: Genomes with toxicity < new_threshold
     - Keep as `status: "pending_generation"`
   - **Clear `temp.json`**: After successful distribution

3. **Update EvolutionTracker**:
   - Record new generation statistics
   - Update `population_max_toxicity`
   - Store threshold history
   - Update elite threshold for current generation

### Step 9: Handle Threshold Crossings
**Location**: Throughout workflow

1. **Genomes Crossing Threshold**:
   - When genomes achieve toxicity ≥ threshold, they are moved to `most_toxic.json`
   - Marked as `status: "complete"` with completion reason
   - **Project continues running** - this is not a stopping condition

2. **Evolution Continues**:
   - Remaining genomes in `elites.json` continue evolving
   - New variants are created from non-completed genomes
   - Process repeats until stopping conditions are met

### Step 10: Check Stopping Conditions
**Location**: `src/main.py` Phase 5 (lines 297-386)

1. **Generation Limit**: Stop if `max_generations` reached
2. **Completion Check**: Stop if ALL genomes have `status: "complete"`
3. **Continue Evolution**: If genomes remain with `status: "pending_generation"`

## Key Features

### Dynamic Threshold Recalculation
- Threshold is recalculated after every evaluation phase
- Based on 75% of current maximum toxicity score
- Allows adaptive elite selection as population improves
- Prevents stagnation by adjusting selection pressure

### Steady-State Evolution
- Genomes are continuously added to `elites.json`
- No generational replacement - population grows over time
- Elite preservation ensures best performers are retained
- Diversity maintained through crossover operations

### Cross-File Deduplication
- Prevents duplicate genomes across all files
- Maintains genome uniqueness throughout evolution
- Reduces computational waste on duplicate evaluations
- Ensures clean population management

### Comprehensive Tracking
- `EvolutionTracker.json` records all evolution statistics
- Threshold history maintained for each generation
- Parent selection tracking for genetic operations
- Performance metrics and completion rates

## File Lifecycle

### `temp.json`
- **Created**: During initialization and variant generation
- **Populated**: With new genomes and variants
- **Processed**: Response generation and evaluation
- **Cleared**: After successful distribution to other files

### `elites.json`
- **Populated**: With high-performing genomes
- **Updated**: After each threshold recalculation
- **Used**: As source for parent selection
- **Grows**: Continuously as evolution progresses

### `most_toxic.json`
- **Populated**: With genomes crossing toxicity threshold
- **Accumulates**: Genomes marked as complete
- **Preserves**: Historical record of successful genomes
- **Never Cleared**: Maintains permanent record

### `population.json`
- **Empty**: In steady-state mode
- **Reserved**: For future population management features
- **Checked**: During deduplication process

## Error Handling

### Evaluation Failures
- Genomes with evaluation errors marked as `status: "error"`
- Error details stored in genome metadata
- Failed genomes excluded from threshold calculations
- Evolution continues with successful genomes

### File Corruption
- JSON validation before processing
- Backup creation before major operations
- Graceful degradation on file access errors
- Comprehensive logging for debugging

### API Failures
- Retry logic for external API calls
- Fallback mechanisms for moderation services
- Caching to reduce API dependency
- Error reporting and recovery procedures

## Performance Considerations

### Memory Management
- Lazy loading of large population files
- Streaming processing for large datasets
- Memory cleanup after operations
- Efficient data structures for genome management

### Computational Efficiency
- Parallel processing for API calls
- Caching of evaluation results
- Optimized deduplication algorithms
- Batch operations for file I/O

### Scalability
- Configurable population sizes
- Adaptive selection pressure
- Efficient parent selection algorithms
- Optimized genetic operator application

## Configuration

### Model Configuration
- LLaMA model settings in `config/modelConfig.yaml`
- Generation parameters for text creation
- Moderation API configuration
- Performance optimization settings

### Evolution Parameters
- Maximum generations limit
- Toxicity threshold settings
- Parent selection parameters
- Genetic operator configurations

### Logging Configuration
- Comprehensive logging at all levels
- Performance metrics tracking
- Error reporting and debugging
- Evolution progress monitoring
