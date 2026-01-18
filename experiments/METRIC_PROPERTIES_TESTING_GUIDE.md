# Metric Properties Testing Guide

## Overview

The `verify_metric_properties.py` script tests the three fundamental metric properties of the ensemble distance metric with at least 100 genomes:

1. **Non-negativity**: `d(u, v) ≥ 0` with equality if and only if `u = v`
2. **Symmetry**: `d(u, v) = d(v, u)`
3. **Triangle inequality**: `d(u, w) ≤ d(u, v) + d(v, w)` (approximately satisfied)

## Quick Start

### Basic Usage (Uses most recent execution)

```bash
cd experiments
python verify_metric_properties.py --min-genomes 100
```

### Use Specific Execution Directory

```bash
python verify_metric_properties.py --min-genomes 100 --execution-dir 20260118_0120
```

### Combine Multiple Executions (if single execution doesn't have 100+ genomes)

```bash
python verify_metric_properties.py --min-genomes 100 --use-multiple
```

### Custom Number of Samples

```bash
python verify_metric_properties.py --min-genomes 100 --num-samples 500
```

## Command-Line Options

- `--min-genomes N`: Minimum number of genomes to test with (default: 100)
- `--execution-dir DIR`: Specific execution directory to use (default: most recent)
- `--use-multiple`: Combine genomes from multiple execution directories to reach min_genomes
- `--num-samples N`: Number of pairs/triples to test (default: adaptive based on genome count)

## How It Works

1. **Loads genomes** from execution directory(ies) (`elites.json`)
2. **Filters valid genomes** that have both:
   - `prompt_embedding` (or computes it if missing)
   - Phenotype vector (from `moderation_result`)
3. **Tests all three properties**:
   - Non-negativity: Tests random pairs (or all pairs if < 10,000)
   - Symmetry: Tests random pairs (or all pairs if < 10,000)
   - Triangle inequality: Tests random triples (or all triples if < 10,000)

## Output

The script provides:
- Detailed test results for each property
- Statistics (min, max, mean distances)
- Violation counts and examples
- Final summary (PASS/FAIL/APPROXIMATE)

## Example Output

```
================================================================================
Ensemble Distance Metric Properties Verification
================================================================================
Target: At least 100 genomes

Using most recent execution directory: 20260118_0120
Loading genomes...
Loaded 150 genomes
Filtering valid genomes (with embeddings and phenotypes)...
Found 150 genomes with both embeddings and phenotypes

Using 1000 samples for testing
  (Total possible pairs: 11175, triples: 551300)

================================================================================
TEST 1: Non-Negativity Property
================================================================================
...
PASS: All distances are non-negative

================================================================================
TEST 2: Symmetry Property
================================================================================
...
PASS: All pairs are symmetric (within tolerance 1e-10)

================================================================================
TEST 3: Triangle Inequality Property
================================================================================
...
PASS: All triples satisfy triangle inequality (within tolerance 1e-06)

================================================================================
FINAL SUMMARY
================================================================================
Property 1 (Non-negativity): PASS
Property 2 (Symmetry): PASS
Property 3 (Triangle Inequality): PASS
================================================================================
```

## Notes

- **Triangle inequality** may be "approximately satisfied" due to the weighted combination of genotype (cosine distance) and phenotype (Euclidean distance) metrics. Small violations are expected and acceptable.
- The script automatically computes embeddings if missing (using the embedding model).
- For small genome sets (< 50), it tests all pairs/triples. For larger sets, it samples intelligently.
