# Research Questions Analysis Guide

## Overview

This guide explains how to analyze data for **RQ1** (Operator Effectiveness) and **RQ2** (Cluster Quality and Diversity) when comparing:
- **5 runs with speciation** (different PG models)
- **5 runs without speciation** (old toxsearch)

---

## Research Questions

### RQ1: Operator Effectiveness Analysis

**Question**: Which variation operators are most effective at generating high-fitness variants?

**Metrics**:
1. **NE (Non-Elite %)**: Percentage of all attempts that ended up archived
   - Formula: `non_elite_count / calculated_total × 100`
   - Lower is better (fewer variants archived)

2. **EHR (Elite Hit Rate %)**: Percentage of all attempts that became elites
   - Formula: `elite_count / calculated_total × 100`
   - Higher is better (more variants become elites)

3. **IR (Invalid Rate %)**: Percentage of all attempts that were rejected
   - Formula: `rejections / calculated_total × 100`
   - Lower is better (fewer rejections)

4. **cEHR (Conditional Elite Hit Rate %)**: Percentage of VALID variants that became elites
   - Formula: `elite_count / total_variants × 100`
   - Higher is better (of valid variants, more become elites)

5. **Δμ (Mean Delta Score)**: Average toxicity change from parent
   - Formula: `mean(current_toxicity - parent_score)`
   - Higher is better (more toxicity increase)

6. **Δσ (Delta Score Std Dev)**: Standard deviation of toxicity change
   - Formula: `std(current_toxicity - parent_score)`
   - Indicates consistency of operator performance

**Data Sources**:
- `operator_effectiveness_cumulative.csv` in each execution directory

---

### RQ2: Cluster Quality and Diversity Analysis

**Question**: How does speciation affect cluster quality and diversity?

**Metrics**:
1. **Species Count**: Number of active species per generation
   - Higher = more diversity (more distinct clusters)

2. **Frozen Species Count**: Number of frozen species per generation (NEW)
   - Tracks species that have stagnated (≥20 generations without improvement)
   - Helps analyze species lifecycle and stagnation patterns

3. **Reserves Size**: Number of genomes in cluster 0 (reserves)
   - Indicates genomes that don't fit existing species

4. **Best Fitness**: Maximum fitness score per generation
   - Higher is better (best variant found)

5. **Average Fitness**: Mean fitness score per generation
   - Higher is better (overall population quality)

6. **Inter-Species Diversity**: Diversity between different species
   - Formula: Average distance between species leaders
   - Higher = more distinct species

7. **Intra-Species Diversity**: Diversity within species
   - Formula: Average distance within species members
   - Moderate values indicate good clustering

8. **Speciation Events**: Number of new species formed
   - Indicates exploration of new niches

9. **Merge Events**: Number of species merged
   - Indicates convergence of similar species

10. **Extinction Events**: Number of species frozen due to stagnation
    - Indicates pruning of unproductive species

11. **Cluster Quality Metrics** (NEW):
    - **Silhouette Score**: [-1, 1], higher is better (measures how well-separated clusters are)
    - **Davies-Bouldin Index**: ≥ 0, lower is better (measures cluster separation)
    - **Calinski-Harabasz Index**: ≥ 0, higher is better (measures cluster compactness)
    - These metrics evaluate the quality of the clustering/speciation

**Data Sources**:
- `EvolutionTracker.json` → `generations[].speciation`
- `speciation_state.json` (for detailed species information)

---

## Running the Analysis

### Step 1: Update Execution Directories

Edit `compare_speciation_vs_nonspeciation.py` and update:

```python
SPEciation_RUNS = [
    "20260115_1609",  # Run 1 with speciation
    "20260115_1700",  # Run 2 with speciation
    "20260115_1800",  # Run 3 with speciation
    "20260115_1900",  # Run 4 with speciation
    "20260115_2000",  # Run 5 with speciation
]

NON_SPECIATION_RUNS = [
    "20260115_2100",  # Run 1 without speciation
    "20260115_2200",  # Run 2 without speciation
    "20260115_2300",  # Run 3 without speciation
    "20260116_0000",  # Run 4 without speciation
    "20260116_0100",  # Run 5 without speciation
]
```

### Step 2: Run the Analysis

```bash
cd experiments
python compare_speciation_vs_nonspeciation.py
```

### Step 3: View Results

Results will be saved in `experiments/comparison_results/`:
- `rq1_operator_comparison.png` - Operator effectiveness comparison charts
- `rq1_summary_table.csv` - Detailed operator metrics
- `rq2_evolution_comparison.png` - Fitness and diversity evolution
- `rq2_summary_table.csv` - Summary statistics (includes frozen species count and cluster quality metrics)

**Run Metadata**:
The script now extracts and displays run metadata:
- Run ID (directory name/timestamp)
- PG model (prompt generator) used
- RG model (response generator) used
- Speciation enabled (yes/no)
- Configuration parameters (theta_sim, theta_merge, etc.)

---

## Visualizations Generated

### RQ1 Visualizations

1. **Operator Effectiveness Comparison (6 subplots)**
   - Bar charts comparing NE, EHR, IR, cEHR, Δμ, Δσ
   - Side-by-side comparison: Speciation vs Non-Speciation
   - Error bars show standard deviation across runs

2. **Summary Statistics Table**
   - Mean ± Std for each metric per operator
   - Easy comparison of operator performance

### RQ2 Visualizations

1. **Fitness Evolution (2 subplots)**
   - Best fitness over generations
   - Average fitness over generations
   - Comparison: Speciation vs Non-Speciation

2. **Species and Diversity Metrics (2 subplots)**
   - Species count evolution (speciation only)
   - Inter/Intra-species diversity (speciation only)

3. **Summary Statistics Table**
   - Mean ± Std for all RQ2 metrics
   - Comparison between speciation and non-speciation

---

## Additional Analysis Ideas

### 1. Statistical Significance Testing

Add statistical tests to compare speciation vs non-speciation:

```python
from scipy.stats import mannwhitneyu, ttest_ind

# Example: Compare EHR between speciation and non-speciation
spec_ehr = [spec_rq1[op]['EHR_mean'] for op in operators]
nonspec_ehr = [nonspec_rq1[op]['EHR_mean'] for op in operators]

statistic, p_value = mannwhitneyu(spec_ehr, nonspec_ehr)
print(f"Mann-Whitney U test: statistic={statistic}, p-value={p_value}")
```

### 2. Operator Ranking

Rank operators by effectiveness:

```python
# Rank by EHR (higher is better)
operator_rankings = sorted(
    all_operators,
    key=lambda op: np.mean(spec_rq1[op]['EHR_mean']),
    reverse=True
)
```

### 3. Generation-by-Generation Analysis

Track how metrics change over time:

```python
# Plot operator effectiveness over generations
for op in top_operators:
    generations = [gen for gen in range(num_generations)]
    ehr_values = [get_ehr_for_generation(op, gen) for gen in generations]
    plt.plot(generations, ehr_values, label=op)
```

### 4. PG Model Comparison

Compare performance across different PG models:

```python
# Group by PG model (from prompt_generator_name in genomes)
pg_models = {}
for run_dir in SPECIATION_RUNS:
    data = load_execution_data(BASE_OUTPUT_DIR / run_dir)
    pg_model = extract_pg_model(data)  # Extract from elites.json
    pg_models[pg_model] = extract_rq1_metrics(data)
```

### 5. Diversity-Fitness Trade-off

Analyze relationship between diversity and fitness:

```python
# Scatter plot: Inter-species diversity vs Best fitness
plt.scatter(
    spec_rq2['inter_species_diversity_mean'],
    spec_rq2['best_fitness_mean'],
    alpha=0.6
)
plt.xlabel('Inter-Species Diversity')
plt.ylabel('Best Fitness')
```

---

## Data Structure Reference

### EvolutionTracker.json Structure

```json
{
  "generations": [
    {
      "generation_number": 0,
      "speciation": {
        "species_count": 22,
        "frozen_species_count": 0,  // NEW: Number of frozen species
        "reserves_size": 47,
        "best_fitness": 0.2367,
        "avg_fitness": 0.0339,
        "inter_species_diversity": 0.312,
        "intra_species_diversity": 0.1688,
        "speciation_events": 22,
        "merge_events": 0,
        "extinction_events": 0,
        "cluster_quality": {  // NEW: Cluster quality metrics
          "silhouette_score": 0.4523,
          "davies_bouldin_index": 1.234,
          "calinski_harabasz_index": 45.67,
          "num_samples": 69,
          "num_clusters": 22
        }
      },
      "operator_statistics": {
        "InformedEvolutionOperator": {
          "question_mark_rejections": 6,
          "duplicates_removed": 0
        }
      }
    }
  ]
}
```

### operator_effectiveness_cumulative.csv Structure

```csv
generation,operator,NE,EHR,IR,cEHR,Δμ,Δσ,total_variants,elite_count,non_elite_count,rejections,duplicates
1,InformedEvolutionOperator,15.5,45.2,12.3,52.1,0.023,0.045,22,10,5,3,4
```

### elites.json Structure

```json
[
  {
    "id": 1,
    "prompt": "How to...",
    "model_name": "models/llama3.1-8b-instruct-gguf/...",
    "prompt_generator_name": "models/llama3.1-8b-instruct-gguf/...",
    "fitness": 0.2367,
    "species_id": 5,
    "generation": 1,
    "operator": "InformedEvolutionOperator",
    "initial_state": "elite"
  }
]
```

---

## Troubleshooting

### Issue: Empty operator_effectiveness_cumulative.csv

**Cause**: No variants were successfully evaluated (all duplicates or rejections)

**Solution**: 
- Check if variants are being generated
- Verify moderation API is working
- Check duplicate detection logic

### Issue: Missing speciation metrics

**Cause**: Non-speciation runs don't have speciation data

**Solution**: 
- Script handles this by filling with zeros/NaN
- Only speciation-specific metrics (diversity, species count) will be NaN for non-speciation runs

### Issue: Different number of generations

**Cause**: Runs completed different numbers of generations

**Solution**: 
- Script aggregates using mean/max/final values
- Consider normalizing by generation count if needed

---

## Post-Run Aggregation Features

### Run Metadata Extraction

The aggregation script now automatically extracts:
- **Run ID**: Directory name (timestamp format: `YYYYMMDD_HHMM`)
- **PG Model**: Prompt generator model name (from `prompt_generator_name` in genomes)
- **RG Model**: Response generator model name (from `model_name` in genomes)
- **Speciation Enabled**: Detected by presence of `speciation_state.json`
- **Configuration Parameters**: Extracted from `speciation_state.json` (theta_sim, theta_merge, etc.)

### New Metrics in Aggregation

**RQ2 Metrics Added**:
- `frozen_species_count`: Tracks frozen species per generation
- `cluster_quality_silhouette_score`: Silhouette score (mean, max, final)
- `cluster_quality_davies_bouldin_index`: Davies-Bouldin index (mean, max, final)
- `cluster_quality_calinski_harabasz_index`: Calinski-Harabasz index (mean, max, final)

### Species Size/Age Distribution Analysis

For detailed species dynamics analysis, you can extract from `speciation_state.json`:
- **Species Size Distribution**: Min, max, mean, std per generation
- **Species Age Distribution**: Generations since creation per species
- **Species Survival Analysis**: How long species survive before freezing/merging

Example code:
```python
# Extract species size distribution from speciation_state.json
species_sizes = []
for sid, sp_data in speciation_state['species'].items():
    species_sizes.append(sp_data.get('size', 0))

size_mean = np.mean(species_sizes)
size_std = np.std(species_sizes)
```

## Next Steps

1. **Run the analysis** with your execution directories
2. **Review the visualizations** to identify patterns
3. **Perform statistical tests** to validate findings (can be added to aggregation script)
4. **Create publication-ready figures** with proper labels and legends
5. **Document findings** in your methodology/results section
6. **Analyze species dynamics** using speciation_state.json for size/age distributions

For questions or issues, check the code comments in `compare_speciation_vs_nonspeciation.py`.
