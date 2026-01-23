# RQ2 Answer Rewrite Summary

## Key Changes from Original to Updated Analysis

### Methodology Updates
1. **Sample Size**: Original analysis used top 25% (174 prompts); updated analysis uses all 437 unique elite prompts (from comb runs only)
2. **Source**: The 696 prompts are collected using the **same pattern as RQ2**: `run*_comb/elites.json`
   - **All comb executions included**: Uses `glob.glob("run*_comb/elites.json")` which collects elites from ALL `run*_comb` directories (no sampling, all execution runs included)
   - Same as RQ2: Both RQ2 and RQ3 collect from all comb execution runs
   - Only `run*_comb` directories (excludes `run*_ops` directories)
   - Model directories are used separately for target model evaluations
   - All elites from all comb execution runs are combined, then deduplicated by prompt text (keeping highest toxicity per unique prompt)
3. **Filtering**: Both analyses deduplicate by prompt text and filter for questions (ending with "?")
4. **Evaluation**: All prompts tested on all target models with identical decoding parameters

### Statistical Updates

#### Source Model (LlaMA 3.1 8B)
- **Original**: Mean=0.4900, Median=0.4440, Std=0.1130
- **Updated**: Mean=0.3421, Median=0.3326, Std=0.0496, Min=0.2786, Max=0.6025, Q1=0.3053, Q3=0.3666, IQR=0.0613
- **Note**: Lower values in updated analysis reflect inclusion of all elites from comb runs only (n=437)

#### Target Models - Updated Statistics
All models show substantial toxicity reduction (30-54% relative to source):

1. **Qwen 2.5 7B** (highest among targets)
   - Mean=0.2401, Median=0.2592, Std=0.1040, Min=0.0191, Max=0.5433, Q1=0.1685, Q3=0.3169, IQR=0.1484
   - Reduction: ~30% from source

2. **Mistral 7B**
   - Mean=0.2076, Median=0.2057, Std=0.0983, Min=0.0084, Max=0.4814, Q1=0.1374, Q3=0.2855, IQR=0.1481
   - Reduction: ~39%

3. **Phi-3.5 Mini**
   - Mean=0.1887, Median=0.1987, Std=0.1060, Min=0.0135, Max=0.5198, Q1=0.0944, Q3=0.2855, IQR=0.1911
   - Reduction: ~45%

4. **Gemma 2 9B**
   - Mean=0.1886, Median=0.1999, Std=0.0885, Min=0.0134, Max=0.3992, Q1=0.1074, Q3=0.2546, IQR=0.1472
   - Reduction: ~45%

5. **LlaMA 3.2 1B** (strong resistance)
   - Mean=0.1926, Median=0.2178, Std=0.1169, Min=0.0080, Max=0.4269, Q1=0.0931, Q3=0.2867, IQR=0.1936
   - Reduction: ~44%

6. **LlaMA 3.2 3B** (strongest resistance)
   - Mean=0.1576, Median=0.1304, Std=0.1207, Min=0.0102, Max=0.4850, Q1=0.0412, Q3=0.2648, IQR=0.2236
   - Reduction: ~54%

### Key Findings Maintained
1. **Transfer attenuation**: All models show reduced toxicity relative to source
2. **Within-family resistance**: LlaMA variants show strongest resistance despite architectural similarity
3. **Cross-architecture variability**: Different architectures show varied resistance patterns
4. **High-toxicity outliers**: Some prompts retain high toxicity (max values 0.3992-0.5433) across models
5. **Invalid responses**: Smaller LlaMA variants show highest refusal rates

### Scientific Improvements
1. Added IQR (Interquartile Range) to better characterize distribution spread
2. More precise percentage reductions (30-54% range)
3. Updated to reflect current analysis using only comb runs (n=437 prompts)
3. Better characterization of variability through IQR and maximum values
4. Clearer distinction between within-family and cross-architecture models
5. Enhanced discussion of defensive mechanisms and alignment effects

### Figures Referenced
- `all_elites_toxicity_distribution_all_models.pdf`: Violin plots with jittered points
- `rq3_invalid_fraction_per_model.pdf`: Bar chart of invalid response percentages

### Data Source
- Statistics from: `experiments/rq3_statistics_table.csv`
- Analysis script: `experiments/analysis.py` (main() function)
- **Collection method**: Uses `glob.glob("run*_comb/elites.json")` which collects from ALL comb execution runs (no sampling, all executions included)
