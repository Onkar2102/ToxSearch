#!/usr/bin/env python3
"""
RQ2: Operator Performance Analysis

This script analyzes operator performance metrics across multiple runs:
- NE: Non-elite percentage
- EHR: Elite Hit Rate
- IR: Invalid Rate  
- cEHR: Conditional Elite Hit Rate
- Δμ: Mean delta score (toxicity - parent_score)
- Δσ: Standard deviation of delta score

The script:
1. Processes all comb runs and generates metrics tables
2. Performs non-parametric statistical tests (Kruskal-Wallis, Mann-Whitney U)
3. Calculates effect sizes (Cohen's r) and 95% confidence intervals
4. Exports results to CSV and PDF files
"""

# Imports and Helper Functions
import os
import glob
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table
from scipy.stats import kruskal, mannwhitneyu, norm
from itertools import combinations

# Helper function to flatten operator_statistics column
def flatten_operator_statistics(df, col="operator_statistics"):
    """Flatten nested operator_statistics dictionary into separate columns"""
    if col not in df.columns:
        return df
    all_keys = set()
    for ops in df[col]:
        if isinstance(ops, dict):
            all_keys.update(ops.keys())
    
    for op_key in all_keys:
        flat_rows = []
        for ops in df[col]:
            if isinstance(ops, dict) and op_key in ops and isinstance(ops[op_key], dict):
                prefix = f"operator_statistics_{op_key}_"
                row = {prefix + subk: subv for subk, subv in ops[op_key].items()}
                flat_rows.append(row)
            else:
                flat_rows.append({})
        flat_df = pd.DataFrame(flat_rows)
        df = pd.concat([df.reset_index(drop=True), flat_df.reset_index(drop=True)], axis=1)
    df = df.drop(columns=[col])
    return df

# Define crossover operators (others are mutations)
CROSSOVER_OPERATORS = {'SemanticSimilarityCrossover', 'SemanticFusionCrossover'}


# Process all comb runs and generate final table
# This cell processes all run*_comb directories and creates the final metrics table

# Setup paths - use script's directory as reference
script_dir = os.path.dirname(os.path.abspath(__file__))
# Script is in experiments/, so go up one level to project root
project_root = os.path.dirname(script_dir)
base_data_dir = os.path.join(project_root, "data", "outputs")
base_data_dir = os.path.normpath(base_data_dir)

# Find all comb runs
pattern = os.path.join(base_data_dir, "run*_comb")
run_dirs = sorted(glob.glob(pattern))
run_dirs = [os.path.basename(d.rstrip('/')) for d in run_dirs]

if not run_dirs:
    raise ValueError(f"No comb run directories found in {base_data_dir}")

print(f"Found {len(run_dirs)} comb runs: {run_dirs}")

def process_single_run(run_dir):
    """Process a single run directory and return metrics per operator"""
    data_dir = os.path.join(base_data_dir, run_dir)
    
    if not os.path.exists(data_dir):
        return None
    
    # Load all files
    dfs = {}
    filenames = [f for f in os.listdir(data_dir) if not f.startswith(".") and os.path.isfile(os.path.join(data_dir, f))]
    
    for fname in filenames:
        file_path = os.path.join(data_dir, fname)
        ext = os.path.splitext(fname)[1].lower()
        try:
            if fname == "EvolutionTracker.json":
                with open(file_path, 'r') as f:
                    jdata = json.load(f)
                if 'generations' in jdata and isinstance(jdata['generations'], list):
                    df = pd.DataFrame(jdata['generations'])
                    if "operator_statistics" in df.columns:
                        df = flatten_operator_statistics(df, col="operator_statistics")
                else:
                    df = pd.json_normalize(jdata)
            elif ext == ".json":
                try:
                    df = pd.read_json(file_path)
                except Exception:
                    with open(file_path, "r") as f:
                        jdata = json.load(f)
                    if isinstance(jdata, list):
                        df = pd.DataFrame(jdata)
                    elif isinstance(jdata, dict):
                        df = pd.json_normalize(jdata)
            else:
                continue
        except Exception:
            continue
        
        if df is not None:
            df_name = os.path.splitext(fname)[0]
            dfs[df_name] = df
    
    # Flatten nested structures in elites, non_elites, under_performing
    for label in ['elites', 'non_elites', 'under_performing']:
        if label in dfs:
            df = dfs[label]
            cols_to_flatten = []
            for col in df.columns:
                sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample_val, (dict, list)):
                    cols_to_flatten.append(col)
            try:
                for col in cols_to_flatten:
                    flattened = pd.json_normalize(df[col])
                    flattened.columns = [f"{col}_{c}" for c in flattened.columns]
                    df = df.drop(columns=[col]).reset_index(drop=True)
                    df = pd.concat([df, flattened], axis=1)
                dfs[label] = df
            except Exception:
                pass
    
    # Create unified_df
    group_labels = ['elites', 'non_elites', 'under_performing']
    selected_dfs = []
    for label in group_labels:
        if label in dfs:
            df = dfs[label].copy()
            if df.empty:
                continue
            df['_source_group'] = label
            selected_dfs.append(df)
    
    if not selected_dfs:
        return None
    
    unified_df = pd.concat(selected_dfs, ignore_index=True, sort=False)
    
    # Calculate delta_score
    unified_df['delta_score'] = unified_df['moderation_result_google.scores.toxicity'] - unified_df['parent_score']
    
    # Get EvolutionTracker_df
    EvolutionTracker_df = dfs.get('EvolutionTracker', None)
    if EvolutionTracker_df is None:
        return None
    
    # Create operator vs initial_state crosstab
    operator_vs_initial_state = pd.crosstab(
        unified_df['operator'].fillna('Initial Seed'),
        unified_df['initial_state'].fillna('none')
    )
    operator_vs_initial_state['total'] = operator_vs_initial_state.sum(axis=1)
    
    # Get operator statistics columns
    operator_stats_cols = [col for col in EvolutionTracker_df.columns if col.startswith('operator_statistics_')]
    pattern_question = re.compile(r'operator_statistics_(.*?)_question_mark_rejections')
    pattern_duplicates = re.compile(r'operator_statistics_(.*?)_duplicates_removed')
    
    operator_names = set()
    for col in operator_stats_cols:
        m_q = pattern_question.match(col)
        m_d = pattern_duplicates.match(col)
        if m_q:
            operator_names.add(m_q.group(1))
        if m_d:
            operator_names.add(m_d.group(1))
    
    # Calculate delta stats (do this early as it's independent)
    operator_delta_stats = unified_df.groupby('operator')['delta_score'].agg(['mean', 'std']).round(2)
    
    # Build result DataFrame directly (avoid intermediate DataFrames)
    result_data = {}
    all_operators_set = set(operator_names) | set(operator_vs_initial_state.index) - {'Initial Seed'}
    
    for operator in sorted(all_operators_set):
        # Get counts from operator_vs_initial_state
        if operator in operator_vs_initial_state.index:
            elite = operator_vs_initial_state.loc[operator, 'elite'] if 'elite' in operator_vs_initial_state.columns else 0
            non_elite = operator_vs_initial_state.loc[operator, 'non_elite'] if 'non_elite' in operator_vs_initial_state.columns else 0
            total = operator_vs_initial_state.loc[operator, 'total']
        else:
            elite = non_elite = total = 0
        
        # Get cleaning stats (only for operators in operator_names)
        if operator in operator_names:
            col_q = f'operator_statistics_{operator}_question_mark_rejections'
            col_d = f'operator_statistics_{operator}_duplicates_removed'
            question_removed = EvolutionTracker_df[col_q].sum() if col_q in EvolutionTracker_df.columns else 0
            duplicates_removed = EvolutionTracker_df[col_d].sum() if col_d in EvolutionTracker_df.columns else 0
        else:
            question_removed = duplicates_removed = 0
        
        # Calculate total (including removed items)
        calculated_total = total + question_removed + duplicates_removed
        
        if calculated_total == 0:
            continue
        
        # Calculate metrics directly as percentages (handle division by zero)
        NE = (non_elite / calculated_total * 100).round(2) if calculated_total > 0 else 0.0
        EHR = (elite / calculated_total * 100).round(2) if calculated_total > 0 else 0.0
        IR = (question_removed / calculated_total * 100).round(2) if calculated_total > 0 else 0.0
        cEHR = (elite / total * 100).round(2) if total > 0 else 0.0
        
        # Get delta stats
        delta_mean = operator_delta_stats.loc[operator, 'mean'] if operator in operator_delta_stats.index else np.nan
        delta_std = operator_delta_stats.loc[operator, 'std'] if operator in operator_delta_stats.index else np.nan
        
        result_data[operator] = {
            'NE': NE,
            'EHR': EHR,
            'IR': IR,
            'cEHR': cEHR,
            'Δμ': delta_mean,
            'Δσ': delta_std
        }
    
    result_df = pd.DataFrame(result_data).T
    return result_df[['NE', 'EHR', 'IR', 'cEHR', 'Δμ', 'Δσ']]

# Process all runs
all_run_results = {}
for run_dir in run_dirs:
    run_match = re.search(r'run(\d+)_comb', run_dir)
    if run_match:
        run_key = run_match.group(1)
    else:
        run_key = run_dir.replace('run', '').replace('_comb', '')
    
    result = process_single_run(run_dir)
    if result is not None:
        all_run_results[run_key] = result
        print(f"Processed {run_dir} -> E{run_key}")

# Get all unique operators across all runs
all_operators = set()
for run_key, df in all_run_results.items():
    all_operators.update(df.index.tolist())

# Get all run keys sorted numerically
sorted_run_keys = sorted(all_run_results.keys(), key=lambda x: int(x) if x.isdigit() else 999)

# Create table data
table_rows = []
for operator in sorted(all_operators):
    # Get data for each run
    run_data = {}
    for run_key in sorted_run_keys:
        if run_key in all_run_results and operator in all_run_results[run_key].index:
            run_data[run_key] = all_run_results[run_key].loc[operator]
        else:
            run_data[run_key] = None
    
    # Add individual run rows
    for run_key in sorted_run_keys:
        if run_data[run_key] is not None:
            row_data = run_data[run_key]
            table_rows.append({
                'Operator': operator,
                'Exec': f'E{run_key}',
                'NE': row_data['NE'],
                'EHR': row_data['EHR'],
                'IR': row_data['IR'],
                'cEHR': row_data['cEHR'],
                'Δμ': row_data['Δμ'],
                'Δσ': row_data['Δσ'],
                'is_mean': False
            })
        else:
            table_rows.append({
                'Operator': operator,
                'Exec': f'E{run_key}',
                'NE': np.nan,
                'EHR': np.nan,
                'IR': np.nan,
                'cEHR': np.nan,
                'Δμ': np.nan,
                'Δσ': np.nan,
                'is_mean': False
            })
    
    # Calculate mean across all valid runs
    valid_runs = [run_data[k] for k in sorted_run_keys if run_data[k] is not None]
    if valid_runs:
        mean_data = pd.DataFrame(valid_runs).mean()
        table_rows.append({
            'Operator': operator,
            'Exec': 'Mean',
            'NE': round(mean_data['NE'], 2),
            'EHR': round(mean_data['EHR'], 2),
            'IR': round(mean_data['IR'], 2),
            'cEHR': round(mean_data['cEHR'], 2),
            'Δμ': round(mean_data['Δμ'], 2),
            'Δσ': round(mean_data['Δσ'], 2),
            'is_mean': True
        })

# Create DataFrame
final_table_df = pd.DataFrame(table_rows)

# Display table
print("\n" + "="*100)
print("RQ2: Operator Performance Metrics Across Multiple Runs")
print("="*100)
print("\nFinal Table DataFrame:")
print(final_table_df)

# Create formatted table for PDF export
fig, ax = plt.subplots(figsize=(16, len(final_table_df) * 0.5 + 2))
ax.axis('tight')
ax.axis('off')

headers = ['Operator', 'Exec', 'NE', 'EHR', 'IR', 'cEHR', 'Δμ', 'Δσ']
table_data = [[row['Operator'], row['Exec'],
               f"{row['NE']:.2f}" if not pd.isna(row['NE']) else 'N/A',
               f"{row['EHR']:.2f}" if not pd.isna(row['EHR']) else 'N/A',
               f"{row['IR']:.2f}" if not pd.isna(row['IR']) else 'N/A',
               f"{row['cEHR']:.2f}" if not pd.isna(row['cEHR']) else 'N/A',
               f"{row['Δμ']:.2f}" if not pd.isna(row['Δμ']) else 'N/A',
               f"{row['Δσ']:.2f}" if not pd.isna(row['Δσ']) else 'N/A']
              for _, row in final_table_df.iterrows()]

table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

# Style header
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style mean rows and alternate row colors (combine iteration)
for row_idx, (_, row) in enumerate(final_table_df.iterrows(), start=1):
    for j in range(len(headers)):
        if row['is_mean']:
            table[(row_idx, j)].set_facecolor('#B3E5FC')
            table[(row_idx, j)].set_text_props(weight='bold')
        else:
            table[(row_idx, j)].set_facecolor('#f0f0f0' if row_idx % 2 == 0 else 'white')

plt.title('RQ2: Operator Performance Metrics (Rates % and Deltas)', fontsize=14, fontweight='bold', pad=20)

# Save to PDF - use script's directory as output directory
output_dir = script_dir

filename_pdf = os.path.join(output_dir, "rq2_operator_metrics_table.pdf")
if os.path.exists(filename_pdf):
    os.remove(filename_pdf)
plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
plt.close()

print(f"\nTable saved to: {filename_pdf}")

# Create simplified table for report (one row per operator - mean values only)
simplified_table_df = final_table_df[final_table_df['is_mean'] == True].copy()
simplified_table_df = simplified_table_df[['Operator', 'NE', 'EHR', 'IR', 'cEHR', 'Δμ', 'Δσ']].copy()
simplified_table_df = simplified_table_df.sort_values('Operator').reset_index(drop=True)

print("\n" + "="*100)
print("Simplified Table for Report (One Row Per Operator - Mean Across All Runs)")
print("="*100)
print("\nSimplified Table DataFrame:")
print(simplified_table_df)

# Create formatted PDF table for simplified version
fig, ax = plt.subplots(figsize=(14, len(simplified_table_df) * 0.4 + 2))
ax.axis('tight')
ax.axis('off')

headers = ['Operator', 'NE (%)', 'EHR (%)', 'IR (%)', 'cEHR (%)', 'Δμ', 'Δσ']
table_data = [[row['Operator'],
               f"{row['NE']:.2f}" if not pd.isna(row['NE']) else 'N/A',
               f"{row['EHR']:.2f}" if not pd.isna(row['EHR']) else 'N/A',
               f"{row['IR']:.2f}" if not pd.isna(row['IR']) else 'N/A',
               f"{row['cEHR']:.2f}" if not pd.isna(row['cEHR']) else 'N/A',
               f"{row['Δμ']:.2f}" if not pd.isna(row['Δμ']) else 'N/A',
               f"{row['Δσ']:.2f}" if not pd.isna(row['Δσ']) else 'N/A']
              for _, row in simplified_table_df.iterrows()]

table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.0)

# Style header
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows (alternate colors)
for row_idx in range(1, len(table_data) + 1):
    for j in range(len(headers)):
        table[(row_idx, j)].set_facecolor('#f0f0f0' if row_idx % 2 == 0 else 'white')

plt.title('RQ2: Operator Performance Metrics (Mean Across All Runs)', fontsize=14, fontweight='bold', pad=20)

# Save simplified table to PDF
filename_simplified = os.path.join(output_dir, "rq2_operator_metrics_simplified.pdf")
if os.path.exists(filename_simplified):
    os.remove(filename_simplified)
plt.savefig(filename_simplified, dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSimplified table saved to: {filename_simplified}")

# Export to CSV
csv_filename = os.path.join(output_dir, "rq2_operator_metrics_simplified.csv")
simplified_table_df.to_csv(csv_filename, index=False)
print(f"Simplified table exported to: {csv_filename}")

# Non-Parametric Statistical Analysis
# Kruskal-Wallis H-test for each metric, followed by post-hoc Mann-Whitney U tests

# Prepare data for statistical tests (exclude mean rows, only use individual runs)
# Ensure final_table_df exists (created in previous cell)
if 'final_table_df' not in globals() or final_table_df.empty:
    raise ValueError("final_table_df not found. Please run the main processing cell first.")

test_data_df = final_table_df[final_table_df['is_mean'] == False].copy()

# Metrics to test
metrics = ['EHR', 'cEHR', 'IR', 'NE', 'Δμ', 'Δσ']
metric_names = {
    'EHR': 'Elite Hit Rate (%)',
    'cEHR': 'Conditional Elite Hit Rate (%)',
    'IR': 'Invalid Rate (%)',
    'NE': 'Non-Elite Percentage (%)',
    'Δμ': 'Mean Delta Score',
    'Δσ': 'Delta Score Std Dev'
}

# Store results
statistical_results = {}

print("="*100)
print("Non-Parametric Statistical Analysis")
print("="*100)

for metric in metrics:
    print(f"\n{'='*100}")
    print(f"Metric: {metric_names[metric]}")
    print(f"{'='*100}")
    
    # Prepare data: operator -> list of values across runs
    operator_data = {}
    for operator in sorted(all_operators):
        operator_df = test_data_df[(test_data_df['Operator'] == operator) & 
                                    (test_data_df[metric].notna())]
        values = operator_df[metric].dropna().tolist()
        if len(values) > 0:
            operator_data[operator] = values
    
    if len(operator_data) < 2:
        print(f"Insufficient data for {metric}")
        continue
    
    # Kruskal-Wallis H-test (tests if any operators differ)
    # Filter out operators with no data
    operators_with_data = [op for op in sorted(operator_data.keys()) if len(operator_data[op]) > 0]
    
    if len(operators_with_data) < 2:
        print(f"Insufficient operators with data for {metric} (need at least 2 operators)")
        continue
    
    groups = [operator_data[op] for op in operators_with_data]
    operators_list = operators_with_data
    
    try:
        h_statistic, p_value = kruskal(*groups)
        print(f"\nKruskal-Wallis H-test:")
        print(f"  H-statistic: {h_statistic:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        
        if p_value < 0.05:
            print(f"  → Significant difference found (p < 0.05)")
        else:
            print(f"  → No significant difference (p >= 0.05)")
        
        statistical_results[metric] = {
            'kruskal_wallis': {
                'h_statistic': h_statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'operator_data': operator_data,
            'operators': operators_list
        }
        
        # Post-hoc Mann-Whitney U tests (pairwise comparisons) if Kruskal-Wallis is significant
        if p_value < 0.05:
            print(f"\nPost-hoc Mann-Whitney U tests (Bonferroni corrected):")
            pairwise_results = []
            operator_pairs = list(combinations(operators_list, 2))
            num_comparisons = len(operator_pairs)
            bonferroni_alpha = 0.05 / num_comparisons
            
            print(f"  Number of comparisons: {num_comparisons}")
            print(f"  Bonferroni corrected α: {bonferroni_alpha:.6f}")
            print(f"\n  Significant pairwise differences (p < {bonferroni_alpha:.6f}):")
            print(f"  (Format: Operator1 >/< Operator2 (p-value, U-statistic, effect size r, 95% CI))")
            
            significant_pairs = []
            for op1, op2 in operator_pairs:
                try:
                    # Check if both operators have data
                    if len(operator_data[op1]) == 0 or len(operator_data[op2]) == 0:
                        continue
                    
                    data1 = np.array(operator_data[op1])
                    data2 = np.array(operator_data[op2])
                    n1, n2 = len(data1), len(data2)
                    
                    u_statistic, p_val = mannwhitneyu(
                        data1, 
                        data2, 
                        alternative='two-sided'
                    )
                    is_significant = p_val < bonferroni_alpha
                    
                    # Calculate mean difference
                    mean1 = np.mean(data1)
                    mean2 = np.mean(data2)
                    mean_diff = mean1 - mean2
                    
                    # Calculate effect size (Cohen's r for Mann-Whitney U)
                    # r = Z / sqrt(N), where Z = (U - n1*n2/2) / sqrt(n1*n2*(n1+n2+1)/12)
                    expected_u = n1 * n2 / 2
                    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                    if std_u > 0:
                        z_score = (u_statistic - expected_u) / std_u
                        n_total = n1 + n2
                        effect_size_r = z_score / np.sqrt(n_total)
                    else:
                        effect_size_r = np.nan
                    
                    # Calculate 95% confidence interval for mean difference using bootstrap
                    # Bootstrap method: resample with replacement and calculate mean difference
                    n_bootstrap = 1000
                    bootstrap_diffs = []
                    for _ in range(n_bootstrap):
                        sample1 = np.random.choice(data1, size=n1, replace=True)
                        sample2 = np.random.choice(data2, size=n2, replace=True)
                        bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))
                    
                    # Calculate 95% CI (2.5th and 97.5th percentiles)
                    ci_lower = np.percentile(bootstrap_diffs, 2.5)
                    ci_upper = np.percentile(bootstrap_diffs, 97.5)
                    
                    if is_significant:
                        direction = ">" if mean1 > mean2 else "<"
                        print(f"    {op1} {direction} {op2} (p={p_val:.6f}, U={u_statistic:.2f}, r={effect_size_r:.3f}, 95% CI=[{ci_lower:.2f}, {ci_upper:.2f}])")
                        significant_pairs.append((op1, op2, p_val, mean1, mean2))
                    
                    pairwise_results.append({
                        'operator1': op1,
                        'operator2': op2,
                        'u_statistic': u_statistic,
                        'p_value': p_val,
                        'significant': is_significant,
                        'mean1': mean1,
                        'mean2': mean2,
                        'mean_diff': mean_diff,
                        'effect_size_r': effect_size_r,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper
                    })
                except Exception as e:
                    print(f"    Error comparing {op1} vs {op2}: {e}")
                    continue
            
            if not significant_pairs:
                print(f"    None (after Bonferroni correction)")
            
            # Print summary of effect sizes for all comparisons
            print(f"\n  Effect Size Summary (all comparisons):")
            effect_sizes = [p.get('effect_size_r', np.nan) for p in pairwise_results if not np.isnan(p.get('effect_size_r', np.nan))]
            if len(effect_sizes) > 0:
                small_effects = sum(1 for r in effect_sizes if abs(r) < 0.1)
                medium_effects = sum(1 for r in effect_sizes if 0.1 <= abs(r) < 0.3)
                large_effects = sum(1 for r in effect_sizes if abs(r) >= 0.3)
                print(f"    Small (|r| < 0.1): {small_effects}")
                print(f"    Medium (0.1 ≤ |r| < 0.3): {medium_effects}")
                print(f"    Large (|r| ≥ 0.3): {large_effects}")
            
            statistical_results[metric]['pairwise'] = pairwise_results
            statistical_results[metric]['significant_pairs'] = significant_pairs
        
        # Summary statistics per operator
        print(f"\nSummary Statistics by Operator:")
        summary_stats = []
        for operator in operators_list:
            values = operator_data[operator]
            summary_stats.append({
                'Operator': operator,
                'Mean': np.mean(values),
                'Median': np.median(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values),
                'N': len(values)
            })
        
        summary_df = pd.DataFrame(summary_stats)
        # Sort by mean (descending for EHR, cEHR; ascending for others)
        if metric in ['EHR', 'cEHR']:
            summary_df = summary_df.sort_values('Mean', ascending=False)
        else:
            summary_df = summary_df.sort_values('Mean', ascending=True)
        print("\nSummary Statistics by Operator:")
        print(summary_df)
        
    except Exception as e:
        print(f"Error in Kruskal-Wallis test: {e}")
        continue

print(f"\n{'='*100}")
print("Statistical Analysis Complete")
print(f"{'='*100}")

# Create summary of significant findings
print("\n" + "="*100)
print("Summary of Significant Findings")
print("="*100)

significant_metrics = [m for m in metrics if m in statistical_results and 
                       statistical_results[m]['kruskal_wallis']['significant']]

if significant_metrics:
    print(f"\nMetrics with significant operator differences (p < 0.05):")
    for metric in significant_metrics:
        p_val = statistical_results[metric]['kruskal_wallis']['p_value']
        num_sig_pairs = len(statistical_results[metric].get('significant_pairs', []))
        print(f"  - {metric_names[metric]}: p={p_val:.6f}, {num_sig_pairs} significant pairwise differences")
else:
    print("\nNo metrics showed significant operator differences (p >= 0.05)")

# Export statistical results to CSV
# Use script's directory as output directory (already defined above)
if 'output_dir' not in globals():
    output_dir = script_dir

# Export summary statistics
summary_all = []
for metric in metrics:
    if metric in statistical_results:
        for operator in statistical_results[metric]['operators']:
            values = statistical_results[metric]['operator_data'][operator]
            summary_all.append({
                'Metric': metric_names[metric],
                'Operator': operator,
                'Mean': np.mean(values),
                'Median': np.median(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values),
                'N': len(values)
            })

if summary_all:
    summary_df_all = pd.DataFrame(summary_all)
    csv_filename = os.path.join(output_dir, "rq2_statistical_summary.csv")
    summary_df_all.to_csv(csv_filename, index=False)
    print(f"\nSummary statistics exported to: {csv_filename}")

# Export pairwise comparison results
pairwise_all = []
for metric in metrics:
    if metric in statistical_results and 'pairwise' in statistical_results[metric]:
        for pair in statistical_results[metric]['pairwise']:
            pairwise_all.append({
                'Metric': metric_names[metric],
                'Operator1': pair['operator1'],
                'Operator2': pair['operator2'],
                'U_statistic': pair['u_statistic'],
                'p_value': pair['p_value'],
                'Significant': pair['significant'],
                'Mean1': pair['mean1'],
                'Mean2': pair['mean2'],
                'Mean_Diff': pair.get('mean_diff', np.nan),
                'Effect_Size_r': pair.get('effect_size_r', np.nan),
                'CI_Lower': pair.get('ci_lower', np.nan),
                'CI_Upper': pair.get('ci_upper', np.nan)
            })

if pairwise_all:
    pairwise_df = pd.DataFrame(pairwise_all)
    csv_filename = os.path.join(output_dir, "rq2_pairwise_comparisons.csv")
    pairwise_df.to_csv(csv_filename, index=False)
    print(f"Pairwise comparisons exported to: {csv_filename}")
