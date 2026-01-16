"""
Comprehensive Analysis: Speciation vs Non-Speciation Comparison

This script analyzes and compares:
- RQ1: Operator Effectiveness (NE, EHR, IR, cEHR, Δμ, Δσ)
- RQ2: Cluster Quality and Diversity (species count, diversity metrics, fitness evolution)

For:
- 5 runs with speciation (different PG models)
- 5 runs without speciation (old toxsearch)
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Configuration
BASE_OUTPUT_DIR = Path("../data/outputs")
RESULTS_DIR = Path("comparison_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Define your execution directories here
# Update these paths to match your actual execution directories
SPEciation_RUNS = [
    # Add your 5 speciation run directories here
    # Example: "20260115_1609", "20260115_1700", etc.
]

NON_SPECIATION_RUNS = [
    # Add your 5 non-speciation (old toxsearch) run directories here
    # Example: "20260115_1800", "20260115_1900", etc.
]


def load_execution_data(output_dir: Path):
    """Load all data from an execution directory."""
    data = {
        'evolution_tracker': None,
        'speciation_state': None,
        'elites': [],
        'reserves': [],
        'archive': [],
        'operator_effectiveness': None,
    }
    
    try:
        # Load EvolutionTracker
        et_path = output_dir / "EvolutionTracker.json"
        if et_path.exists():
            with open(et_path, 'r') as f:
                data['evolution_tracker'] = json.load(f)
        
        # Load speciation state (only for speciation runs)
        ss_path = output_dir / "speciation_state.json"
        if ss_path.exists():
            with open(ss_path, 'r') as f:
                data['speciation_state'] = json.load(f)
        
        # Load elites
        elites_path = output_dir / "elites.json"
        if elites_path.exists():
            with open(elites_path, 'r') as f:
                elites_data = json.load(f)
                data['elites'] = elites_data if isinstance(elites_data, list) else []
        
        # Load reserves
        reserves_path = output_dir / "reserves.json"
        if reserves_path.exists():
            with open(reserves_path, 'r') as f:
                reserves_data = json.load(f)
                data['reserves'] = reserves_data if isinstance(reserves_data, list) else []
        
        # Load archive
        archive_path = output_dir / "archive.json"
        if archive_path.exists():
            with open(archive_path, 'r') as f:
                archive_data = json.load(f)
                data['archive'] = archive_data if isinstance(archive_data, list) else []
        
        # Load operator effectiveness CSV
        oe_path = output_dir / "operator_effectiveness_cumulative.csv"
        if oe_path.exists():
            try:
                data['operator_effectiveness'] = pd.read_csv(oe_path)
            except:
                pass
    
    except Exception as e:
        print(f"Error loading data from {output_dir}: {e}")
    
    return data


def extract_rq1_metrics(data):
    """Extract RQ1 (Operator Effectiveness) metrics."""
    metrics = {
        'operators': {},
        'summary': {}
    }
    
    # From operator_effectiveness CSV
    if data['operator_effectiveness'] is not None and not data['operator_effectiveness'].empty:
        df = data['operator_effectiveness']
        for _, row in df.iterrows():
            op = row.get('operator', 'Unknown')
            if op not in metrics['operators']:
                metrics['operators'][op] = {
                    'NE': [], 'EHR': [], 'IR': [], 'cEHR': [], 'Δμ': [], 'Δσ': []
                }
            
            if pd.notna(row.get('NE')):
                metrics['operators'][op]['NE'].append(row['NE'])
            if pd.notna(row.get('EHR')):
                metrics['operators'][op]['EHR'].append(row['EHR'])
            if pd.notna(row.get('IR')):
                metrics['operators'][op]['IR'].append(row['IR'])
            if pd.notna(row.get('cEHR')):
                metrics['operators'][op]['cEHR'].append(row['cEHR'])
            if pd.notna(row.get('Δμ')):
                metrics['operators'][op]['Δμ'].append(row['Δμ'])
            if pd.notna(row.get('Δσ')):
                metrics['operators'][op]['Δσ'].append(row['Δσ'])
    
    # Calculate summary statistics
    for op, values in metrics['operators'].items():
        metrics['summary'][op] = {}
        for metric_name, metric_values in values.items():
            if metric_values:
                metrics['summary'][op][f'{metric_name}_mean'] = np.mean(metric_values)
                metrics['summary'][op][f'{metric_name}_std'] = np.std(metric_values)
                metrics['summary'][op][f'{metric_name}_count'] = len(metric_values)
    
    return metrics


def extract_rq2_metrics(data):
    """Extract RQ2 (Cluster Quality and Diversity) metrics."""
    metrics = {
        'generations': [],
        'species_count': [],
        'reserves_size': [],
        'best_fitness': [],
        'avg_fitness': [],
        'inter_species_diversity': [],
        'intra_species_diversity': [],
        'merge_events': [],
        'extinction_events': [],
        'speciation_events': [],
    }
    
    et = data.get('evolution_tracker')
    if not et:
        return metrics
    
    generations = et.get('generations', [])
    for gen in generations:
        gen_num = gen.get('generation_number')
        if gen_num is None:
            continue
        
        metrics['generations'].append(gen_num)
        
        # Speciation metrics
        spec = gen.get('speciation', {})
        if spec:
            metrics['species_count'].append(spec.get('species_count', 0))
            metrics['reserves_size'].append(spec.get('reserves_size', 0))
            metrics['best_fitness'].append(spec.get('best_fitness', 0))
            metrics['avg_fitness'].append(spec.get('avg_fitness', 0))
            metrics['inter_species_diversity'].append(spec.get('inter_species_diversity', 0))
            metrics['intra_species_diversity'].append(spec.get('intra_species_diversity', 0))
            metrics['merge_events'].append(spec.get('merge_events', 0))
            metrics['extinction_events'].append(spec.get('extinction_events', 0))
            metrics['speciation_events'].append(spec.get('speciation_events', 0))
        else:
            # Non-speciation run - fill with zeros or N/A
            metrics['species_count'].append(0)
            metrics['reserves_size'].append(0)
            metrics['best_fitness'].append(gen.get('max_score_variants', 0))
            metrics['avg_fitness'].append(gen.get('avg_fitness', 0))
            metrics['inter_species_diversity'].append(np.nan)
            metrics['intra_species_diversity'].append(np.nan)
            metrics['merge_events'].append(0)
            metrics['extinction_events'].append(0)
            metrics['speciation_events'].append(0)
    
    return metrics


def aggregate_runs(run_dirs, run_type="speciation"):
    """Aggregate metrics across multiple runs."""
    all_rq1 = defaultdict(lambda: defaultdict(list))
    all_rq2 = defaultdict(list)
    
    for run_dir in run_dirs:
        output_path = BASE_OUTPUT_DIR / run_dir
        if not output_path.exists():
            print(f"Warning: {output_path} does not exist, skipping...")
            continue
        
        data = load_execution_data(output_path)
        
        # RQ1 metrics
        rq1 = extract_rq1_metrics(data)
        for op, op_metrics in rq1['summary'].items():
            for metric_name, metric_value in op_metrics.items():
                all_rq1[op][metric_name].append(metric_value)
        
        # RQ2 metrics
        rq2 = extract_rq2_metrics(data)
        for key, values in rq2.items():
            if key != 'generations':
                # Aggregate across generations (mean, max, final)
                if values:
                    all_rq2[f'{key}_mean'].append(np.nanmean(values))
                    all_rq2[f'{key}_max'].append(np.nanmax(values) if not all(np.isnan(values)) else np.nan)
                    all_rq2[f'{key}_final'].append(values[-1] if values else np.nan)
    
    return all_rq1, all_rq2


def create_rq1_visualizations(spec_rq1, nonspec_rq1):
    """Create RQ1 operator effectiveness visualizations."""
    
    # 1. Operator Effectiveness Comparison (Bar Chart)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    metrics = ['NE', 'EHR', 'IR', 'cEHR', 'Δμ', 'Δσ']
    metric_labels = {
        'NE': 'Non-Elite %',
        'EHR': 'Elite Hit Rate %',
        'IR': 'Invalid Rate %',
        'cEHR': 'Conditional EHR %',
        'Δμ': 'Mean Delta Score',
        'Δσ': 'Delta Score Std Dev'
    }
    
    # Get all operators
    all_ops = set()
    for op_dict in [spec_rq1, nonspec_rq1]:
        all_ops.update(op_dict.keys())
    all_ops = sorted(all_ops)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        spec_means = []
        spec_stds = []
        nonspec_means = []
        nonspec_stds = []
        op_names = []
        
        for op in all_ops:
            spec_key = f'{metric}_mean'
            nonspec_key = f'{metric}_mean'
            
            if op in spec_rq1 and spec_key in spec_rq1[op]:
                spec_vals = spec_rq1[op][spec_key]
                spec_means.append(np.mean(spec_vals) if spec_vals else 0)
                spec_stds.append(np.std(spec_vals) if spec_vals else 0)
            else:
                spec_means.append(0)
                spec_stds.append(0)
            
            if op in nonspec_rq1 and nonspec_key in nonspec_rq1[op]:
                nonspec_vals = nonspec_rq1[op][nonspec_key]
                nonspec_means.append(np.mean(nonspec_vals) if nonspec_vals else 0)
                nonspec_stds.append(np.std(nonspec_vals) if nonspec_vals else 0)
            else:
                nonspec_means.append(0)
                nonspec_stds.append(0)
            
            op_names.append(op[:20])  # Truncate long names
        
        x = np.arange(len(op_names))
        width = 0.35
        
        ax.bar(x - width/2, spec_means, width, yerr=spec_stds, label='Speciation', alpha=0.8, color='#2ecc71')
        ax.bar(x + width/2, nonspec_means, width, yerr=nonspec_stds, label='Non-Speciation', alpha=0.8, color='#e74c3c')
        
        ax.set_xlabel('Operator')
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(f'{metric_labels[metric]} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(op_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'rq1_operator_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Summary Statistics Table
    summary_data = []
    for op in all_ops:
        row = {'Operator': op}
        
        for metric in metrics:
            spec_key = f'{metric}_mean'
            nonspec_key = f'{metric}_mean'
            
            if op in spec_rq1 and spec_key in spec_rq1[op]:
                spec_vals = spec_rq1[op][spec_key]
                row[f'Spec_{metric}'] = f"{np.mean(spec_vals):.2f} ± {np.std(spec_vals):.2f}" if spec_vals else "N/A"
            else:
                row[f'Spec_{metric}'] = "N/A"
            
            if op in nonspec_rq1 and nonspec_key in nonspec_rq1[op]:
                nonspec_vals = nonspec_rq1[op][nonspec_key]
                row[f'NonSpec_{metric}'] = f"{np.mean(nonspec_vals):.2f} ± {np.std(nonspec_vals):.2f}" if nonspec_vals else "N/A"
            else:
                row[f'NonSpec_{metric}'] = "N/A"
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(RESULTS_DIR / 'rq1_summary_table.csv', index=False)
    
    print("\nRQ1 Summary Table saved to comparison_results/rq1_summary_table.csv")


def create_rq2_visualizations(spec_rq2, nonspec_rq2):
    """Create RQ2 cluster quality and diversity visualizations."""
    
    # 1. Fitness Evolution Over Generations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Best Fitness
    ax = axes[0, 0]
    spec_best = spec_rq2.get('best_fitness_mean', [])
    nonspec_best = nonspec_rq2.get('best_fitness_mean', [])
    
    if spec_best and nonspec_best:
        ax.plot(range(len(spec_best)), spec_best, 'o-', label='Speciation', linewidth=2, markersize=8, color='#2ecc71')
        ax.plot(range(len(nonspec_best)), nonspec_best, 's-', label='Non-Speciation', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Best Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Average Fitness
    ax = axes[0, 1]
    spec_avg = spec_rq2.get('avg_fitness_mean', [])
    nonspec_avg = nonspec_rq2.get('avg_fitness_mean', [])
    
    if spec_avg and nonspec_avg:
        ax.plot(range(len(spec_avg)), spec_avg, 'o-', label='Speciation', linewidth=2, markersize=8, color='#2ecc71')
        ax.plot(range(len(nonspec_avg)), nonspec_avg, 's-', label='Non-Speciation', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Fitness')
        ax.set_title('Average Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Species Count (only for speciation)
    ax = axes[1, 0]
    spec_species = spec_rq2.get('species_count_mean', [])
    
    if spec_species:
        ax.plot(range(len(spec_species)), spec_species, 'o-', label='Speciation', linewidth=2, markersize=8, color='#2ecc71')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Species Count')
        ax.set_title('Species Count Evolution (Speciation Only)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Diversity Metrics
    ax = axes[1, 1]
    spec_inter = spec_rq2.get('inter_species_diversity_mean', [])
    spec_intra = spec_rq2.get('intra_species_diversity_mean', [])
    
    if spec_inter and spec_intra:
        ax.plot(range(len(spec_inter)), spec_inter, 'o-', label='Inter-Species Diversity', linewidth=2, markersize=8, color='#3498db')
        ax.plot(range(len(spec_intra)), spec_intra, 's-', label='Intra-Species Diversity', linewidth=2, markersize=8, color='#9b59b6')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Diversity Score')
        ax.set_title('Diversity Metrics (Speciation Only)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'rq2_evolution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Summary Statistics
    summary_data = {
        'Metric': [],
        'Speciation_Mean': [],
        'Speciation_Std': [],
        'NonSpeciation_Mean': [],
        'NonSpeciation_Std': []
    }
    
    metrics_to_compare = [
        ('best_fitness', 'Best Fitness'),
        ('avg_fitness', 'Average Fitness'),
        ('species_count', 'Species Count'),
        ('inter_species_diversity', 'Inter-Species Diversity'),
        ('intra_species_diversity', 'Intra-Species Diversity'),
    ]
    
    for metric_key, metric_label in metrics_to_compare:
        spec_vals = spec_rq2.get(f'{metric_key}_mean', [])
        nonspec_vals = nonspec_rq2.get(f'{metric_key}_mean', [])
        
        summary_data['Metric'].append(metric_label)
        summary_data['Speciation_Mean'].append(f"{np.mean(spec_vals):.4f}" if spec_vals else "N/A")
        summary_data['Speciation_Std'].append(f"{np.std(spec_vals):.4f}" if spec_vals else "N/A")
        summary_data['NonSpeciation_Mean'].append(f"{np.mean(nonspec_vals):.4f}" if nonspec_vals else "N/A")
        summary_data['NonSpeciation_Std'].append(f"{np.std(nonspec_vals):.4f}" if nonspec_vals else "N/A")
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(RESULTS_DIR / 'rq2_summary_table.csv', index=False)
    
    print("RQ2 Summary Table saved to comparison_results/rq2_summary_table.csv")


def main():
    """Main analysis function."""
    print("=" * 80)
    print("Speciation vs Non-Speciation Comparison Analysis")
    print("=" * 80)
    
    # Check if run directories are specified
    if not SPEciation_RUNS or not NON_SPECIATION_RUNS:
        print("\n⚠️  WARNING: Please update SPEciation_RUNS and NON_SPECIATION_RUNS lists in the script!")
        print("\nExample:")
        print("SPEciation_RUNS = ['20260115_1609', '20260115_1700', ...]")
        print("NON_SPECIATION_RUNS = ['20260115_1800', '20260115_1900', ...]")
        return
    
    print(f"\nAnalyzing {len(SPEciation_RUNS)} speciation runs...")
    print(f"Analyzing {len(NON_SPECIATION_RUNS)} non-speciation runs...")
    
    # Aggregate metrics
    print("\n[1/3] Aggregating speciation run metrics...")
    spec_rq1, spec_rq2 = aggregate_runs(SPEciation_RUNS, "speciation")
    
    print("[2/3] Aggregating non-speciation run metrics...")
    nonspec_rq1, nonspec_rq2 = aggregate_runs(NON_SPECIATION_RUNS, "non-speciation")
    
    # Create visualizations
    print("[3/3] Creating visualizations...")
    create_rq1_visualizations(spec_rq1, nonspec_rq1)
    create_rq2_visualizations(spec_rq2, nonspec_rq2)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print(f"Results saved to: {RESULTS_DIR.absolute()}")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - rq1_operator_comparison.png")
    print("  - rq1_summary_table.csv")
    print("  - rq2_evolution_comparison.png")
    print("  - rq2_summary_table.csv")


if __name__ == "__main__":
    main()
