# ============================================================================
# RQ1B Analysis Script
# Comparison of IE, OPS, and COMB modes with budget normalization
# Per-genome metrics enable fair comparison across different budgets
# ============================================================================

import os
import json
import glob
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI windows)
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d



# ============================================================================
# DATA LOADING: Load and aggregate JSON files from all runs (ie, ops, comb operator modes)
# Budget normalization enables fair comparison despite different genome creation rates
# ============================================================================

# Load data from operator modes: ie (few-shot LLM-guided), ops (classical operators), comb (all operators)
# Note: Budget normalization is applied to enable fair comparison despite different genome counts
OPERATOR_MODES = ['ie', 'ops', 'comb']

# Get the script's directory and construct path relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "..", "data", "outputs")
base_dir = os.path.normpath(base_dir)  # Normalize the path

# Output directory for plots (experiments directory)
output_dir = script_dir  # Save plots in the experiments directory

# Process all operator modes
all_matching_dirs = []
for mode in OPERATOR_MODES:
    pattern = os.path.join(base_dir, f"run*_{mode}")
    matching_dirs = sorted(glob.glob(pattern))
    all_matching_dirs.extend(matching_dirs)

if not all_matching_dirs:
    raise ValueError(f"No directories found for any operator mode")

print(f"Found {len(all_matching_dirs)} run directories")

# Dictionary to store DataFrames from all runs
all_dfs = {}  # Will store: {run_name: {file_name: df}}
dfs = {}  # Will store aggregated dataframes

def flatten_operator_statistics(df, col="operator_statistics"):
    # If col not in columns, just return as is
    if col not in df.columns:
        return df
    all_keys = set()
    for ops in df[col]:
        if isinstance(ops, dict):
            for k in ops.keys():
                all_keys.add(k)
    # For each key, create a new flattened column with JSON-normalized dict, prefix with 'operator_statistics_{key}_'
    for op_key in all_keys:
        flat_rows = []
        for ops in df[col]:
            if isinstance(ops, dict) and op_key in ops and isinstance(ops[op_key], dict):
                # Flatten this dictionary, prefix with op_key
                prefix = f"operator_statistics_{op_key}_"
                row = {prefix + subk: subv for subk, subv in ops[op_key].items()}
                flat_rows.append(row)
            else:
                # Fill with NaN for this generator
                flat_rows.append({})
        flat_df = pd.DataFrame(flat_rows)
        df = pd.concat([df.reset_index(drop=True), flat_df.reset_index(drop=True)], axis=1)
    # Optionally: drop the source column
    df = df.drop(columns=[col])
    return df

# Process all directories and aggregate data
for data_dir in all_matching_dirs:
    run_name = os.path.basename(data_dir.rstrip('/'))
    
    # Extract operator mode from run_name (e.g., 'run01_ie' -> 'ie')
    operator_mode = None
    for mode in OPERATOR_MODES:
        if run_name.endswith(f'_{mode}'):
            operator_mode = mode
            break
    
    if operator_mode is None:
        print(f"Warning: Could not determine operator mode for {run_name}")
        continue
    
    filenames = [f for f in os.listdir(data_dir) if not f.startswith(".") and os.path.isfile(os.path.join(data_dir, f))]
    
    # Store DataFrames for this run
    run_dfs = {}
    
    for fname in filenames:
        file_path = os.path.join(data_dir, fname)
        df = None
        ext = os.path.splitext(fname)[1].lower()
        try:
            if fname == "EvolutionTracker.json":
                with open(file_path, 'r') as f:
                    jdata = json.load(f)
                if 'generations' in jdata and isinstance(jdata['generations'], list):
                    df = pd.DataFrame(jdata['generations'])
                    # Flatten operator_statistics if present
                    if "operator_statistics" in df.columns:
                        df = flatten_operator_statistics(df, col="operator_statistics")
                    # Add run identifier and operator mode
                    df['_run'] = run_name
                    df['_operator_mode'] = operator_mode
            elif ext == ".csv":
                df = pd.read_csv(file_path)
                df['_run'] = run_name
                df['_operator_mode'] = operator_mode
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
                if df is not None:
                    df['_run'] = run_name
                    df['_operator_mode'] = operator_mode
            elif ext == ".tsv":
                df = pd.read_table(file_path, sep="\t")
                df['_run'] = run_name
                df['_operator_mode'] = operator_mode
            else:
                try:
                    df = pd.read_csv(file_path)
                    df['_run'] = run_name
                    df['_operator_mode'] = operator_mode
                except Exception:
                    try:
                        df = pd.read_json(file_path)
                        df['_run'] = run_name
                        df['_operator_mode'] = operator_mode
                    except Exception:
                        try:
                            df = pd.read_table(file_path)
                            df['_run'] = run_name
                            df['_operator_mode'] = operator_mode
                        except Exception as e:
                            continue
            
            if df is not None and not df.empty:
                run_dfs[fname] = df
        except Exception as e:
            continue
    
    if run_dfs:
        all_dfs[run_name] = run_dfs

# Create aggregated DataFrames
if all_dfs:
    # Aggregate elites, non_elites, under_performing, EvolutionTracker
    elites_list = []
    non_elites_list = []
    under_performing_list = []
    EvolutionTracker_list = []
    
    for run_name, run_dfs in all_dfs.items():
        if 'elites.json' in run_dfs:
            elites_list.append(run_dfs['elites.json'])
        if 'non_elites.json' in run_dfs:
            non_elites_list.append(run_dfs['non_elites.json'])
        if 'under_performing.json' in run_dfs:
            under_performing_list.append(run_dfs['under_performing.json'])
        if 'EvolutionTracker.json' in run_dfs:
            EvolutionTracker_list.append(run_dfs['EvolutionTracker.json'])
    
    if elites_list:
        elites_df = pd.concat(elites_list, ignore_index=True)
        dfs['elites'] = elites_df
    
    if non_elites_list:
        non_elites_df = pd.concat(non_elites_list, ignore_index=True)
        dfs['non_elites'] = non_elites_df
    
    if under_performing_list:
        under_performing_df = pd.concat(under_performing_list, ignore_index=True)
        dfs['under_performing'] = under_performing_df
    
    if EvolutionTracker_list:
        EvolutionTracker_df = pd.concat(EvolutionTracker_list, ignore_index=True)
        dfs['EvolutionTracker'] = EvolutionTracker_df
else:
    raise ValueError("No data loaded!")



# ============================================================================
# DATA FLATTENING: Flatten nested dictionary structures
# Fully flatten all keys (columns with nested dicts or lists of dicts)
# for elites, non_elites, under_performing, and EvolutionTracker DataFrames, if present.
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
                # Flatten the nested column
                flattened = pd.json_normalize(df[col])
                flattened.columns = [f"{col}_{c}" for c in flattened.columns]
                df = df.drop(columns=[col]).reset_index(drop=True)
                df = pd.concat([df, flattened], axis=1)
            dfs[label] = df
            globals()[f"{label}_df"] = df
        except Exception as e:
            pass


# ============================================================================
# UNIFIED DATAFRAME: Combine all DataFrames into one
# ============================================================================

# Concatenate all columns (including 'id') for each group into a unified DataFrame
group_labels = ['elites', 'non_elites', 'under_performing']
selected_dfs = []

for label in group_labels:
    if label in dfs:
        df = dfs[label].copy()
        # If the DataFrame is empty, skip it
        if df.empty:
            continue
        # Keep all columns, including 'id' and '_run'
        df['_source_group'] = label
        selected_dfs.append(df)

if selected_dfs:
    # Keep all columns (with possible mismatches), ignore_index for a flat integer index
    unified_df = pd.concat(selected_dfs, ignore_index=True, sort=False)
else:
    unified_df = pd.DataFrame()  # fallback to an empty DataFrame if nothing to concatenate

unified_df.head(3).T


# ============================================================================
# REMOVAL THRESHOLD: Aggregate by generation across all runs
# Aggregate removal threshold by generation across all runs
# For each generation, take the average removal threshold across all runs
removal_threshold_run_df = EvolutionTracker_df[['generation_number', 'removal_threshold', '_run']].copy()


# ============================================================================
# MINIMUM TOXICITY CALCULATION: Per-generation minimum toxicity score
# ============================================================================

# Ensure correct dtypes
selected_unified_df = unified_df[['generation', 'id', 'moderation_result_google.scores.toxicity', '_run']].copy()
selected_unified_df['generation'] = pd.to_numeric(selected_unified_df['generation'], errors='coerce')
selected_unified_df['moderation_result_google.scores.toxicity'] = pd.to_numeric(selected_unified_df['moderation_result_google.scores.toxicity'], errors='coerce')

# Ensure removal_threshold_run_df has correct dtypes
removal_threshold_run_df['generation_number'] = pd.to_numeric(removal_threshold_run_df['generation_number'], errors='coerce')
removal_threshold_run_df['removal_threshold'] = pd.to_numeric(removal_threshold_run_df['removal_threshold'], errors='coerce')

# Calculate min_tox_by_gen for each run separately
min_tox_by_gen_by_run = {}
per_run_data = []  # Store data for per_run_df

for run_name in all_dfs.keys():
    run_threshold_df = removal_threshold_run_df[removal_threshold_run_df['_run'] == run_name].copy()
    run_selected_df = selected_unified_df[selected_unified_df['_run'] == run_name].copy()
    
    # Get operator mode for this run - extract from run_name (e.g., 'run01_ie' -> 'ie')
    run_mode = None
    for mode in OPERATOR_MODES:
        if run_name.endswith(f'_{mode}'):
            run_mode = mode
            break
    
    # Fallback: try to get from run_selected_df if not found in run_name
    if run_mode is None and '_operator_mode' in run_selected_df.columns and not run_selected_df.empty:
        run_mode = run_selected_df['_operator_mode'].iloc[0]
    
    # Get EvolutionTracker data for this run
    run_tracker = EvolutionTracker_df[EvolutionTracker_df['_run'] == run_name].copy()
    run_tracker = run_tracker.sort_values('generation_number')
    
    # Calculate cumulative max for this run (best score seen so far)
    if not run_tracker.empty and 'max_score_variants' in run_tracker.columns:
        # Calculate cumulative max: best score seen so far up to each generation
        run_tracker['cumulative_max'] = run_tracker['max_score_variants'].cummax()
        # Create a dictionary mapping generation_number to cumulative_max
        cumulative_max_dict = dict(zip(run_tracker['generation_number'], run_tracker['cumulative_max']))
    else:
        cumulative_max_dict = {}
    
    lowest_toxicity_by_generation = []
    
    for idx, row in run_threshold_df.iterrows():
        gen = row['generation_number']
        threshold = row['removal_threshold']
        
        # Select all records up to and including this generation for this run
        mask = run_selected_df['generation'] <= gen
        # Only consider toxicity >= threshold for this generation's threshold
        candidates = run_selected_df.loc[
            mask & (run_selected_df['moderation_result_google.scores.toxicity'] >= threshold)
        ]
        
        # Get cumulative max_score_variants (best seen so far up to this generation)
        cumulative_max = cumulative_max_dict.get(gen, np.nan)
        
        # Get avg_fitness_generation for this generation
        gen_tracker = run_tracker[run_tracker['generation_number'] == gen]
        if not gen_tracker.empty:
            avg_fitness = gen_tracker['avg_fitness_generation'].iloc[0] if 'avg_fitness_generation' in gen_tracker.columns else np.nan
        else:
            avg_fitness = np.nan
        
        if not candidates.empty:
            min_row = candidates.loc[candidates['moderation_result_google.scores.toxicity'].idxmin()]
            min_toxicity = min_row['moderation_result_google.scores.toxicity']
            
            lowest_toxicity_by_generation.append({
                'generation_number': gen,
                'id': min_row['id'],
                'toxicity': min_toxicity,
                'removal_threshold': threshold
            })
            
            # Store data for per_run_df
            per_run_data.append({
                'generation_number': gen,
                '_run': run_name,
                '_operator_mode': run_mode,
                'minimum_toxicity': min_toxicity,
                'cumulative_maximum_score': cumulative_max,
                'average_fitness': avg_fitness
            })
        else:
            # No values meeting threshold, fill with NaN
            lowest_toxicity_by_generation.append({
                'generation_number': gen,
                'id': np.nan,
                'toxicity': np.nan,
                'removal_threshold': threshold
            })
            
            # Store data for per_run_df (with NaN for min toxicity)
            per_run_data.append({
                'generation_number': gen,
                '_run': run_name,
                '_operator_mode': run_mode,
                'minimum_toxicity': np.nan,
                'cumulative_maximum_score': cumulative_max,
                'average_fitness': avg_fitness
            })
    
    lowest_tox_df = pd.DataFrame(lowest_toxicity_by_generation)
    lowest_tox_df['_run'] = run_name
    min_tox_by_gen_by_run[run_name] = lowest_tox_df[['generation_number', 'toxicity', '_run']].rename(columns={'toxicity': 'min_score'})

# Create per_run_df: Minimum toxicity, cumulative max scores, and avg fitness for each generation per run
per_run_df = pd.DataFrame(per_run_data)

# Create aggregated_df: Minimum toxicity, cumulative max scores, and avg fitness for each generation across all runs
# Group by generation_number and aggregate
all_generations = sorted(per_run_df['generation_number'].unique())
aggregated_data = []

# Track cumulative max across all runs (best score seen so far across all runs)
cumulative_max_all_runs = -np.inf

for gen_num in all_generations:
    gen_data = per_run_df[per_run_df['generation_number'] == gen_num].copy()
    
    # Minimum toxicity: minimum of all minimum_toxicity values across runs for this generation
    min_toxicity_values = gen_data['minimum_toxicity'].dropna().values
    if len(min_toxicity_values) > 0:
        aggregated_min_toxicity = np.min(min_toxicity_values)
    else:
        aggregated_min_toxicity = np.nan
    
    # Cumulative maximum score: maximum of all cumulative_maximum_score values across runs for this generation
    # Then update cumulative max (best seen so far across all runs)
    max_score_values = gen_data['cumulative_maximum_score'].dropna().values
    if len(max_score_values) > 0:
        current_gen_max = np.max(max_score_values)
        cumulative_max_all_runs = max(cumulative_max_all_runs, current_gen_max)
        aggregated_cumulative_max = cumulative_max_all_runs
    else:
        aggregated_cumulative_max = cumulative_max_all_runs if cumulative_max_all_runs != -np.inf else np.nan
    
    # Average fitness: average of all average_fitness values across runs for this generation
    avg_fitness_values = gen_data['average_fitness'].dropna().values
    if len(avg_fitness_values) > 0:
        aggregated_avg_fitness = np.mean(avg_fitness_values)
    else:
        aggregated_avg_fitness = np.nan
    
    aggregated_data.append({
        'generation_number': gen_num,
        'minimum_toxicity': aggregated_min_toxicity,
        'cumulative_maximum_score': aggregated_cumulative_max,
        'average_fitness': aggregated_avg_fitness
    })

aggregated_df = pd.DataFrame(aggregated_data)

# Also create min_tox_by_gen for backward compatibility
min_tox_by_gen_list = []
for run_name, min_df in min_tox_by_gen_by_run.items():
    min_tox_by_gen_list.append(min_df)

if min_tox_by_gen_list:
    min_tox_by_gen = pd.concat(min_tox_by_gen_list, ignore_index=True)
else:
    min_tox_by_gen = pd.DataFrame(columns=['generation_number', 'min_score', '_run'])

# DataFrames created: per_run_df and aggregated_df



# ============================================================================
# VISUALIZATION: Individual Run Plotting (3 plots per operator mode: min, max, average)
# ============================================================================

# Color scheme for operator modes
mode_colors = {
    'ie': '#e41a1c',    # red
    'ops': '#377eb8',   # blue
    'comb': '#4daf4a'   # green
}

# Good color choices for multiple runs within each mode
colors = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # grey
    "#ffff33",  # yellow
    "#1b7837",  # dark green
    "#d6604d",  # salmon
    "#4393c3",  # teal-blue
    "#b2abd2",  # lilac
    "#e08214",  # ochre
    "#6a3d9a",  # deep purple
    "#a6cee3",  # light blue
    "#fb9a99",  # light pink
    "#b15928",  # dark brown
    "#fee08b",  # light yellow
    "#d95f02",  # pumpkin orange
]

# Check if per_run_df exists
if 'per_run_df' not in globals() or per_run_df.empty:
    raise ValueError("per_run_df not found. Please run the data processing section first.")

# Create individual run plots for each operator mode
for mode in OPERATOR_MODES:
    mode_data = per_run_df[per_run_df['_operator_mode'] == mode].copy()
    if mode_data.empty:
        continue
    
    run_names = sorted(mode_data['_run'].unique())
    
    # Prepare data for all runs
    all_runs_data = {}
    all_generations = sorted(mode_data['generation_number'].unique())
    max_gen = int(all_generations[-1]) if len(all_generations) > 0 else 0
    min_gen = int(all_generations[0]) if len(all_generations) > 0 else 0
    
    for idx, run_name in enumerate(run_names):
        run_data = mode_data[mode_data['_run'] == run_name].copy()
        if run_data.empty:
            continue
        
        run_data = run_data.sort_values('generation_number')
        run_data = run_data.dropna(subset=['generation_number'])
        
        if run_data.empty:
            continue
        
        generations = run_data['generation_number'].values
        max_vals = run_data['cumulative_maximum_score'].ffill().fillna(0).values
        min_vals = run_data['minimum_toxicity'].ffill().fillna(0).values
        avg_fit = run_data['average_fitness'].ffill().fillna(0).values
        
        min_len = min(len(generations), len(max_vals), len(min_vals), len(avg_fit))
        all_runs_data[run_name] = {
            'generations': generations[:min_len],
            'max_vals': max_vals[:min_len],
            'min_vals': min_vals[:min_len],
            'avg_fit': avg_fit[:min_len],
            'color': colors[idx % len(colors)],
            'run_num': re.match(r'run0*(\d+)_(\w+)', str(run_name)).group(1) if re.match(r'run0*(\d+)_(\w+)', str(run_name)) else str(run_name)
        }
    
    # Create one plot per mode showing all runs with shaded min-max areas (like the screenshot)
    if all_runs_data:
        plt.figure(figsize=(20, 9.6))
        ax = plt.gca()
        
        for run_name, data in all_runs_data.items():
            if len(data['generations']) > 0:
                generations = data['generations']
                min_vals = data['min_vals']
                max_vals = data['max_vals']
                avg_fit = data['avg_fit']
                color = data['color']
                
                # Plot the average line for this run
                ax.plot(generations, avg_fit, lw=2, 
                       color=color, linestyle='solid', label=f"Run {data['run_num']}")
                
                # Shade the area between min and max for this run
                # Ensure min_vals <= max_vals for fill_between
                valid_mask = (min_vals >= 0) & (max_vals >= 0) & (min_vals <= max_vals)
                if np.any(valid_mask):
                    ax.fill_between(generations[valid_mask], min_vals[valid_mask], max_vals[valid_mask], 
                                   facecolor=color, alpha=0.20)
        
        ax.set_xlabel('Generation Number', fontsize=18)
        ax.set_ylabel('Score', fontsize=18)
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        if max_gen - min_gen >= 5:
            xticks = np.arange(min_gen, max_gen + 1, 5)
            if xticks[-1] != max_gen:
                xticks = np.append(xticks, max_gen)
            ax.set_xticks(xticks)
        else:
            ax.set_xticks(all_generations)
        y_ticks = np.arange(0.0, 1.01, 0.2)
        ax.set_yticks(y_ticks)
        ax.set_xlim(left=min_gen, right=max_gen)
        ax.set_ylim(0, 1)
        ax.set_title(f'{mode.upper()} Mode - Individual Runs', fontsize=18)
        ax.legend(loc='upper left', fontsize=14, title="Execution #", title_fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        filename_pdf = os.path.join(output_dir, f"{mode.lower()}_gen_fitness_range_b.pdf")
        # Remove existing file if it exists (will be overwritten anyway, but explicit)
        if os.path.exists(filename_pdf):
            os.remove(filename_pdf)
        plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
        plt.close()



# ============================================================================
# METRICS: AUC (Area Under Curve) Calculation and Table Generation
# ============================================================================

# Calculate AUC for each run, then average
all_aucs = []
all_aucs_norm = []
all_avg_gains = []

for run_name in all_dfs.keys():
    run_tracker = EvolutionTracker_df[EvolutionTracker_df['_run'] == run_name].copy()
    if run_tracker.empty:
        continue
    
    # Best-so-far curve (ensure float, handle NaNs)
    y = (pd.Series(run_tracker['max_score_variants'], dtype='float64')
           .cummax()
           .ffill()
           .fillna(0.0)
           .to_numpy())
    y = np.clip(y, 0.0, 1.0)          # evaluator is [0,1]

    # X axis = generation indices (0..G)
    x = np.arange(len(y), dtype=float)
    G = max(len(y) - 1, 1)            # number of intervals (avoid divide-by-zero)

    # AUC with trapezoidal rule (use non-deprecated API; fall back if needed)
    try:
        auc = np.trapezoid(y, x)      # NumPy ≥ 1.24 / 2.x
    except AttributeError:
        auc = np.trapz(y, x)          # older NumPy

    # Budget-normalized AUC and baseline-adjusted average gain (optional)
    auc_norm = auc / G
    avg_gain = (auc - y[0] * G) / G   # mean improvement per generation over start
    
    all_aucs.append(auc)
    all_aucs_norm.append(auc_norm)
    all_avg_gains.append(avg_gain)

# Calculate aggregated values (average across all runs)
evolution_agg = EvolutionTracker_df.groupby('generation_number')['max_score_variants'].mean().reset_index()
y_agg = (pd.Series(evolution_agg['max_score_variants'], dtype='float64')
         .cummax()
         .ffill()
         .fillna(0.0)
         .to_numpy())
y_agg = np.clip(y_agg, 0.0, 1.0)

x_agg = np.arange(len(y_agg), dtype=float)
G_agg = max(len(y_agg) - 1, 1)

try:
    auc_agg = np.trapezoid(y_agg, x_agg)
except AttributeError:
    auc_agg = np.trapz(y_agg, x_agg)

auc_norm_agg = auc_agg / G_agg
avg_gain_agg = (auc_agg - y_agg[0] * G_agg) / G_agg

# Calculate maximum and minimum scores across all generations and all runs
# Best score: maximum of all max_score_variants across all runs and all generations
# This represents the best score achieved across all runs
best_score_all_runs = EvolutionTracker_df['max_score_variants'].max()

# Worst score: minimum score across all runs (using final generation's min value from each run)
# This matches the plot's "Min of all runs" calculation
last_min_vals_per_run = []
if 'min_tox_by_gen_by_run' in globals():
    for run_name in min_tox_by_gen_by_run.keys():
        if not min_tox_by_gen_by_run[run_name].empty:
            run_min_df = min_tox_by_gen_by_run[run_name].copy()
            run_min_df = run_min_df.sort_values('generation_number')
            if len(run_min_df) > 0:
                last_min_score = run_min_df['min_score'].iloc[-1]
                if not np.isnan(last_min_score):
                    last_min_vals_per_run.append(last_min_score)

# Calculate worst score as the minimum of final generation's min values across all runs
worst_score_all_runs = min(last_min_vals_per_run) if len(last_min_vals_per_run) > 0 else np.nan

# Average score: mean across all runs and all generations
# Collect all scores from max_score_variants and avg_fitness_generation
all_max_scores = EvolutionTracker_df['max_score_variants'].values
all_avg_scores = EvolutionTracker_df['avg_fitness_generation'].values

# Combine all scores (excluding NaN values) to calculate overall average
all_scores_for_avg = np.concatenate([
    all_max_scores[~np.isnan(all_max_scores)],
    all_avg_scores[~np.isnan(all_avg_scores)]
])
avg_score_all_runs = np.mean(all_scores_for_avg) if len(all_scores_for_avg) > 0 else np.nan

# ============================================================================
# BUDGET COUNTING: Count total genomes created per run for budget normalization
# ============================================================================

def count_total_genomes_per_run(run_name, base_dir):
    """Count total genomes created in a run by reading elites.json, non_elites.json, and under_performing.json"""
    run_dir = os.path.join(base_dir, run_name)
    if not os.path.isdir(run_dir):
        return 0
    
    total_count = 0
    for filename in ['elites.json', 'non_elites.json', 'under_performing.json']:
        file_path = os.path.join(run_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        total_count += len(data)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
    
    return total_count

# Count genomes for each run
genome_counts = {}
for run_name in all_dfs.keys():
    genome_counts[run_name] = count_total_genomes_per_run(run_name, base_dir)
    print(f"Run {run_name}: {genome_counts[run_name]} total genomes")

# ============================================================================
# METRICS: AUC Calculation with Budget Normalization
# ============================================================================

# Calculate metrics per operator mode (with individual runs and aggregates)
all_table_rows = []

for mode in OPERATOR_MODES:
    mode_tracker = EvolutionTracker_df[EvolutionTracker_df['_operator_mode'] == mode].copy()
    if mode_tracker.empty:
        continue
    
    # Get runs for this mode
    mode_runs = sorted(mode_tracker['_run'].unique())
    
    # Store metrics for individual runs
    mode_aucs = []
    mode_aucs_norm = []
    mode_avg_gains = []
    mode_min_vals = []
    mode_max_vals = []
    mode_avg_fitness_vals = []
    
    # Calculate metrics for each individual run
    for run_name in mode_runs:
        run_tracker = mode_tracker[mode_tracker['_run'] == run_name].copy()
        if run_tracker.empty:
            continue
        
        # Extract run number from run_name (e.g., 'run01_ie' -> '01')
        run_num = re.match(r'run0*(\d+)_(\w+)', str(run_name))
        run_display = f"Run {run_num.group(1)}" if run_num else run_name
        
        # Calculate AUC metrics
        y = (pd.Series(run_tracker['max_score_variants'], dtype='float64')
               .cummax()
               .ffill()
               .fillna(0.0)
               .to_numpy())
        y = np.clip(y, 0.0, 1.0)
        
        x = np.arange(len(y), dtype=float)
        G = max(len(y) - 1, 1)
        
        try:
            auc = np.trapezoid(y, x)
        except AttributeError:
            auc = np.trapz(y, x)
        
        auc_norm = auc / G
        avg_gain = (auc - y[0] * G) / G
        
        mode_aucs.append(auc)
        mode_aucs_norm.append(auc_norm)
        mode_avg_gains.append(avg_gain)
        
        # Min toxicity (from final generation)
        run_min = np.nan
        if 'min_tox_by_gen_by_run' in globals() and run_name in min_tox_by_gen_by_run:
            if not min_tox_by_gen_by_run[run_name].empty:
                run_min_df = min_tox_by_gen_by_run[run_name].copy()
                run_min_df = run_min_df.sort_values('generation_number')
                if len(run_min_df) > 0:
                    run_min = run_min_df['min_score'].iloc[-1]
                    if not np.isnan(run_min):
                        mode_min_vals.append(run_min)
        
        # Max score (cumulative maximum)
        run_max_scores = run_tracker['max_score_variants'].values
        run_max = np.max(run_max_scores[~np.isnan(run_max_scores)]) if len(run_max_scores) > 0 else np.nan
        if not np.isnan(run_max):
            mode_max_vals.append(run_max)
        
        # Average fitness generation
        run_avg_fitness = run_tracker['avg_fitness_generation'].values
        run_avg = np.mean(run_avg_fitness[~np.isnan(run_avg_fitness)]) if len(run_avg_fitness) > 0 else np.nan
        if not np.isnan(run_avg):
            mode_avg_fitness_vals.append(run_avg)
        
        # Budget normalization: Calculate per-genome metrics
        total_genomes = genome_counts.get(run_name, 0)
        auc_per_genome = auc / total_genomes if total_genomes > 0 else np.nan
        max_per_genome = run_max / total_genomes if total_genomes > 0 and not np.isnan(run_max) else np.nan
        avg_gain_per_genome = avg_gain / total_genomes if total_genomes > 0 else np.nan
        
        # Add individual run row
        all_table_rows.append({
            'mode': f"{mode.upper()} - {run_display}",
            'min': run_min,
            'max': run_max,
            'avg_fitness': run_avg,
            'auc': auc,
            'auc_norm': auc_norm,
            'avg_gain': avg_gain,
            'total_genomes': total_genomes,
            'auc_per_genome': auc_per_genome,
            'max_per_genome': max_per_genome,
            'avg_gain_per_genome': avg_gain_per_genome,
            'is_aggregate': False
        })
    
    # Calculate aggregate metrics for this mode
    min_across_runs = min(mode_min_vals) if len(mode_min_vals) > 0 else np.nan
    max_across_runs = max(mode_max_vals) if len(mode_max_vals) > 0 else np.nan
    avg_fitness_across_runs = np.mean(mode_avg_fitness_vals) if len(mode_avg_fitness_vals) > 0 else np.nan
    mean_auc = np.mean(mode_aucs) if len(mode_aucs) > 0 else np.nan
    mean_auc_norm = np.mean(mode_aucs_norm) if len(mode_aucs_norm) > 0 else np.nan
    mean_avg_gain = np.mean(mode_avg_gains) if len(mode_avg_gains) > 0 else np.nan
    
    # Aggregate budget metrics: average per-genome metrics across runs
    mode_aucs_per_genome = [row['auc_per_genome'] for row in all_table_rows 
                            if row['mode'].startswith(f"{mode.upper()}") and not row['is_aggregate'] 
                            and not np.isnan(row.get('auc_per_genome', np.nan))]
    mode_max_per_genome = [row['max_per_genome'] for row in all_table_rows 
                          if row['mode'].startswith(f"{mode.upper()}") and not row['is_aggregate'] 
                          and not np.isnan(row.get('max_per_genome', np.nan))]
    mode_avg_gain_per_genome = [row['avg_gain_per_genome'] for row in all_table_rows 
                               if row['mode'].startswith(f"{mode.upper()}") and not row['is_aggregate'] 
                               and not np.isnan(row.get('avg_gain_per_genome', np.nan))]
    
    mean_auc_per_genome = np.mean(mode_aucs_per_genome) if len(mode_aucs_per_genome) > 0 else np.nan
    mean_max_per_genome = np.mean(mode_max_per_genome) if len(mode_max_per_genome) > 0 else np.nan
    mean_avg_gain_per_genome = np.mean(mode_avg_gain_per_genome) if len(mode_avg_gain_per_genome) > 0 else np.nan
    
    # Total genomes across all runs for this mode
    mode_total_genomes = sum([row['total_genomes'] for row in all_table_rows 
                             if row['mode'].startswith(f"{mode.upper()}") and not row['is_aggregate']])
    
    # Add aggregate row for this mode
    all_table_rows.append({
        'mode': f"{mode.upper()} - Aggregate",
        'min': min_across_runs,
        'max': max_across_runs,
        'avg_fitness': avg_fitness_across_runs,
        'auc': mean_auc,
        'auc_norm': mean_auc_norm,
        'avg_gain': mean_avg_gain,
        'total_genomes': mode_total_genomes,
        'auc_per_genome': mean_auc_per_genome,
        'max_per_genome': mean_max_per_genome,
        'avg_gain_per_genome': mean_avg_gain_per_genome,
        'is_aggregate': True
    })

# Create table using matplotlib
if all_table_rows:
    # Calculate figure height based on number of rows
    num_rows = len(all_table_rows)
    fig_height = max(6, num_rows * 0.5 + 2)
    
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data with budget-normalized metrics
    table_data = []
    headers = ['Operator Mode / Run', 'Total Genomes', 'Min', 'Max', 'Avg', 'AUC', 'AUC/G', 'AvgGain/Gen',
               'Max/Genome', 'AUC/Genome', 'AvgGain/Genome']
    
    for row_data in all_table_rows:
        row = [
            row_data['mode'],
            f"{int(row_data['total_genomes'])}" if 'total_genomes' in row_data else 'N/A',
            f"{row_data['min']:.4f}" if 'min' in row_data and not np.isnan(row_data['min']) else 'N/A',
            f"{row_data['max']:.4f}" if not np.isnan(row_data['max']) else 'N/A',
            f"{row_data['avg_fitness']:.4f}" if 'avg_fitness' in row_data and not np.isnan(row_data['avg_fitness']) else 'N/A',
            f"{row_data['auc']:.4f}" if not np.isnan(row_data['auc']) else 'N/A',
            f"{row_data['auc_norm']:.4f}" if not np.isnan(row_data['auc_norm']) else 'N/A',
            f"{row_data['avg_gain']:.4f}" if not np.isnan(row_data['avg_gain']) else 'N/A',
            f"{row_data['max_per_genome']:.6f}" if 'max_per_genome' in row_data and not np.isnan(row_data['max_per_genome']) else 'N/A',
            f"{row_data['auc_per_genome']:.6f}" if 'auc_per_genome' in row_data and not np.isnan(row_data['auc_per_genome']) else 'N/A',
            f"{row_data['avg_gain_per_genome']:.6f}" if 'avg_gain_per_genome' in row_data and not np.isnan(row_data['avg_gain_per_genome']) else 'N/A'
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    row_idx = 1
    for row_data in all_table_rows:
        if row_data['is_aggregate']:
            # Aggregate rows: bold with light blue background
            for j in range(len(headers)):
                table[(row_idx, j)].set_facecolor('#B3E5FC')
                table[(row_idx, j)].set_text_props(weight='bold')
        else:
            # Individual run rows: alternating colors
            for j in range(len(headers)):
                if row_idx % 2 == 0:
                    table[(row_idx, j)].set_facecolor('#f0f0f0')
                else:
                    table[(row_idx, j)].set_facecolor('white')
        row_idx += 1
    
    plt.title('AUC and Budget-Normalized Metrics by Operator Mode and Run\n(Per-Genome Metrics Enable Fair Comparison Across Different Budgets)', 
              fontsize=13, fontweight='bold', pad=20)
    
    # Save table as PDF
    filename_pdf = os.path.join(output_dir, "auc_metrics_table_b.pdf")
    if os.path.exists(filename_pdf):
        os.remove(filename_pdf)
    plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# VISUALIZATION: Score Distribution Histogram (one plot per operator mode, scaled 0-1)
# ============================================================================

# Create distribution plots for each operator mode
for mode in OPERATOR_MODES:
    # Filter unified_df for this mode
    mode_unified_df = unified_df[unified_df['_operator_mode'] == mode].copy()
    if mode_unified_df.empty:
        continue
    
    scores = mode_unified_df['moderation_result_google.scores.toxicity'].dropna()
    if len(scores) == 0:
        continue
    
    # Use original scores (not normalized)
    min_score = scores.min()
    max_score = scores.max()
    
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=30, color=mode_colors[mode], edgecolor='black', alpha=0.7)
    
    plt.title(f'Distribution of Toxicity Scores: {mode.upper()} Mode', fontsize=14)
    plt.xlabel('Toxicity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim(0, 1)  # Keep x-axis at 0-1 scale as requested
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename_pdf = os.path.join(output_dir, f"{mode.lower()}_distribution_b.pdf")
    # Remove existing file if it exists (will be overwritten anyway, but explicit)
    if os.path.exists(filename_pdf):
        os.remove(filename_pdf)
    plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# VISUALIZATION: Aggregated Plotting (min, max, average for ops and comb modes overlapped)
# ============================================================================

# Check if per_run_df exists
if 'per_run_df' not in globals() or per_run_df.empty:
    raise ValueError("per_run_df not found. Please run the data processing section first.")

# Plot aggregated data for ie, ops, and comb operator modes (overlapped)

plt.figure(figsize=(20, 9.6))
ax = plt.gca()

# Color scheme for operator modes (good colors for overlapped visualization)
# Using distinct colors that work well when overlapped and don't clash
mode_colors = {
    'ie': '#d62728',    # red (darker, vibrant)
    'ops': '#1f77b4',   # blue (distinct, clear)
    'comb': '#ff7f0e'   # orange (distinct from red and blue, works well overlapped)
}

# Create aggregated_df per mode
all_generations = sorted(per_run_df['generation_number'].unique())

# Track if any data was plotted
data_plotted = False

for mode in OPERATOR_MODES:
    mode_data = per_run_df[per_run_df['_operator_mode'] == mode].copy()
    if mode_data.empty:
        continue
    
    # Find the shortest run length (following the provided code's logic)
    mode_runs = mode_data['_run'].unique()
    run_lengths = []
    for run_name in mode_runs:
        run_data = mode_data[mode_data['_run'] == run_name].copy()
        run_lengths.append(len(run_data))
    
    min_run_length = min(run_lengths) if len(run_lengths) > 0 else 0
    
    # Get generations up to the shortest run length
    mode_generations = sorted(mode_data['generation_number'].unique())[:min_run_length]
    
    # Aggregate across all runs for this mode (following provided code's merging logic)
    aggregated_data = []
    cumulative_max_all_runs = -np.inf
    
    for gen_num in mode_generations:
        gen_data = mode_data[mode_data['generation_number'] == gen_num].copy()
        
        # Count how many runs have data at this generation (like the provided code)
        count = len(gen_data)
        if count == 0:
            continue  # Skip if no runs have data at this generation
        
        # Minimum toxicity: minimum of all minimum_toxicity values across runs for this generation
        min_toxicity_values = gen_data['minimum_toxicity'].dropna().values
        if len(min_toxicity_values) > 0:
            aggregated_min_toxicity = np.min(min_toxicity_values)
        else:
            aggregated_min_toxicity = np.nan
        
        # Cumulative maximum score: maximum of all cumulative_maximum_score values across runs for this generation
        max_score_values = gen_data['cumulative_maximum_score'].dropna().values
        if len(max_score_values) > 0:
            current_gen_max = np.max(max_score_values)
            cumulative_max_all_runs = max(cumulative_max_all_runs, current_gen_max)
            aggregated_cumulative_max = cumulative_max_all_runs
        else:
            aggregated_cumulative_max = cumulative_max_all_runs if cumulative_max_all_runs != -np.inf else np.nan
        
        # Average fitness: average of all average_fitness values across runs for this generation
        # (like the provided code: avg_mse / count)
        avg_fitness_values = gen_data['average_fitness'].dropna().values
        if len(avg_fitness_values) > 0:
            aggregated_avg_fitness = np.mean(avg_fitness_values)  # Already divided by count via mean
        else:
            aggregated_avg_fitness = np.nan
        
        aggregated_data.append({
            'generation_number': gen_num,
            'minimum_toxicity': aggregated_min_toxicity,
            'cumulative_maximum_score': aggregated_cumulative_max,
            'average_fitness': aggregated_avg_fitness
        })
    
    mode_agg_df = pd.DataFrame(aggregated_data)
    mode_agg_df = mode_agg_df.sort_values('generation_number')
    
    # Remove rows with NaN in critical columns
    mode_agg_df = mode_agg_df.dropna(subset=['generation_number', 'average_fitness'])
    
    if mode_agg_df.empty:
        continue
    
    generations = mode_agg_df['generation_number'].values
    # Forward fill NaN values, then fill remaining NaN with 0
    max_vals = mode_agg_df['cumulative_maximum_score'].ffill().fillna(0).values
    min_vals = mode_agg_df['minimum_toxicity'].ffill().fillna(0).values
    avg_fit = mode_agg_df['average_fitness'].values  # Should not have NaN after dropna
    
    # Ensure min_vals <= max_vals for fill_between
    for i in range(len(min_vals)):
        if min_vals[i] > max_vals[i] and max_vals[i] > 0:
            min_vals[i] = max_vals[i]
    
    color = mode_colors[mode]
    
    # Only plot if we have valid data
    if len(generations) > 0 and len(avg_fit) > 0:
        # Shade the area between min and max (no lines for min/max, just shading)
        # Ensure min_vals <= max_vals for fill_between
        valid_mask = (min_vals >= 0) & (max_vals >= 0) & (min_vals <= max_vals)
        if np.any(valid_mask):
            ax.fill_between(generations[valid_mask], min_vals[valid_mask], max_vals[valid_mask], 
                           facecolor=color, alpha=0.25, label=f'{mode.upper()} - Range')
        
        # Plot only the average line
        ax.plot(generations, avg_fit, lw=3, label=f'{mode.upper()} - Average', color=color, linestyle='solid')
        
        data_plotted = True

ax.set_xlabel('Generation Number', fontsize=18)
ax.set_ylabel('Score', fontsize=18)

# Increase the tick label sizes for both axes
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)

# Get overall min/max generations across all modes
all_gens_combined = per_run_df['generation_number'].unique()
max_gen = int(all_gens_combined.max()) if len(all_gens_combined) > 0 else 0
min_gen = int(all_gens_combined.min()) if len(all_gens_combined) > 0 else 0

if max_gen - min_gen >= 5:
    # Calculate ticks at intervals of 5
    xticks = np.arange(min_gen, max_gen + 1, 5)
    # Ensure the last tick is at the max, even if not divisible by 5
    if xticks[-1] != max_gen:
        xticks = np.append(xticks, max_gen)
    ax.set_xticks(xticks)
else:
    # Too few generations, just use all
    ax.set_xticks(sorted(all_gens_combined))

# Y: Set ticks at exactly 0.2, 0.4, ..., 1.0
y_ticks = np.arange(0.2, 1.01, 0.2)
ax.set_yticks(y_ticks)

ax.set_xlim(left=min_gen, right=max_gen)
ax.set_ylim(0, 1)
ax.set_title('All Operator Modes - Aggregated (Overlapped)', fontsize=18)

# Add legend only if data was plotted
if data_plotted:
    ax.legend(loc='upper left', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Only save the plot if data was actually plotted
if data_plotted:
    # === Save the aggregated plot ===
    plot_type = "aggregated_gen_fitness_range"
    filename_pdf = os.path.join(output_dir, f"all_modes_{plot_type}_b.pdf")
    # Remove existing file if it exists (will be overwritten anyway, but explicit)
    if os.path.exists(filename_pdf):
        os.remove(filename_pdf)
    plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
plt.close()  # Close the figure to free memory


# ============================================================================
# VISUALIZATION: Single Unified Efficiency Plot (Temporal Efficiency)
# ============================================================================
# Single plot showing efficiency over generations - complementary to table metrics

try:
    print(f"\n[Unified Efficiency Plot] Creating temporal efficiency visualization...")
    
    if 'per_run_df' in globals() and not per_run_df.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        plot_data_exists = False
        
        for mode in OPERATOR_MODES:
            mode_tracker = EvolutionTracker_df[EvolutionTracker_df['_operator_mode'] == mode].copy()
            if mode_tracker.empty:
                continue
            
            mode_runs = sorted(mode_tracker['_run'].unique())
            mode_efficiency_curves = []
            
            for run_name in mode_runs:
                run_tracker = mode_tracker[mode_tracker['_run'] == run_name].copy()
                if run_tracker.empty:
                    continue
                
                total_genomes = genome_counts.get(run_name, 1)
                if total_genomes == 0:
                    continue
                
                # Calculate cumulative AUC/Genome up to each generation
                generations = sorted(run_tracker['generation_number'].unique())
                cumulative_efficiency = []
                
                for gen in generations:
                    gen_data = run_tracker[run_tracker['generation_number'] <= gen]
                    if gen_data.empty:
                        cumulative_efficiency.append(0)
                        continue
                    
                    y = (pd.Series(gen_data['max_score_variants'], dtype='float64')
                         .cummax().ffill().fillna(0.0).to_numpy())
                    y = np.clip(y, 0.0, 1.0)
                    x = np.arange(len(y), dtype=float)
                    
                    try:
                        auc = np.trapezoid(y, x)
                    except AttributeError:
                        auc = np.trapz(y, x)
                    
                    # Normalize by total genomes to get efficiency
                    efficiency = auc / total_genomes
                    cumulative_efficiency.append(efficiency)
                
                if len(generations) > 0 and len(cumulative_efficiency) > 0:
                    mode_efficiency_curves.append((generations, cumulative_efficiency))
            
            # Plot average efficiency curve for this mode with shaded std dev
            if mode_efficiency_curves:
                all_gens = set()
                for gens, _ in mode_efficiency_curves:
                    all_gens.update(gens)
                all_gens = sorted(all_gens)
                
                if all_gens:
                    interpolated_curves = []
                    for gens, effs in mode_efficiency_curves:
                        if len(gens) == len(effs):
                            # Interpolate to common generation points
                            interp_effs = np.interp(all_gens, gens, effs)
                            interpolated_curves.append(interp_effs)
                    
                    if interpolated_curves:
                        avg_efficiency = np.mean(interpolated_curves, axis=0)
                        
                        # Plot average efficiency line
                        ax.plot(all_gens, avg_efficiency, lw=3.5, 
                               label=f'{mode.upper()}', 
                               color=mode_colors[mode], linestyle='solid', marker='o', 
                               markersize=4, markevery=max(1, len(all_gens)//10))
                        plot_data_exists = True
        
        if plot_data_exists:
            ax.set_xlabel('Generation', fontsize=13, fontweight='bold')
            ax.set_ylabel('Cumulative AUC per Genome', fontsize=13, fontweight='bold')
            ax.set_title('Efficiency Over Generations',
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax.tick_params(labelsize=11)
            
            # Set axes to start from (0, 0) and end at generation 50
            ax.set_xlim(left=0, right=50)
            ax.set_ylim(bottom=0)
            
            # Place legend in top-left (where the blue box was)
            ax.legend(loc='upper left', fontsize=11, framealpha=0.9, ncol=1)
            
            plt.tight_layout()
            
            # Save unified plot
            filename_pdf = os.path.join(output_dir, "unified_efficiency.pdf")
            if os.path.exists(filename_pdf):
                os.remove(filename_pdf)
            plt.savefig(filename_pdf, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[Unified Efficiency Plot] Saved: {filename_pdf}")
        else:
            print(f"[Unified Efficiency Plot] No data available for temporal efficiency plot")
    else:
        print(f"[Unified Efficiency Plot] per_run_df not available")
except Exception as e:
    print(f"[Unified Efficiency Plot] Error: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# Main execution block
# ============================================================================
if __name__ == "__main__":
    saved_files = []
    for mode in OPERATOR_MODES:
        saved_files.append(f"{mode.lower()}_gen_fitness_range_b.pdf")
        saved_files.append(f"{mode.lower()}_distribution_b.pdf")
    saved_files.append("all_modes_aggregated_gen_fitness_range_b.pdf")
    saved_files.append("auc_metrics_table_b.pdf")
    saved_files.append("unified_efficiency.pdf")
    
    print(f"\nRQ1B Analysis completed. Generated {len(saved_files)} plots and tables in {output_dir}")
    print(f"  - Unified efficiency plot: unified_efficiency.pdf")

