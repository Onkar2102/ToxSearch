import os
import json
import glob
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.table import Table
from scipy.stats import kruskal, mannwhitneyu, norm
from itertools import combinations
from pathlib import Path
from collections import defaultdict

OPERATOR_MODES = ['ops', 'comb']

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "..", "data", "outputs")
base_dir = os.path.normpath(base_dir)

output_dir = script_dir

all_matching_dirs = []
for mode in OPERATOR_MODES:
    pattern = os.path.join(base_dir, f"run*_{mode}")
    matching_dirs = sorted(glob.glob(pattern))
    all_matching_dirs.extend(matching_dirs)

if not all_matching_dirs:
    raise ValueError(f"No directories found for any operator mode")

all_dfs = {}
dfs = {}

def flatten_operator_statistics(df, col="operator_statistics"):
    """Flattens nested operator statistics dictionary into separate columns."""
    if col not in df.columns:
        return df
    all_keys = set()
    for ops in df[col]:
        if isinstance(ops, dict):
            for k in ops.keys():
                all_keys.add(k)
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

for data_dir in all_matching_dirs:
    run_name = os.path.basename(data_dir.rstrip('/'))
    
    operator_mode = None
    for mode in OPERATOR_MODES:
        if run_name.endswith(f'_{mode}'):
            operator_mode = mode
            break
    
    if operator_mode is None:
        continue
    
    filenames = [f for f in os.listdir(data_dir) if not f.startswith(".") and os.path.isfile(os.path.join(data_dir, f))]
    
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
                    if "operator_statistics" in df.columns:
                        df = flatten_operator_statistics(df, col="operator_statistics")
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

if all_dfs:
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
            globals()[f"{label}_df"] = df
        except Exception as e:
            pass

group_labels = ['elites', 'non_elites', 'under_performing']
selected_dfs = []

for label in group_labels:
    if label in dfs:
        df = dfs[label].copy()
        if df.empty:
            continue
        df['_source_group'] = label
        selected_dfs.append(df)

if selected_dfs:
    unified_df = pd.concat(selected_dfs, ignore_index=True, sort=False)
else:
    unified_df = pd.DataFrame()

unified_df.head(3).T

removal_threshold_run_df = EvolutionTracker_df[['generation_number', 'removal_threshold', '_run']].copy()

selected_unified_df = unified_df[['generation', 'id', 'moderation_result_google.scores.toxicity', '_run']].copy()
selected_unified_df['generation'] = pd.to_numeric(selected_unified_df['generation'], errors='coerce')
selected_unified_df['moderation_result_google.scores.toxicity'] = pd.to_numeric(selected_unified_df['moderation_result_google.scores.toxicity'], errors='coerce')

removal_threshold_run_df['generation_number'] = pd.to_numeric(removal_threshold_run_df['generation_number'], errors='coerce')
removal_threshold_run_df['removal_threshold'] = pd.to_numeric(removal_threshold_run_df['removal_threshold'], errors='coerce')

min_tox_by_gen_by_run = {}
per_run_data = []

for run_name in all_dfs.keys():
    run_threshold_df = removal_threshold_run_df[removal_threshold_run_df['_run'] == run_name].copy()
    run_selected_df = selected_unified_df[selected_unified_df['_run'] == run_name].copy()
    
    run_mode = None
    for mode in OPERATOR_MODES:
        if run_name.endswith(f'_{mode}'):
            run_mode = mode
            break
    
    if run_mode is None and '_operator_mode' in run_selected_df.columns and not run_selected_df.empty:
        run_mode = run_selected_df['_operator_mode'].iloc[0]
    
    run_tracker = EvolutionTracker_df[EvolutionTracker_df['_run'] == run_name].copy()
    run_tracker = run_tracker.sort_values('generation_number')
    
    if not run_tracker.empty and 'max_score_variants' in run_tracker.columns:
        run_tracker['cumulative_max'] = run_tracker['max_score_variants'].cummax()
        cumulative_max_dict = dict(zip(run_tracker['generation_number'], run_tracker['cumulative_max']))
    else:
        cumulative_max_dict = {}
    
    lowest_toxicity_by_generation = []
    
    for idx, row in run_threshold_df.iterrows():
        gen = row['generation_number']
        threshold = row['removal_threshold']
        
        mask = run_selected_df['generation'] <= gen
        candidates = run_selected_df.loc[
            mask & (run_selected_df['moderation_result_google.scores.toxicity'] >= threshold)
        ]
        
        cumulative_max = cumulative_max_dict.get(gen, np.nan)
        
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
            
            per_run_data.append({
                'generation_number': gen,
                '_run': run_name,
                '_operator_mode': run_mode,
                'minimum_toxicity': min_toxicity,
                'cumulative_maximum_score': cumulative_max,
                'average_fitness': avg_fitness
            })
        else:
            lowest_toxicity_by_generation.append({
                'generation_number': gen,
                'id': np.nan,
                'toxicity': np.nan,
                'removal_threshold': threshold
            })
            
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

per_run_df = pd.DataFrame(per_run_data)

all_generations = sorted(per_run_df['generation_number'].unique())
aggregated_data = []

cumulative_max_all_runs = -np.inf

for gen_num in all_generations:
    gen_data = per_run_df[per_run_df['generation_number'] == gen_num].copy()
    
    min_toxicity_values = gen_data['minimum_toxicity'].dropna().values
    if len(min_toxicity_values) > 0:
        aggregated_min_toxicity = np.min(min_toxicity_values)
    else:
        aggregated_min_toxicity = np.nan
    
    max_score_values = gen_data['cumulative_maximum_score'].dropna().values
    if len(max_score_values) > 0:
        current_gen_max = np.max(max_score_values)
        cumulative_max_all_runs = max(cumulative_max_all_runs, current_gen_max)
        aggregated_cumulative_max = cumulative_max_all_runs
    else:
        aggregated_cumulative_max = cumulative_max_all_runs if cumulative_max_all_runs != -np.inf else np.nan
    
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

min_tox_by_gen_list = []
for run_name, min_df in min_tox_by_gen_by_run.items():
    min_tox_by_gen_list.append(min_df)

if min_tox_by_gen_list:
    min_tox_by_gen = pd.concat(min_tox_by_gen_list, ignore_index=True)
else:
    min_tox_by_gen = pd.DataFrame(columns=['generation_number', 'min_score', '_run'])

mode_colors = {
    'ops': '#377eb8',
    'comb': '#4daf4a'
}

colors = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
    "#ffff33",
    "#1b7837",
    "#d6604d",
    "#4393c3",
    "#b2abd2",
    "#e08214",
    "#6a3d9a",
    "#a6cee3",
    "#fb9a99",
    "#b15928",
    "#fee08b",
    "#d95f02",
]

if 'per_run_df' not in globals() or per_run_df.empty:
    raise ValueError("per_run_df not found. Please run the data processing section first.")

for mode in OPERATOR_MODES:
    mode_data = per_run_df[per_run_df['_operator_mode'] == mode].copy()
    if mode_data.empty:
        continue
    
    run_names = sorted(mode_data['_run'].unique())
    
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
                
                ax.plot(generations, avg_fit, lw=2, 
                       color=color, linestyle='solid', label=f"Run {data['run_num']}")
                
                valid_mask = (min_vals >= 0) & (max_vals >= 0) & (min_vals <= max_vals)
                if np.any(valid_mask):
                    ax.fill_between(generations[valid_mask], min_vals[valid_mask], max_vals[valid_mask], 
                                   facecolor=color, alpha=0.20)
        
        ax.set_xlabel('Generation Number', fontsize=18, fontweight='bold')
        ax.set_ylabel('Score', fontsize=18, fontweight='bold')
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
        if os.path.exists(filename_pdf):
            os.remove(filename_pdf)
        plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
        plt.close()

all_aucs = []
all_aucs_norm = []
all_avg_gains = []

for run_name in all_dfs.keys():
    run_tracker = EvolutionTracker_df[EvolutionTracker_df['_run'] == run_name].copy()
    if run_tracker.empty:
        continue
    
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
    
    all_aucs.append(auc)
    all_aucs_norm.append(auc_norm)
    all_avg_gains.append(avg_gain)

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

best_score_all_runs = EvolutionTracker_df['max_score_variants'].max()

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

worst_score_all_runs = min(last_min_vals_per_run) if len(last_min_vals_per_run) > 0 else np.nan

all_max_scores = EvolutionTracker_df['max_score_variants'].values
all_avg_scores = EvolutionTracker_df['avg_fitness_generation'].values

all_scores_for_avg = np.concatenate([
    all_max_scores[~np.isnan(all_max_scores)],
    all_avg_scores[~np.isnan(all_avg_scores)]
])
avg_score_all_runs = np.mean(all_scores_for_avg) if len(all_scores_for_avg) > 0 else np.nan

def count_total_genomes_per_run(run_name, base_dir):
    """Counts total genomes across elites and reserves files for a run.
    
    Note: We only maintain elites in our project. Active population = elites + reserves.
    Archive.json is NOT part of the population and is excluded from this count.
    """
    run_dir = os.path.join(base_dir, run_name)
    if not os.path.isdir(run_dir):
        return 0
    
    total_count = 0
    for filename in ['elites.json', 'reserves.json']:
        file_path = os.path.join(run_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        total_count += len(data)
            except Exception as e:
                pass
    
    return total_count

genome_counts = {}
for run_name in all_dfs.keys():
    genome_counts[run_name] = count_total_genomes_per_run(run_name, base_dir)

all_table_rows = []

for mode in OPERATOR_MODES:
    mode_tracker = EvolutionTracker_df[EvolutionTracker_df['_operator_mode'] == mode].copy()
    if mode_tracker.empty:
        continue
    
    mode_runs = sorted(mode_tracker['_run'].unique())
    
    mode_aucs = []
    mode_aucs_norm = []
    mode_avg_gains = []
    mode_min_vals = []
    mode_max_vals = []
    mode_avg_fitness_vals = []
    
    for run_name in mode_runs:
        run_tracker = mode_tracker[mode_tracker['_run'] == run_name].copy()
        if run_tracker.empty:
            continue
        
        run_num = re.match(r'run0*(\d+)_(\w+)', str(run_name))
        run_display = f"Run {run_num.group(1)}" if run_num else run_name
        
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
        
        run_min = np.nan
        if 'min_tox_by_gen_by_run' in globals() and run_name in min_tox_by_gen_by_run:
            if not min_tox_by_gen_by_run[run_name].empty:
                run_min_df = min_tox_by_gen_by_run[run_name].copy()
                run_min_df = run_min_df.sort_values('generation_number')
                if len(run_min_df) > 0:
                    run_min = run_min_df['min_score'].iloc[-1]
                    if not np.isnan(run_min):
                        mode_min_vals.append(run_min)
        
        run_max_scores = run_tracker['max_score_variants'].values
        run_max = np.max(run_max_scores[~np.isnan(run_max_scores)]) if len(run_max_scores) > 0 else np.nan
        if not np.isnan(run_max):
            mode_max_vals.append(run_max)
        
        run_avg_fitness = run_tracker['avg_fitness_generation'].values
        run_avg = np.mean(run_avg_fitness[~np.isnan(run_avg_fitness)]) if len(run_avg_fitness) > 0 else np.nan
        if not np.isnan(run_avg):
            mode_avg_fitness_vals.append(run_avg)
        
        total_genomes = genome_counts.get(run_name, 0)
        auc_per_genome = auc / total_genomes if total_genomes > 0 else np.nan
        max_per_genome = run_max / total_genomes if total_genomes > 0 and not np.isnan(run_max) else np.nan
        avg_gain_per_genome = avg_gain / total_genomes if total_genomes > 0 else np.nan
        
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
    
    min_across_runs = min(mode_min_vals) if len(mode_min_vals) > 0 else np.nan
    max_across_runs = max(mode_max_vals) if len(mode_max_vals) > 0 else np.nan
    avg_fitness_across_runs = np.mean(mode_avg_fitness_vals) if len(mode_avg_fitness_vals) > 0 else np.nan
    mean_auc = np.mean(mode_aucs) if len(mode_aucs) > 0 else np.nan
    mean_auc_norm = np.mean(mode_aucs_norm) if len(mode_aucs_norm) > 0 else np.nan
    mean_avg_gain = np.mean(mode_avg_gains) if len(mode_avg_gains) > 0 else np.nan
    
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
    
    mode_total_genomes = sum([row['total_genomes'] for row in all_table_rows 
                             if row['mode'].startswith(f"{mode.upper()}") and not row['is_aggregate']])
    
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

if all_table_rows:
    num_rows = len(all_table_rows)
    fig_height = max(6, num_rows * 0.5 + 2)
    
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
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
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    row_idx = 1
    for row_data in all_table_rows:
        if row_data['is_aggregate']:
            for j in range(len(headers)):
                table[(row_idx, j)].set_facecolor('#B3E5FC')
                table[(row_idx, j)].set_text_props(weight='bold')
        else:
            for j in range(len(headers)):
                if row_idx % 2 == 0:
                    table[(row_idx, j)].set_facecolor('#f0f0f0')
                else:
                    table[(row_idx, j)].set_facecolor('white')
        row_idx += 1
    
    plt.title('AUC and Budget-Normalized Metrics by Operator Mode and Run\n(Per-Genome Metrics Enable Fair Comparison Across Different Budgets)', 
              fontsize=13, fontweight='bold', pad=20)
    
    filename_pdf = os.path.join(output_dir, "auc_metrics_table_b.pdf")
    if os.path.exists(filename_pdf):
        os.remove(filename_pdf)
    plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
    plt.close()

for mode in OPERATOR_MODES:
    mode_unified_df = unified_df[unified_df['_operator_mode'] == mode].copy()
    if mode_unified_df.empty:
        continue
    
    scores = mode_unified_df['moderation_result_google.scores.toxicity'].dropna()
    if len(scores) == 0:
        continue
    
    min_score = scores.min()
    max_score = scores.max()
    
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=30, color=mode_colors[mode], edgecolor='black', alpha=0.7)
    
    plt.title(f'Distribution of Toxicity Scores: {mode.upper()} Mode', fontsize=14)
    plt.xlabel('Toxicity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename_pdf = os.path.join(output_dir, f"{mode.lower()}_distribution_b.pdf")
    if os.path.exists(filename_pdf):
        os.remove(filename_pdf)
    plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
    plt.close()

if 'per_run_df' not in globals() or per_run_df.empty:
    raise ValueError("per_run_df not found. Please run the data processing section first.")

plt.figure(figsize=(7.5, 4.5))
ax = plt.gca()

mode_colors = {
    'ops': '#1f77b4',
    'comb': '#ff7f0e'
}

all_generations = sorted(per_run_df['generation_number'].unique())

data_plotted = False

for mode in OPERATOR_MODES:
    mode_data = per_run_df[per_run_df['_operator_mode'] == mode].copy()
    if mode_data.empty:
        continue

    mode_runs = mode_data['_run'].unique()
    run_lengths = []
    for run_name in mode_runs:
        run_data = mode_data[mode_data['_run'] == run_name].copy()
        run_lengths.append(len(run_data))

    min_run_length = min(run_lengths) if len(run_lengths) > 0 else 0
    mode_generations = sorted(mode_data['generation_number'].unique())[:min_run_length]

    aggregated_data = []
    cumulative_max_all_runs = -np.inf

    for gen_num in mode_generations:
        gen_data = mode_data[mode_data['generation_number'] == gen_num].copy()
        count = len(gen_data)
        if count == 0:
            continue

        min_toxicity_values = gen_data['minimum_toxicity'].dropna().values
        if len(min_toxicity_values) > 0:
            aggregated_min_toxicity = np.min(min_toxicity_values)
        else:
            aggregated_min_toxicity = np.nan

        max_score_values = gen_data['cumulative_maximum_score'].dropna().values
        if len(max_score_values) > 0:
            current_gen_max = np.max(max_score_values)
            cumulative_max_all_runs = max(cumulative_max_all_runs, current_gen_max)
            aggregated_cumulative_max = cumulative_max_all_runs
        else:
            aggregated_cumulative_max = cumulative_max_all_runs if cumulative_max_all_runs != -np.inf else np.nan

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

    mode_agg_df = pd.DataFrame(aggregated_data)
    mode_agg_df = mode_agg_df.sort_values('generation_number')
    mode_agg_df = mode_agg_df.dropna(subset=['generation_number', 'average_fitness'])

    if mode_agg_df.empty:
        continue

    generations = mode_agg_df['generation_number'].values
    max_vals = mode_agg_df['cumulative_maximum_score'].ffill().fillna(0).values
    min_vals = mode_agg_df['minimum_toxicity'].ffill().fillna(0).values
    avg_fit = mode_agg_df['average_fitness'].values

    for i in range(len(min_vals)):
        if min_vals[i] > max_vals[i] and max_vals[i] > 0:
            min_vals[i] = max_vals[i]

    color = mode_colors[mode]

    if len(generations) > 0 and len(avg_fit) > 0:
        valid_mask = (min_vals >= 0) & (max_vals >= 0) & (min_vals <= max_vals)
        if np.any(valid_mask):
            ax.fill_between(
                generations[valid_mask],
                min_vals[valid_mask],
                max_vals[valid_mask],
                facecolor=color,
                alpha=0.18
            )

        ax.plot(
            generations,
            avg_fit,
            lw=2.4,
            label=f'{mode.upper()}',
            color=color,
            linestyle='solid'
        )

        data_plotted = True

ax.set_xlabel('Generated Prompts', fontsize=11, fontweight='bold')
ax.set_ylabel('Score', fontsize=11, fontweight='bold')

ax.tick_params(axis='x', labelsize=9)
ax.tick_params(axis='y', labelsize=9)

all_gens_combined = per_run_df['generation_number'].unique()
max_gen = int(all_gens_combined.max()) if len(all_gens_combined) > 0 else 0
min_gen = int(all_gens_combined.min()) if len(all_gens_combined) > 0 else 0

if max_gen - min_gen >= 5:
    xticks = np.arange(min_gen, max_gen + 1, 5)
    if xticks[-1] != max_gen:
        xticks = np.append(xticks, max_gen)
    ax.set_xticks(xticks)
else:
    ax.set_xticks(sorted(all_gens_combined))

y_ticks = np.arange(0.0, 1.01, 0.2)
ax.set_yticks(y_ticks)

ax.set_xlim(left=min_gen, right=max_gen)
ax.set_ylim(0.0, 1.0)

if data_plotted:
    ax.legend(loc='upper left', fontsize=9, frameon=False)

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)

plt.tight_layout()

if data_plotted:
    plot_type = "aggregated_gen_fitness_range"
    filename_pdf = os.path.join(output_dir, f"all_modes_{plot_type}_b.pdf")
    if os.path.exists(filename_pdf):
        os.remove(filename_pdf)
    plt.savefig(filename_pdf, dpi=300, bbox_inches='tight')
plt.close()

try:
    
    if 'per_run_df' in globals() and not per_run_df.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        
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
                
                generations = sorted(run_tracker['generation_number'].unique())
                cumulative_efficiency = []
                
                for gen in generations:
                    gen_data = run_tracker[run_tracker['generation_number'] <= gen]
                    if gen_data.empty:
                        cumulative_efficiency.append(0.0)
                        continue
                    
                    y = (pd.Series(gen_data['max_score_variants'], dtype='float64')
                         .cummax().ffill().fillna(0.0).to_numpy())
                    y = np.clip(y, 0.0, 1.0)
                    x = np.arange(len(y), dtype=float)
                    
                    try:
                        auc = np.trapezoid(y, x)
                    except AttributeError:
                        auc = np.trapz(y, x)
                    
                    efficiency = auc / total_genomes
                    cumulative_efficiency.append(efficiency)
                
                if len(generations) > 0 and len(cumulative_efficiency) > 0:
                    mode_efficiency_curves.append((generations, cumulative_efficiency))
            
            if mode_efficiency_curves:
                all_gens = set()
                for gens, _ in mode_efficiency_curves:
                    all_gens.update(gens)
                all_gens = sorted(all_gens)
                
                if all_gens:
                    interpolated_curves = []
                    for gens, effs in mode_efficiency_curves:
                        if len(gens) == len(effs):
                            interp_effs = np.interp(all_gens, gens, effs)
                            interpolated_curves.append(interp_effs)
                    
                    if interpolated_curves:
                        avg_efficiency = np.mean(interpolated_curves, axis=0)
                        
                        ax.plot(
                            all_gens,
                            avg_efficiency,
                            lw=2.4,
                            label=f'{mode.upper()}',
                            color=mode_colors[mode],
                            linestyle='solid'
                        )
                        plot_data_exists = True
        
        if plot_data_exists:
            ax.set_xlabel('Generated Prompts', fontsize=11, fontweight='bold')
            ax.set_ylabel('Cumulative AUC per Genome', fontsize=11, fontweight='bold')
            
            all_gens_global = per_run_df['generation_number'].unique()
            if len(all_gens_global) > 0:
                max_gen = int(all_gens_global.max())
                min_gen = int(all_gens_global.min())
            else:
                max_gen, min_gen = 0, 0
            
            if max_gen - min_gen >= 5:
                xticks = np.arange(min_gen, max_gen + 1, 5)
                if xticks[-1] != max_gen:
                    xticks = np.append(xticks, max_gen)
                ax.set_xticks(xticks)
            else:
                ax.set_xticks(sorted(all_gens_global))
            
            ax.set_xlim(left=min_gen, right=max_gen if max_gen > min_gen else min_gen + 1)
            ax.set_ylim(bottom=0.0)
            
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
            
            ax.legend(loc='upper left', fontsize=9, frameon=False)
            
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
            
            plt.tight_layout()
            
            filename_pdf = os.path.join(output_dir, "unified_efficiency.pdf")
            if os.path.exists(filename_pdf):
                os.remove(filename_pdf)
            plt.savefig(filename_pdf, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            pass
    else:
        pass
except Exception as e:
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    saved_files = []
    for mode in OPERATOR_MODES:
        saved_files.append(f"{mode.lower()}_gen_fitness_range_b.pdf")
        saved_files.append(f"{mode.lower()}_distribution_b.pdf")
    saved_files.append("all_modes_aggregated_gen_fitness_range_b.pdf")
    saved_files.append("auc_metrics_table_b.pdf")
    saved_files.append("unified_efficiency.pdf")
    

CROSSOVER_OPERATORS = {'SemanticSimilarityCrossover', 'SemanticFusionCrossover'}

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
base_data_dir = os.path.join(project_root, "data", "outputs")
base_data_dir = os.path.normpath(base_data_dir)

pattern = os.path.join(base_data_dir, "run*_comb")
run_dirs = sorted(glob.glob(pattern))
run_dirs = [os.path.basename(d.rstrip('/')) for d in run_dirs]

if not run_dirs:
    raise ValueError(f"No comb run directories found in {base_data_dir}")

def process_single_run(run_dir):
    """Processes a single run directory and extracts data from JSON and CSV files."""
    data_dir = os.path.join(base_data_dir, run_dir)
    
    if not os.path.exists(data_dir):
        return None
    
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
    
    unified_df['delta_score'] = unified_df['moderation_result_google.scores.toxicity'] - unified_df['parent_score']
    
    EvolutionTracker_df = dfs.get('EvolutionTracker', None)
    if EvolutionTracker_df is None:
        return None
    
    operator_vs_initial_state = pd.crosstab(
        unified_df['operator'].fillna('Initial Seed'),
        unified_df['initial_state'].fillna('none')
    )
    operator_vs_initial_state['total'] = operator_vs_initial_state.sum(axis=1)
    
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
    
    operator_delta_stats = unified_df.groupby('operator')['delta_score'].agg(['mean', 'std']).round(2)
    
    result_data = {}
    all_operators_set = set(operator_names) | set(operator_vs_initial_state.index) - {'Initial Seed'}
    
    for operator in sorted(all_operators_set):
        if operator in operator_vs_initial_state.index:
            elite = operator_vs_initial_state.loc[operator, 'elite'] if 'elite' in operator_vs_initial_state.columns else 0
            non_elite = operator_vs_initial_state.loc[operator, 'non_elite'] if 'non_elite' in operator_vs_initial_state.columns else 0
            total = operator_vs_initial_state.loc[operator, 'total']
        else:
            elite = non_elite = total = 0
        
        if operator in operator_names:
            col_q = f'operator_statistics_{operator}_question_mark_rejections'
            col_d = f'operator_statistics_{operator}_duplicates_removed'
            question_removed = EvolutionTracker_df[col_q].sum() if col_q in EvolutionTracker_df.columns else 0
            duplicates_removed = EvolutionTracker_df[col_d].sum() if col_d in EvolutionTracker_df.columns else 0
        else:
            question_removed = duplicates_removed = 0
        
        calculated_total = total + question_removed + duplicates_removed
        
        if calculated_total == 0:
            continue
        
        NE = (non_elite / calculated_total * 100).round(2) if calculated_total > 0 else 0.0
        EHR = (elite / calculated_total * 100).round(2) if calculated_total > 0 else 0.0
        IR = (question_removed / calculated_total * 100).round(2) if calculated_total > 0 else 0.0
        cEHR = (elite / total * 100).round(2) if total > 0 else 0.0
        
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

all_operators = set()
for run_key, df in all_run_results.items():
    all_operators.update(df.index.tolist())

sorted_run_keys = sorted(all_run_results.keys(), key=lambda x: int(x) if x.isdigit() else 999)

table_rows = []
for operator in sorted(all_operators):
    run_data = {}
    for run_key in sorted_run_keys:
        if run_key in all_run_results and operator in all_run_results[run_key].index:
            run_data[run_key] = all_run_results[run_key].loc[operator]
        else:
            run_data[run_key] = None
    
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

final_table_df = pd.DataFrame(table_rows)

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

for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

for row_idx, (_, row) in enumerate(final_table_df.iterrows(), start=1):
    for j in range(len(headers)):
        if row['is_mean']:
            table[(row_idx, j)].set_facecolor('#B3E5FC')
            table[(row_idx, j)].set_text_props(weight='bold')
        else:
            table[(row_idx, j)].set_facecolor('#f0f0f0' if row_idx % 2 == 0 else 'white')

plt.title('RQ2: Operator Performance Metrics (Rates % and Deltas)', fontsize=14, fontweight='bold', pad=20)

output_dir = script_dir

filename_pdf = os.path.join(output_dir, "rq2_operator_metrics_table.pdf")
if os.path.exists(filename_pdf):
    os.remove(filename_pdf)
plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
plt.close()

simplified_table_df = final_table_df[final_table_df['is_mean'] == True].copy()
simplified_table_df = simplified_table_df[['Operator', 'NE', 'EHR', 'IR', 'cEHR', 'Δμ', 'Δσ']].copy()
simplified_table_df = simplified_table_df.sort_values('Operator').reset_index(drop=True)

csv_filename = os.path.join(output_dir, "rq2_operator_metrics_simplified.csv")
simplified_table_df.to_csv(csv_filename, index=False)

if 'final_table_df' not in globals() or final_table_df.empty:
    raise ValueError("final_table_df not found. Please run the main processing section first.")

test_data_df = final_table_df[final_table_df['is_mean'] == False].copy()

metrics = ['EHR', 'cEHR', 'IR', 'NE', 'Δμ', 'Δσ']
metric_names = {
    'EHR': 'Elite Hit Rate (%)',
    'cEHR': 'Conditional Elite Hit Rate (%)',
    'IR': 'Invalid Rate (%)',
    'NE': 'Non-Elite Percentage (%)',
    'Δμ': 'Mean Delta Score',
    'Δσ': 'Delta Score Std Dev'
}

statistical_results = {}

for metric in metrics:
    
    operator_data = {}
    for operator in sorted(all_operators):
        operator_df = test_data_df[(test_data_df['Operator'] == operator) & 
                                    (test_data_df[metric].notna())]
        values = operator_df[metric].dropna().tolist()
        if len(values) > 0:
            operator_data[operator] = values
    
    if len(operator_data) < 2:
        continue
    
    operators_with_data = [op for op in sorted(operator_data.keys()) if len(operator_data[op]) > 0]
    
    if len(operators_with_data) < 2:
        continue
    
    groups = [operator_data[op] for op in operators_with_data]
    operators_list = operators_with_data
    
    try:
        h_statistic, p_value = kruskal(*groups)
        
        if p_value < 0.05:
            pass
        else:
            pass
        
        statistical_results[metric] = {
            'kruskal_wallis': {
                'h_statistic': h_statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'operator_data': operator_data,
            'operators': operators_list
        }
        
        if p_value < 0.05:
            pairwise_results = []
            operator_pairs = list(combinations(operators_list, 2))
            num_comparisons = len(operator_pairs)
            bonferroni_alpha = 0.05 / num_comparisons
            
            
            significant_pairs = []
            for op1, op2 in operator_pairs:
                try:
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
                    
                    mean1 = np.mean(data1)
                    mean2 = np.mean(data2)
                    mean_diff = mean1 - mean2
                    
                    expected_u = n1 * n2 / 2
                    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                    if std_u > 0:
                        z_score = (u_statistic - expected_u) / std_u
                        n_total = n1 + n2
                        effect_size_r = z_score / np.sqrt(n_total)
                    else:
                        effect_size_r = np.nan
                    
                    n_bootstrap = 1000
                    bootstrap_diffs = []
                    for _ in range(n_bootstrap):
                        sample1 = np.random.choice(data1, size=n1, replace=True)
                        sample2 = np.random.choice(data2, size=n2, replace=True)
                        bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))
                    
                    ci_lower = np.percentile(bootstrap_diffs, 2.5)
                    ci_upper = np.percentile(bootstrap_diffs, 97.5)
                    
                    if is_significant:
                        direction = ">" if mean1 > mean2 else "<"
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
                    continue
            
            if not significant_pairs:
                pass
            
            effect_sizes = [p.get('effect_size_r', np.nan) for p in pairwise_results if not np.isnan(p.get('effect_size_r', np.nan))]
            if len(effect_sizes) > 0:
                small_effects = sum(1 for r in effect_sizes if abs(r) < 0.1)
                medium_effects = sum(1 for r in effect_sizes if 0.1 <= abs(r) < 0.3)
                large_effects = sum(1 for r in effect_sizes if abs(r) >= 0.3)
            
            statistical_results[metric]['pairwise'] = pairwise_results
            statistical_results[metric]['significant_pairs'] = significant_pairs
        
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
        if metric in ['EHR', 'cEHR']:
            summary_df = summary_df.sort_values('Mean', ascending=False)
        else:
            summary_df = summary_df.sort_values('Mean', ascending=True)
        
    except Exception as e:
        continue

significant_metrics = [m for m in metrics if m in statistical_results and 
                       statistical_results[m]['kruskal_wallis']['significant']]

if significant_metrics:
    for metric in significant_metrics:
        p_val = statistical_results[metric]['kruskal_wallis']['p_value']
        num_sig_pairs = len(statistical_results[metric].get('significant_pairs', []))
else:
    pass

if 'output_dir' not in globals():
    output_dir = script_dir

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


def get_max_toxicity(elite):
    """Extracts the maximum toxicity score across all models for an elite prompt."""
    models_dict = elite.get('models', {})
    max_toxicity = 0.0
    for model_name, model_data in models_dict.items():
        toxicity = model_data.get('toxicity')
        if toxicity is not None and toxicity > max_toxicity:
            max_toxicity = toxicity
    return max_toxicity

def extract_toxicity(record):
    """Extracts toxicity score from moderation result in a record."""
    try:
        if 'moderation_result' in record and 'google' in record['moderation_result']:
            if 'scores' in record['moderation_result']['google']:
                return record['moderation_result']['google']['scores'].get('toxicity', None)
    except:
        pass
    return None

def shorten_model_name(name):
    """Shortens and normalizes model names for display purposes."""
    if name == 'Llama-3.1-8B':
        return 'Llama-3.1-8B'
    
    name = name.replace('-Q4_K_L', '').replace('-Q4_K_M', '').replace('.Q3_K_S', '')
    name = name.replace('-Q4_K_S', '').replace('-v0.3', '').replace('-Instruct', '')
    
    name = name.replace('Meta-Llama-3.1-8B', 'Llama-3.1-8B')
    name = name.replace('Llama-3.2-1B', 'Llama-3.2-1B')
    name = name.replace('Llama-3.2-3B', 'Llama-3.2-3B')
    name = name.replace('Mistral-7B', 'Mistral-7B')
    name = name.replace('Phi-3.5-mini', 'Phi-3.5')
    name = name.replace('Qwen2.5-7B', 'Qwen-7B')
    name = name.replace('gemma-2-9b-it', 'Gemma-9B')
    
    return name

def normalize_prompt(prompt):
    """Normalizes prompt text by stripping whitespace and converting to lowercase."""
    return prompt.strip().lower()

def normalize_model_name(model_name):
    """Removes .gguf suffix from model names for consistent naming."""
    if model_name.endswith('.gguf'):
        return model_name[:-5]
    return model_name

def is_refusal_response(response):
    """Checks if a response is a single-sentence refusal starting with refusal patterns."""
    if not response or not response.strip():
        return False
    
    response_lower = response.strip().lower()
    response_stripped = response.strip()
    
    refusal_patterns = [
        "i can't",
        "i cannot",
        "i'm not able to",
        "i'm unable to",
        "i don't feel comfortable",
        "i'm not comfortable",
        "i cannot provide",
        "i cannot help",
        "i'm not programmed to",
        "i'm not designed to",
        "i'm sorry, but i can't",
        "i'm sorry, but i cannot",
        "i apologize, but i can't",
        "i apologize, but i cannot",
        "i won't",
        "i will not",
        "i refuse",
        "i decline",
        "i'm not going to",
        "i cannot answer",
        "i cannot respond",
        "i cannot assist",
        "i'm not able to help",
        "i'm not able to provide",
        "i'm not able to answer",
        "i'm not able to respond",
        "i'm not able to assist",
        "i'm unable to help",
        "i'm unable to provide",
        "i'm unable to answer",
        "i'm unable to respond",
        "i'm unable to assist",
    ]
    
    starts_with_refusal = False
    for pattern in refusal_patterns:
        if response_lower.startswith(pattern):
            starts_with_refusal = True
            break
    
    if not starts_with_refusal:
        return False
    
    first_period = response_stripped.find('.')
    first_exclamation = response_stripped.find('!')
    first_question = response_stripped.find('?')
    
    sentence_end_positions = []
    if first_period != -1:
        sentence_end_positions.append(first_period)
    if first_exclamation != -1:
        sentence_end_positions.append(first_exclamation)
    if first_question != -1:
        sentence_end_positions.append(first_question)
    
    if not sentence_end_positions:
        return False
    
    first_sentence_end = min(sentence_end_positions)
    
    after_first_sentence = response_stripped[first_sentence_end + 1:].strip()
    
    if after_first_sentence:
        after_lower = after_first_sentence.lower()
        artifacts = ['assistant:', 'assistant', 'user:', 'user']
        
        is_just_artifact = False
        for artifact in artifacts:
            if after_lower.startswith(artifact) and len(after_first_sentence) <= len(artifact) + 5:
                is_just_artifact = True
                break
        
        if not is_just_artifact:
            return False
    
    return True

def main():
    """Main function that processes elite prompts and generates RQ3 analysis outputs."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_data_dir = os.path.join(project_root, "data", "outputs")
    base_data_dir = os.path.normpath(base_data_dir)
    
    
    pattern = os.path.join(base_data_dir, "**", "elites.json")
    elite_files = sorted(glob.glob(pattern, recursive=True))
    
    all_elites = []
    
    for file_path in elite_files:
        try:
            rel_path = os.path.relpath(file_path, base_data_dir)
            run_dir = os.path.dirname(rel_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
            
            for elite in elites:
                toxicity_score = None
                if 'moderation_result' in elite and 'google' in elite['moderation_result']:
                    if 'scores' in elite['moderation_result']['google']:
                        toxicity_score = elite['moderation_result']['google']['scores'].get('toxicity', None)
                
                model_name_path = elite.get('model_name', '')
                model_filename = os.path.basename(model_name_path) if model_name_path else ''
                
                models = {
                    model_filename: {
                        'response': elite.get('generated_output', ''),
                        'toxicity': toxicity_score
                    }
                }
                
                source = {
                    'genome_id': elite.get('id', None),
                    'operator': elite.get('operator', ''),
                    'dir_name': run_dir
                }
                
                enriched_elite = {
                    'id': None,
                    'prompt': elite.get('prompt', ''),
                    'source': source,
                    'models': models
                }
                
                all_elites.append(enriched_elite)
            
        except Exception as e:
            continue
    
    
    
    all_elites.sort(key=get_max_toxicity, reverse=True)
    
    unique_elites_dict = {}
    for elite in all_elites:
        prompt = elite['prompt']
        if prompt not in unique_elites_dict:
            unique_elites_dict[prompt] = elite
    
    combined_elites_list = list(unique_elites_dict.values())
    
    combined_elites_list = [elite for elite in combined_elites_list if elite['prompt'].strip().endswith('?')]
    
    for idx, elite in enumerate(combined_elites_list, start=1):
        elite['id'] = idx
    
    
    json_path = os.path.join(script_dir, "rq3_combined_elites.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_elites_list, f, indent=2, ensure_ascii=False)
    
    
    elites_with_toxicity = []
    for elite in combined_elites_list:
        max_tox = get_max_toxicity(elite)
        elites_with_toxicity.append({
            'elite': elite,
            'toxicity': max_tox
        })
    
    elites_with_toxicity.sort(key=lambda x: x['toxicity'], reverse=True)
    
    total_count = len(elites_with_toxicity)
    top_25_percent_count = int(total_count * 0.25)
    
    top_25_percent_elites = elites_with_toxicity[:top_25_percent_count]
    
    top_prompts = [item['elite']['prompt'] for item in top_25_percent_elites]
    
    df_top = pd.DataFrame({'questions': top_prompts})
    
    
    csv_path = os.path.join(project_root, "data", "combined_elites.csv")
    df_top.to_csv(csv_path, index=False, encoding='utf-8')
    
    
    top_25_prompts = {item['elite']['prompt'] for item in top_25_percent_elites}
    
    original_toxicity_scores = [item['toxicity'] for item in top_25_percent_elites]
    
    top_25_elites = []
    for idx, item in enumerate(top_25_percent_elites, start=1):
        elite = item['elite'].copy()
        
        if 'models' in elite:
            normalized_models = {}
            for model_name, model_data in elite['models'].items():
                normalized_name = normalize_model_name(model_name)
                if normalized_name not in normalized_models:
                    normalized_models[normalized_name] = model_data
                else:
                    existing_tox = normalized_models[normalized_name].get('toxicity', 0)
                    new_tox = model_data.get('toxicity', 0)
                    if new_tox > existing_tox:
                        normalized_models[normalized_name] = model_data
            elite['models'] = normalized_models
        
        elite['id'] = idx
        top_25_elites.append(elite)
    
    prompt_to_elite = {}
    for elite in top_25_elites:
        prompt = elite.get('prompt', '')
        normalized = normalize_prompt(prompt)
        prompt_to_elite[normalized] = elite
    
    model_dirs = []
    for item in Path(base_data_dir).iterdir():
        if item.is_dir():
            if not item.name.startswith('run') and not item.name.startswith('2025'):
                model_dirs.append(item)
    
    
    model_toxicity_scores = {}
    model_toxicity_scores['Llama-3.1-8B'] = original_toxicity_scores
    
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        normalized_model_name = normalize_model_name(model_name)
        
        all_model_records = []
        
        elites_path = model_dir / "elites.json"
        if elites_path.exists():
            try:
                with open(elites_path, 'r', encoding='utf-8') as f:
                    all_model_records.extend(json.load(f))
            except Exception as e:
                pass
        
        non_elites_path = model_dir / "non_elites.json"
        if non_elites_path.exists():
            try:
                with open(non_elites_path, 'r', encoding='utf-8') as f:
                    all_model_records.extend(json.load(f))
            except Exception as e:
                pass
        
        under_performing_path = model_dir / "under_performing.json"
        if under_performing_path.exists():
            try:
                with open(under_performing_path, 'r', encoding='utf-8') as f:
                    all_model_records.extend(json.load(f))
            except Exception as e:
                pass
        
        for record in all_model_records:
            prompt = record.get('prompt', '')
            normalized = normalize_prompt(prompt)
            
            if normalized in prompt_to_elite:
                elite = prompt_to_elite[normalized]
                
                response = record.get('generated_output', '')
                toxicity = extract_toxicity(record)
                
                if 'models' not in elite:
                    elite['models'] = {}
                
                if normalized_model_name not in elite['models']:
                    elite['models'][normalized_model_name] = {
                        'response': response,
                        'toxicity': toxicity
                    }
                else:
                    existing_tox = elite['models'][normalized_model_name].get('toxicity', 0) or 0
                    new_tox = toxicity if toxicity is not None else 0
                    if new_tox > existing_tox:
                        elite['models'][normalized_model_name] = {
                            'response': response,
                            'toxicity': toxicity
                        }
        
        if all_model_records:
            model_scores = []
            for record in all_model_records:
                prompt = record.get('prompt', '')
                normalized_prompt = prompt.strip().lower()
                top_25_normalized = {p.strip().lower() for p in top_25_prompts}
                
                if normalized_prompt in top_25_normalized:
                    toxicity = extract_toxicity(record)
                    if toxicity is not None:
                        model_scores.append(toxicity)
            
            if model_scores:
                model_toxicity_scores[model_name] = model_scores
    
    target_model_name = "Meta-Llama-3.1-8B-Instruct.Q3_K_S"
    normalized_target = normalize_model_name(target_model_name)
    
    run_dirs = []
    for item in Path(base_data_dir).iterdir():
        if item.is_dir() and item.name.startswith('run'):
            run_dirs.append(item)
    
    for run_dir in sorted(run_dirs):
        all_run_records = []
        
        for json_file in ['elites.json', 'non_elites.json', 'under_performing.json']:
            json_path = run_dir / json_file
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        run_data = json.load(f)
                        all_run_records.extend(run_data)
                except Exception as e:
                    pass
        
        for record in all_run_records:
            prompt = record.get('prompt', '')
            normalized = normalize_prompt(prompt)
            
            if normalized in prompt_to_elite:
                elite = prompt_to_elite[normalized]
                
                if 'models' not in elite:
                    elite['models'] = {}
                
                if normalized_target not in elite['models']:
                    response = record.get('generated_output', '')
                    toxicity = extract_toxicity(record)
                    
                    if response or toxicity is not None:
                        elite['models'][normalized_target] = {
                            'response': response,
                            'toxicity': toxicity
                        }
    
    all_model_names = set()
    for elite in top_25_elites:
        models = elite.get('models', {})
        all_model_names.update(models.keys())
    
    all_7_models = sorted(all_model_names)
    
    output_path = os.path.join(script_dir, "rq3_top25_elites_with_models.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(top_25_elites, f, indent=2, ensure_ascii=False)
    
    missing_responses_per_model = {}
    missing_scores_per_model = {}
    missing_models_per_model = {}
    refusal_responses_per_model = {}
    
    for model_name in all_7_models:
        missing_responses_per_model[model_name] = 0
        missing_scores_per_model[model_name] = 0
        missing_models_per_model[model_name] = 0
        refusal_responses_per_model[model_name] = 0
    
    for elite in top_25_elites:
        models = elite.get('models', {})
        for model_name in all_7_models:
            if model_name not in models:
                missing_models_per_model[model_name] += 1
            else:
                response = models[model_name].get('response', '')
                toxicity = models[model_name].get('toxicity')
                
                if not response or not response.strip():
                    missing_responses_per_model[model_name] += 1
                else:
                    if is_refusal_response(response):
                        refusal_responses_per_model[model_name] += 1
                
                if toxicity is None:
                    missing_scores_per_model[model_name] += 1
    
    model_names = sorted(all_7_models)
    model_names_short = []
    for name in model_names:
        if 'Llama-3.2-1B' in name:
            model_names_short.append('Llama-3.2-1B')
        elif 'Llama-3.2-3B' in name:
            model_names_short.append('Llama-3.2-3B')
        elif 'Meta-Llama-3.1-8B' in name:
            model_names_short.append('Llama-3.1-8B')
        elif 'Mistral-7B' in name:
            model_names_short.append('Mistral-7B')
        elif 'Phi-3.5' in name:
            model_names_short.append('Phi-3.5-mini')
        elif 'Qwen2.5-7B' in name:
            model_names_short.append('Qwen2.5-7B')
        elif 'gemma-2-9b' in name:
            model_names_short.append('gemma-2-9b')
        else:
            model_names_short.append(name[:20])
    
    total_prompts = len(top_25_elites)
    invalid_counts = []
    for model_name in model_names:
        missing_model = missing_models_per_model[model_name]
        missing_response = missing_responses_per_model[model_name]
        refusals = refusal_responses_per_model[model_name]
        invalid = missing_model + missing_response + refusals
        invalid_counts.append(invalid)
    
    invalid_pct = [100.0 * inv / total_prompts for inv in invalid_counts]
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    x = np.arange(len(model_names))
    bars = ax.bar(x, invalid_pct, width=0.6)
    
    for bar, pct in zip(bars, invalid_pct):
        height = bar.get_height()
        y_pos = bar.get_y() + height / 2.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{pct:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
        )
    
    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_ylabel('Invalid responses (%)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names_short, rotation=45, ha='right', fontsize=9)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    max_pct = max(invalid_pct) if invalid_pct else 0
    ax.set_ylim(0, max_pct * 1.3 if max_pct > 0 else 10.0)
    
    plt.tight_layout()
    
    histogram_path = os.path.join(script_dir, "rq3_invalid_fraction_per_model.pdf")
    plt.savefig(histogram_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    medians = {
        label: (np.median(scores) if len(scores) > 0 else np.nan)
        for label, scores in model_toxicity_scores.items()
    }
    ordered_full_labels = [
        k for k, _ in sorted(
            [(k, v) for k, v in medians.items() if not np.isnan(v)],
            key=lambda kv: kv[1],
            reverse=True
        )
    ]

    box_data = [model_toxicity_scores[k] for k in ordered_full_labels]
    box_labels = [shorten_model_name(k) for k in ordered_full_labels]

    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12
    })
    
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=300)
    
    positions = np.arange(1, len(box_data) + 1, dtype=float)

    vp = ax.violinplot(
        box_data,
        positions=positions,
        widths=0.8,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    palette = ['#4A90E2', '#7ED321', '#F5A623', '#BD10E0', '#50E3C2',
               '#B8E986', '#9013FE', '#D0021B', '#417505', '#8B572A']

    for i, body in enumerate(vp['bodies']):
        color = palette[i % len(palette)]
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(0.55)
        body.set_linewidth(0.9)

    for i, scores in enumerate(box_data, start=1):
        if len(scores) == 0:
            continue
        q1, med, q3 = np.percentile(scores, [25, 50, 75])
        ax.vlines(i, q1, q3, colors='black', lw=1.3, zorder=3)
        ax.scatter([i], [med], s=18, color='black', zorder=4)

    rng = np.random.default_rng(42)
    for i, scores in enumerate(box_data, start=1):
        if len(scores) == 0:
            continue
        x = rng.normal(loc=i, scale=0.04, size=len(scores))
        ax.scatter(x, scores, s=8, color='black', alpha=0.25, linewidths=0, zorder=2)

    ax.set_xlim(0.5, len(box_data) + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Toxicity Score', fontsize=11, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(box_labels, rotation=25, ha='right', fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=1.2)
    plot_path = Path(script_dir) / "top_25_percent_elites_toxicity_distribution_all_models.pdf"
    fig.savefig(plot_path, bbox_inches='tight', format='pdf')
    plt.close(fig)

    

    stats_rows = []
    for model_name in sorted(model_toxicity_scores.keys()):
        scores = model_toxicity_scores[model_name]
        stats_rows.append({
            'Model': shorten_model_name(model_name),
            'n': len(scores),
            'Mean': np.mean(scores),
            'Median': np.median(scores),
            'Std': np.std(scores),
            'Min': np.min(scores),
            'Max': np.max(scores),
            'Q1': np.percentile(scores, 25),
            'Q3': np.percentile(scores, 75),
            'IQR': np.percentile(scores, 75) - np.percentile(scores, 25)
        })

    stats_df = pd.DataFrame(stats_rows)

    pub_df = stats_df.drop(columns=['n'])

    csv_path = Path(script_dir) / "rq3_statistics_table.csv"
    stats_df.to_csv(csv_path, index=False, float_format='%.4f')

if __name__ == "__main__":
    main()

