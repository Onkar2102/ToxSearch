
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

OPERATOR_MODES = ['ie', 'ops', 'comb']

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
    'ie': '#e41a1c',
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

plt.figure(figsize=(20, 9.6))
ax = plt.gca()

mode_colors = {
    'ie': '#d62728',
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
            ax.fill_between(generations[valid_mask], min_vals[valid_mask], max_vals[valid_mask], 
                           facecolor=color, alpha=0.25, label=f'{mode.upper()} - Range')
        
        ax.plot(generations, avg_fit, lw=3, label=f'{mode.upper()} - Average', color=color, linestyle='solid')
        
        data_plotted = True

ax.set_xlabel('Generation Number', fontsize=18)
ax.set_ylabel('Score', fontsize=18)

ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)

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

y_ticks = np.arange(0.2, 1.01, 0.2)
ax.set_yticks(y_ticks)

ax.set_xlim(left=min_gen, right=max_gen)
ax.set_ylim(0, 1)
ax.set_title('All Operator Modes - Aggregated (Overlapped)', fontsize=18)

if data_plotted:
    ax.legend(loc='upper left', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()

if data_plotted:
    plot_type = "aggregated_gen_fitness_range"
    filename_pdf = os.path.join(output_dir, f"all_modes_{plot_type}_b.pdf")
    if os.path.exists(filename_pdf):
        os.remove(filename_pdf)
    plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
plt.close()

try:
    
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
            
            ax.set_xlim(left=0, right=50)
            ax.set_ylim(bottom=0)
            
            ax.legend(loc='upper left', fontsize=11, framealpha=0.9, ncol=1)
            
            plt.tight_layout()
            
            filename_pdf = os.path.join(output_dir, "unified_efficiency.pdf")
            if os.path.exists(filename_pdf):
                os.remove(filename_pdf)
            plt.savefig(filename_pdf, dpi=300, bbox_inches='tight')
            plt.close()
        else:
    else:
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
    

