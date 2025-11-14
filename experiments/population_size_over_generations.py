import os
import json
import glob
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

OPERATOR_MODES = ['ie', 'ops', 'comb']

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "..", "data", "outputs")
base_dir = os.path.normpath(base_dir)

output_dir = os.path.join(script_dir, "population_trend")
os.makedirs(output_dir, exist_ok=True)

all_matching_dirs = []
for mode in OPERATOR_MODES:
    pattern = os.path.join(base_dir, f"run*_{mode}")
    matching_dirs = sorted(glob.glob(pattern))
    all_matching_dirs.extend(matching_dirs)

if not all_matching_dirs:
    raise ValueError(f"No directories found for any operator mode")

def count_population_per_generation(run_dir):
    elites_path = os.path.join(run_dir, "elites.json")
    non_elites_path = os.path.join(run_dir, "non_elites.json")
    under_performing_path = os.path.join(run_dir, "under_performing.json")
    
    elites_genomes = []
    non_elites_genomes = []
    under_performing_genomes = []
    
    if os.path.exists(elites_path):
        try:
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_genomes = json.load(f)
        except Exception as e:
            pass
    
    if os.path.exists(non_elites_path):
        try:
            with open(non_elites_path, 'r', encoding='utf-8') as f:
                non_elites_genomes = json.load(f)
        except Exception as e:
            pass
    
    if os.path.exists(under_performing_path):
        try:
            with open(under_performing_path, 'r', encoding='utf-8') as f:
                under_performing_genomes = json.load(f)
        except Exception as e:
            pass
    
    elites_counts = defaultdict(int)
    non_elites_counts = defaultdict(int)
    under_performing_counts = defaultdict(int)
    
    for genome in elites_genomes:
        if genome and "generation" in genome:
            gen_num = genome.get("generation")
            if gen_num is not None:
                elites_counts[gen_num] += 1
    
    for genome in non_elites_genomes:
        if genome and "generation" in genome:
            gen_num = genome.get("generation")
            if gen_num is not None:
                non_elites_counts[gen_num] += 1
    
    for genome in under_performing_genomes:
        if genome and "generation" in genome:
            gen_num = genome.get("generation")
            if gen_num is not None:
                under_performing_counts[gen_num] += 1
    
    all_gens = set(list(elites_counts.keys()) + list(non_elites_counts.keys()) + list(under_performing_counts.keys()))
    
    if not all_gens:
        return {}, {}, {}, {}, {}, {}
    
    max_gen = max(all_gens)
    
    elites_cumulative = {}
    non_elites_cumulative = {}
    under_performing_cumulative = {}
    total_cumulative = {}
    
    elites_total = 0
    non_elites_total = 0
    under_performing_total = 0
    
    prev_elites = 0
    prev_non_elites = 0
    prev_under_performing = 0
    prev_total = 0
    
    for gen_num in range(max_gen + 1):
        if gen_num in elites_counts:
            elites_total += elites_counts[gen_num]
        if gen_num in non_elites_counts:
            non_elites_total += non_elites_counts[gen_num]
        if gen_num in under_performing_counts:
            under_performing_total += under_performing_counts[gen_num]
        
        current_total = elites_total + non_elites_total + under_performing_total
        
        if current_total < prev_total:
            elites_total = prev_elites
            non_elites_total = prev_non_elites
            under_performing_total = prev_under_performing
            current_total = prev_total
        
        elites_cumulative[gen_num] = elites_total
        non_elites_cumulative[gen_num] = non_elites_total
        under_performing_cumulative[gen_num] = under_performing_total
        total_cumulative[gen_num] = current_total
        
        prev_elites = elites_total
        prev_non_elites = non_elites_total
        prev_under_performing = under_performing_total
        prev_total = current_total
    
    total_per_gen = {}
    elites_non_elites_per_gen = {}
    for gen_num in range(max_gen + 1):
        total_per_gen[gen_num] = (elites_counts.get(gen_num, 0) + 
                                  non_elites_counts.get(gen_num, 0) + 
                                  under_performing_counts.get(gen_num, 0))
        elites_non_elites_per_gen[gen_num] = (elites_counts.get(gen_num, 0) + 
                                             non_elites_counts.get(gen_num, 0))
    
    return (dict(sorted(elites_cumulative.items())), 
            dict(sorted(non_elites_cumulative.items())), 
            dict(sorted(under_performing_cumulative.items())),
            dict(sorted(total_cumulative.items())),
            dict(sorted(total_per_gen.items())),
            dict(sorted(elites_non_elites_per_gen.items())))

all_data = []

for run_dir in all_matching_dirs:
    run_name = os.path.basename(run_dir)
    
    operator_mode = None
    for mode in OPERATOR_MODES:
        if run_name.endswith(f'_{mode}'):
            operator_mode = mode
            break
    
    if operator_mode is None:
        continue
    
    elites_cumulative, non_elites_cumulative, under_performing_cumulative, total_cumulative, total_per_gen, elites_non_elites_per_gen = count_population_per_generation(run_dir)
    
    for gen_num in total_cumulative.keys():
        all_data.append({
            'run': run_name,
            'mode': operator_mode,
            'generation': gen_num,
            'elites': elites_cumulative.get(gen_num, 0),
            'non_elites': non_elites_cumulative.get(gen_num, 0),
            'under_performing': under_performing_cumulative.get(gen_num, 0),
            'total': total_cumulative.get(gen_num, 0),
            'genomes_generated': total_per_gen.get(gen_num, 0),
            'elites_non_elites': elites_non_elites_per_gen.get(gen_num, 0)
        })

if not all_data:
    raise ValueError("No population data found")

mode_data_dict = defaultdict(lambda: defaultdict(list))

for item in all_data:
    mode_data_dict[item['mode']][item['run']].append({
        'generation': item['generation'],
        'elites': item['elites'],
        'non_elites': item['non_elites'],
        'under_performing': item['under_performing'],
        'total': item['total']
    })

if HAS_MATPLOTLIB:
    mode_color_map = {
        'ie': '#d62728',
        'ops': '#1f77b4',
        'comb': '#ff7f0e'
    }
    
    for mode in OPERATOR_MODES:
        if mode not in mode_data_dict:
            continue
        
        run_data_dict = mode_data_dict[mode]
        run_names = sorted(run_data_dict.keys())
        
        all_runs_data = {}
        all_generations_set = set()
        for run_items in run_data_dict.values():
            for item in run_items:
                all_generations_set.add(item['generation'])
        all_generations = sorted(all_generations_set)
        max_gen = int(all_generations[-1]) if len(all_generations) > 0 else 0
        min_gen = int(all_generations[0]) if len(all_generations) > 0 else 0
        
        mode_color = mode_color_map.get(mode, '#000000')
        
        for run_name in run_names:
            run_items = run_data_dict[run_name]
            if not run_items:
                continue
            
            run_items_sorted = sorted(run_items, key=lambda x: x['generation'])
            
            generations = [item['generation'] for item in run_items_sorted]
            elites_sizes = [item['elites'] for item in run_items_sorted]
            non_elites_sizes = [item['non_elites'] for item in run_items_sorted]
            under_performing_sizes = [item['under_performing'] for item in run_items_sorted]
            total_sizes = [item['total'] for item in run_items_sorted]
            
            all_runs_data[run_name] = {
                'generations': generations,
                'elites': elites_sizes,
                'non_elites': non_elites_sizes,
                'under_performing': under_performing_sizes,
                'total': total_sizes,
                'run_num': run_name.replace(f'_{mode}', '').replace('run', '')
            }
            
            plt.figure(figsize=(20, 9.6))
            ax = plt.gca()
            
            ax.fill_between(generations, 0, under_performing_sizes, 
                           alpha=0.6, color='#d62728', label='Under-Performing')
            ax.fill_between(generations, under_performing_sizes, 
                           [u + n for u, n in zip(under_performing_sizes, non_elites_sizes)],
                           alpha=0.6, color='#1f77b4', label='Non-Elites')
            ax.fill_between(generations, 
                           [u + n for u, n in zip(under_performing_sizes, non_elites_sizes)],
                           total_sizes,
                           alpha=0.6, color='#2ca02c', label='Elites')
            
            ax.plot(generations, total_sizes, lw=3, 
                   color='#000000', linestyle='solid', marker='o', markersize=5,
                   label='Total')
            
            ax.set_xlabel('Generation Number', fontsize=18, fontweight='bold')
            ax.set_ylabel('Cumulative Population Size', fontsize=18, fontweight='bold')
            ax.tick_params(axis='x', labelsize=13)
            ax.tick_params(axis='y', labelsize=13)
            
            if max_gen - min_gen >= 5:
                xticks = list(range(min_gen, max_gen + 1, 5))
                if xticks[-1] != max_gen:
                    xticks.append(max_gen)
                ax.set_xticks(xticks)
            else:
                ax.set_xticks(all_generations)
            
            ax.set_xlim(left=min_gen, right=max_gen)
            ax.set_ylim(bottom=0)
            ax.set_title(f'{run_name} - Population Size Over Generations', fontsize=18)
            ax.legend(loc='upper left', fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename_pdf = os.path.join(output_dir, f"{run_name}.pdf")
            if os.path.exists(filename_pdf):
                os.remove(filename_pdf)
            plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
            plt.close()
        
        if all_runs_data:
            all_gens_in_mode = set()
            for data in all_runs_data.values():
                all_gens_in_mode.update(data['generations'])
            agg_generations = sorted(all_gens_in_mode)
            
            agg_sum_elites = []
            agg_sum_non_elites = []
            agg_sum_under_performing = []
            agg_sum_total = []
            
            for gen in agg_generations:
                sum_elites = 0
                sum_non_elites = 0
                sum_under_performing = 0
                sum_total = 0
                
                for run_name, data in all_runs_data.items():
                    if gen in data['generations']:
                        idx = data['generations'].index(gen)
                        sum_elites += data['elites'][idx]
                        sum_non_elites += data['non_elites'][idx]
                        sum_under_performing += data['under_performing'][idx]
                        sum_total += data['total'][idx]
                    else:
                        if data['generations']:
                            last_gen = max([g for g in data['generations'] if g < gen], default=None)
                            if last_gen is not None:
                                last_idx = data['generations'].index(last_gen)
                                sum_elites += data['elites'][last_idx]
                                sum_non_elites += data['non_elites'][last_idx]
                                sum_under_performing += data['under_performing'][last_idx]
                                sum_total += data['total'][last_idx]
                
                agg_sum_elites.append(sum_elites)
                agg_sum_non_elites.append(sum_non_elites)
                agg_sum_under_performing.append(sum_under_performing)
                agg_sum_total.append(sum_total)
            
            plt.figure(figsize=(20, 9.6))
            ax = plt.gca()
            
            if agg_generations:
                bottom_under_performing = [0] * len(agg_generations)
                top_under_performing = agg_sum_under_performing
                bottom_non_elites = agg_sum_under_performing
                top_non_elites = [u + n for u, n in zip(agg_sum_under_performing, agg_sum_non_elites)]
                bottom_elites = top_non_elites
                top_elites = agg_sum_total
                
                ax.fill_between(agg_generations, bottom_under_performing, top_under_performing,
                               alpha=0.6, color='#d62728', label='Under-Performing')
                ax.fill_between(agg_generations, bottom_non_elites, top_non_elites,
                               alpha=0.6, color='#1f77b4', label='Non-Elites')
                ax.fill_between(agg_generations, bottom_elites, top_elites,
                               alpha=0.6, color='#2ca02c', label='Elites')
                
                ax.plot(agg_generations, agg_sum_total, lw=3, 
                       color='#000000', linestyle='solid', marker='o', markersize=5,
                       label='Total')
            
            ax.set_xlabel('Generation Number', fontsize=18, fontweight='bold')
            ax.set_ylabel('Cumulative Population Size', fontsize=18, fontweight='bold')
            ax.tick_params(axis='x', labelsize=13)
            ax.tick_params(axis='y', labelsize=13)
            
            if max_gen - min_gen >= 5:
                xticks = list(range(min_gen, max_gen + 1, 5))
                if xticks[-1] != max_gen:
                    xticks.append(max_gen)
                ax.set_xticks(xticks)
            else:
                ax.set_xticks(all_generations)
            
            ax.set_xlim(left=min_gen, right=max_gen)
            ax.set_ylim(bottom=0)
            ax.set_title(f'{mode.upper()} Mode - Aggregated Population Size Over Generations', fontsize=18)
            ax.legend(loc='upper left', fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename_pdf = os.path.join(output_dir, f"{mode.lower()}_aggregated.pdf")
            if os.path.exists(filename_pdf):
                os.remove(filename_pdf)
            plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
            plt.close()

all_generations_set = set()
for item in all_data:
    all_generations_set.add(item['generation'])

aggregated_data_elites = defaultdict(lambda: defaultdict(list))
aggregated_data_non_elites = defaultdict(lambda: defaultdict(list))
aggregated_data_under_performing = defaultdict(lambda: defaultdict(list))
aggregated_data_total = defaultdict(lambda: defaultdict(list))
aggregated_data_genomes_generated = defaultdict(lambda: defaultdict(list))
aggregated_data_elites_non_elites = defaultdict(lambda: defaultdict(list))

for item in all_data:
    aggregated_data_elites[item['mode']][item['generation']].append(item['elites'])
    aggregated_data_non_elites[item['mode']][item['generation']].append(item['non_elites'])
    aggregated_data_under_performing[item['mode']][item['generation']].append(item['under_performing'])
    aggregated_data_total[item['mode']][item['generation']].append(item['total'])
    aggregated_data_genomes_generated[item['mode']][item['generation']].append(item['genomes_generated'])
    aggregated_data_elites_non_elites[item['mode']][item['generation']].append(item['elites_non_elites'])

mode_colors = {
    'ie': '#d62728',
    'ops': '#1f77b4',
    'comb': '#ff7f0e'
}

if HAS_MATPLOTLIB and aggregated_data_total:
    plt.figure(figsize=(8, 4.5))
    ax = plt.gca()
    
    for mode in OPERATOR_MODES:
        if mode not in aggregated_data_total:
            continue
        
        mode_gen_total = aggregated_data_total[mode]
        mode_gen_elites = aggregated_data_elites[mode]
        mode_gen_non_elites = aggregated_data_non_elites[mode]
        all_gens_for_mode = sorted(mode_gen_total.keys())
        
        if not all_gens_for_mode:
            continue
        
        mode_runs_data = {}
        for run_name in mode_data_dict[mode].keys():
            run_items = mode_data_dict[mode][run_name]
            if run_items:
                run_items_sorted = sorted(run_items, key=lambda x: x['generation'])
                mode_runs_data[run_name] = {
                    'generations': [item['generation'] for item in run_items_sorted],
                    'totals': [item['total'] for item in run_items_sorted],
                    'elites': [item['elites'] for item in run_items_sorted],
                    'non_elites': [item['non_elites'] for item in run_items_sorted]
                }
        
        sum_cumulative_total = []
        sum_cumulative_elites_non_elites = []
        prev_total = 0
        prev_elites_non_elites = 0
        
        for gen in all_gens_for_mode:
            current_total = 0
            current_elites_non_elites = 0
            
            for run_name, run_data in mode_runs_data.items():
                if gen in run_data['generations']:
                    idx = run_data['generations'].index(gen)
                    current_total += run_data['totals'][idx]
                    current_elites_non_elites += run_data['elites'][idx] + run_data['non_elites'][idx]
                else:
                    if run_data['generations']:
                        last_gen = max([g for g in run_data['generations'] if g < gen], default=None)
                        if last_gen is not None:
                            last_idx = run_data['generations'].index(last_gen)
                            current_total += run_data['totals'][last_idx]
                            current_elites_non_elites += run_data['elites'][last_idx] + run_data['non_elites'][last_idx]
            
            if current_total < prev_total:
                current_total = prev_total
            if current_elites_non_elites < prev_elites_non_elites:
                current_elites_non_elites = prev_elites_non_elites
            
            sum_cumulative_total.append(current_total)
            sum_cumulative_elites_non_elites.append(current_elites_non_elites)
            prev_total = current_total
            prev_elites_non_elites = current_elites_non_elites
        
        color = mode_colors[mode]
        
        ax.plot(
            sum_cumulative_total, sum_cumulative_elites_non_elites, linewidth=2.2,
            color=color, linestyle='solid',
            label=f'{mode.upper()}'
        )
    
    all_cumulative_total = []
    all_cumulative_elites_non_elites = []
    for mode in OPERATOR_MODES:
        if mode not in aggregated_data_total:
            continue
        mode_gen_total = aggregated_data_total[mode]
        mode_gen_elites = aggregated_data_elites[mode]
        mode_gen_non_elites = aggregated_data_non_elites[mode]
        all_gens_for_mode = sorted(mode_gen_total.keys())
        
        for gen in all_gens_for_mode:
            total_sum = sum(mode_gen_total[gen])
            elites_sum = sum(mode_gen_elites[gen])
            non_elites_sum = sum(mode_gen_non_elites[gen])
            all_cumulative_total.append(total_sum)
            all_cumulative_elites_non_elites.append(elites_sum + non_elites_sum)
    
    if all_cumulative_total:
        max_x = max(all_cumulative_total)
        max_y = max(all_cumulative_elites_non_elites) if all_cumulative_elites_non_elites else 0
    else:
        max_x = max_y = 0
    
    ax.set_xlim(left=0, right=max_x * 1.1 if max_x > 0 else 1000)
    ax.set_ylim(bottom=0, top=max_y * 1.1 if max_y > 0 else 1000)
    ax.set_xlabel('Count of Generated Prompts', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count of Elites and Non-Elites', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.legend(loc='upper left', fontsize=9, frameon=True)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
    
    plt.tight_layout()
    
    filename_pdf = os.path.join(output_dir, "all_modes_aggregated.pdf")
    if os.path.exists(filename_pdf):
        os.remove(filename_pdf)
    plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
    plt.close()

print("\n" + "="*60)
print("FINAL COUNTS (Cumulative across all executions)")
print("="*60)

for mode in OPERATOR_MODES:
    if mode not in aggregated_data_total:
        continue
    
    mode_gen_total = aggregated_data_total[mode]
    mode_gen_elites = aggregated_data_elites[mode]
    mode_gen_non_elites = aggregated_data_non_elites[mode]
    mode_gen_under_performing = aggregated_data_under_performing[mode]
    
    if not mode_gen_total:
        continue
    
    max_gen = max(mode_gen_total.keys())
    
    final_total = sum(mode_gen_total[max_gen])
    final_elites = sum(mode_gen_elites[max_gen])
    final_non_elites = sum(mode_gen_non_elites[max_gen])
    final_under_performing = sum(mode_gen_under_performing[max_gen])
    
    print(f"\n{mode.upper()} Mode (Generation {max_gen}):")
    print(f"  Total:           {final_total:,}")
    print(f"  Elites:          {final_elites:,}")
    print(f"  Non-Elites:      {final_non_elites:,}")
    print(f"  Under-Performing: {final_under_performing:,}")
    print(f"  Active (E+N):    {final_elites + final_non_elites:,}")

print("\n" + "="*60)

if HAS_MATPLOTLIB:
    print("Population size over generations plots generated successfully!")
else:
    print("Matplotlib not available. Population data collected but plots not generated.")

