#!/usr/bin/env python3
"""
RQ3: Cross-Model Transferability Analysis

Analyzes whether toxic prompts evolved on one model transfer to other models 
with different architectures or alignment tuning.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def get_max_toxicity(elite):
    """Get the maximum toxicity score from all models for this elite."""
    models_dict = elite.get('models', {})
    max_toxicity = 0.0
    for model_name, model_data in models_dict.items():
        toxicity = model_data.get('toxicity')
        if toxicity is not None and toxicity > max_toxicity:
            max_toxicity = toxicity
    return max_toxicity


def extract_toxicity(record):
    """Extract toxicity score from moderation_result."""
    try:
        if 'moderation_result' in record and 'google' in record['moderation_result']:
            if 'scores' in record['moderation_result']['google']:
                return record['moderation_result']['google']['scores'].get('toxicity', None)
    except:
        pass
    return None


def shorten_model_name(name):
    """Shorten model names for better readability."""
    if name == 'Llama-3.1-8B':
        return 'Llama-3.1-8B'
    
    # Remove common suffixes
    name = name.replace('-Q4_K_L', '').replace('-Q4_K_M', '').replace('.Q3_K_S', '')
    name = name.replace('-Q4_K_S', '').replace('-v0.3', '').replace('-Instruct', '')
    
    # Shorten specific model names
    name = name.replace('Meta-Llama-3.1-8B', 'Llama-3.1-8B')
    name = name.replace('Llama-3.2-1B', 'Llama-3.2-1B')
    name = name.replace('Llama-3.2-3B', 'Llama-3.2-3B')
    name = name.replace('Mistral-7B', 'Mistral-7B')
    name = name.replace('Phi-3.5-mini', 'Phi-3.5')
    name = name.replace('Qwen2.5-7B', 'Qwen-7B')
    name = name.replace('gemma-2-9b-it', 'Gemma-9B')
    
    return name


def main():
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_data_dir = os.path.join(project_root, "data", "outputs")
    base_data_dir = os.path.normpath(base_data_dir)
    
    print("="*80)
    print("RQ3: Cross-Model Transferability Analysis")
    print("="*80)
    
    # Step 1: Read all elites.json files
    print("\nStep 1: Reading all elites.json files...")
    pattern = os.path.join(base_data_dir, "**", "elites.json")
    elite_files = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(elite_files)} elites.json files")
    
    # Step 2: Load and combine all elites.json files
    print("\nStep 2: Loading and combining all elites...")
    all_elites = []
    
    for file_path in elite_files:
        try:
            # Extract run directory from path
            rel_path = os.path.relpath(file_path, base_data_dir)
            run_dir = os.path.dirname(rel_path)  # e.g., "run01_comb"
            
            # Load JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
            
            # Process each elite
            for elite in elites:
                # Extract toxicity score
                toxicity_score = None
                if 'moderation_result' in elite and 'google' in elite['moderation_result']:
                    if 'scores' in elite['moderation_result']['google']:
                        toxicity_score = elite['moderation_result']['google']['scores'].get('toxicity', None)
                
                # Extract model filename from model path
                model_name_path = elite.get('model_name', '')
                model_filename = os.path.basename(model_name_path) if model_name_path else ''
                
                # Create models dictionary with source model
                models = {
                    model_filename: {
                        'response': elite.get('generated_output', ''),
                        'toxicity': toxicity_score
                    }
                }
                
                # Source metadata
                source = {
                    'genome_id': elite.get('id', None),
                    'operator': elite.get('operator', ''),
                    'dir_name': run_dir
                }
                
                enriched_elite = {
                    'id': None,  # Will be assigned later
                    'prompt': elite.get('prompt', ''),
                    'source': source,
                    'models': models
                }
                
                all_elites.append(enriched_elite)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print(f"Loaded {len(all_elites)} total elite records")
    
    # Step 3: Deduplicate based on prompt (keep highest toxicity)
    print("\nStep 3: Deduplicating and filtering prompts...")
    
    # Sort by toxicity (descending) so highest toxicity comes first
    all_elites.sort(key=get_max_toxicity, reverse=True)
    
    # Create dictionary to track unique prompts (keep first occurrence = highest toxicity)
    unique_elites_dict = {}
    for elite in all_elites:
        prompt = elite['prompt']
        if prompt not in unique_elites_dict:
            unique_elites_dict[prompt] = elite
    
    # Convert back to list
    combined_elites_list = list(unique_elites_dict.values())
    
    # Filter: Keep only prompts that end with a question mark
    combined_elites_list = [elite for elite in combined_elites_list if elite['prompt'].strip().endswith('?')]
    
    # Re-assign IDs after filtering
    for idx, elite in enumerate(combined_elites_list, start=1):
        elite['id'] = idx
    
    print(f"After deduplication and filtering: {len(combined_elites_list)} unique prompts with question marks")
    
    # Step 4: Save combined and deduplicated data to JSON
    print("\nStep 4: Saving combined data to JSON...")
    json_path = os.path.join(script_dir, "rq3_combined_elites.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_elites_list, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {json_path}")
    print(f"Total records: {len(combined_elites_list)}")
    
    # Step 5: Save top 25% elites to CSV
    print("\nStep 5: Selecting top 25% elites and saving to CSV...")
    
    # Calculate toxicity for each elite and sort by toxicity (descending)
    elites_with_toxicity = []
    for elite in combined_elites_list:
        max_tox = get_max_toxicity(elite)
        elites_with_toxicity.append({
            'elite': elite,
            'toxicity': max_tox
        })
    
    # Sort by toxicity (descending)
    elites_with_toxicity.sort(key=lambda x: x['toxicity'], reverse=True)
    
    # Calculate top 25% (75th percentile and above)
    total_count = len(elites_with_toxicity)
    top_25_percent_count = int(total_count * 0.25)
    print(f"Top 25% count: {top_25_percent_count} out of {total_count}")
    
    # Get top 25% elites (75th percentile and above)
    top_25_percent_elites = elites_with_toxicity[:top_25_percent_count]
    
    # Extract prompts from top 25% elites
    top_prompts = [item['elite']['prompt'] for item in top_25_percent_elites]
    
    # Create DataFrame with just one column "questions"
    df_top = pd.DataFrame({'questions': top_prompts})
    
    print(f"DataFrame created: {len(df_top)} rows, {len(df_top.columns)} columns")
    print(f"Toxicity range: {min(item['toxicity'] for item in top_25_percent_elites):.4f} - {max(item['toxicity'] for item in top_25_percent_elites):.4f}")
    
    # Save DataFrame to CSV in data/ directory
    csv_path = os.path.join(project_root, "data", "combined_elites.csv")
    df_top.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved {len(df_top)} prompts (top 25% by toxicity, 75th percentile and above)")
    
    # Step 6: Create box plot for top 25% elites toxicity distribution across all models
    print("\nStep 6: Creating box plot for toxicity distribution across all models...")
    
    # Get top 25% prompts
    top_25_prompts = {item['elite']['prompt'] for item in top_25_percent_elites}
    
    # Extract original toxicity scores
    original_toxicity_scores = [item['toxicity'] for item in top_25_percent_elites]
    
    # Find all model directories in data/outputs
    model_dirs = []
    for item in Path(base_data_dir).iterdir():
        if item.is_dir():
            # Check if it's a model directory (not a run directory like run01_comb)
            if not item.name.startswith('run') and not item.name.startswith('2025'):
                model_dirs.append(item)
    
    print(f"Found {len(model_dirs)} model directories")
    
    # Collect toxicity scores per model
    model_toxicity_scores = {}
    model_toxicity_scores['Llama-3.1-8B'] = original_toxicity_scores
    
    # Read elites.json, non_elites.json, and under_performing.json from each model directory
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        
        # Collect all records from all three files
        all_model_records = []
        
        # Read elites.json
        elites_path = model_dir / "elites.json"
        if elites_path.exists():
            try:
                with open(elites_path, 'r', encoding='utf-8') as f:
                    all_model_records.extend(json.load(f))
            except Exception as e:
                print(f"  Error reading elites.json for {model_name}: {e}")
        
        # Read non_elites.json
        non_elites_path = model_dir / "non_elites.json"
        if non_elites_path.exists():
            try:
                with open(non_elites_path, 'r', encoding='utf-8') as f:
                    all_model_records.extend(json.load(f))
            except Exception as e:
                print(f"  Error reading non_elites.json for {model_name}: {e}")
        
        # Read under_performing.json
        under_performing_path = model_dir / "under_performing.json"
        if under_performing_path.exists():
            try:
                with open(under_performing_path, 'r', encoding='utf-8') as f:
                    all_model_records.extend(json.load(f))
            except Exception as e:
                print(f"  Error reading under_performing.json for {model_name}: {e}")
        
        if all_model_records:
            # Match prompts with top 25% prompts
            model_scores = []
            for record in all_model_records:
                prompt = record.get('prompt', '')
                # Normalize prompt for matching (strip and lowercase)
                normalized_prompt = prompt.strip().lower()
                top_25_normalized = {p.strip().lower() for p in top_25_prompts}
                
                if normalized_prompt in top_25_normalized:
                    toxicity = extract_toxicity(record)
                    if toxicity is not None:
                        model_scores.append(toxicity)
            
            if model_scores:
                model_toxicity_scores[model_name] = model_scores
                print(f"  {model_name}: {len(model_scores)} matching prompts (from {len(all_model_records)} total records)")
            else:
                print(f"  {model_name}: 0 matching prompts (from {len(all_model_records)} total records)")
    
    # Prepare data for improved distribution plot (ordered by median)
    # Compute per-model medians and order descending; skip empty groups
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

    # Map ordered labels to data and shortened tick labels
    box_data = [model_toxicity_scores[k] for k in ordered_full_labels]
    box_labels = [shorten_model_name(k) for k in ordered_full_labels]


    # Create publication-ready violin + jitter plot (raincloud-style without seaborn)
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

    # Violin plot (distribution shape) with clean styling
    vp = ax.violinplot(
        box_data,
        positions=positions,
        widths=0.8,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    # Palette (color-blind friendly tones)
    palette = ['#4A90E2', '#7ED321', '#F5A623', '#BD10E0', '#50E3C2',
               '#B8E986', '#9013FE', '#D0021B', '#417505', '#8B572A']

    # Style violins
    for i, body in enumerate(vp['bodies']):
        color = palette[i % len(palette)]
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(0.55)
        body.set_linewidth(0.9)

    # Overlay IQR bars and median dots
    for i, scores in enumerate(box_data, start=1):
        if len(scores) == 0:
            continue
        q1, med, q3 = np.percentile(scores, [25, 50, 75])
        ax.vlines(i, q1, q3, colors='black', lw=1.3, zorder=3)   # IQR bar
        ax.scatter([i], [med], s=18, color='black', zorder=4)     # median dot

    # Add jittered points to show sample distribution
    rng = np.random.default_rng(42)  # reproducible jitter
    for i, scores in enumerate(box_data, start=1):
        if len(scores) == 0:
            continue
        x = rng.normal(loc=i, scale=0.04, size=len(scores))
        ax.scatter(x, scores, s=8, color='black', alpha=0.25, linewidths=0, zorder=2)

    # Axes and style
    ax.set_xlim(0.5, len(box_data) + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Toxicity Score', fontsize=11, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(box_labels, rotation=25, ha='right', fontsize=9)

    # Minimalist spines and subtle grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # Tight layout and save (same filename used in LaTeX)
    fig.tight_layout(pad=1.2)
    plot_path = Path(script_dir) / "top_25_percent_elites_toxicity_distribution_all_models.pdf"
    fig.savefig(plot_path, bbox_inches='tight', format='pdf')
    plt.close(fig)

    print(f"\nDistribution plot saved to: {plot_path}")
    
    # Step 7: Create statistical analysis table
    print("\nStep 7: Creating statistical analysis table...")

    # Calculate statistics for each model
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

    # Create DataFrame
    stats_df = pd.DataFrame(stats_rows)

    # Public-facing table excludes sample counts for publication
    pub_df = stats_df.drop(columns=['n'])

    # Format numbers for display
    display_df = pub_df.copy()
    for col in ['Mean', 'Median', 'Std', 'Min', 'Max', 'Q1', 'Q3', 'IQR']:
        display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')

    print("Statistical Analysis Table for RQ3:")
    print("="*100)
    print(display_df.to_string(index=False))

    # Create PDF table using matplotlib with publication styling
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif'
    })
    
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data with formatted numbers (3 decimal places for readability)
    table_data = []
    for _, row in pub_df.iterrows():
        table_data.append([
            row['Model'],
            f"{row['Mean']:.3f}",
            f"{row['Median']:.3f}",
            f"{row['Std']:.3f}",
            f"{row['Min']:.3f}",
            f"{row['Max']:.3f}",
            f"{row['Q1']:.3f}",
            f"{row['Q3']:.3f}",
            f"{row['IQR']:.3f}"
        ])

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q1', 'Q3', 'IQR'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Publication-friendly styling
    n_cols = len(['Model', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q1', 'Q3', 'IQR'])
    
    # Header row styling
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor('#E8E8E8')
        cell.set_edgecolor('black')
        cell.set_linewidth(1.2)
        cell.set_text_props(weight='bold', color='black', fontsize=10)
        cell.set_height(0.08)
    
    # Body cells styling with alternating row colors
    for i in range(1, len(table_data) + 1):
        for j in range(n_cols):
            cell = table[(i, j)]
            # Alternate row colors for better readability
            if i % 2 == 0:
                cell.set_facecolor('#F5F5F5')
            else:
                cell.set_facecolor('white')
            cell.set_edgecolor('#CCCCCC')
            cell.set_linewidth(0.8)
            cell.set_text_props(color='black', fontsize=9)
            cell.set_height(0.06)
    
    # No title here; captions are added in LaTeX

    # Save as PDF
    pdf_path = Path(script_dir) / "rq3_statistics_table.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()

    print(f"\nStatistics table PDF saved to: {pdf_path}")

    # Also save as CSV for reference
    csv_path = Path(script_dir) / "rq3_statistics_table.csv"
    stats_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Statistics table CSV saved to: {csv_path}")

    print("\n" + "="*80)
    print("RQ3 Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

