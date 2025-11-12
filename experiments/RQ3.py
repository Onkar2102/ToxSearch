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
    if name == 'original':
        return 'Original'
    
    # Remove common suffixes
    name = name.replace('-Q4_K_L', '').replace('-Q4_K_M', '').replace('.Q3_K_S', '')
    name = name.replace('-Q4_K_S', '').replace('-v0.3', '').replace('-Instruct', '')
    
    # Shorten specific model names
    name = name.replace('Meta-Llama-3.1-8B', 'Llama-3.1-8B')
    name = name.replace('Llama-3.2-1B', 'Llama-1B')
    name = name.replace('Llama-3.2-3B', 'Llama-3B')
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
    model_toxicity_scores['original'] = original_toxicity_scores
    
    # Read elites.json from each model directory
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        elites_path = model_dir / "elites.json"
        
        if elites_path.exists():
            try:
                with open(elites_path, 'r', encoding='utf-8') as f:
                    elites = json.load(f)
                
                # Match prompts with top 25% prompts
                model_scores = []
                for elite in elites:
                    prompt = elite.get('prompt', '')
                    # Normalize prompt for matching (strip and lowercase)
                    normalized_prompt = prompt.strip().lower()
                    top_25_normalized = {p.strip().lower() for p in top_25_prompts}
                    
                    if normalized_prompt in top_25_normalized:
                        toxicity = extract_toxicity(elite)
                        if toxicity is not None:
                            model_scores.append(toxicity)
                
                if model_scores:
                    model_toxicity_scores[model_name] = model_scores
                    print(f"  {model_name}: {len(model_scores)} matching prompts")
            except Exception as e:
                print(f"  Error reading {model_name}: {e}")
    
    # Prepare data for box plot with shortened names
    box_data = []
    box_labels = []
    short_name_map = {}  # Map short names to full names for statistics
    
    for label in sorted(model_toxicity_scores.keys()):
        box_data.append(model_toxicity_scores[label])
        short_name = shorten_model_name(label)
        box_labels.append(short_name)
        short_name_map[short_name] = label
    
    # Create box plot with better sizing
    fig, ax = plt.subplots(figsize=(16, 8))
    
    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, 
                    showmeans=True, meanline=True, widths=0.7)
    
    # Color boxes with better colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
    for i, patch in enumerate(bp['boxes']):
        color = colors[i % len(colors)]
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    
    # Customize plot
    ax.set_ylabel('Toxicity Score', fontsize=18, fontweight='bold')
    ax.set_xlabel('Model', fontsize=18, fontweight='bold')
    ax.set_title('Toxicity Distribution', 
                 fontsize=18, fontweight='bold', pad=25)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # Set y-axis limits for better visibility
    ax.set_ylim([0.3, 0.9])
    
    # Explicitly remove any legend if it exists
    if hasattr(ax, 'legend_') and ax.legend_ is not None:
        ax.legend_.remove()
    
    plt.tight_layout()
    
    # Save plot as PDF
    plot_path = Path(script_dir) / "top_25_percent_elites_toxicity_distribution_all_models.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"\nBox plot saved to: {plot_path}")
    print(f"\nToxicity Statistics:")
    for label in sorted(model_toxicity_scores.keys()):
        scores = model_toxicity_scores[label]
        print(f"\n{label}:")
        print(f"  Count: {len(scores)}")
        print(f"  Mean: {np.mean(scores):.4f}")
        print(f"  Median: {np.median(scores):.4f}")
        print(f"  Std: {np.std(scores):.4f}")
        print(f"  Min: {np.min(scores):.4f}")
        print(f"  Max: {np.max(scores):.4f}")
    
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
    
    # Format numbers for display
    display_df = stats_df.copy()
    for col in ['Mean', 'Median', 'Std', 'Min', 'Max', 'Q1', 'Q3', 'IQR']:
        display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')
    
    print("Statistical Analysis Table for RQ3:")
    print("="*100)
    print(display_df.to_string(index=False))
    
    # Create PDF table using matplotlib
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data with formatted numbers
    table_data = []
    for _, row in stats_df.iterrows():
        table_data.append([
            row['Model'],
            f"{int(row['n'])}",
            f"{row['Mean']:.4f}",
            f"{row['Median']:.4f}",
            f"{row['Std']:.4f}",
            f"{row['Min']:.4f}",
            f"{row['Max']:.4f}",
            f"{row['Q1']:.4f}",
            f"{row['Q3']:.4f}",
            f"{row['IQR']:.4f}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=['Model', 'n', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q1', 'Q3', 'IQR'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(10):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells
    for i in range(1, len(table_data) + 1):
        for j in range(10):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Statistical Analysis of Toxicity Scores Across Models for Top 25% Elites',
              fontsize=14, fontweight='bold', pad=20)
    
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

