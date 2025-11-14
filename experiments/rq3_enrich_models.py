#!/usr/bin/env python3
"""
RQ3: Enrich Combined Elites with All Model Responses

Performs a left join: for each prompt in rq3_combined_elites.json,
finds matching prompts in all model directories and adds their responses
and toxicity scores to the 'models' dictionary.
"""

import os
import json
import glob
from pathlib import Path
from collections import defaultdict


def extract_toxicity(record):
    """Extract toxicity score from moderation_result."""
    try:
        if 'moderation_result' in record and 'google' in record['moderation_result']:
            if 'scores' in record['moderation_result']['google']:
                return record['moderation_result']['google']['scores'].get('toxicity', None)
    except:
        pass
    return None


def normalize_prompt(prompt):
    """Normalize prompt for matching (strip whitespace and lowercase)."""
    return prompt.strip().lower()


def main():
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_data_dir = os.path.join(project_root, "data", "outputs")
    base_data_dir = os.path.normpath(base_data_dir)
    
    combined_elites_path = os.path.join(script_dir, "rq3_combined_elites.json")
    output_path = os.path.join(script_dir, "rq3_combined_elites_enriched.json")
    
    print("="*80)
    print("RQ3: Enriching Combined Elites with All Model Responses")
    print("="*80)
    
    # Step 1: Load combined elites
    print("\nStep 1: Loading combined elites...")
    with open(combined_elites_path, 'r', encoding='utf-8') as f:
        combined_elites = json.load(f)
    
    print(f"Loaded {len(combined_elites)} elite records")
    
    # Step 2: Create a normalized prompt lookup for combined elites
    print("\nStep 2: Creating prompt lookup...")
    prompt_to_elite = {}
    for elite in combined_elites:
        prompt = elite.get('prompt', '')
        normalized = normalize_prompt(prompt)
        prompt_to_elite[normalized] = elite
    
    print(f"Created lookup for {len(prompt_to_elite)} unique prompts")
    
    # Step 3: Find all model directories
    print("\nStep 3: Finding model directories...")
    model_dirs = []
    for item in Path(base_data_dir).iterdir():
        if item.is_dir():
            # Check if it's a model directory (not a run directory like run01_comb)
            if not item.name.startswith('run') and not item.name.startswith('2025'):
                model_dirs.append(item)
    
    print(f"Found {len(model_dirs)} model directories")
    
    # Step 4: For each model directory, read all records and match prompts
    print("\nStep 4: Reading model data and matching prompts...")
    
    model_stats = defaultdict(int)
    
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        print(f"\n  Processing model: {model_name}")
        
        # Collect all records from all three files
        all_model_records = []
        
        # Read elites.json
        elites_path = model_dir / "elites.json"
        if elites_path.exists():
            try:
                with open(elites_path, 'r', encoding='utf-8') as f:
                    elites_data = json.load(f)
                    all_model_records.extend(elites_data)
                    print(f"    Loaded {len(elites_data)} records from elites.json")
            except Exception as e:
                print(f"    Error reading elites.json: {e}")
        
        # Read non_elites.json
        non_elites_path = model_dir / "non_elites.json"
        if non_elites_path.exists():
            try:
                with open(non_elites_path, 'r', encoding='utf-8') as f:
                    non_elites_data = json.load(f)
                    all_model_records.extend(non_elites_data)
                    print(f"    Loaded {len(non_elites_data)} records from non_elites.json")
            except Exception as e:
                print(f"    Error reading non_elites.json: {e}")
        
        # Read under_performing.json
        under_performing_path = model_dir / "under_performing.json"
        if under_performing_path.exists():
            try:
                with open(under_performing_path, 'r', encoding='utf-8') as f:
                    under_performing_data = json.load(f)
                    all_model_records.extend(under_performing_data)
                    print(f"    Loaded {len(under_performing_data)} records from under_performing.json")
            except Exception as e:
                print(f"    Error reading under_performing.json: {e}")
        
        print(f"    Total records for {model_name}: {len(all_model_records)}")
        
        # Match prompts and add to combined elites
        matches_found = 0
        for record in all_model_records:
            prompt = record.get('prompt', '')
            normalized = normalize_prompt(prompt)
            
            if normalized in prompt_to_elite:
                elite = prompt_to_elite[normalized]
                
                # Extract response and toxicity
                response = record.get('generated_output', '')
                toxicity = extract_toxicity(record)
                
                # Add to models dictionary (use model directory name as key)
                if 'models' not in elite:
                    elite['models'] = {}
                
                # Add model data
                elite['models'][model_name] = {
                    'response': response,
                    'toxicity': toxicity
                }
                
                matches_found += 1
                model_stats[model_name] += 1
        
        print(f"    Matched {matches_found} prompts for {model_name}")
    
    # Step 5: Print statistics
    print("\n" + "="*80)
    print("Enrichment Statistics:")
    print("="*80)
    print(f"Total elites: {len(combined_elites)}")
    print(f"\nMatches per model:")
    for model_name in sorted(model_stats.keys()):
        print(f"  {model_name}: {model_stats[model_name]} matches")
    
    # Count how many models each elite has
    model_counts = defaultdict(int)
    for elite in combined_elites:
        num_models = len(elite.get('models', {}))
        model_counts[num_models] += 1
    
    print(f"\nElites by number of models:")
    for num_models in sorted(model_counts.keys()):
        print(f"  {num_models} model(s): {model_counts[num_models]} elites")
    
    # Step 6: Save enriched data
    print(f"\nStep 5: Saving enriched data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_elites, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(combined_elites)} enriched elite records")
    print(f"Output file: {output_path}")
    
    print("\n" + "="*80)
    print("Enrichment Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

