#!/usr/bin/env python3
"""
RQ3: Analyze Enriched Combined Elites

Analyzes rq3_combined_elites_enriched.json to:
1. Count total number of genomes (elites)
2. Check if all prompts have toxicity scores
3. Check if all prompts have responses from all models
"""

import os
import json
from collections import defaultdict, Counter


def main():
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    enriched_path = os.path.join(script_dir, "rq3_combined_elites_enriched.json")
    
    print("="*80)
    print("RQ3: Analyzing Enriched Combined Elites")
    print("="*80)
    
    # Load enriched elites
    print("\nLoading enriched elites...")
    with open(enriched_path, 'r', encoding='utf-8') as f:
        enriched_elites = json.load(f)
    
    total_genomes = len(enriched_elites)
    print(f"Total number of genomes (elites): {total_genomes}")
    
    # Collect all unique model names
    all_model_names = set()
    for elite in enriched_elites:
        models = elite.get('models', {})
        all_model_names.update(models.keys())
    
    all_model_names = sorted(all_model_names)
    print(f"\nTotal unique models found: {len(all_model_names)}")
    print("Models:", ", ".join(all_model_names))
    
    # Analyze each elite
    print("\n" + "="*80)
    print("Analysis Results:")
    print("="*80)
    
    # Statistics
    elites_with_all_models = 0
    elites_missing_models = 0
    elites_missing_scores = 0
    elites_missing_responses = 0
    
    # Track which models are missing for which elites
    missing_models_count = Counter()
    missing_scores_count = Counter()
    missing_responses_count = Counter()
    
    # Track per-model statistics
    model_stats = defaultdict(lambda: {'with_score': 0, 'with_response': 0, 'total': 0})
    
    for elite in enriched_elites:
        prompt = elite.get('prompt', '')
        models = elite.get('models', {})
        num_models = len(models)
        
        # Check if has all models
        if num_models == len(all_model_names):
            elites_with_all_models += 1
        else:
            elites_missing_models += 1
            missing_models = set(all_model_names) - set(models.keys())
            for model in missing_models:
                missing_models_count[model] += 1
        
        # Check each model for score and response
        for model_name in all_model_names:
            model_stats[model_name]['total'] += 1
            
            if model_name in models:
                model_data = models[model_name]
                response = model_data.get('response', '')
                toxicity = model_data.get('toxicity', None)
                
                if toxicity is None:
                    elites_missing_scores += 1
                    missing_scores_count[model_name] += 1
                else:
                    model_stats[model_name]['with_score'] += 1
                
                if not response or response.strip() == '':
                    elites_missing_responses += 1
                    missing_responses_count[model_name] += 1
                else:
                    model_stats[model_name]['with_response'] += 1
            else:
                # Model not present for this elite
                missing_models_count[model_name] += 1
    
    # Print summary statistics
    print(f"\n1. Total Genomes: {total_genomes}")
    
    print(f"\n2. Model Coverage:")
    print(f"   Elites with ALL models ({len(all_model_names)} models): {elites_with_all_models} ({elites_with_all_models/total_genomes*100:.1f}%)")
    print(f"   Elites missing some models: {elites_missing_models} ({elites_missing_models/total_genomes*100:.1f}%)")
    
    print(f"\n3. Missing Models (which models are missing for how many elites):")
    if missing_models_count:
        for model, count in missing_models_count.most_common():
            percentage = (count / total_genomes) * 100
            print(f"   {model}: missing for {count} elites ({percentage:.1f}%)")
    else:
        print("   All elites have all models!")
    
    print(f"\n4. Toxicity Scores:")
    print(f"   Elites with missing scores: {elites_missing_scores}")
    if missing_scores_count:
        print(f"   Missing scores by model:")
        for model, count in missing_scores_count.most_common():
            percentage = (count / total_genomes) * 100
            print(f"     {model}: {count} missing ({percentage:.1f}%)")
    else:
        print("   All prompts have toxicity scores for all models!")
    
    print(f"\n5. Responses:")
    print(f"   Elites with missing responses: {elites_missing_responses}")
    if missing_responses_count:
        print(f"   Missing responses by model:")
        for model, count in missing_responses_count.most_common():
            percentage = (count / total_genomes) * 100
            print(f"     {model}: {count} missing ({percentage:.1f}%)")
    else:
        print("   All prompts have responses for all models!")
    
    # Per-model detailed statistics
    print(f"\n6. Per-Model Statistics:")
    print(f"{'Model':<40} {'Total':<10} {'With Score':<12} {'With Response':<15} {'Score %':<10} {'Response %':<12}")
    print("-" * 100)
    for model_name in all_model_names:
        stats = model_stats[model_name]
        total = stats['total']
        with_score = stats['with_score']
        with_response = stats['with_response']
        score_pct = (with_score / total * 100) if total > 0 else 0
        response_pct = (with_response / total * 100) if total > 0 else 0
        print(f"{model_name:<40} {total:<10} {with_score:<12} {with_response:<15} {score_pct:<10.1f} {response_pct:<12.1f}")
    
    # Overall completeness
    print(f"\n7. Overall Completeness:")
    total_possible = total_genomes * len(all_model_names)
    total_with_scores = sum(stats['with_score'] for stats in model_stats.values())
    total_with_responses = sum(stats['with_response'] for stats in model_stats.values())
    
    score_completeness = (total_with_scores / total_possible * 100) if total_possible > 0 else 0
    response_completeness = (total_with_responses / total_possible * 100) if total_possible > 0 else 0
    
    print(f"   Score completeness: {total_with_scores}/{total_possible} ({score_completeness:.1f}%)")
    print(f"   Response completeness: {total_with_responses}/{total_possible} ({response_completeness:.1f}%)")
    
    # Check for elites with complete data
    complete_elites = 0
    for elite in enriched_elites:
        models = elite.get('models', {})
        is_complete = True
        
        # Check if has all models
        if len(models) != len(all_model_names):
            is_complete = False
        else:
            # Check if all models have both score and response
            for model_name in all_model_names:
                if model_name not in models:
                    is_complete = False
                    break
                model_data = models[model_name]
                if model_data.get('toxicity') is None or not model_data.get('response', '').strip():
                    is_complete = False
                    break
        
        if is_complete:
            complete_elites += 1
    
    print(f"\n8. Fully Complete Elites:")
    print(f"   Elites with all models, all scores, and all responses: {complete_elites} ({complete_elites/total_genomes*100:.1f}%)")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

