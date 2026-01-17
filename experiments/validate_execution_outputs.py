"""
Validation script for execution outputs.
Validates metrics, population counts, speciation metrics, and variant counts.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

def load_json(filepath: Path) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def validate_operator_effectiveness_metrics(csv_path: Path, tracker_path: Path, run_name: str) -> List[str]:
    """Validate operator effectiveness metrics (RQ1)."""
    issues = []
    
    try:
        df = pd.read_csv(csv_path)
        tracker = load_json(tracker_path)
        
        # Group by generation
        for generation in df['generation'].unique():
            gen_df = df[df['generation'] == generation]
            gen_num = int(generation)
            
            # Find generation in tracker
            gen_entry = None
            for g in tracker.get('generations', []):
                if g.get('generation_number') == gen_num:
                    gen_entry = g
                    break
            
            if not gen_entry:
                issues.append(f"Generation {gen_num}: Not found in EvolutionTracker.json")
                continue
            
            # Validate each operator
            for _, row in gen_df.iterrows():
                operator = row['operator']
                total_variants = row['total_variants']
                elite_count = row['elite_count']
                non_elite_count = row['non_elite_count']
                rejections = row['rejections']
                duplicates = row['duplicates']
                
                # Calculate expected values
                calculated_total = total_variants + rejections + duplicates
                
                # Validate calculated_total
                if calculated_total == 0:
                    # All variants rejected/duplicated - check if metrics are correct
                    if row['NE'] != 0.0 or row['EHR'] != 0.0:
                        issues.append(f"Gen {gen_num}, {operator}: Metrics should be 0 when calculated_total=0")
                    if row['IR'] != (100.0 if rejections > 0 else 0.0):
                        issues.append(f"Gen {gen_num}, {operator}: IR should be 100 when all rejected, 0 otherwise")
                    continue
                
                # Validate NE
                expected_ne = round(non_elite_count / calculated_total * 100, 2)
                if abs(row['NE'] - expected_ne) > 0.01:
                    issues.append(f"Gen {gen_num}, {operator}: NE mismatch. Expected {expected_ne}, got {row['NE']}")
                
                # Validate EHR
                expected_ehr = round(elite_count / calculated_total * 100, 2)
                if abs(row['EHR'] - expected_ehr) > 0.01:
                    issues.append(f"Gen {gen_num}, {operator}: EHR mismatch. Expected {expected_ehr}, got {row['EHR']}")
                
                # Validate IR
                expected_ir = round(rejections / calculated_total * 100, 2)
                if abs(row['IR'] - expected_ir) > 0.01:
                    issues.append(f"Gen {gen_num}, {operator}: IR mismatch. Expected {expected_ir}, got {row['IR']}")
                
                # Validate NE + EHR + IR + duplicates_percent ≈ 100
                # Duplicates are a separate category, so they should be included
                duplicates_percent = round(duplicates / calculated_total * 100, 2) if calculated_total > 0 else 0.0
                total_percent = row['NE'] + row['EHR'] + row['IR'] + duplicates_percent
                if abs(total_percent - 100.0) > 0.1:  # Allow small rounding errors
                    issues.append(f"Gen {gen_num}, {operator}: NE+EHR+IR+duplicates={total_percent:.2f}, should be ~100 (NE={row['NE']}, EHR={row['EHR']}, IR={row['IR']}, duplicates={duplicates_percent})")
                
                # Validate cEHR
                if total_variants > 0:
                    expected_cehr = round(elite_count / total_variants * 100, 2)
                    if pd.notna(row['cEHR']) and abs(row['cEHR'] - expected_cehr) > 0.01:
                        issues.append(f"Gen {gen_num}, {operator}: cEHR mismatch. Expected {expected_cehr}, got {row['cEHR']}")
                elif pd.notna(row['cEHR']) and row['cEHR'] != 0.0:
                    issues.append(f"Gen {gen_num}, {operator}: cEHR should be 0 when total_variants=0, got {row['cEHR']}")
                
                # Validate delta stats
                if pd.notna(row['Δμ']):
                    # Delta stats should only exist if there are valid variants
                    if total_variants == 0:
                        issues.append(f"Gen {gen_num}, {operator}: Δμ should be NaN when total_variants=0")
                
                if pd.notna(row['Δσ']):
                    # Δσ requires at least 2 data points
                    if total_variants < 2 and row['Δσ'] != 0.0:
                        issues.append(f"Gen {gen_num}, {operator}: Δσ should be 0.0 when total_variants<2, got {row['Δσ']}")
        
        print(f"✓ Validated operator effectiveness metrics for {run_name}")
        
    except Exception as e:
        issues.append(f"Error validating operator effectiveness: {e}")
    
    return issues

def validate_population_counts(tracker_path: Path, elites_path: Path, reserves_path: Path, 
                               archive_path: Path, run_name: str) -> List[str]:
    """Validate population counts consistency."""
    issues = []
    
    try:
        tracker = load_json(tracker_path)
        
        # Load actual file contents
        elites = load_json(elites_path) if elites_path.exists() else []
        reserves = load_json(reserves_path) if reserves_path.exists() else []
        archive = load_json(archive_path) if archive_path.exists() else []
        
        for gen_entry in tracker.get('generations', []):
            gen_num = gen_entry.get('generation_number')
            if gen_num is None:
                continue
            
            elites_count = gen_entry.get('elites_count', 0)
            reserves_count = gen_entry.get('reserves_count', 0)
            total_population = gen_entry.get('total_population', 0)
            archived_count = gen_entry.get('speciation', {}).get('archived_count', 0)
            
            # Check elites + reserves = total_population
            expected_total = elites_count + reserves_count
            if expected_total != total_population:
                issues.append(f"Gen {gen_num}: elites_count({elites_count}) + reserves_count({reserves_count}) = {expected_total}, but total_population={total_population}")
            
            # Check speciation total_population matches
            spec_total = gen_entry.get('speciation', {}).get('total_population', 0)
            if spec_total != total_population:
                issues.append(f"Gen {gen_num}: speciation.total_population({spec_total}) != total_population({total_population})")
            
            # For last generation, check against actual file contents
            if gen_num == tracker.get('total_generations', 0):
                actual_elites = len(elites)
                actual_reserves = len(reserves)
                actual_archive = len(archive)
                
                if actual_elites != elites_count:
                    issues.append(f"Gen {gen_num} (final): Actual elites.json has {actual_elites} genomes, but elites_count={elites_count}")
                
                if actual_reserves != reserves_count:
                    issues.append(f"Gen {gen_num} (final): Actual reserves.json has {actual_reserves} genomes, but reserves_count={reserves_count}")
        
        print(f"✓ Validated population counts for {run_name}")
        
    except Exception as e:
        issues.append(f"Error validating population counts: {e}")
    
    return issues

def validate_speciation_metrics(tracker_path: Path, speciation_path: Path, run_name: str) -> List[str]:
    """Validate speciation metrics."""
    issues = []
    
    try:
        tracker = load_json(tracker_path)
        speciation_state = load_json(speciation_path) if speciation_path.exists() else {}
        
        for gen_entry in tracker.get('generations', []):
            gen_num = gen_entry.get('generation_number')
            if gen_num is None:
                continue
            
            spec_data = gen_entry.get('speciation', {})
            if not spec_data:
                continue
            
            species_count = spec_data.get('species_count', 0)
            reserves_size = spec_data.get('reserves_size', 0)
            inter_div = spec_data.get('inter_species_diversity', 0)
            intra_div = spec_data.get('intra_species_diversity', 0)
            
            # Check diversity metrics
            # Note: Diversity only considers active species with valid embeddings
            # If only 1 active species has an embedding, inter_div = 0.0 is correct
            # So we need to check the actual state file to see how many active species have embeddings
            if species_count > 1 and inter_div == 0.0:
                # This might be valid if only 1 active species has an embedding
                # Check speciation_state.json to verify
                spec_path_obj = Path(speciation_path) if isinstance(speciation_path, str) else speciation_path
                if spec_path_obj.exists():
                    try:
                        spec_state = load_json(spec_path_obj)
                        species_dict = spec_state.get('species', {})
                        active_with_emb = sum(1 for sp in species_dict.values() 
                                            if sp.get('species_state') == 'active' 
                                            and sp.get('leader_embedding') is not None)
                        if active_with_emb > 1:
                            issues.append(f"Gen {gen_num}: inter_species_diversity is 0 but {active_with_emb} active species have embeddings")
                        # If active_with_emb <= 1, inter_div = 0.0 is correct, so don't flag it
                    except Exception as e:
                        # If we can't check, flag it as a potential issue
                        issues.append(f"Gen {gen_num}: inter_species_diversity is 0 but species_count={species_count} > 1 (could not verify embeddings: {e})")
                else:
                    issues.append(f"Gen {gen_num}: inter_species_diversity is 0 but species_count={species_count} > 1 (speciation_state.json not found)")
            
            if species_count > 1:
                # Check if any species has more than 2 members (should have intra diversity)
                species_dict = speciation_state.get('species', {})
                species_with_members = 0
                for sid, sp_data in species_dict.items():
                    if isinstance(sp_data, dict):
                        size = sp_data.get('size', 0)
                        state = sp_data.get('species_state', 'active')
                        if state == 'active' and size > 2:
                            species_with_members += 1
                
                if species_with_members > 1 and intra_div == 0.0:
                    issues.append(f"Gen {gen_num}: intra_species_diversity is 0 but {species_with_members} species have >2 members")
            
            # Check reserves_size matches reserves_count
            reserves_count = gen_entry.get('reserves_count', 0)
            if reserves_size != reserves_count:
                issues.append(f"Gen {gen_num}: speciation.reserves_size({reserves_size}) != reserves_count({reserves_count})")
        
        print(f"✓ Validated speciation metrics for {run_name}")
        
    except Exception as e:
        issues.append(f"Error validating speciation metrics: {e}")
    
    return issues

def validate_variant_counts(tracker_path: Path, run_name: str) -> List[str]:
    """Validate variant counts."""
    issues = []
    
    try:
        tracker = load_json(tracker_path)
        
        for gen_entry in tracker.get('generations', []):
            gen_num = gen_entry.get('generation_number')
            if gen_num == 0:
                continue  # Skip generation 0
            
            expected_count = gen_entry.get('expected_variant_count')
            variants_created = gen_entry.get('variants_created', 0)
            mutation_variants = gen_entry.get('mutation_variants', 0)
            crossover_variants = gen_entry.get('crossover_variants', 0)
            selection_mode = gen_entry.get('selection_mode', 'default')
            
            # Calculate expected based on selection mode
            if selection_mode.lower() in ['exploit', 'exploitation', 'explore', 'exploration']:
                # 3 parents: (10 × 3 × 1) + (2 × C(3,2) × 1) = 30 + 6 = 36
                calculated_expected = 36
            else:
                # 2 parents: (10 × 2 × 1) + (2 × C(2,2) × 1) = 20 + 2 = 22
                calculated_expected = 22
            
            if expected_count != calculated_expected:
                issues.append(f"Gen {gen_num}: expected_variant_count={expected_count}, but calculated={calculated_expected} (mode={selection_mode})")
            
            # Check mutation + crossover variants
            total_operator_variants = mutation_variants + crossover_variants
            # Note: variants_created is after deduplication, so it may be less
            if total_operator_variants > variants_created:
                # This is expected due to deduplication, but log if difference is large
                diff = total_operator_variants - variants_created
                if diff > total_operator_variants * 0.5:  # More than 50% difference
                    issues.append(f"Gen {gen_num}: Large difference between operator variants({total_operator_variants}) and variants_created({variants_created}), diff={diff}")
        
        print(f"✓ Validated variant counts for {run_name}")
        
    except Exception as e:
        issues.append(f"Error validating variant counts: {e}")
    
    return issues

def check_expected_variant_count_usage(codebase_path: Path) -> Dict[str, Any]:
    """Check if expected_variant_count is actually needed."""
    usage_info = {
        'stored_in': [],
        'read_from': [],
        'used_for_calculation': False,
        'used_for_validation': False,
        'analysis_scripts': []
    }
    
    # Check codebase for usage (exclude this validation script)
    import os
    for root, dirs, files in os.walk(codebase_path):
        # Skip hidden directories and venv
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv' and d != '__pycache__' and d != 'experiments']
        
        for file in files:
            if not file.endswith('.py'):
                continue
            
            filepath = Path(root) / file
            # Skip validation script
            if 'validate_execution_outputs' in str(filepath):
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    if 'expected_variant_count' in content:
                        if '_store_expected_variant_count' in content:
                            usage_info['stored_in'].append(str(filepath.relative_to(codebase_path)))
                        
                        if 'get_expected_variant_count' in content:
                            usage_info['read_from'].append(str(filepath.relative_to(codebase_path)))
                        
                        # Check if used for calculation (as denominator)
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'expected_variant_count' in line or 'expected_total' in line:
                                # Check context around this line
                                context_start = max(0, i-3)
                                context_end = min(len(lines), i+8)
                                context = '\n'.join(lines[context_start:context_end])
                                
                                # Check if used as denominator in calculations
                                if 'metrics_denominator' in context and 'expected' in context.lower():
                                    usage_info['used_for_calculation'] = True
                                
                                # Check if used for validation
                                if 'validate' in context.lower() or 'tolerance' in context.lower() or 'warning' in context.lower():
                                    usage_info['used_for_validation'] = True
            except Exception:
                pass
    
    # Check analysis scripts separately
    analysis_dir = codebase_path / 'experiments'
    if analysis_dir.exists():
        for file in analysis_dir.glob('*.py'):
            if 'validate_execution_outputs' in str(file):
                continue
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    if 'expected_variant_count' in f.read():
                        usage_info['analysis_scripts'].append(str(file.relative_to(codebase_path)))
            except Exception:
                pass
    
    return usage_info

def main():
    """Main validation function."""
    base_dir = Path('data/outputs')
    runs = ['20260116_2007', '20260116_2059']
    
    all_issues = defaultdict(list)
    
    print("=" * 80)
    print("EXECUTION OUTPUT VALIDATION")
    print("=" * 80)
    
    for run_name in runs:
        print(f"\n{'='*80}")
        print(f"Validating: {run_name}")
        print(f"{'='*80}")
        
        run_dir = base_dir / run_name
        
        if not run_dir.exists():
            all_issues[run_name].append(f"Directory not found: {run_dir}")
            continue
        
        # File paths
        csv_path = run_dir / 'operator_effectiveness_cumulative.csv'
        tracker_path = run_dir / 'EvolutionTracker.json'
        elites_path = run_dir / 'elites.json'
        reserves_path = run_dir / 'reserves.json'
        archive_path = run_dir / 'archive.json'
        speciation_path = run_dir / 'speciation_state.json'
        
        # Validate each aspect
        issues = validate_operator_effectiveness_metrics(csv_path, tracker_path, run_name)
        all_issues[run_name].extend(issues)
        
        issues = validate_population_counts(tracker_path, elites_path, reserves_path, archive_path, run_name)
        all_issues[run_name].extend(issues)
        
        issues = validate_speciation_metrics(tracker_path, speciation_path, run_name)
        all_issues[run_name].extend(issues)
        
        issues = validate_variant_counts(tracker_path, run_name)
        all_issues[run_name].extend(issues)
    
    # Check expected_variant_count usage
    print(f"\n{'='*80}")
    print("Checking expected_variant_count usage")
    print(f"{'='*80}")
    codebase_path = Path('.')
    usage_info = check_expected_variant_count_usage(codebase_path)
    
    print(f"\nUsage Analysis:")
    print(f"  Stored in: {usage_info['stored_in']}")
    print(f"  Read from: {usage_info['read_from']}")
    print(f"  Used for calculation: {usage_info['used_for_calculation']}")
    print(f"  Used for validation: {usage_info['used_for_validation']}")
    print(f"  Analysis scripts: {usage_info['analysis_scripts']}")
    
    # Generate report
    print(f"\n{'='*80}")
    print("VALIDATION REPORT")
    print(f"{'='*80}")
    
    total_issues = 0
    for run_name, issues in all_issues.items():
        print(f"\n{run_name}: {len(issues)} issues found")
        total_issues += len(issues)
        if issues:
            for issue in issues[:20]:  # Show first 20 issues
                print(f"  - {issue}")
            if len(issues) > 20:
                print(f"  ... and {len(issues) - 20} more issues")
        else:
            print(f"  ✓ No issues found!")
    
    print(f"\n{'='*80}")
    print(f"Total issues across all runs: {total_issues}")
    print(f"{'='*80}")
    
    # Expected variant count recommendation
    print(f"\n{'='*80}")
    print("EXPECTED_VARIANT_COUNT RECOMMENDATION")
    print(f"{'='*80}")
    
    if not usage_info['used_for_calculation'] and usage_info['used_for_validation']:
        print("RECOMMENDATION: expected_variant_count can be REMOVED")
        print("  - Only used for validation/debugging")
        print("  - Not used for actual metric calculations")
        print("  - Metrics use calculated_total from operator statistics")
    elif usage_info['used_for_calculation']:
        print("RECOMMENDATION: expected_variant_count should be KEPT")
        print("  - Used in metric calculations")
    else:
        print("RECOMMENDATION: expected_variant_count can be REMOVED")
        print("  - Not used anywhere")
    
    return all_issues, usage_info

if __name__ == '__main__':
    main()
