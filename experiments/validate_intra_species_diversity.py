#!/usr/bin/env python3
"""
Validation script to check why intra_species_diversity is 0.0.

This script:
1. Loads elites.json and checks if species have multiple members
2. Verifies embeddings are present
3. Checks if compute_diversity_metrics is receiving correct data
4. Simulates the diversity calculation to identify the issue
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

def load_execution_data(output_dir: Path):
    """Load elites.json and EvolutionTracker.json from execution directory."""
    elites_path = output_dir / "elites.json"
    tracker_path = output_dir / "EvolutionTracker.json"
    speciation_path = output_dir / "speciation_state.json"
    
    if not elites_path.exists():
        print(f"ERROR: {elites_path} not found")
        return None, None, None
    
    if not tracker_path.exists():
        print(f"ERROR: {tracker_path} not found")
        return None, None, None
    
    with open(elites_path, 'r', encoding='utf-8') as f:
        elites = json.load(f)
    
    with open(tracker_path, 'r', encoding='utf-8') as f:
        tracker = json.load(f)
    
    speciation_state = None
    if speciation_path.exists():
        with open(speciation_path, 'r', encoding='utf-8') as f:
            speciation_state = json.load(f)
    
    return elites, tracker, speciation_state


def analyze_species_members(elites):
    """Analyze species membership from elites.json."""
    species_members = defaultdict(list)
    species_embeddings = defaultdict(int)
    
    for genome in elites:
        species_id = genome.get("species_id")
        if species_id is not None and species_id > 0:
            species_members[species_id].append(genome)
            if "prompt_embedding" in genome and genome.get("prompt_embedding") is not None:
                species_embeddings[species_id] += 1
    
    return species_members, species_embeddings


def check_diversity_calculation_requirements(species_members, species_embeddings):
    """Check if requirements for intra-species diversity calculation are met."""
    issues = []
    
    # Check species with multiple members
    species_with_multiple = {sid: members for sid, members in species_members.items() if len(members) >= 2}
    
    if not species_with_multiple:
        issues.append("❌ No species have multiple members (all species have only 1 member)")
        return issues
    
    print(f"✅ Found {len(species_with_multiple)} species with ≥2 members")
    
    # Check embeddings
    species_with_embeddings = {}
    for sid, members in species_with_multiple.items():
        members_with_emb = sum(1 for m in members if "prompt_embedding" in m and m.get("prompt_embedding") is not None)
        if members_with_emb >= 2:
            species_with_embeddings[sid] = members_with_emb
        else:
            issues.append(f"⚠️  Species {sid}: {len(members)} members, but only {members_with_emb} have embeddings (need ≥2)")
    
    if not species_with_embeddings:
        issues.append("❌ No species have ≥2 members with embeddings")
        return issues
    
    print(f"✅ Found {len(species_with_embeddings)} species with ≥2 members that have embeddings")
    
    # Check if embeddings are valid (not empty lists)
    for sid, count in list(species_with_embeddings.items()):
        members = species_members[sid]
        valid_embeddings = 0
        for member in members:
            emb = member.get("prompt_embedding")
            if emb is not None:
                if isinstance(emb, list) and len(emb) > 0:
                    valid_embeddings += 1
                elif isinstance(emb, np.ndarray) and emb.size > 0:
                    valid_embeddings += 1
        
        if valid_embeddings < 2:
            issues.append(f"⚠️  Species {sid}: {count} members with embeddings, but only {valid_embeddings} have valid (non-empty) embeddings")
            del species_with_embeddings[sid]
    
    if not species_with_embeddings:
        issues.append("❌ No species have ≥2 members with valid (non-empty) embeddings")
        return issues
    
    print(f"✅ Found {len(species_with_embeddings)} species with ≥2 members that have valid embeddings")
    
    return issues


def find_zero_diversity_generations(tracker):
    """Find generations where intra_species_diversity is 0.0 but species_count > 1."""
    zero_diversity_gens = []
    
    generations = tracker.get("generations", [])
    for gen in generations:
        gen_num = gen.get("generation_number")
        spec = gen.get("speciation", {})
        intra_div = spec.get("intra_species_diversity", 0.0)
        species_count = spec.get("species_count", 0)
        
        if intra_div == 0.0 and species_count > 1:
            zero_diversity_gens.append({
                "generation": gen_num,
                "species_count": species_count,
                "frozen_species_count": spec.get("frozen_species_count", 0),
                "inter_species_diversity": spec.get("inter_species_diversity", 0.0),
                "total_population": spec.get("total_population", 0),
            })
    
    return zero_diversity_gens


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_intra_species_diversity.py <output_dir>")
        print("Example: python validate_intra_species_diversity.py data/outputs/20260117_1335")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    if not output_dir.exists():
        print(f"ERROR: Directory {output_dir} does not exist")
        sys.exit(1)
    
    print(f"Validating intra-species diversity for: {output_dir.name}\n")
    
    # Load data
    elites, tracker, speciation_state = load_execution_data(output_dir)
    if elites is None or tracker is None:
        sys.exit(1)
    
    print(f"Loaded {len(elites)} genomes from elites.json\n")
    
    # Analyze species membership
    species_members, species_embeddings = analyze_species_members(elites)
    print(f"Found {len(species_members)} species in elites.json")
    print(f"Species size distribution: {dict(Counter(len(members) for members in species_members.values()))}\n")
    
    # Check diversity calculation requirements
    print("Checking diversity calculation requirements...")
    issues = check_diversity_calculation_requirements(species_members, species_embeddings)
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ All requirements met for intra-species diversity calculation")
    
    # Find generations with zero diversity
    print("\n" + "="*60)
    print("Generations with intra_species_diversity = 0.0:")
    print("="*60)
    zero_diversity_gens = find_zero_diversity_generations(tracker)
    
    if not zero_diversity_gens:
        print("✅ No generations found with intra_species_diversity = 0.0 and species_count > 1")
    else:
        for gen_info in zero_diversity_gens:
            print(f"\nGeneration {gen_info['generation']}:")
            print(f"  species_count: {gen_info['species_count']}")
            print(f"  frozen_species_count: {gen_info['frozen_species_count']}")
            print(f"  inter_species_diversity: {gen_info['inter_species_diversity']}")
            print(f"  total_population: {gen_info['total_population']}")
            print(f"  intra_species_diversity: 0.0 ⚠️")
    
    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    if issues:
        print("❌ Intra-species diversity calculation will fail due to missing data")
        print("\nPossible causes:")
        print("  1. Embeddings not preserved in elites.json")
        print("  2. Species only have 1 member (leader only)")
        print("  3. Embeddings are empty or invalid")
    else:
        print("✅ Data looks good - intra-species diversity should be calculable")
        if zero_diversity_gens:
            print("\n⚠️  However, generations show 0.0 diversity - this suggests:")
            print("  1. elites_path not being passed to compute_diversity_metrics()")
            print("  2. Timing issue - calculation happens before elites.json is written")
            print("  3. Species.members in-memory only has leader, not loading from elites.json")


if __name__ == "__main__":
    main()
