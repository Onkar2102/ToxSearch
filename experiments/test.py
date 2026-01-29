#!/usr/bin/env python3
"""
Test script to read population_max_toxicity from all speciated runs.
"""

import json
from pathlib import Path

# Paths
PROJ = Path(__file__).parent.parent.resolve()
BASE = PROJ / "data" / "outputs"

# Speciated runs (runs 01-05)
RUNS_SPECIATION = [BASE / f"run0{i}_speciated" for i in range(1, 11)]

print("="*80)
print("Population Max Toxicity for Speciated Runs")
print("="*80)
print()

for run_dir in RUNS_SPECIATION:
    tracker_path = run_dir / "EvolutionTracker.json"
    
    if not tracker_path.exists():
        print(f"{run_dir.name}: EvolutionTracker.json not found")
        continue
    
    try:
        with open(tracker_path, 'r', encoding='utf-8') as f:
            tracker = json.load(f)
        
        population_max_toxicity = tracker.get("population_max_toxicity")
        
        if population_max_toxicity is not None:
            print(f"{run_dir.name}: {population_max_toxicity:.4f}")
        else:
            print(f"{run_dir.name}: population_max_toxicity field not found")
    
    except Exception as e:
        print(f"{run_dir.name}: Error reading file - {e}")

print()
print("="*80)
