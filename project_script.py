#!/usr/bin/env python3
"""
Simple Experiment Runner

Just edit the EXPERIMENTS list below and run: python project_script.py
"""

import subprocess
import sys
import time

# =============================================================================
# Define your experiments here (one command per line)
# =============================================================================

EXPERIMENTS = [
    "python src/main.py --operators ie --generations 30",
    "python src/main.py --operators cm --generations 30", 
    "python src/main.py --operators all --generations 30",
]

# =============================================================================
# Run experiments
# =============================================================================

print(f"Starting {len(EXPERIMENTS)} experiments...")
print()

for i, cmd in enumerate(EXPERIMENTS, 1):
    print("=" * 50)
    print(f"Experiment {i}/{len(EXPERIMENTS)}")
    print("=" * 50)
    print(f"Command: {cmd}")
    print()
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"\n✓ Experiment {i} completed")
    except subprocess.CalledProcessError:
        print(f"\n✗ Experiment {i} failed")
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user!")
        sys.exit(1)
    
    print()
    print("Waiting 5 seconds...")
    time.sleep(5)
    print()

print("All experiments completed!")

