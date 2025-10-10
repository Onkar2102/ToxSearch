#!/usr/bin/env python3
"""
Debug script to test population index calculation.
"""

import sys
import os
sys.path.append('src')

from utils.population_io import get_population_files_info
import json

def main():
    print("Testing population index calculation...")
    
    try:
        info = get_population_files_info("outputs")
        print("Population files info:")
        print(json.dumps(info, indent=2))
        
        # Check files directly
        with open('outputs/elites.json', 'r') as f:
            elites = json.load(f)
        print(f"\nDirect elites count: {len(elites)}")
        
        with open('outputs/Population.json', 'r') as f:
            population = json.load(f)
        print(f"Direct population count: {len(population)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
