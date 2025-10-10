#!/usr/bin/env python3
"""
Debug script to test parent selection.
"""

import sys
import os
sys.path.append('src')

from ea.ParentSelector import ParentSelector
import json

def main():
    print("Testing parent selection...")
    
    # Initialize parent selector
    selector = ParentSelector("toxicity", log_file=None)
    
    # Test adaptive tournament selection
    try:
        print("Calling adaptive_tournament_selection...")
        selector.adaptive_tournament_selection(evolution_tracker=None)
        print("Parent selection completed")
        
        # Check if parents.json was created
        with open('outputs/parents.json', 'r') as f:
            parents_data = json.load(f)
        
        print(f"Parents.json content: {parents_data}")
        
    except Exception as e:
        print(f"Error in parent selection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
