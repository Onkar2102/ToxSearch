"""
Helper script to find execution directories for analysis.

This script scans the outputs directory and helps identify:
- Speciation runs (have speciation_state.json)
- Non-speciation runs (no speciation_state.json)
- Recent runs sorted by timestamp
"""

import json
from pathlib import Path
from datetime import datetime

BASE_OUTPUT_DIR = Path("../data/outputs")

def check_speciation(output_dir: Path) -> bool:
    """Check if a directory contains speciation data."""
    speciation_file = output_dir / "speciation_state.json"
    return speciation_file.exists()

def get_pg_model(output_dir: Path) -> str:
    """Extract PG model name from elites.json if available."""
    elites_file = output_dir / "elites.json"
    if not elites_file.exists():
        return "Unknown"
    
    try:
        with open(elites_file, 'r') as f:
            elites = json.load(f)
            if isinstance(elites, list) and len(elites) > 0:
                first_genome = elites[0]
                pg_model = first_genome.get('prompt_generator_name', 'Unknown')
                if pg_model and pg_model != 'Unknown':
                    # Extract just the model name
                    return Path(pg_model).name if pg_model else "Unknown"
    except:
        pass
    
    return "Unknown"

def get_execution_info(output_dir: Path):
    """Get execution information."""
    info = {
        'directory': output_dir.name,
        'has_speciation': check_speciation(output_dir),
        'pg_model': get_pg_model(output_dir),
        'timestamp': None,
    }
    
    # Try to extract timestamp from directory name (YYYYMMDD_HHMM format)
    try:
        parts = output_dir.name.split('_')
        if len(parts) >= 2:
            date_str = parts[0]
            time_str = parts[1]
            if len(date_str) == 8 and len(time_str) == 4:
                dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
                info['timestamp'] = dt
    except:
        pass
    
    return info

def main():
    """Find and list execution directories."""
    if not BASE_OUTPUT_DIR.exists():
        print(f"Error: {BASE_OUTPUT_DIR} does not exist")
        return
    
    print("=" * 80)
    print("Execution Directory Finder")
    print("=" * 80)
    
    speciation_runs = []
    nonspeciation_runs = []
    
    # Scan all directories
    for output_dir in sorted(BASE_OUTPUT_DIR.iterdir()):
        if not output_dir.is_dir():
            continue
        
        # Skip if it doesn't look like an execution directory
        if not any(output_dir.glob("*.json")):
            continue
        
        info = get_execution_info(output_dir)
        
        if info['has_speciation']:
            speciation_runs.append(info)
        else:
            nonspeciation_runs.append(info)
    
    # Sort by timestamp (most recent first)
    speciation_runs.sort(key=lambda x: x['timestamp'] or datetime.min, reverse=True)
    nonspeciation_runs.sort(key=lambda x: x['timestamp'] or datetime.min, reverse=True)
    
    print(f"\n📊 Found {len(speciation_runs)} speciation runs:")
    print("-" * 80)
    for i, run in enumerate(speciation_runs[:10], 1):  # Show top 10
        pg_model = run['pg_model']
        timestamp = run['timestamp'].strftime("%Y-%m-%d %H:%M") if run['timestamp'] else "Unknown"
        print(f"{i:2d}. {run['directory']:20s} | PG: {pg_model:40s} | {timestamp}")
    
    print(f"\n📊 Found {len(nonspeciation_runs)} non-speciation runs:")
    print("-" * 80)
    for i, run in enumerate(nonspeciation_runs[:10], 1):  # Show top 10
        timestamp = run['timestamp'].strftime("%Y-%m-%d %H:%M") if run['timestamp'] else "Unknown"
        print(f"{i:2d}. {run['directory']:20s} | {timestamp}")
    
    print("\n" + "=" * 80)
    print("Python Code Snippet for compare_speciation_vs_nonspeciation.py:")
    print("=" * 80)
    
    if speciation_runs:
        print("\nSPEciation_RUNS = [")
        for run in speciation_runs[:5]:  # Top 5
            print(f'    "{run["directory"]}",  # PG: {run["pg_model"]}')
        print("]")
    
    if nonspeciation_runs:
        print("\nNON_SPECIATION_RUNS = [")
        for run in nonspeciation_runs[:5]:  # Top 5
            print(f'    "{run["directory"]}",')
        print("]")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
