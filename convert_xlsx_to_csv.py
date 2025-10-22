#!/usr/bin/env python3
"""
Convert existing Excel files to CSV format.
Run this script to convert prompt.xlsx and prompt_extended.xlsx to CSV.
"""

import pandas as pd
from pathlib import Path

def convert_files():
    """Convert Excel files to CSV"""
    data_dir = Path("data")
    
    files_to_convert = [
        ("prompt.xlsx", "prompt.csv"),
        ("prompt_extended.xlsx", "prompt_extended.csv")
    ]
    
    print("=" * 80)
    print("Converting Excel files to CSV")
    print("=" * 80)
    print()
    
    for xlsx_name, csv_name in files_to_convert:
        xlsx_path = data_dir / xlsx_name
        csv_path = data_dir / csv_name
        
        if xlsx_path.exists():
            try:
                print(f"Converting {xlsx_name}...")
                df = pd.read_excel(xlsx_path)
                print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
                
                df.to_csv(csv_path, index=False)
                print(f"  ✅ Saved as {csv_name}")
                
                # Verify
                df_verify = pd.read_csv(csv_path)
                if len(df_verify) == len(df):
                    print(f"  ✅ Verified {len(df_verify)} rows")
                else:
                    print(f"  ⚠️  Row count mismatch: {len(df)} -> {len(df_verify)}")
                print()
            except Exception as e:
                print(f"  ❌ Error converting {xlsx_name}: {e}")
                print()
        else:
            print(f"ℹ️  {xlsx_name} not found, skipping")
            print()
    
    print("=" * 80)
    print("Conversion complete!")
    print("=" * 80)
    print()
    print("The project now uses CSV files:")
    print("  - data/prompt.csv (main prompt file)")
    print("  - data/prompt_extended.csv (extended dataset)")
    print("  - data/harmful_questions.csv (full dataset)")
    print()
    print("You can safely delete the .xlsx files if desired:")
    print("  rm data/prompt.xlsx data/prompt_extended.xlsx")
    print()

if __name__ == "__main__":
    convert_files()

