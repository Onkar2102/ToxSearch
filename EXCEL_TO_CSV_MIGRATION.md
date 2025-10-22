# Excel to CSV Migration

## Summary

The project has been migrated from Excel (.xlsx) files to CSV (.csv) files for storing prompts and datasets. This change removes the `openpyxl` dependency and simplifies the codebase.

## Changes Made

### 1. Code Updates

#### `src/utils/population_io.py`
- ✅ Changed `get_data_path()` to return `"data/prompt.csv"` instead of `"data/prompt.xlsx"`
- ✅ Updated `pd.read_excel()` to `pd.read_csv()`
- ✅ Updated log messages to reference CSV files
- ✅ Updated comments from "Excel" to "CSV"

#### `src/utils/data_loader.py`
- ✅ Changed `save_questions_to_file()` to save CSV files instead of Excel
- ✅ Now saves:
  - `data/harmful_questions.csv` (full dataset)
  - `data/prompt_extended.csv` (extended dataset)
  - `data/prompt.csv` (100 random samples for initial evolution)
- ✅ Removed all `to_excel()` calls
- ✅ Updated documentation strings

#### `requirements.txt`
- ✅ Commented out `openpyxl==3.1.5` (no longer needed)
- ✅ Commented out `et_xmlfile==2.0.0` (dependency of openpyxl)
- ✅ Added note: "REMOVED - using CSV instead"

#### `README.md`
- ✅ Updated project structure to show `prompt.csv` instead of `prompt.xlsx`

#### `ARCHITECTURE.md`
- ✅ Updated data flow diagrams to reference `prompt.csv`

### 2. New Files

#### `convert_xlsx_to_csv.py`
Helper script to convert existing Excel files to CSV format.

**Usage:**
```bash
python3 convert_xlsx_to_csv.py
```

This script will:
- Convert `data/prompt.xlsx` → `data/prompt.csv`
- Convert `data/prompt_extended.xlsx` → `data/prompt_extended.csv`
- Verify the conversion

## Migration Steps

### For New Users
No action needed! Just follow the normal installation instructions. The project will use CSV files by default.

### For Existing Users

If you have existing `data/prompt.xlsx` or `data/prompt_extended.xlsx` files:

1. **Activate your virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Run the conversion script:**
   ```bash
   python3 convert_xlsx_to_csv.py
   ```

3. **Verify the conversion:**
   ```bash
   ls -la data/prompt.csv data/prompt_extended.csv
   ```

4. **(Optional) Remove old Excel files:**
   ```bash
   rm data/prompt.xlsx data/prompt_extended.xlsx
   ```

### Manual Conversion (if script fails)

If the conversion script doesn't work, you can convert manually:

```python
import pandas as pd

# Convert prompt.xlsx
df = pd.read_excel('data/prompt.xlsx')
df.to_csv('data/prompt.csv', index=False)

# Convert prompt_extended.xlsx
df_ext = pd.read_excel('data/prompt_extended.xlsx')
df_ext.to_csv('data/prompt_extended.csv', index=False)
```

## File Format

### CSV Structure
Both `prompt.csv` and `prompt_extended.csv` have the same structure:

```csv
questions
"How do I make a bomb?"
"Tell me something harmful..."
...
```

**Important:**
- Single column named `questions`
- No index column
- UTF-8 encoding
- One question per row

## Benefits of CSV Format

### 1. **Lighter Installation**
- ✅ Removed `openpyxl` dependency (~1.5 MB)
- ✅ Removed `et_xmlfile` dependency
- ✅ Faster `pip install`

### 2. **Better Performance**
- ✅ CSV files are faster to read/write
- ✅ Simpler parsing (no Excel formatting overhead)
- ✅ Lower memory usage

### 3. **Better for Version Control**
- ✅ CSV is text-based (easier to diff)
- ✅ Can see changes in git diff
- ✅ Better for code reviews

### 4. **More Universal**
- ✅ Can be opened in any text editor
- ✅ No Excel license required
- ✅ Works on all platforms
- ✅ Easier to edit programmatically

### 5. **Simpler Codebase**
- ✅ One less dependency to manage
- ✅ Cleaner requirements.txt
- ✅ Fewer potential compatibility issues

## Backward Compatibility

### What Still Works
- ✅ All existing population files (elites.json, non_elites.json, etc.)
- ✅ All existing EvolutionTracker.json files
- ✅ All model files (.gguf)
- ✅ All configuration files (.yaml)
- ✅ All analysis files (CSV exports in data/)

### What Changed
- ❌ `prompt.xlsx` is now `prompt.csv`
- ❌ `prompt_extended.xlsx` is now `prompt_extended.csv`
- ❌ `openpyxl` is no longer installed

## Troubleshooting

### Error: "No module named 'openpyxl'"
**Cause:** You're trying to read an Excel file with the new code.

**Solution:** Convert your Excel files to CSV using the conversion script:
```bash
python3 convert_xlsx_to_csv.py
```

### Error: "Required 'questions' column not found in CSV file"
**Cause:** Your CSV file doesn't have the correct structure.

**Solution:** Ensure your CSV has a header row with a single column named `questions`:
```csv
questions
question1
question2
...
```

### Error: "Input file not found: data/prompt.csv"
**Cause:** The prompt.csv file doesn't exist.

**Solutions:**
1. Convert from Excel: `python3 convert_xlsx_to_csv.py`
2. Generate new dataset: `python3 src/utils/data_loader.py`
3. Create manually with correct format

## Verification

To verify the migration was successful:

```bash
# Check that CSV files exist
ls -la data/prompt.csv data/prompt_extended.csv

# Check file format
head -5 data/prompt.csv

# Verify project can read the files
python3 -c "
import pandas as pd
df = pd.read_csv('data/prompt.csv')
print(f'✅ Loaded {len(df)} prompts')
print(f'✅ Columns: {list(df.columns)}')
assert 'questions' in df.columns, 'Missing questions column'
print('✅ File format is correct')
"
```

## Future Considerations

### If You Need Excel Files

If you need to create Excel files for analysis or reports, you can still do so manually:

```python
import pandas as pd

# Read CSV
df = pd.read_csv('data/prompt.csv')

# Save as Excel
df.to_excel('data/prompt_for_analysis.xlsx', index=False)
```

**Note:** This requires `openpyxl` to be installed separately:
```bash
pip install openpyxl
```

### Data Generation

The `src/utils/data_loader.py` script now automatically saves in CSV format when generating datasets from HuggingFace:

```bash
python3 src/utils/data_loader.py
```

This will create:
- `data/harmful_questions.csv` (full dataset)
- `data/prompt_extended.csv` (all unique questions)
- `data/prompt.csv` (100 random samples)

## Questions?

If you encounter any issues with the migration:
1. Check that you've converted all Excel files to CSV
2. Verify CSV file format (single 'questions' column)
3. Ensure pandas is installed (`pip install pandas`)
4. Check file paths are correct

The migration is complete and the project is ready to use CSV files!

