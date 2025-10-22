from datasets import load_dataset
import pandas as pd
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_english_questions(ds, source_name, split_name):
    """
    Extract 'question' column from a HuggingFace datasets.Dataset object.
    - ds: HuggingFace Dataset object (already restricted to an English split)
    - source_name: Name of the source dataset.
    - split_name: Name of the split ('train', 'en', etc.)
    Returns: DataFrame with ['questions']
    """
    try:
        # Try to find the question column (case-insensitive fallback)
        col = 'question'
        if col not in ds.column_names:
            alt_cols = [c for c in ds.column_names if c.lower() == 'question']
            if alt_cols:
                col = alt_cols[0]
                logger.info(f"Using alternative column '{col}' for {source_name}")
            else:
                logger.warning(f"No 'question' column found in {source_name}. Available columns: {ds.column_names}")
                return pd.DataFrame(columns=['questions'])

        # Convert the column to python list
        q_col = ds[col]
        try:
            q_list = q_col.to_pylist()
        except AttributeError:
            q_list = list(q_col)
        
        df = pd.DataFrame({"questions": pd.Series(q_list, dtype="string")})
        logger.info(f"Extracted {len(df)} questions from {source_name} ({split_name})")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting questions from {source_name}: {e}")
        return pd.DataFrame(columns=['questions'])

def load_harmful_datasets():
    """
    Load and combine harmful question datasets from HuggingFace.
    For now, only use CategoricalHarmfulQA.
    Returns: DataFrame with unique questions from CategoricalHarmfulQA
    """
    all_questions = []
    try:
        # ---- CategoricalHarmfulQA ----
        logger.info("Loading CategoricalHarmfulQA dataset...")
        categorical_ds = load_dataset("declare-lab/CategoricalHarmfulQA", split="en")
        categorical_questions = extract_english_questions(categorical_ds, "CategoricalHarmfulQA", "en")
        all_questions.append(categorical_questions)
    except Exception as e:
        logger.error(f"Failed to load CategoricalHarmfulQA dataset: {e}")

    if not all_questions:
        logger.error("No datasets could be loaded!")
        return pd.DataFrame(columns=['questions'])

    # Use only 'questions' columns and combine
    logger.info("Combining datasets...")
    combined_df = pd.concat(all_questions, ignore_index=True)

    # Basic cleaning and deduplication
    logger.info("Cleaning data...")
    combined_df['questions'] = combined_df['questions'].astype(str).str.strip()
    combined_df = combined_df.dropna(subset=['questions'])
    combined_df = combined_df.drop_duplicates(subset=['questions'])

    return combined_df

def save_questions_to_file(questions_df, filename=os.path.join("data", "harmful_questions.csv")):
    """
    Save questions DataFrame to CSV and Excel files.

    Args:
        questions_df: DataFrame with questions
        filename: Output filename (default: "data/harmful_questions.csv")
    """
    # Ensure the data/ directory exists
    outdir = os.path.dirname(filename)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    success = True

    # Save as CSV
    try:
        questions_df[['questions']].to_csv(filename, index=False, header=True)
        logger.info(f"Saved {len(questions_df)} unique questions to {filename}")
    except Exception as e:
        logger.error(f"Failed to save questions to {filename}: {e}")
        success = False

    # Save as Excel (.xlsx)
    excel_filename = "data/prompt.xlsx"
    try:
        questions_df[['questions']].to_excel(excel_filename, index=False, header=True)
        logger.info(f"Saved {len(questions_df)} unique questions to {excel_filename}")
    except Exception as e:
        logger.error(f"Failed to save questions to {excel_filename}: {e}")
        success = False

    return success

def get_questions_as_list(questions_df):
    """
    Extract questions as a simple list for easy use in other modules.

    Args:
        questions_df: DataFrame with questions

    Returns:
        List of question strings
    """
    return questions_df['questions'].tolist()

if __name__ == "__main__":
    # Load and process datasets
    all_questions = load_harmful_datasets()

    if not all_questions.empty:
        logger.info(f"Number of unique questions: {len(all_questions)}")
        # Save to data/harmful_questions.csv and also to data/harmful_questions.xlsx
        saved = save_questions_to_file(all_questions)
        if saved:
            print("="*50)
            print("Saved questions to: data/harmful_questions.csv and data/harmful_questions.xlsx")
            print("="*50)
            print("SAMPLE QUESTIONS:")
            print("="*50)
            print(all_questions.sample(min(5, len(all_questions))))
            print("="*50)
            print(f"TOTAL UNIQUE QUESTIONS: {len(all_questions)}")
            print("="*50)
        else:
            logger.error("Failed to save harmful questions to file.")
    else:
        logger.error("No questions were loaded!")
