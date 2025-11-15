from datasets import load_dataset
import pandas as pd
import logging
import os

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
        col = 'question'
        if col not in ds.column_names:
            alt_cols = [c for c in ds.column_names if c.lower() == 'question']
            if alt_cols:
                col = alt_cols[0]
                logger.info(f"Using alternative column '{col}' for {source_name}")
            else:
                logger.warning(f"No 'question' column found in {source_name}. Available columns: {ds.column_names}")
                return pd.DataFrame(columns=['questions'])

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

def extract_harmfulqa_questions(ds, source_name, split_name):
    """
    Extract English questions from the HarmfulQA dataset format:
    The dataset has a "contexts" column which is a list of lists of questions,
    and "contexts_language" which includes a "en" split.
    """
    try:
        if "contexts" in ds.column_names:
            context_lists = ds["contexts"]
            all_questions = []
            for context_entry in context_lists:
                if isinstance(context_entry, (list, tuple)):
                    all_questions.extend([q for q in context_entry if isinstance(q, str)])
            df = pd.DataFrame({"questions": pd.Series(all_questions, dtype="string")})
            logger.info(f"Extracted {len(df)} questions from 'contexts' in {source_name} ({split_name})")
            return df
        elif "question" in ds.column_names:
            return extract_english_questions(ds, source_name, split_name)
        else:
            logger.warning(f"No 'contexts' or 'question' column found in {source_name}. Available columns: {ds.column_names}")
            return pd.DataFrame(columns=['questions'])
    except Exception as e:
        logger.error(f"Error extracting HarmfulQA questions from {source_name}: {e}")
        return pd.DataFrame(columns=['questions'])

def load_harmful_datasets():
    """
    Load and combine harmful question datasets from HuggingFace.
    Uses CategoricalHarmfulQA and HarmfulQA.
    Returns: DataFrame with unique questions from all datasets.
    """
    all_questions = []

    try:
        logger.info("Loading CategoricalHarmfulQA dataset...")
        try:
            categorical_ds = load_dataset("declare-lab/CategoricalHarmfulQA", split="en")
            logger.info("CategoricalHarmfulQA: using 'en' split")
        except Exception as e_en:
            logger.warning(f"CategoricalHarmfulQA 'en' split failed: {e_en}; falling back to 'train'")
            categorical_ds = load_dataset("declare-lab/CategoricalHarmfulQA", split="train")
            logger.info("CategoricalHarmfulQA: using 'train' split")
        categorical_questions = extract_english_questions(categorical_ds, "CategoricalHarmfulQA", "en_or_train")
        if not categorical_questions.empty:
            all_questions.append(categorical_questions)
        else:
            logger.warning("CategoricalHarmfulQA produced 0 questions after extraction.")
    except Exception as e:
        logger.error(f"Failed to load CategoricalHarmfulQA dataset: {e}")

    try:
        logger.info("Loading HarmfulQA dataset (prefer 'en' split)...")
        try:
            harmfulqa_ds = load_dataset("declare-lab/HarmfulQA", split="en")
            logger.info("HarmfulQA: using 'en' split")
        except Exception as e_en:
            logger.warning(f"HarmfulQA 'en' split failed: {e_en}; falling back to 'train'")
            harmfulqa_ds = load_dataset("declare-lab/HarmfulQA", split="train")
            logger.info("HarmfulQA: using 'train' split")
        harmfulqa_questions = extract_harmfulqa_questions(harmfulqa_ds, "HarmfulQA", "en_or_train")
        if not harmfulqa_questions.empty:
            all_questions.append(harmfulqa_questions)
        else:
            logger.warning("HarmfulQA produced 0 questions after extraction.")
    except Exception as e:
        logger.error(f"Failed to load HarmfulQA dataset: {e}")

    if not any(not df.empty for df in all_questions):
        logger.error("No datasets could be loaded!")
        return pd.DataFrame(columns=['questions'])

    logger.info("Combining datasets...")
    combined_df = pd.concat([df for df in all_questions if not df.empty], ignore_index=True)

    logger.info("Cleaning data...")
    combined_df['questions'] = combined_df['questions'].astype(str).str.strip()
    combined_df = combined_df.dropna(subset=['questions'])
    combined_df = combined_df.drop_duplicates(subset=['questions'])

    logger.info(f"Combined unique questions: {len(combined_df)}")
    return combined_df

def save_questions_to_file(questions_df, filename=os.path.join("data", "harmful_questions.csv")):
    """
    Save questions DataFrame to CSV files.

    Args:
        questions_df: DataFrame with questions (should already be UNIQUE for prompt_extended.csv!)
        filename: Output filename (default: "data/harmful_questions.csv")
    """
    outdir = os.path.dirname(filename)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    success = True

    try:
        questions_df[['questions']].to_csv(filename, index=False, header=True)
        logger.info(f"Saved {len(questions_df)} unique questions to {filename}")
    except Exception as e:
        logger.error(f"Failed to save questions to {filename}: {e}")
        success = False

    extended_filename = "data/prompt_extended.csv"
    try:
        questions_df[['questions']].to_csv(extended_filename, index=False, header=True)
        logger.info(f"Saved {len(questions_df)} unique questions to {extended_filename} (combined/unique)")
    except Exception as e:
        logger.error(f"Failed to save questions to {extended_filename}: {e}")
        success = False

    try:
        prompt_filename = "data/prompt.csv"
        sample_df = questions_df.sample(n=min(100, len(questions_df)), random_state=42)
        sample_df[['questions']].to_csv(prompt_filename, index=False, header=True)
        logger.info(f"Saved {len(sample_df)} randomly selected questions to {prompt_filename}")
    except Exception as e:
        logger.error(f"Failed to save prompt.csv: {e}")
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
    all_questions = load_harmful_datasets()

    if not all_questions.empty:
        logger.info(f"Number of unique questions: {len(all_questions)}")
        saved = save_questions_to_file(all_questions)
        if saved:
            print("="*50)
            print("Saved questions to: data/harmful_questions.csv, data/prompt_extended.csv, and data/prompt.csv")
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
