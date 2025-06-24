"""Evolution utilities to break circular imports."""

import json
import os
from utils.custom_logging import get_logger


def append_parents_by_generation_entry(prompt_id, generation_number, parent_ids, parent_type, logger):
    """Append a parents-by-generation entry to ParentsByGenerationTracker.json
    
    This tracks which parents were used for each generation, not individual variants.
    Much more efficient than tracking every single variant.
    """
    generation_entry = {
        "generation_number": generation_number,
        "parent_ids": parent_ids,
        "parent_type": parent_type  # "mutation" or "crossover"
    }
    try:
        tracker_path = "outputs/ParentsByGenerationTracker.json"
        if os.path.exists(tracker_path):
            with open(tracker_path, 'r', encoding='utf-8') as f:
                tracker = json.load(f)
        else:
            tracker = []
        
        # Find existing entry for this prompt_id
        existing_entry = None
        for entry in tracker:
            if entry["prompt_id"] == prompt_id:
                existing_entry = entry
                break
        
        if existing_entry:
            # Check if this generation already exists
            generation_exists = False
            for gen in existing_entry["generations"]:
                if gen["generation_number"] == generation_number and gen["parent_type"] == parent_type:
                    # Update existing generation entry
                    gen["parent_ids"] = parent_ids
                    generation_exists = True
                    break
            
            if not generation_exists:
                # Append new generation info
                existing_entry["generations"].append(generation_entry)
        else:
            # Create new entry for this prompt_id
            new_entry = {
                "prompt_id": prompt_id,
                "generations": [generation_entry]
            }
            tracker.append(new_entry)
        
        with open(tracker_path, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, indent=4, ensure_ascii=False)
        logger.debug(f"Appended generation {generation_number} for prompt_id={prompt_id}, parent_type={parent_type}, parent_ids={parent_ids}")
    except Exception as e:
        logger.error(f"Failed to append parents by generation entry: {e}", exc_info=True)
        raise 