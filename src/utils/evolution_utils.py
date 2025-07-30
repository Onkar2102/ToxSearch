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


def sync_parents_tracker_with_evolution_tracker(logger):
    """Sync ParentsByGenerationTracker with EvolutionTracker to ensure all generations are tracked.
    
    This function reads the EvolutionTracker.json and ensures that ParentsByGenerationTracker.json
    has entries for all generations that have been run, even if the current population doesn't
    contain all those generations.
    """
    try:
        evolution_tracker_path = "outputs/EvolutionTracker.json"
        parents_tracker_path = "outputs/ParentsByGenerationTracker.json"
        
        if not os.path.exists(evolution_tracker_path):
            logger.warning("EvolutionTracker.json not found, cannot sync parents tracker")
            return
        
        # Load evolution tracker
        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
            evolution_tracker = json.load(f)
        
        # Load current parents tracker
        if os.path.exists(parents_tracker_path):
            with open(parents_tracker_path, 'r', encoding='utf-8') as f:
                parents_tracker = json.load(f)
        else:
            parents_tracker = []
        
        # For each prompt_id in evolution tracker, ensure parents tracker has all generations
        for evolution_entry in evolution_tracker:
            prompt_id = evolution_entry["prompt_id"]
            generations = evolution_entry.get("generations", [])
            
            # Find or create parents tracker entry for this prompt_id
            parents_entry = None
            for entry in parents_tracker:
                if entry["prompt_id"] == prompt_id:
                    parents_entry = entry
                    break
            
            if parents_entry is None:
                parents_entry = {
                    "prompt_id": prompt_id,
                    "generations": []
                }
                parents_tracker.append(parents_entry)
            
            # Check which generations are missing from parents tracker
            existing_generations = set()
            for gen in parents_entry["generations"]:
                existing_generations.add((gen["generation_number"], gen["parent_type"]))
            
            # Add missing generations based on evolution tracker
            for gen in generations:
                generation_number = gen["generation_number"]
                
                # For generation 0, add initial entry if missing
                if generation_number == 0 and (0, "initial") not in existing_generations:
                    parents_entry["generations"].append({
                        "generation_number": 0,
                        "parent_ids": None,
                        "parent_type": "initial"
                    })
                    existing_generations.add((0, "initial"))
                    logger.info(f"Added missing generation 0 entry for prompt_id {prompt_id}")
                
                # For other generations, add mutation and crossover entries if missing
                if generation_number > 0:
                    # Add mutation entry if missing
                    if (generation_number, "mutation") not in existing_generations:
                        # Try to infer parent from evolution tracker
                        parent_id = None
                        if gen.get("mutation") and "variants created" in gen["mutation"]:
                            # This generation had mutation, so we need to infer the parent
                            # We'll use a placeholder since we don't have the exact parent info
                            parent_id = f"gen_{generation_number-1}_best"
                        
                        parents_entry["generations"].append({
                            "generation_number": generation_number,
                            "parent_ids": [parent_id] if parent_id else None,
                            "parent_type": "mutation"
                        })
                        existing_generations.add((generation_number, "mutation"))
                        logger.info(f"Added missing generation {generation_number} mutation entry for prompt_id {prompt_id}")
                    
                    # Add crossover entry if missing
                    if (generation_number, "crossover") not in existing_generations:
                        # Try to infer parents from evolution tracker
                        parent_ids = None
                        if gen.get("crossover") and "variants created" in gen["crossover"]:
                            # This generation had crossover, so we need to infer the parents
                            # We'll use placeholders since we don't have the exact parent info
                            parent_ids = [f"gen_{generation_number-1}_parent_{i}" for i in range(5)]
                        
                        parents_entry["generations"].append({
                            "generation_number": generation_number,
                            "parent_ids": parent_ids,
                            "parent_type": "crossover"
                        })
                        existing_generations.add((generation_number, "crossover"))
                        logger.info(f"Added missing generation {generation_number} crossover entry for prompt_id {prompt_id}")
        
        # Save updated parents tracker
        with open(parents_tracker_path, 'w', encoding='utf-8') as f:
            json.dump(parents_tracker, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Synced ParentsByGenerationTracker with EvolutionTracker")
        
    except Exception as e:
        logger.error(f"Failed to sync parents tracker with evolution tracker: {e}", exc_info=True)
        raise 