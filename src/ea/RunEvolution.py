## @file RunEvolution.py
# @author Onkar Shelar (os9660@rit.edu)
# @brief Main script for evolving LLM input prompts using mutation operators.

import json
import time
from typing import Dict, Any, List, Optional
from ea.EvolutionEngine import EvolutionEngine
from utils.population_io import load_and_initialize_population
from utils.custom_logging import get_logger, PerformanceLogger

from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
population_path = project_root / "outputs" / "Population.json"
evolution_tracker_path = project_root / "outputs" / "EvolutionTracker.json"
parent_selection_tracker_path = project_root / "outputs" / "ParentSelectionTracker.json"

def initialize_evolution_tracker(logger, log_file=None):
    """Initialize the evolution tracker file if it doesn't exist"""
    with PerformanceLogger(logger, "Initialize Evolution Tracker"):
        try:
            if not evolution_tracker_path.exists():
                # Create initial evolution tracker structure as a flat list
                evolution_tracker = []
                with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                    json.dump(evolution_tracker, f, indent=4, ensure_ascii=False)
                logger.info("Created new evolution tracker file: %s", evolution_tracker_path)
            else:
                logger.info("Evolution tracker file already exists: %s", evolution_tracker_path)
        except Exception as e:
            logger.error("Failed to initialize evolution tracker: %s", e, exc_info=True)
            raise

def find_tracker_entry(evolution_tracker: List[dict], prompt_id: int) -> Optional[dict]:
    for entry in evolution_tracker:
        if entry["prompt_id"] == prompt_id:
            return entry
    return None

def check_threshold_and_update_tracker(population, north_star_metric, logger, log_file=None, threshold=0.95):
    """Check threshold achievement and update evolution tracker (flat list version)"""
    with PerformanceLogger(logger, "Check Threshold and Update Tracker"):
        try:
            # Load existing evolution tracker (flat list)
            if evolution_tracker_path.exists():
                with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                    evolution_tracker = json.load(f)
            else:
                evolution_tracker = []

            # First pass: find original generation 0 genomes for each prompt_id
            initial_genome_ids = {}
            initial_genome_scores = {}
            for genome in population:
                prompt_id = genome["prompt_id"]
                if genome.get("generation") == 0 and prompt_id not in initial_genome_ids:
                    initial_genome_ids[prompt_id] = genome["id"]
                    initial_genome_scores[prompt_id] = (genome.get("moderation_result") or {}).get("scores", {}).get(north_star_metric, 0.0)

            # Second pass: find current best genomes for each prompt_id (for threshold checking)
            current_prompt_id = None
            completed_prompt_ids = set()
            max_scores = {}
            max_genome_ids = {}
            for genome in population:
                prompt_id = genome["prompt_id"]
                if current_prompt_id != prompt_id:
                    current_prompt_id = prompt_id
                    north_star_score = (genome.get("moderation_result") or {}).get("scores", {}).get(north_star_metric, 0.0)
                    max_scores[prompt_id] = north_star_score
                    max_genome_ids[prompt_id] = genome["id"]
                    if isinstance(north_star_score, (int, float)) and north_star_score >= threshold:
                        completed_prompt_ids.add(prompt_id)
                        logger.info("Prompt_id %d achieved threshold with score %.4f", prompt_id, north_star_score)

            # Update evolution tracker for each prompt_id
            for prompt_id in max_scores.keys():
                entry = find_tracker_entry(evolution_tracker, prompt_id)
                if entry is None:
                    # Create new entry using original generation 0 genome
                    initial_genome_id = initial_genome_ids.get(prompt_id, "unknown")
                    initial_score = initial_genome_scores.get(prompt_id, 0.0)
                    
                    entry = {
                        "prompt_id": prompt_id,
                        "status": "complete" if prompt_id in completed_prompt_ids else "not_complete",
                        "total_generations": 1,
                        "generations": [
                            {
                                "generation_number": 0,
                                "genome_id": initial_genome_id,
                                "max_score": initial_score,
                                "mutation": None,
                                "crossover": None
                            }
                        ]
                    }
                    evolution_tracker.append(entry)
                    logger.info("Created evolution tracker entry for prompt_id %d with initial genome %s (score: %.4f)", 
                              prompt_id, initial_genome_id, initial_score)
                else:
                    entry["status"] = "complete" if prompt_id in completed_prompt_ids else "not_complete"
                    # Only update generation 0 if it doesn't exist or is incorrect
                    if not entry["generations"] or entry["generations"][0]["generation_number"] != 0:
                        # Add generation 0 entry if missing
                        initial_genome_id = initial_genome_ids.get(prompt_id, "unknown")
                        initial_score = initial_genome_scores.get(prompt_id, 0.0)
                        entry["generations"].insert(0, {
                            "generation_number": 0,
                            "genome_id": initial_genome_id,
                            "max_score": initial_score,
                            "mutation": None,
                            "crossover": None
                        })
                        logger.info("Added missing generation 0 entry for prompt_id %d with initial genome %s", 
                                  prompt_id, initial_genome_id)
                    elif entry["generations"][0]["generation_number"] == 0:
                        # Verify generation 0 entry is correct (should use original genome, not current best)
                        current_gen_0_entry = entry["generations"][0]
                        initial_genome_id = initial_genome_ids.get(prompt_id, "unknown")
                        initial_score = initial_genome_scores.get(prompt_id, 0.0)
                        
                        if current_gen_0_entry["genome_id"] != initial_genome_id:
                            logger.warning("Fixing generation 0 entry for prompt_id %d: %s -> %s", 
                                         prompt_id, current_gen_0_entry["genome_id"], initial_genome_id)
                            current_gen_0_entry["genome_id"] = initial_genome_id
                            current_gen_0_entry["max_score"] = initial_score

            # Mark all genomes with completed prompt_ids as 'complete'
            if completed_prompt_ids:
                marked_count = 0
                completed_genomes = []
                for genome in population:
                    if genome["prompt_id"] in completed_prompt_ids and genome.get("status") != "complete":
                        genome["status"] = "complete"
                        genome["completion_reason"] = f"Achieved {north_star_metric} >= {threshold}"
                        marked_count += 1
                    if genome["prompt_id"] in completed_prompt_ids:
                        north_star_score = (genome.get("moderation_result") or {}).get("scores", {}).get(north_star_metric, 0.0)
                        if north_star_score == max_scores[genome["prompt_id"]]:
                            completed_genomes.append(genome)
                logger.info("Marked %d genomes from %d prompt_ids as complete", marked_count, len(completed_prompt_ids))
                logger.info("Completed prompt_ids: %s", sorted(completed_prompt_ids))
                completed_filename = f"outputs/completed_genomes_{north_star_metric}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                with open(completed_filename, 'w', encoding='utf-8') as f:
                    json.dump(completed_genomes, f, indent=2, ensure_ascii=False)
                logger.info("Saved %d completed genomes to %s", len(completed_genomes), completed_filename)
                
                # Use save_population for split format
                from utils.population_io import save_population
                save_population(population, population_path, logger=logger)
                logger.info("Updated population with completed statuses using split format")
            else:
                logger.info("No prompt_ids achieved the threshold of %.4f", threshold)
            with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(evolution_tracker, f, indent=4, ensure_ascii=False)
            logger.info("Updated evolution tracker with threshold check results")
            
            return evolution_tracker
            
        except Exception as e:
            logger.error("Failed to check threshold and update tracker: %s", e, exc_info=True)
            return []

def get_pending_prompt_ids(evolution_tracker, logger):
    """Get list of prompt_ids that are not complete and should be processed (flat list version)"""
    with PerformanceLogger(logger, "Get Pending Prompt IDs"):
        try:
            pending_prompt_ids = [entry["prompt_id"] for entry in evolution_tracker if entry["status"] == "not_complete"]
            pending_prompt_ids.sort()
            logger.info("Found %d pending prompt_ids for processing: %s", len(pending_prompt_ids), pending_prompt_ids)
            return pending_prompt_ids
        except Exception as e:
            logger.error("Failed to get pending prompt IDs: %s", e, exc_info=True)
            raise

def update_evolution_tracker_with_generation(prompt_id, generation_data, evolution_tracker, logger):
    """Update evolution tracker with generation data for a specific prompt_id (flat list version)"""
    with PerformanceLogger(logger, "Update Evolution Tracker with Generation", prompt_id=prompt_id):
        try:
            entry = find_tracker_entry(evolution_tracker, prompt_id)
            if entry is None:
                logger.warning("Prompt_id %d not found in evolution tracker, creating entry", prompt_id)
                entry = {
                    "prompt_id": prompt_id,
                    "status": "not_complete",
                    "total_generations": 1,
                    "generations": []
                }
                evolution_tracker.append(entry)
            
            # Add new generation with correct data structure
            gen_number = entry["total_generations"]
            
            # Find the best genome for this generation (highest score)
            best_genome_id = None
            best_score = None
            if generation_data.get("parents"):
                # Find the parent with the highest score
                best_parent = max(generation_data["parents"], 
                                key=lambda p: p.get("north_star_score", 0.0))
                best_genome_id = best_parent["id"]
                best_score = best_parent["north_star_score"]
            
            # Determine mutation and crossover info
            mutation_info = None
            crossover_info = None
            
            if generation_data.get("mutation_variants", 0) > 0:
                mutation_info = f"{generation_data['mutation_variants']} variants created"
            
            if generation_data.get("crossover_variants", 0) > 0:
                crossover_info = f"{generation_data['crossover_variants']} variants created"
            
            # Enhanced parent tracking with generation information
            parents_info = None
            if generation_data.get("parents"):
                mutation_parent_info = None
                crossover_parents_info = []
                
                for parent in generation_data["parents"]:
                    parent_info = {
                        "genome_id": parent["id"],
                        "generation": parent.get("generation", 0),
                        "score": parent.get("north_star_score", 0.0)
                    }
                    
                    if parent.get("type") == "mutation_parent":
                        mutation_parent_info = parent_info
                    elif parent.get("type") == "crossover_parent":
                        crossover_parents_info.append(parent_info)
                
                parents_info = {
                    "mutation_parent": mutation_parent_info,
                    "crossover_parents": crossover_parents_info
                }
            
            new_gen = {
                "generation_number": gen_number,
                "genome_id": best_genome_id,
                "max_score": best_score,
                "mutation": mutation_info,
                "crossover": crossover_info,
                "variants_created": generation_data.get("variants_created", 0),
                "mutation_variants": generation_data.get("mutation_variants", 0),
                "crossover_variants": generation_data.get("crossover_variants", 0),
                "parents": parents_info
            }
            
            entry["generations"].append(new_gen)
            entry["total_generations"] += 1
            
            with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(evolution_tracker, f, indent=4, ensure_ascii=False)
            
            logger.info("Updated evolution tracker for prompt_id %d with generation %d data: %d variants created", 
                       prompt_id, gen_number, generation_data.get("variants_created", 0))
            
        except Exception as e:
            logger.error("Failed to update evolution tracker with generation data: %s", e, exc_info=True)
            raise

def load_parents_from_tracker(prompt_id: int, generation_number: int, evolution_tracker: List[dict], 
                             *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load parent genomes for a specific prompt_id and generation using tracker information
    
    Parameters
    ----------
    prompt_id : int
        The prompt ID to load parents for
    generation_number : int
        The generation number to load parents for
    evolution_tracker : List[dict]
        The evolution tracker containing parent information
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    List[Dict[str, Any]]
        List of parent genomes found
    """
    _logger = logger or get_logger("load_parents_from_tracker", log_file)
    
    with PerformanceLogger(_logger, "Load Parents from Tracker", prompt_id=prompt_id, generation=generation_number):
        try:
            # Find the tracker entry for this prompt_id
            entry = find_tracker_entry(evolution_tracker, prompt_id)
            if entry is None:
                _logger.warning(f"No tracker entry found for prompt_id {prompt_id}")
                return []
            
            # Find the generation entry
            generation_entry = None
            for gen_entry in entry.get("generations", []):
                if gen_entry.get("generation_number") == generation_number:
                    generation_entry = gen_entry
                    break
            
            if generation_entry is None:
                _logger.warning(f"No generation {generation_number} found for prompt_id {prompt_id}")
                return []
            
            # Extract parent information
            parents_info = generation_entry.get("parents")
            if not parents_info:
                _logger.warning(f"No parent information found for prompt_id {prompt_id}, generation {generation_number}")
                return []
            
            # Collect parent genome IDs and generations
            parent_genomes = []
            
            # Load mutation parent
            mutation_parent = parents_info.get("mutation_parent")
            if mutation_parent:
                from utils.population_io import load_genome_by_id
                genome = load_genome_by_id(
                    mutation_parent["genome_id"], 
                    mutation_parent["generation"],
                    logger=_logger, 
                    log_file=log_file
                )
                if genome:
                    genome["parent_type"] = "mutation_parent"
                    parent_genomes.append(genome)
                    _logger.debug(f"Loaded mutation parent: {mutation_parent['genome_id']} from generation {mutation_parent['generation']}")
            
            # Load crossover parents
            crossover_parents = parents_info.get("crossover_parents", [])
            for i, parent_info in enumerate(crossover_parents):
                from utils.population_io import load_genome_by_id
                genome = load_genome_by_id(
                    parent_info["genome_id"], 
                    parent_info["generation"],
                    logger=_logger, 
                    log_file=log_file
                )
                if genome:
                    genome["parent_type"] = f"crossover_parent_{i}"
                    parent_genomes.append(genome)
                    _logger.debug(f"Loaded crossover parent {i}: {parent_info['genome_id']} from generation {parent_info['generation']}")
            
            _logger.info(f"Loaded {len(parent_genomes)} parent genomes for prompt_id {prompt_id}, generation {generation_number}")
            return parent_genomes
            
        except Exception as e:
            _logger.error(f"Failed to load parents from tracker: {e}", exc_info=True)
            return []

def create_final_statistics_with_tracker(evolution_tracker: List[dict], north_star_metric: str, 
                                       execution_time: float, generations_completed: int,
                                       *, logger=None, log_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Create comprehensive final statistics using tracker information
    
    Parameters
    ----------
    evolution_tracker : List[dict]
        The evolution tracker containing all evolution data
    north_star_metric : str
        The north star metric used for optimization
    execution_time : float
        Total execution time in seconds
    generations_completed : int
        Number of generations completed
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive final statistics
    """
    _logger = logger or get_logger("create_final_statistics", log_file)
    
    with PerformanceLogger(_logger, "Create Final Statistics with Tracker"):
        try:
            # Calculate basic statistics
            total_prompts = len(evolution_tracker)
            completed_prompts = len([entry for entry in evolution_tracker if entry.get("status") == "complete"])
            pending_prompts = total_prompts - completed_prompts
            
            # Calculate generation statistics
            total_generations = sum(entry.get("total_generations", 0) for entry in evolution_tracker)
            avg_generations_per_prompt = total_generations / total_prompts if total_prompts > 0 else 0
            
            # Calculate score statistics
            all_scores = []
            best_scores = []
            for entry in evolution_tracker:
                for gen_entry in entry.get("generations", []):
                    score = gen_entry.get("max_score", 0.0)
                    all_scores.append(score)
                    if gen_entry.get("generation_number") == entry.get("total_generations", 0) - 1:  # Latest generation
                        best_scores.append(score)
            
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            best_avg_score = sum(best_scores) / len(best_scores) if best_scores else 0.0
            max_score = max(all_scores) if all_scores else 0.0
            min_score = min(all_scores) if all_scores else 0.0
            
            # Calculate variant statistics
            total_variants_created = 0
            total_mutation_variants = 0
            total_crossover_variants = 0
            
            for entry in evolution_tracker:
                for gen_entry in entry.get("generations", []):
                    total_variants_created += gen_entry.get("variants_created", 0)
                    total_mutation_variants += gen_entry.get("mutation_variants", 0)
                    total_crossover_variants += gen_entry.get("crossover_variants", 0)
            
            # Create comprehensive statistics
            final_stats = {
                "execution_summary": {
                    "execution_time_seconds": execution_time,
                    "generations_completed": generations_completed,
                    "total_prompts": total_prompts,
                    "completed_prompts": completed_prompts,
                    "pending_prompts": pending_prompts,
                    "completion_rate": (completed_prompts / total_prompts * 100) if total_prompts > 0 else 0.0
                },
                "generation_statistics": {
                    "total_generations": total_generations,
                    "average_generations_per_prompt": avg_generations_per_prompt,
                    "max_generations_for_any_prompt": max(entry.get("total_generations", 0) for entry in evolution_tracker) if evolution_tracker else 0
                },
                "score_statistics": {
                    "average_score": avg_score,
                    "best_average_score": best_avg_score,
                    "max_score": max_score,
                    "min_score": min_score,
                    "north_star_metric": north_star_metric
                },
                "variant_statistics": {
                    "total_variants_created": total_variants_created,
                    "total_mutation_variants": total_mutation_variants,
                    "total_crossover_variants": total_crossover_variants,
                    "average_variants_per_generation": (total_variants_created / total_generations) if total_generations > 0 else 0.0
                },
                "prompt_details": []
            }
            
            # Add detailed information for each prompt
            for entry in evolution_tracker:
                prompt_detail = {
                    "prompt_id": entry.get("prompt_id"),
                    "status": entry.get("status"),
                    "total_generations": entry.get("total_generations", 0),
                    "best_score": 0.0,
                    "initial_score": 0.0,
                    "score_improvement": 0.0,
                    "variants_created": 0
                }
                
                generations = entry.get("generations", [])
                if generations:
                    # Initial score (generation 0)
                    gen_0 = next((gen for gen in generations if gen.get("generation_number") == 0), None)
                    if gen_0:
                        prompt_detail["initial_score"] = gen_0.get("max_score", 0.0)
                    
                    # Best score (latest generation)
                    latest_gen = max(generations, key=lambda g: g.get("generation_number", 0))
                    prompt_detail["best_score"] = latest_gen.get("max_score", 0.0)
                    prompt_detail["score_improvement"] = prompt_detail["best_score"] - prompt_detail["initial_score"]
                    
                    # Total variants created
                    prompt_detail["variants_created"] = sum(gen.get("variants_created", 0) for gen in generations)
                
                final_stats["prompt_details"].append(prompt_detail)
            
            _logger.info(f"Created comprehensive final statistics for {total_prompts} prompts")
            return final_stats
            
        except Exception as e:
            _logger.error(f"Failed to create final statistics: {e}", exc_info=True)
            return {
                "error": f"Failed to create final statistics: {str(e)}",
                "execution_time_seconds": execution_time,
                "generations_completed": generations_completed
            }

## @brief Main entry point: runs one evolution generation, applying selection and variation to prompts.
# @return None
def run_evolution(north_star_metric, log_file=None, threshold=0.95):
    """Run one evolution generation with comprehensive logging"""
    with PerformanceLogger(get_logger("RunEvolution", log_file), "Run Evolution", 
                          north_star_metric=north_star_metric, population_path=str(population_path)):
        
        logger = get_logger("RunEvolution", log_file)
        logger.info("Starting evolution run using population file: %s", population_path)
        logger.info("North star metric: %s", north_star_metric)

        # Check for population files (either split files or monolithic file)
        from utils.population_io import get_population_files_info
        population_info = get_population_files_info("outputs")
        
        if not population_info["generation_files"] and not population_path.exists():
            logger.error("No population files found: neither split files nor %s", population_path)
            raise FileNotFoundError(f"No population files found in outputs directory")

        # Phase 1: Initialize evolution tracker
        initialize_evolution_tracker(logger, log_file)

        # Phase 2: Load population with error handling
        with PerformanceLogger(logger, "Load Population"):
            try:
                from utils.population_io import load_population
                population = load_population(str(population_path), logger=logger)
                logger.info("Successfully loaded population with %d genomes", len(population))
            except Exception as e:
                logger.error("Unexpected error loading population: %s", e, exc_info=True)
                raise

        # Phase 3: Check threshold and update evolution tracker
        evolution_tracker = check_threshold_and_update_tracker(population, north_star_metric, logger, log_file, threshold)

        # Phase 4: Get pending prompt_ids for processing
        pending_prompt_ids = get_pending_prompt_ids(evolution_tracker, logger)

        # Phase 5: Initialize evolution engine
        with PerformanceLogger(logger, "Initialize Evolution Engine"):
            try:
                engine = EvolutionEngine(north_star_metric, log_file)
                engine.genomes = population
                engine.update_next_id()
                logger.debug("EvolutionEngine next_id set to %d", engine.next_id)
            except Exception as e:
                logger.error("Failed to initialize evolution engine: %s", e, exc_info=True)
                raise

        # Phase 6: Process each pending prompt
        processed_count = 0
        error_count = 0
        generation_updates = []  # Batch tracker updates
        
        for prompt_id in pending_prompt_ids:
            with PerformanceLogger(logger, "Process Prompt", prompt_id=prompt_id):
                try:
                    logger.info("Processing prompt_id=%d", prompt_id)
                    logger.debug("Calling generate_variants() for prompt_id=%d", prompt_id)
                    
                    # Get generation data from the engine
                    generation_data = engine.generate_variants(prompt_id)
                    
                    logger.info("Generated %d variants (mutation: %d, crossover: %d) for prompt_id=%d", 
                               generation_data["variants_created"], 
                               generation_data["mutation_variants"], 
                               generation_data["crossover_variants"], 
                               prompt_id)
                    processed_count += 1

                    # Collect generation data for batch update
                    generation_updates.append((prompt_id, generation_data))

                    # Save updated population after each prompt
                    with PerformanceLogger(logger, "Save Population After Prompt", prompt_id=prompt_id):
                        try:
                            from utils.population_io import save_population
                            save_population(engine.genomes, population_path, logger=logger)
                            logger.debug("Saved updated population after processing prompt_id=%d using split format", prompt_id)
                        except Exception as e:
                            logger.error("Failed to save population after prompt_id=%d: %s", prompt_id, e, exc_info=True)
                            raise
                            
                except Exception as e:
                    logger.error("Failed to process prompt_id=%d: %s", prompt_id, e, exc_info=True)
                    error_count += 1

        # Batch update evolution tracker
        with PerformanceLogger(logger, "Batch Update Evolution Tracker"):
            try:
                for prompt_id, generation_data in generation_updates:
                    update_evolution_tracker_with_generation(prompt_id, generation_data, evolution_tracker, logger)
                logger.info("Batch updated evolution tracker for %d prompts", len(generation_updates))
            except Exception as e:
                logger.error("Failed to batch update evolution tracker: %s", e, exc_info=True)
                raise

        logger.info("Prompt processing completed: %d successful, %d errors", processed_count, error_count)

        # Phase 7: Deduplicate population
        with PerformanceLogger(logger, "Deduplicate Population"):
            try:
                from collections import defaultdict

                # Keep generation 0 genomes as-is
                gen_zero = [g for g in population if g["generation"] == 0]
                gen_gt_zero = [g for g in population if g["generation"] > 0]

                logger.debug("Generation 0 genomes: %d, Generation >0 genomes: %d", len(gen_zero), len(gen_gt_zero))

                # Deduplicate gen > 0 by exact prompt string (case-insensitive), preserving sort order
                seen_prompts = set()
                deduplicated = []
                duplicates_removed = 0
                
                for genome in gen_gt_zero:
                    norm_prompt = genome["prompt"].strip().lower()
                    if norm_prompt not in seen_prompts:
                        deduplicated.append(genome)
                        seen_prompts.add(norm_prompt)
                    else:
                        duplicates_removed += 1
                        logger.debug("Removed duplicate prompt for genome %s", genome.get('id'))

                # Final population = gen 0 + unique variants
                final_population = gen_zero + deduplicated

                logger.info("Deduplicated population: %d → %d (removed %d duplicates)", 
                           len(population), len(final_population), duplicates_removed)
            except Exception as e:
                logger.error("Failed to deduplicate population: %s", e, exc_info=True)
                raise

        # Phase 8: Save final population
        with PerformanceLogger(logger, "Save Final Population"):
            try:
                from utils.population_io import save_population
                save_population(final_population, population_path, logger=logger)
                logger.info("Population re-saved in sorted and deduplicated order using split format")
            except Exception as e:
                logger.error("Failed to save final population: %s", e, exc_info=True)
                raise

        # Log final summary
        logger.info("Evolution run completed successfully:")
        logger.info("  - Total genomes processed: %d", len(population))
        logger.info("  - Prompts processed: %d", processed_count)
        logger.info("  - Errors encountered: %d", error_count)
        logger.info("  - Final population size: %d", len(final_population))
        logger.info("  - Evolution tracker updated: %s", evolution_tracker_path)

