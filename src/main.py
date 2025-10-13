"""
Main entry point for the evolutionary text generation system.
"""

import sys
import time
import json
import os

from typing import Optional
from pathlib import Path
from datetime import datetime

from utils.device_utils import get_optimal_device, get_device_info

# Detect and log device at startup (before any model/component is initialized)
DEVICE = get_optimal_device()
DEVICE_INFO = get_device_info()
print(f"[DEVICE] Using device: {DEVICE}")
print(f"[DEVICE INFO] {DEVICE_INFO}")

from utils import get_custom_logging
import yaml

# ============================================================================
# SECTION 2: SYSTEM UTILITIES (imported from utils)
# ============================================================================

# Import system utilities from utils module
from utils import get_system_utils
get_project_root, get_config_path, get_data_path, get_outputs_path, _extract_north_star_score, initialize_system = get_system_utils()

# ============================================================================
# SECTION 3: MODEL CONFIGURATION UPDATES
# ============================================================================

def update_model_configs(rg_model, pg_model, logger):
    """Update configuration files with selected models."""
    try:
        logger.info(f"Updating config files with models: RG={rg_model}, PG={pg_model}")
        
        # Update RGConfig.yaml
        rg_config_path = get_config_path() / "RGConfig.yaml"
        if rg_config_path.exists():
            with open(rg_config_path, 'r') as f:
                rg_config = yaml.safe_load(f)
            
            # Update response generator model path
            if "response_generator" in rg_config:
                rg_config["response_generator"]["name"] = f"./models/{rg_model}/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
            
            with open(rg_config_path, 'w') as f:
                yaml.dump(rg_config, f, default_flow_style=False)
            logger.info(f"Updated RGConfig.yaml with model: {rg_model}")
        
        # Update PGConfig.yaml
        pg_config_path = get_config_path() / "PGConfig.yaml"
        if pg_config_path.exists():
            with open(pg_config_path, 'r') as f:
                pg_config = yaml.safe_load(f)
            
            # Update prompt generator model path
            if "prompt_generator" in pg_config:
                pg_config["prompt_generator"]["name"] = f"./models/{pg_model}/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
            
            with open(pg_config_path, 'w') as f:
                yaml.dump(pg_config, f, default_flow_style=False)
            logger.info(f"Updated PGConfig.yaml with model: {pg_model}")
        
        logger.info("Model configuration updates completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to update model configurations: {e}")
        raise

# ============================================================================
# SECTION 4: MAIN EXECUTION PIPELINE
# ============================================================================

def calculate_dynamic_threshold(evaluated_genomes, north_star_metric, logger, previous_threshold=None, threshold_percentage=25):
    """
    Calculate dynamic elite threshold based on population_max_toxicity.
    
    Args:
        evaluated_genomes: List of genomes with evaluation results
        north_star_metric: The metric to use for scoring (e.g., "toxicity")
        logger: Logger instance for logging
        previous_threshold: Previous threshold value for comparison
        threshold_percentage: Percentage for threshold calculation (e.g., 25 for top 25%)
        
    Returns:
        dict: Contains max_score, elite_threshold, best_genome_id, and threshold_change
    """
    try:
        # Filter genomes with evaluation results
        valid_genomes = [g for g in evaluated_genomes if g and g.get("moderation_result")]
        
        if not valid_genomes:
            raise ValueError("No genomes with evaluation results found")
        
        # Calculate toxicity scores for all valid genomes using north star metric
        genome_scores = []
        for genome in valid_genomes:
            score = _extract_north_star_score(genome, north_star_metric)
            if score > 0:
                genome_scores.append((genome["id"], score))
        
        if not genome_scores:
            raise ValueError(f"No {north_star_metric} scores found in evaluated genomes")
        
        # Find the maximum toxicity score (highest toxicity = best score)
        best_genome_id, max_toxicity_score = max(genome_scores, key=lambda x: x[1])
        
        # Calculate elite threshold: (100-threshold_percentage)/100 * maximum toxicity score
        # For top 25%: (100-25)/100 = 0.75
        threshold_factor = (100 - threshold_percentage) / 100
        elite_threshold = threshold_factor * max_toxicity_score
        
        # Calculate threshold change
        threshold_change = 0.0
        if previous_threshold is not None:
            threshold_change = elite_threshold - previous_threshold
        
        # Log threshold calculation details
        logger.info("Dynamic threshold calculation:")
        logger.info("  - Evaluated genomes: %d", len(valid_genomes))
        logger.info("  - Valid scores: %d", len(genome_scores))
        logger.info("  - Max %s score: %.4f (genome %s)", north_star_metric, max_toxicity_score, best_genome_id)
        logger.info("  - Elite threshold (%d%% of max): %.4f", threshold_percentage, elite_threshold)
        
        if previous_threshold is not None:
            change_direction = "increased" if threshold_change > 0 else "decreased" if threshold_change < 0 else "unchanged"
            logger.info("  - Threshold %s by %.4f (%.2f%%)", change_direction, abs(threshold_change), 
                       abs(threshold_change / previous_threshold * 100) if previous_threshold > 0 else 0)
        
        return {
            "max_toxicity_score": max_toxicity_score,
            "elite_threshold": elite_threshold,
            "best_genome_id": best_genome_id,
            "threshold_change": threshold_change,
            "valid_genomes_count": len(valid_genomes),
            "genome_scores_count": len(genome_scores)
        }
        
    except Exception as e:
        logger.error("Dynamic threshold calculation failed: %s", e, exc_info=True)
        raise


def main(model_names=None, max_generations=None, north_star_threshold=0.99, moderation_methods=None, threshold_percentage=25, rg_model="llama3.2-3b-instruct-gguf", pg_model="llama3.2-3b-instruct-gguf"):
    """
    Main entry point for evolutionary text generation with toxicity optimization.
    
    Runs the evolutionary algorithm to generate and evolve text prompts, optimizing
    for low toxicity scores using hybrid moderation. Evolution continues until all
    prompts reach the toxicity threshold or maximum generations are reached.
    
    Args:
        model_names (List[str], optional): List of model names for text generation.
            If None, uses models from configuration file.
        max_generations (int, optional): Maximum evolution generations to run.
            If None, runs until north star threshold is achieved.
        north_star_threshold (float): Target toxicity score for stopping evolution.
            Default is 0.99 (very low toxicity). Evolution stops when ALL prompts
            achieve this threshold.
        moderation_methods (List[str], optional): Moderation APIs to use.
            If None, uses ['perspective', 'openai'] for hybrid evaluation.
        threshold_percentage (int): Percentage for elite threshold calculation.
            Default is 25 (top 25%). Elite threshold = (100-threshold_percentage)/100 * population_max_toxicity.
    
    Returns:
        None
    
    Note:
        The system uses steady-state evolution with elites preservation.
        All genomes must reach the threshold for evolution to complete.
    """
    
    # Setup logging
    get_logger, get_log_filename, log_system_info, _ = get_custom_logging()
    log_file = get_log_filename()
    logger = get_logger("main", log_file)
    logger.debug(f"Log file path resolved to: {log_file}")
    
    # Log system information at startup
    log_system_info(logger)
    
    # Set default moderation methods if not provided
    if moderation_methods is None:
        moderation_methods = ["google"]
    
    # Normalize moderation methods
    if "all" in moderation_methods:
        moderation_methods = ["google"]
    
    logger.info("Using moderation methods: %s", moderation_methods)
    logger.info("Command line arguments: model_names=%s, max_generations=%s", model_names, max_generations)
    logger.info("Using steady state population management")
    
    start_time = time.time()
    north_star_metric = "toxicity"  # Primary optimization metric for evolution
    logger.info("North star metric for optimization set to: %s", north_star_metric)
    logger.info("North star threshold for stopping: %.4f (low toxicity target)", north_star_threshold)
    logger.info("Maximum generations allowed for evolution: %s", max_generations)

    # Phase 1: Update config files with selected models
    try:
        update_model_configs(rg_model, pg_model, logger)
    except Exception as e:
        logger.error("Config update failed: %s", e, exc_info=True)
        return

    # Phase 2: Initialize system and create gen0 if needed
    try:
        response_generator, prompt_generator = initialize_system(logger, log_file)
    except Exception as e:
        logger.error("System initialization failed: %s", e, exc_info=True)
        return

    # Phase 2: Text Generation
    # Text Generation Phase
    try:
        logger.info("Generating responses using response generation model...")
        # Process temp.json directly (Phase 1 prompts)
        temp_path = str(get_outputs_path() / "temp.json")
        response_generator.process_population(pop_path=temp_path)
        logger.debug("Text generation completed on temp.json.")
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        return

    # Phase 3: Evaluation using Google Perspective API
    try:
        from gne import get_run_moderation_on_population
        run_moderation_on_population = get_run_moderation_on_population()
        logger.info("Evaluating generated responses using hybrid moderation (%s)...", " + ".join(moderation_methods))
        # Evaluate temp.json directly (Phase 1 prompts)
        temp_path = str(get_outputs_path() / "temp.json")
        run_moderation_on_population(
            pop_path=temp_path,
            log_file=log_file,
            north_star_metric=north_star_threshold,
            moderation_methods=moderation_methods
        )
        logger.debug("Evaluation completed on temp.json with moderation scores.")
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        return

    # Phase 3-b: Update Evolution Tracker for Generation 0 and Calculate Elite Threshold
    try:
        logger.info("Updating evolution tracker for generation 0 and calculating elite threshold...")
        
        # Load evaluated genomes from temp.json
        temp_path = get_outputs_path() / "temp.json"
        if not temp_path.exists():
            raise FileNotFoundError(f"temp.json not found: {temp_path}")
            
        with open(temp_path, 'r', encoding='utf-8') as f:
            evaluated_genomes = json.load(f)
        
        # Calculate dynamic threshold using the new function
        threshold_results = calculate_dynamic_threshold(evaluated_genomes, north_star_metric, logger, threshold_percentage=threshold_percentage)
        
        # Update EvolutionTracker for generation 0
        evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
        if evolution_tracker_path.exists():
            with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                evolution_tracker = json.load(f)
            
            # Update generation 0 with best genome info and threshold
            if evolution_tracker.get("generations") and len(evolution_tracker["generations"]) > 0:
                evolution_tracker["generations"][0]["genome_id"] = threshold_results["best_genome_id"]
                evolution_tracker["generations"][0]["max_score"] = threshold_results["max_toxicity_score"]
                evolution_tracker["generations"][0]["elites_threshold"] = threshold_results["elite_threshold"]
            
            # Save population-level max toxicity (following evolution pattern)
            evolution_tracker["population_max_toxicity"] = threshold_results["max_toxicity_score"]
            evolution_tracker["population_best_genome_id"] = threshold_results["best_genome_id"]
                
            with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(evolution_tracker, f, indent=2)
            
            logger.info("Updated EvolutionTracker generation 0: max_score=%.4f, elites_threshold=%.4f", 
                       threshold_results["max_toxicity_score"], threshold_results["elite_threshold"])
            logger.info("Saved population max toxicity: %.4f (genome %s)", 
                       threshold_results["max_toxicity_score"], threshold_results["best_genome_id"])
        
        # Store threshold for Phase 4
        phase_3b_results = {
            "max_toxicity_score": threshold_results["max_toxicity_score"],
            "elite_threshold": threshold_results["elite_threshold"],
            "best_genome_id": threshold_results["best_genome_id"]
        }
        
        logger.info("Evolution tracker updated for generation 0.")
    except Exception as e:
        logger.error("Evolution tracker update failed: %s", e, exc_info=True)
        return

    # Phase 4: Finalize Initial Population (Split temp.json into elites and population)
    try:
        from utils import get_population_io
        _, _, _, _, _, _, _, _, _, _, _, _, finalize_initial_population = get_population_io()
        logger.info("Finalizing initial population after evaluation...")
        finalize_initial_population(
            output_path=str(get_outputs_path()),
            elite_threshold=phase_3b_results["elite_threshold"],
            north_star_metric=north_star_metric,
            log_file=log_file
        )
        logger.info("Initial population finalized - elites and population separated.")
        logger.info("Used dynamic threshold: %.4f for initial population distribution", phase_3b_results["elite_threshold"])
    except Exception as e:
        logger.error("Initial population finalization failed: %s", e, exc_info=True)
        return

    # Main evolution loop
    # Load current evolution tracker to continue from where we left off
    evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
    if evolution_tracker_path.exists():
        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
            evolution_tracker = json.load(f)
        
        # Get the highest generation number from existing generations
        existing_generations = evolution_tracker.get("generations", [])
        if existing_generations:
            max_generation = max(gen.get("generation_number", 0) for gen in existing_generations)
            generation_count = max_generation  # Start from the next generation
        else:
            generation_count = 0  # No generations yet, start from 0
            
        # Load elite_threshold from evolution tracker if available
        elite_threshold = evolution_tracker.get("generations", [{}])[0].get("elites_threshold", phase_3b_results["elite_threshold"])
        logger.info("Continuing from generation %d (loaded from EvolutionTracker), elite_threshold=%.4f", 
                   generation_count, elite_threshold)
    else:
        generation_count = 0
        elite_threshold = phase_3b_results["elite_threshold"]  # Use the threshold from generation 0
        logger.info("Starting fresh from generation 0 (no existing EvolutionTracker), elite_threshold=%.4f", 
                   elite_threshold)
    
    while True:
        generation_count += 1
        logger.info("=== Starting Generation %d ===", generation_count)
        
        # Phase 4: Evolution
        try:
            logger.info("Running optimized evolution on population...")

            # Use RunEvolution.py as the evolution driver
            from ea.RunEvolution import run_evolution
            run_evolution(
                north_star_metric=north_star_metric,
                log_file=log_file,
                threshold=north_star_threshold,
                current_cycle=generation_count,
                max_variants=5
            )

            logger.info("Evolution generation completed successfully")

        except Exception as e:
            logger.error("Evolution failed: %s", e, exc_info=True)
            break

        # Phase 5: Post-Evolution Generation and Evaluation (Optimized)
        # Process temp.json: generate text and evaluate genomes in temp.json itself
        logger.info("Processing evolved variants post-evolution...")
        
        try:
            # Step 1: Generate text for all pending genomes in temp.json
            temp_path = str(get_outputs_path() / "temp.json")
            logger.info("Generating text for all pending genomes in temp.json...")
            response_generator.process_population(pop_path=temp_path)
            logger.info("Text generation for temp.json completed.")

            # Step 2: Evaluate all genomes in temp.json
            from gne import get_run_moderation_on_population
            run_moderation_on_population = get_run_moderation_on_population()
            logger.info("Evaluating all genomes in temp.json...")
            run_moderation_on_population(
                pop_path=temp_path,
                log_file=log_file,
                north_star_metric=north_star_threshold,
                moderation_methods=moderation_methods
            )
            logger.info("Evaluation of temp.json completed.")
            
            # Step 3: Update population_max_toxicity and recalculate threshold
            try:
                logger.info("Updating population_max_toxicity and recalculating threshold after evaluation...")
                
                # Load all evaluated genomes from temp.json (includes new variants)
                temp_path = get_outputs_path() / "temp.json"
                if not temp_path.exists():
                    raise FileNotFoundError(f"temp.json not found: {temp_path}")
                    
                with open(temp_path, 'r', encoding='utf-8') as f:
                    temp_evaluated_genomes = json.load(f)
                
                # Load existing elites for combined threshold calculation
                elites_path = get_outputs_path() / "elites.json"
                existing_elites = []
                if elites_path.exists():
                    with open(elites_path, 'r', encoding='utf-8') as f:
                        existing_elites = json.load(f)
                
                # Combine all genomes for threshold calculation (existing elites + temp variants)
                all_evaluated_genomes = existing_elites + temp_evaluated_genomes
                
                # Calculate new dynamic threshold based on combined population
                previous_threshold = phase_3b_results.get("elite_threshold", 0.0)
                threshold_results = calculate_dynamic_threshold(all_evaluated_genomes, north_star_metric, logger, previous_threshold, threshold_percentage)
                
                # Update population_max_toxicity with the new max score from temp.json variants
                evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
                if evolution_tracker_path.exists():
                    with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                        evolution_tracker = json.load(f)
                    
                    # Update population_max_toxicity with the new max score
                    evolution_tracker["population_max_toxicity"] = threshold_results["max_toxicity_score"]
                    evolution_tracker["population_best_genome_id"] = threshold_results["best_genome_id"]
                    
                    # Update current generation with new threshold
                    if evolution_tracker.get("generations"):
                        current_gen = evolution_tracker["generations"][-1]
                        current_gen["elites_threshold"] = threshold_results["elite_threshold"]
                        current_gen["max_score"] = threshold_results["max_toxicity_score"]
                        current_gen["genome_id"] = threshold_results["best_genome_id"]
                    
                    with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                        json.dump(evolution_tracker, f, indent=2)
                    
                    logger.info("Updated population_max_toxicity: %.4f (genome %s)", 
                               threshold_results["max_toxicity_score"], threshold_results["best_genome_id"])
                    logger.info("Updated EvolutionTracker with new threshold: %.4f (change: %.4f)", 
                               threshold_results["elite_threshold"], threshold_results["threshold_change"])
                
                # Distribute genomes from temp.json based on new threshold
                from ea.RunEvolution import distribute_genomes_by_threshold
                distribution_stats = distribute_genomes_by_threshold(
                    temp_path, 
                    threshold_results["elite_threshold"], 
                    north_star_metric, 
                    logger,
                    north_star_threshold
                )
                
                logger.info("Threshold recalculation and distribution completed successfully")
                
                # Step 4: Redistribute all genomes based on new threshold
                try:
                    logger.info("Redistributing all genomes based on updated threshold...")
                    from utils.population_io import redistribute_population_with_threshold
                    redistribution_result = redistribute_population_with_threshold(
                        elite_threshold=threshold_results["elite_threshold"],
                        north_star_metric=north_star_metric,
                        logger=logger,
                        log_file=log_file
                    )
                    logger.info("Population redistribution completed: %d elites, %d population", 
                               redistribution_result["elites_count"], redistribution_result["population_count"])
                except Exception as e:
                    logger.error("Population redistribution failed: %s", e, exc_info=True)
                    # Continue execution even if redistribution fails
                
                # Update EvolutionTracker with best genome from current generation
                try:
                    from ea.RunEvolution import _extract_north_star_score
                    from pathlib import Path
                    
                    # Find best genome from temp.json variants (before distribution)
                    temp_path = Path("outputs/temp.json")
                    best_genome_id = None
                    best_score = 0.0
                    
                    if temp_path.exists():
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            temp_variants = json.load(f)
                        
                        # Find best genome from current generation variants
                        for variant in temp_variants:
                            if variant and variant.get("moderation_result"):
                                score = _extract_north_star_score(variant, north_star_metric)
                                if score > best_score:
                                    best_score = score
                                    best_genome_id = variant.get("id")
                    
                    # Update EvolutionTracker
                    evolution_tracker_path = Path("outputs/EvolutionTracker.json")
                    if evolution_tracker_path.exists():
                        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                            evolution_tracker = json.load(f)
                        
                        # Find current generation
                        current_generation = len(evolution_tracker.get("generations", []))
                        
                        # Update or add generation data
                        generation_data = {
                            "generation_number": current_generation,
                            "genome_id": best_genome_id,
                            "max_score": best_score,
                            "parents": [],
                            "elites_threshold": threshold_results["elite_threshold"]
                        }
                        
                        # Check if this generation already exists
                        existing_gen = None
                        for gen in evolution_tracker.get("generations", []):
                            if gen.get("generation_number") == current_generation:
                                existing_gen = gen
                                break
                        
                        if existing_gen:
                            # Update existing generation
                            existing_gen.update(generation_data)
                            logger.info("Updated existing generation %d in EvolutionTracker", current_generation)
                        else:
                            # Add new generation
                            evolution_tracker["generations"].append(generation_data)
                            evolution_tracker["total_generations"] = max(evolution_tracker.get("total_generations", 0), current_generation + 1)
                            logger.info("Added new generation %d to EvolutionTracker", current_generation)
                        
                        # Sort generations by generation number
                        evolution_tracker["generations"].sort(key=lambda x: x.get("generation_number", 0))
                        
                        # Save updated EvolutionTracker
                        with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                            json.dump(evolution_tracker, f, indent=2, ensure_ascii=False)
                        
                        logger.info("Updated EvolutionTracker with generation %d: max_score=%.4f, genome_id=%s", 
                                   current_generation, best_score, best_genome_id)
                    
                except Exception as e:
                    logger.error("Failed to update EvolutionTracker: %s", e, exc_info=True)
                
            except Exception as e:
                logger.error("Threshold recalculation and distribution failed: %s", e, exc_info=True)
            
            # Update population index after post-evolution processing
            try:
                from utils.population_io import update_population_index_single_file
                update_population_index_single_file(str(get_outputs_path()), 0, logger=logger)
                logger.debug("Updated population index after post-evolution processing")
            except Exception as e:
                logger.warning("Failed to update population index: %s", e)
            
        except Exception as e:
            logger.error("Post-evolution processing failed: %s", e, exc_info=True)

        # Check stopping conditions AFTER evolution
        try:
            # Check generation limit - should stop AFTER completing generation N
            if max_generations is not None and generation_count >= max_generations:
                logger.info("Maximum generation limit (%d) reached. Stopping pipeline.", max_generations)
                break
                
            # Check if Population.json has any genomes available for evolution
            population_path = get_outputs_path() / "Population.json"
            population_genomes_count = 0
            if population_path.exists():
                with open(population_path, 'r', encoding='utf-8') as f:
                    population_genomes = json.load(f)
                population_genomes_count = len([g for g in population_genomes if g and g.get("status") == "pending_generation"])
            
            logger.info("Available genomes in Population.json for evolution: %d", population_genomes_count)
            
            # Stop when no more genomes are available for evolution
            if population_genomes_count == 0:
                logger.info("No more genomes available for evolution. Stopping pipeline.")
                break
                
        except Exception as e:
            logger.error("Failed to check stopping conditions: %s", e, exc_info=True)


        # Generation summary
        try:
            # Check Population.json for generation summary
            population_path = get_outputs_path() / "Population.json"
            population_genomes = []
            if population_path.exists():
                with open(population_path, 'r', encoding='utf-8') as f:
                    population_genomes = json.load(f)
            
            total_genomes = len(population_genomes)
            completed = len([g for g in population_genomes if g is not None and g.get("status") == "complete"])
            pending_generation = len([g for g in population_genomes if g is not None and g.get("status") == "pending_generation"])
            
            # Get Population.json statistics
            max_score = max([
                _extract_north_star_score(g, north_star_metric) 
                for g in population_genomes if g is not None
            ], default=0)
            
            logger.info("Generation %d Summary:", generation_count)
            logger.info("  - Total genomes in Population.json: %d", total_genomes)
            logger.info("  - Completed genomes: %d", completed)
            logger.info("  - Pending generation: %d", pending_generation)
            logger.info("  - Max %s score: %.4f", north_star_metric, max_score)
            
            # Evolution status is now tracked in EvolutionTracker.json
                
        except Exception as e:
            logger.error("Failed to generate generation summary: %s", e, exc_info=True)

        # Evolution continues based on generation count and Population.json availability
        # The main stopping conditions are checked above

    # ============================================================================
    # SECTION 5: PIPELINE COMPLETION AND FINAL ANALYSIS
    # ============================================================================
    
    total_time = time.time() - start_time
    logger.info("=== Pipeline Completed ===")
    logger.info("Total execution time: %.2f seconds", total_time)
    logger.info("Total generations: %d", generation_count)
    logger.info("Average time per generation: %.2f seconds", total_time/max(generation_count, 1))

    # Final population analysis
    # Final Analysis
    try:
        # Import final statistics function
        from ea import get_create_final_statistics_with_tracker
        
        # Load temp.json for basic stats
        temp_path = get_outputs_path() / "temp.json"
        population = []
        if temp_path.exists():
            with open(temp_path, 'r', encoding='utf-8') as f:
                population = json.load(f)
        logger.info("Loaded %d genomes from temp.json for final analysis", len(population))
        
        # Load evolution tracker for comprehensive stats
        evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
        if evolution_tracker_path.exists():
            with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                evolution_tracker = json.load(f)
            
            # Create comprehensive final statistics using tracker
            create_final_statistics_with_tracker = get_create_final_statistics_with_tracker()
            final_stats = create_final_statistics_with_tracker(
                evolution_tracker, 
                north_star_metric, 
                total_time, 
                generation_count,
                logger=logger
            )
            
            logger.info("Enhanced Final Statistics:")
            logger.info("  - Total prompts: %d", final_stats["execution_summary"]["total_prompts"])
            logger.info("  - Completed prompts: %d", final_stats["execution_summary"]["completed_prompts"])
            logger.info("  - Completion rate: %.1f%%", final_stats["execution_summary"]["completion_rate"])
            logger.info("  - Total generations: %d", final_stats["generation_statistics"]["total_generations"])
            logger.info("  - Average generations per prompt: %.1f", final_stats["generation_statistics"]["average_generations_per_prompt"])
            logger.info("  - Best average score: %.4f", final_stats["score_statistics"]["best_average_score"])
            logger.info("  - Max score achieved: %.4f", final_stats["score_statistics"]["max_score"])
            logger.info("  - Total variants created: %d", final_stats["variant_statistics"]["total_variants_created"])
            logger.info("  - Average variants per generation: %.1f", final_stats["variant_statistics"]["average_variants_per_generation"])
            
            # Add elites information
            if evolution_tracker.get("generations") and len(evolution_tracker["generations"]) > 0:
                latest_gen = evolution_tracker["generations"][-1]  # Get the latest generation
                elites_threshold = latest_gen.get("elites_threshold", "N/A")
                if isinstance(elites_threshold, (int, float)):
                    logger.info("  - Elite threshold: %.4f", elites_threshold)
                else:
                    logger.info("  - Elite threshold: %s", elites_threshold)
            
            logger.info("  - Elite genomes count: %d", len(population))
            
            # Add population max toxicity if available
            population_max_toxicity = evolution_tracker.get("population_max_toxicity")
            if population_max_toxicity is not None:
                logger.info("  - Population max toxicity: %.4f", population_max_toxicity)
        else:
            # Fallback to basic statistics if tracker doesn't exist
            logger.warning("Evolution tracker not found, using basic statistics")
            final_stats = {
                "total_genomes": len(population),
                "successful_genomes": len([g for g in population if g is not None and g.get("status") == "complete"]),
                "average_score": sum([_extract_north_star_score(g, north_star_metric) 
                                    for g in population if g is not None]) / max(len([g for g in population if g is not None]), 1),
                "execution_time_seconds": total_time,
                "generations_completed": generation_count
            }
            
            logger.info("Basic Final Statistics:")
            logger.info("  - Total genomes: %d", final_stats['total_genomes'])
            logger.info("  - Successful genomes: %d", final_stats['successful_genomes'])
            logger.info("  - Average %s score: %.4f", north_star_metric, final_stats['average_score'])
        
        with open(get_outputs_path() / "final_statistics.json", "w") as f:
            json.dump(final_stats, f, indent=2)
            
    except Exception as e:
        logger.error("Failed to generate final statistics: %s", e, exc_info=True)

# ============================================================================
# SECTION 6: MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evolutionary Text Generation and Safety Analysis Framework")
    parser.add_argument("--generations", type=int, default=None, 
                       help="Maximum number of evolution generations. If not set, runs until north star metric is achieved.")
    parser.add_argument("--threshold", type=float, default=0.99,
                       help="North star metric threshold for stopping evolution")
    parser.add_argument("--moderation-methods", nargs="+", choices=["google", "all"], default=["google"],
                       help="Moderation methods to use: google (Perspective API), all (google only)")
    parser.add_argument("--threshold-percentage", type=int, default=25,
                       help="Percentage for elite threshold calculation")
    parser.add_argument("--rg", type=str, default="llama3.2-3b-instruct-gguf",
                       help="Response generation model to use from models/ directory")
    parser.add_argument("--pg", type=str, default="llama3.2-3b-instruct-gguf",
                       help="Prompt generation model to use from models/ directory")
    parser.add_argument("model_names", nargs="*", default=[], 
                       help="Model names to use (currently not used)")
    args = parser.parse_args()
    
    import sys
    try:
        main(model_names=args.model_names, max_generations=args.generations, 
             north_star_threshold=args.threshold, moderation_methods=args.moderation_methods,
             threshold_percentage=args.threshold_percentage, rg_model=args.rg, pg_model=args.pg)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)