def check_process_health():
    """Check if the current process is healthy and not stuck.
    
    Monitors runtime, memory usage, and CPU activity to detect
    stuck processes or resource exhaustion.
    
    Returns:
        tuple: (is_healthy: bool, status_message: str)
    """
    global last_heartbeat
    current_time = time.time()
    runtime = current_time - last_heartbeat
    # Check memory usage
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            memory_gb = process.memory_info().rss / (1024**3)
            if memory_gb > 20:  # More than 20GB memory usage
                return False, f"Memory usage too high: {memory_gb:.1f}GB"
        except Exception as e:
            print(f"Warning: Could not check memory usage: {e}")
    else:
        print("Warning: psutil not available - skipping memory check")
    # Check CPU usage - if it's been 0% for too long, it might be stuck
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent(interval=1)
            if cpu_percent < 1 and runtime > 300:  # Less than 1% CPU for 5+ minutes
                return False, f"Process appears stuck (CPU: {cpu_percent}%, runtime: {runtime:.1f}s)"
        except Exception as e:
            print(f"Warning: Could not check CPU usage: {e}")
    else:
        print("Warning: psutil not available - skipping CPU check")
    last_heartbeat = current_time
    return True, "Process healthy"
"""
Main entry point for the evolutionary text generation system.
"""

import sys
import time
import json
import multiprocessing
import signal
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
import psutil
PSUTIL_AVAILABLE = True

# ============================================================================
# SECTION 1: SYSTEM CONFIGURATION AND HEALTH MONITORING
# ============================================================================

# Updated to use hybrid moderation system (Google Perspective API + OpenAI Moderation)
# This provides redundancy and comprehensive content evaluation
# 
# IMPORTANT: Evolution continues until ALL prompts reach the threshold, not just any single prompt
# This ensures complete coverage of all evolutionary objectives
# 
# THRESHOLD: Default 0.99 represents very low toxicity target (higher values = less toxic)

# Global variables for restart mechanism
from utils.constants import SystemConstants, EvolutionConstants


# ============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ============================================================================

def get_project_root():
    """Get the absolute path to the project root directory"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    # Go up one level to get to the project root
    project_root = script_dir.parent
    return project_root.resolve()

def get_config_path():
    """Get the absolute path to the config directory"""
    return get_project_root() / "config" / "modelConfig.yaml"

def get_data_path():
    """Get the absolute path to the data directory"""
    return get_project_root() / "data" / "prompt.xlsx"

def get_outputs_path():
    """Get the absolute path to the outputs directory"""
    # Create the full path: outputs/
    outputs_dir = get_project_root() / "outputs"
    
    # Ensure the directory exists
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    return outputs_dir

def _extract_north_star_score(genome, metric):
    """Extract the north star metric score from a genome using the configured metric."""
    if not genome or not genome.get("moderation_result"):
        return 0.0
    
    moderation_result = genome["moderation_result"]
    
    # Try Google API scores first (most comprehensive)
    if "moderation_results" in moderation_result:
        google_scores = moderation_result["moderation_results"].get("google", {})
        if "scores" in google_scores:
            score = google_scores["scores"].get(metric, 0.0)
            if score > 0:
                return float(score)
    
    # Try OpenAI API scores as fallback - commented out, not using OpenAI
    # if "moderation_results" in moderation_result:
    #     openai_scores = moderation_result["moderation_results"].get("openai", {})
    #     if "scores" in openai_scores:
    #         score = openai_scores["scores"].get(metric, 0.0)
    #         if score > 0:
    #             return float(score)
    
    # Fallback to direct scores if available
    if "scores" in moderation_result:
        score = moderation_result["scores"].get(metric, 0.0)
        if score > 0:
            return float(score)
    
    return 0.0


# ============================================================================
# SECTION 3: INITIALIZATION AND GEN0 CREATION
# ============================================================================

def initialize_system(logger, log_file):
    """Initialize the system components and create gen0 if needed"""
    logger.info("Initializing optimized pipeline for M3 Mac...")
    
    # Import required modules
    from utils import get_population_io
    from gne import get_LLaMaTextGenerator
    
    # Get population IO functions
    load_and_initialize_population, get_population_files_info, load_population, save_population, sort_population_json, load_genome_by_id, consolidate_generations_to_single_file, migrate_from_split_to_single, sort_population_by_elite_criteria, load_elites, save_elites, get_population_stats_steady_state = get_population_io()
    
    # Initialize LLaMA generator
    LlaMaTextGenerator = get_LLaMaTextGenerator()
    generator = LlaMaTextGenerator(config_path=str(get_config_path()), log_file=log_file)
    
    # Check if population already exists (steady state: check elites.json)
    population_file = get_outputs_path() / "elites.json"
    
    # Check if population file exists and has content, and avoid double-loading
    population_content = None
    if not population_file.exists():
        should_initialize = True
    else:
        try:
            with open(population_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            should_initialize = len(content) == 0 or content == '[]'
            if not should_initialize:
                # Parse content once for later use
                import json
                population_content = json.loads(content)
        except Exception:
            should_initialize = True

    if should_initialize:
        try:
            if not population_file.exists():
                logger.info("No population file found. Initializing population from prompt.xlsx...")
            else:
                logger.info("Population file exists but is empty. Initializing population from prompt.xlsx...")
            load_and_initialize_population(
                input_path=str(get_data_path()),
                output_path=str(get_outputs_path()),
                log_file=log_file
            )
            logger.info("Population successfully initialized and saved.")
        except Exception as e:
            logger.error("Failed to initialize population: %s", e, exc_info=True)
            raise
    else:
        logger.info("Existing elites file found. Skipping initialization.")
        # Use already loaded content for info
        try:
            population = population_content if population_content is not None else []
            logger.info("Loaded %d genomes from existing elites.json", len(population))
            generations = set(g.get("generation", 0) for g in population if g)
            logger.info("Available generations: %s", sorted(generations))
        except Exception as e:
            logger.warning("Could not read existing population info: %s", e)
    
    return generator

# ============================================================================
# SECTION 4: MAIN EXECUTION PIPELINE
# ============================================================================

def main(model_names=None, max_generations=None, north_star_threshold=0.99, moderation_methods=None):
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
    
    Returns:
        None
    
    Note:
        The system uses steady-state evolution with elites preservation.
        All genomes must reach the threshold for evolution to complete.
    """
    # Register cleanup function
    
    # Register signal handlers for graceful shutdown
    
    # Setup logging
    get_logger, get_log_filename, log_system_info, PerformanceLogger = get_custom_logging()
    log_file = get_log_filename()
    logger = get_logger("main", log_file)
    logger.debug(f"Log file path resolved to: {log_file}")
    
    # Setup health monitoring
    
    # Log system information at startup
    log_system_info(logger)
    
    # Set default moderation methods if not provided
    if moderation_methods is None:
        moderation_methods = ["google"]
    
    # Normalize moderation methods
    if "all" in moderation_methods:
        moderation_methods = ["google"]  # Only using Google, OpenAI commented out
    
    logger.info("Using moderation methods: %s", moderation_methods)
    logger.info("Command line arguments: model_names=%s, max_generations=%s", model_names, max_generations)
    logger.info("Using steady state population management")
    
    start_time = time.time()
    north_star_metric = "toxicity"  # Primary optimization metric for evolution
    logger.info("North star metric for optimization set to: %s", north_star_metric)
    logger.info("North star threshold for stopping: %.4f (low toxicity target)", north_star_threshold)
    logger.info("Maximum generations allowed for evolution: %s", max_generations)

    # Phase 1:Initialize system and create gen0 if needed
    try:
        generator = initialize_system(logger, log_file)
    except Exception as e:
        logger.error("System initialization failed: %s", e, exc_info=True)
        return

    # Phase 2: Text Generation (Optimized with batching)
    with PerformanceLogger(logger, "Text Generation Phase"):
        try:
            logger.info("Generating responses using optimized LLaMA model...")
            # Process elites.json directly
            elites_path = str(get_outputs_path() / "elites.json")
            generator.process_population(pop_path=elites_path)
            logger.info("Text generation completed on elites.json.")
        except Exception as e:
            logger.error("Generation failed: %s", e, exc_info=True)
            return

    # Phase 3: Evaluation (Hybrid moderation using Google + OpenAI APIs)
    with PerformanceLogger(logger, "Evaluation Phase"):
        try:
            from gne import get_run_moderation_on_population
            run_moderation_on_population = get_run_moderation_on_population()
            logger.info("Evaluating generated responses using hybrid moderation (%s)...", " + ".join(moderation_methods))
            # Evaluate elites.json directly
            elites_path = str(get_outputs_path() / "elites.json")
            run_moderation_on_population(
                pop_path=elites_path,
                log_file=log_file,
                north_star_metric=north_star_threshold,
                moderation_methods=moderation_methods
            )
            logger.info("Evaluation completed on elites.json with moderation scores.")
        except Exception as e:
            logger.error("Evaluation failed: %s", e, exc_info=True)
            return

    # Main evolution loop
    # Load current evolution tracker to continue from where we left off
    evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
    if evolution_tracker_path.exists():
        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
            evolution_tracker = json.load(f)
        generation_count = evolution_tracker.get("total_generations", 0)
        logger.info("Continuing from generation %d (loaded from EvolutionTracker)", generation_count)
    else:
        generation_count = 0
        logger.info("Starting fresh from generation 0 (no existing EvolutionTracker)")
    
    while True:
        generation_count += 1
        logger.info("=== Starting Generation %d ===", generation_count)
        
        # Phase 4: Evolution (Now enabled and optimized)
        with PerformanceLogger(logger, "Evolution Phase"):
            try:
                from ea import get_run_evolution
                run_evolution = get_run_evolution()
                logger.info("Running optimized evolution on population...")
                # Get generation data from evolution engine
                from ea import get_EvolutionEngine
                
                # Load population for evolution from elites.json
                from utils.population_io import load_elites
                elites_path = str(get_outputs_path() / "elites.json")
                population = load_elites(elites_path, log_file=log_file)
                logger.info("Loaded %d genomes from elites.json for evolution", len(population))
                
                EvolutionEngine = get_EvolutionEngine()
                engine = EvolutionEngine(north_star_metric, log_file, current_cycle=generation_count, max_variants=10)
                engine.genomes = population
                engine.update_next_id()
                
                # Get generation data with parent information
                generation_data = engine.generate_variants_global()
                
                logger.info("Generated %d variants (mutation: %d, crossover: %d) globally", 
                           generation_data["variants_created"], 
                           generation_data["mutation_variants"], 
                           generation_data["crossover_variants"])
                
                # Save updated population after evolution
                with PerformanceLogger(logger, "Save Population After Evolution"):
                    try:
                        from utils import get_population_io
                        _, _, _, save_population, _, _, _, _, _, _, _, _ = get_population_io()
                        # Save to elites.json
                        elites_path = str(get_outputs_path() / "elites.json")
                        save_population(engine.genomes, elites_path, logger=logger)
                        logger.debug("Saved updated elites after global evolution")
                    except Exception as e:
                        logger.error("Failed to save population after evolution: %s", e, exc_info=True)
                        raise
            except Exception as e:
                logger.error("Evolution failed: %s", e, exc_info=True)
                break

        # Phase 5: Post-Evolution Generation and Evaluation (Optimized)
        with PerformanceLogger(logger, "Post-Evolution Processing"):
            try:
                logger.info("Processing new variants post-evolution...")
                
                # Reload population to get new variants from elites.json
                from utils.population_io import load_elites
                elites_path = str(get_outputs_path() / "elites.json")
                population = load_elites(elites_path, log_file=log_file)
                logger.info("Loaded %d genomes from elites.json for post-evolution processing", len(population))

                # Check for pending genomes
                pending_generation = [g for g in population if g.get("status") == "pending_generation"]
                pending_evaluation = [g for g in population if g.get("status") == "pending_evaluation"]
                
                logger.info("Found %d genomes pending generation, %d pending evaluation", 
                           len(pending_generation), len(pending_evaluation))
                
                # Process pending generation
                if pending_generation:
                    logger.info("Generating responses for new variants...")
                    # Use dynamic batch size from generator's config
                    # Process elites.json directly
                    elites_path = str(get_outputs_path() / "elites.json")
                    generator.process_population(pop_path=elites_path)  # Will use config batch size automatically
                    
                    # Process pending evaluation
                    logger.info("Evaluating new responses...")
                    # Evaluate elites.json directly
                    elites_path = str(get_outputs_path() / "elites.json")
                    run_moderation_on_population(
                        pop_path=elites_path,
                        log_file=log_file,
                        north_star_metric=north_star_threshold,
                        moderation_methods=moderation_methods
                    )
                    
                    # Update EvolutionTracker with actual scores from this generation
                    logger.info("Updating EvolutionTracker with generation %d results...", generation_count)
                    from ea import get_update_evolution_tracker_with_generation_global
                    
                    update_evolution_tracker_with_generation_global = get_update_evolution_tracker_with_generation_global()
                    
                    # Reload population to get updated scores from elites.json
                    from utils.population_io import load_elites
                    elites_path = str(get_outputs_path() / "elites.json")
                    population = load_elites(elites_path, log_file=log_file)
                    logger.info("Loaded %d genomes from elites.json for evolution tracker update", len(population))
                    
                    # Use generation_data from evolution engine (includes parent information)
                    # generation_data already contains: generation_number, parents, variants_created, mutation_variants, crossover_variants
                    
                    # Load current evolution tracker
                    evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
                    if evolution_tracker_path.exists():
                        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                            evolution_tracker = json.load(f)
                    else:
                        evolution_tracker = {
                            "scope": "global",
                            "status": "not_complete",
                            "total_generations": 1,
                            "generations": []
                        }
                    
                    # Update tracker with actual scores from this generation (includes parent info)
                    update_evolution_tracker_with_generation_global(
                        generation_data, 
                        evolution_tracker, 
                        logger, 
                        population, 
                        north_star_metric
                    )
                    logger.info("EvolutionTracker updated with generation %d results", generation_count)
                    
                    # Redistribute elites to population if elites exceed threshold
                    logger.info("Checking if elites redistribution is needed...")
                        # Elites redistribution logic removed as per update
                
                
            except Exception as e:
                logger.error("Post-evolution processing failed: %s", e, exc_info=True)

        # Check stopping conditions AFTER evolution
        with PerformanceLogger(logger, "Stopping Conditions Check"):
            try:
                # Check generation limit - should stop AFTER completing generation N
                if max_generations is not None and generation_count >= max_generations:
                    logger.info("Maximum generation limit (%d) reached. Stopping pipeline.", max_generations)
                    break
                    
                # Check if ALL prompts have reached the threshold (not just any single prompt)
                from utils.population_io import load_elites
                elites_path = str(get_outputs_path() / "elites.json")
                population = load_elites(elites_path, log_file=log_file)
                logger.info("Loaded %d genomes from elites.json for stopping conditions check", len(population))
                
                # Check global population status
                total_genomes = len([g for g in population if g is not None])
                completed_genomes = [g for g in population if g is not None and g.get("status") == "complete"]
                
                # Get the highest score achieved across all genomes for logging
                max_score = max([
                    _extract_north_star_score(g, north_star_metric) 
                    for g in population if g is not None
                ], default=0)
                
                logger.info("Progress check - Completed genomes: %d/%d, Max %s score: %.4f", 
                           len(completed_genomes), total_genomes, north_star_metric, max_score)
                
                # Stop when all genomes are complete or threshold is achieved
                if len(completed_genomes) == total_genomes and total_genomes > 0:
                    logger.info("ALL genomes (%d) have achieved the threshold! Stopping pipeline.", total_genomes)
                    logger.info("Final max %s score: %.4f", north_star_metric, max_score)
                    break
                elif len(completed_genomes) > 0:
                    logger.info("%d/%d genomes completed. Continuing evolution for remaining genomes...", 
                               len(completed_genomes), total_genomes)
                    
            except Exception as e:
                logger.error("Failed to check stopping conditions: %s", e, exc_info=True)

        # Phase 5: Sort Elites After Evaluation (now outside Phase 4)
        with PerformanceLogger(logger, "Sort Elites After Evaluation"):
            try:
                logger.info("Sorting elites after evaluation by %s DESC, generation DESC, id DESC...", north_star_metric)
                # Load elites, sort them, and save back
                from utils.population_io import load_elites, save_elites, sort_population_by_elite_criteria
                elites_path = str(get_outputs_path() / "elites.json")
                elites = load_elites(elites_path, log_file=log_file)
                sorted_elites = sort_population_by_elite_criteria(elites, north_star_metric, log_file=log_file)
                save_elites(sorted_elites, elites_path, log_file=log_file)
                logger.info("Elites sorted and saved successfully")
            except Exception as e:
                logger.error("Failed to sort elites after evaluation: %s", e, exc_info=True)

        # Generation summary
        with PerformanceLogger(logger, "Generation Summary"):
            try:
                from utils.population_io import load_elites
                elites_path = str(get_outputs_path() / "elites.json")
                population = load_elites(elites_path, log_file=log_file)
                logger.info("Loaded %d genomes from elites.json for generation summary", len(population))
                
                total_genomes = len(population)
                completed = len([g for g in population if g is not None and g.get("status") == "complete"])
                pending_evolution = len([g for g in population if g is not None and g.get("status") == "pending_evolution"])
                
                # Get global population statistics
                max_score = max([
                    _extract_north_star_score(g, north_star_metric) 
                    for g in population if g is not None
                ], default=0)
                
                logger.info("Generation %d Summary:", generation_count)
                logger.info("  - Total genomes: %d", total_genomes)
                logger.info("  - Completed genomes: %d", completed)
                logger.info("  - Pending evolution: %d", pending_evolution)
                logger.info("  - Max %s score: %.4f", north_star_metric, max_score)
                
                # Evolution status is now tracked in EvolutionTracker.json
                    
            except Exception as e:
                logger.error("Failed to generate generation summary: %s", e, exc_info=True)

        # Check if we should continue - improved stopping condition
        if pending_evolution == 0:
            logger.info("No genomes pending evolution. Stopping.")
            break
        elif max_generations is not None and generation_count >= max_generations:
            logger.info("Maximum generation limit (%d) reached. Stopping pipeline.", max_generations)
            break
        elif completed == 0 and pending_evolution > 0:
            # If no genomes are completed but some are pending evolution, continue
            logger.info("Continuing evolution with %d pending genomes.", pending_evolution)
            continue

    # ============================================================================
    # SECTION 5: PIPELINE COMPLETION AND FINAL ANALYSIS
    # ============================================================================
    
    total_time = time.time() - start_time
    logger.info("=== Pipeline Completed ===")
    logger.info("Total execution time: %.2f seconds", total_time)
    logger.info("Total generations: %d", generation_count)
    logger.info("Average time per generation: %.2f seconds", total_time/max(generation_count, 1))

    # Final population analysis
    with PerformanceLogger(logger, "Final Analysis"):
        try:
            # Import final statistics function
            from ea import get_create_final_statistics_with_tracker
            
            # Load population for basic stats
            from utils.population_io import load_elites
            elites_path = str(get_outputs_path() / "elites.json")
            population = load_elites(elites_path, log_file=log_file)
            logger.info("Loaded %d genomes from elites.json for final analysis", len(population))
            
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
                       help="North star metric threshold for stopping evolution (default: 0.99)")
    parser.add_argument("--moderation-methods", nargs="+", choices=["google", "all"], default=["google"],
                       help="Moderation methods to use: 'google' (Perspective API), 'all' (google only). Default: google")
    parser.add_argument("model_names", nargs="*", default=[], 
                       help="Model names to use (currently not used)")
    args = parser.parse_args()
    
    try:
        main(model_names=args.model_names, max_generations=args.generations, 
             north_star_threshold=args.threshold, moderation_methods=args.moderation_methods)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)