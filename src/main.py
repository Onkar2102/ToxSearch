import sys
import time
import json
import os

from typing import Optional
from pathlib import Path
from datetime import datetime

from utils.device_utils import get_optimal_device, get_device_info

# Detect and log device at startup
DEVICE = get_optimal_device()
DEVICE_INFO = get_device_info()
print(f"[DEVICE] Using device: {DEVICE}")
print(f"[DEVICE INFO] {DEVICE_INFO}")

from utils import get_custom_logging
from utils.population_io import calculate_and_update_population_thresholds, redistribute_population_with_threshold, update_population_index_single_file, remove_worse_performing_genomes, remove_worse_performing_genomes_from_all_files, update_adaptive_selection_logic
from gne import get_run_moderation_on_population
from utils import get_population_io
from ea.run_evolution import run_evolution, distribute_genomes_by_threshold
from ea import get_create_final_statistics_with_tracker
import yaml

# ============================================================================
# SECTION 1: SYSTEM UTILITIES
# ============================================================================

# Import system utilities from utils module
from utils import get_system_utils
get_project_root, get_config_path, get_data_path, get_outputs_path, _extract_north_star_score, initialize_system = get_system_utils()

# ============================================================================
# SECTION 2: MODEL CONFIGURATION UPDATES
# ============================================================================

def _is_gguf_path(value: str) -> bool:
    """Return True if the given value looks like a direct GGUF file path."""
    p = Path(value)
    return str(value).lower().endswith(".gguf") and (p.is_absolute() or str(value).startswith("./") or str(value).startswith("models/"))

def update_model_configs(rg_model, pg_model, logger):
    """Update configuration files with selected models.

    Resolves the concrete .gguf file for each alias by scanning models/{alias}.
    
    Model Quantization Preference Order (Performance vs. Size Trade-offs):
    Q4_K_M → Q4_K_S → Q4_0 → Q5_K_M → Q5_K_S → Q4_K → Q3_K_M → Q3_K_L → Q2_K
    
    - Q4_K_M: Best balance of quality and speed (recommended)
    - Q4_K_S: Smaller size, slightly lower quality
    - Q5_K_M: Higher quality, larger size, slower inference
    - Q3_K_M: Lower quality, smaller size, faster inference
    - Q2_K: Lowest quality, smallest size, fastest inference
    """
    try:
        logger.info(f"Updating config files with models: RG={rg_model}, PG={pg_model}")

        def resolve_model_entry(value: str) -> Optional[str]:
            """
            Resolve the provided model value to a concrete path string.
            - If it's a direct .gguf path, return it as-is (relative or absolute)
            - Otherwise, treat it as an alias directory under models/ and pick a file by preference
            """
            if not value:
                return None
            if _is_gguf_path(value):
                return value

            alias = value
            base_dir = get_project_root() / "models" / alias
            if not base_dir.exists():
                logger.warning("Model alias directory not found: %s", base_dir)
                return None
            ggufs = sorted([p for p in base_dir.glob("*.gguf")], key=lambda p: p.name)
            if not ggufs:
                logger.warning("No GGUF files found under: %s", base_dir)
                return None
            pref_order = [
                "Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_K_S", "Q4_K", "Q3_K_M", "Q3_K_L", "Q2_K"
            ]
            for pref in pref_order:
                for f in ggufs:
                    if pref in f.name:
                        rel = Path("./models") / alias / f.name
                        logger.info("Resolved %s -> %s", alias, rel)
                        return str(rel)

        rg_file = resolve_model_entry(rg_model)
        pg_file = resolve_model_entry(pg_model)

        # Validate that we have at least one model resolved
        if not rg_file and not pg_file:
            logger.error("No models could be resolved for RG=%s, PG=%s", rg_model, pg_model)
            raise ValueError(f"No models could be resolved for RG={rg_model}, PG={pg_model}")

        # Update RGConfig.yaml
        rg_config_path = get_config_path() / "RGConfig.yaml"
        if rg_config_path.exists():
            with open(rg_config_path, 'r') as f:
                rg_config = yaml.safe_load(f) or {}

            if rg_file:
                rg_section = rg_config.get("response_generator", {})
                rg_section["name"] = rg_file
                rg_config["response_generator"] = rg_section
                with open(rg_config_path, 'w') as f:
                    yaml.dump(rg_config, f, default_flow_style=False)
                logger.info("Updated RGConfig.yaml with file: %s", rg_file)
            else:
                logger.warning("Skipped RGConfig.yaml update; no file resolved for alias '%s'", rg_model)

        # Update PGConfig.yaml
        pg_config_path = get_config_path() / "PGConfig.yaml"
        if pg_config_path.exists():
            with open(pg_config_path, 'r') as f:
                pg_config = yaml.safe_load(f) or {}

            if pg_file:
                pg_section = pg_config.get("prompt_generator", {})
                pg_section["name"] = pg_file
                pg_config["prompt_generator"] = pg_section
                with open(pg_config_path, 'w') as f:
                    yaml.dump(pg_config, f, default_flow_style=False)
                logger.info("Updated PGConfig.yaml with file: %s", pg_file)
            else:
                logger.warning("Skipped PGConfig.yaml update; no file resolved for alias '%s'", pg_model)

        logger.info("Model configuration updates completed successfully")

    except Exception as e:
        logger.error(f"Failed to update model configurations: {e}")
        raise

# ============================================================================
# SECTION 4: MAIN EXECUTION PIPELINE
# ============================================================================

def main(max_generations=None, north_star_threshold=0.99, moderation_methods=None, threshold_percentage=25, rg_model="models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf", pg_model="models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf", operators="all", max_variants=1, elites_threshold=25, removal_threshold=5, stagnation_limit=5):
    """
    Main entry point for evolutionary text generation with toxicity optimization.
    
    Runs the evolutionary algorithm to generate and evolve text prompts, optimizing
    for low toxicity scores using content moderation (Google Perspective API by
    default). Evolution continues until prompts meet the configured toxicity
    threshold or the maximum number of generations is reached.
    
    Args:
        max_generations (int, optional): Maximum evolution generations to run.
            If None, runs until north star threshold is achieved.
        north_star_threshold (float): Target toxicity score for stopping evolution.
            Default is 0.99 (very low toxicity). Evolution stops when the configured
            stopping criteria are met (typically when prompts reach the threshold
            or the generation limit is reached).
        moderation_methods (List[str], optional): Moderation APIs to use.
            If None, uses ['google'] for evaluation.
        threshold_percentage (int): Percentage for elite threshold calculation.
            Default is 25 (top 25%). Elite threshold = (100-threshold_percentage)/100 * population_max_toxicity.
        operators (str): Operator configuration mode. Options:
            - "ie": Only InformedEvolution operator enabled, all other mutation-crossover operators disabled
            - "cm": All mutation-crossover operators enabled, InformedEvolution disabled
            - "all": All operators enabled (default)
        max_variants (int): Maximum number of variants to generate per evolution cycle.
            Controls how many times the evolution cycle runs. Default is 1.
    
    Returns:
        None
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
    
    start_time = time.time()
    north_star_metric = "toxicity"  # Primary optimization metric for evolution
    logger.info("North star metric for optimization set to: %s", north_star_metric)
    logger.info("North star threshold for stopping: %.4f (low toxicity target)", north_star_threshold)
    logger.info("Using moderation methods: %s", moderation_methods)
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
        run_moderation_on_population = get_run_moderation_on_population()
        logger.info("Evaluating generated responses using moderation (%s)...", " + ".join(moderation_methods))
        temp_path = str(get_outputs_path() / "temp.json")
        run_moderation_on_population(
            pop_path=temp_path,
            log_file=log_file,
            north_star_metric=north_star_metric,
            moderation_methods=moderation_methods
        )
        logger.debug("Evaluation completed on temp.json with moderation scores.")
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        return

    # Update Evolution Tracker for Generation 0 and Calculate Elite Threshold
    try:
        logger.info("Updating evolution tracker for generation 0 and calculating elite threshold...")
        
        # Load evaluated genomes from temp.json
        temp_path = get_outputs_path() / "temp.json"
        if not temp_path.exists():
            raise FileNotFoundError(f"temp.json not found: {temp_path}")
            
        with open(temp_path, 'r', encoding='utf-8') as f:
            evaluated_genomes = json.load(f)
        
        # Calculate dynamic threshold using centralized function
        temp_path_str = str(temp_path)
        evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
        evolution_tracker_path_str = str(evolution_tracker_path)
        
        threshold_results = calculate_and_update_population_thresholds(
            temp_path=temp_path_str,
            evolution_tracker_path=evolution_tracker_path_str,
            north_star_metric=north_star_metric,
            threshold_percentage=elites_threshold,
            logger=logger,
            log_file=log_file
        )
        
        # Only proceed if threshold calculation was successful
        if not threshold_results.get("skipped", False):
            # Update adaptive selection logic
            logger.info("Updating adaptive selection logic...")
            outputs_path = str(get_outputs_path())
            adaptive_results = update_adaptive_selection_logic(
                outputs_path=outputs_path,
                current_max_toxicity=threshold_results["max_toxicity_score"],
                previous_max_toxicity=0.0,  # Initial population - no previous value
                stagnation_limit=stagnation_limit,
                north_star_metric=north_star_metric,
                logger=logger,
                log_file=log_file
            )
            logger.info("Adaptive selection updated: mode=%s, generations_since_improvement=%d, avg_fitness=%.4f, slope=%.4f",
                       adaptive_results["selection_mode"], adaptive_results["generations_since_improvement"],
                       adaptive_results["current_avg_fitness"], adaptive_results["slope_of_avg_fitness"])
            
            phase_3b_results = {
                "max_toxicity_score": threshold_results["max_toxicity_score"],
                "elite_threshold": threshold_results["elite_threshold"],
                "best_genome_id": threshold_results["best_genome_id"]
            }
            logger.info("Evolution tracker updated for generation 0.")
        else:
            logger.warning("Skipping threshold calculation for generation 0 - no evaluated genomes found")
            phase_3b_results = {
                "max_toxicity_score": threshold_results["max_toxicity_score"],
                "elite_threshold": threshold_results["elite_threshold"],
                "best_genome_id": threshold_results["best_genome_id"]
            }
    except Exception as e:
        logger.error("Evolution tracker update failed: %s", e, exc_info=True)
        return

    # Phase 4: Finalize Initial Population (Split temp.json into elites and population)
    try:
        _, _, _, _, _, _, _, _, _, _, _, _, finalize_initial_population = get_population_io()
        logger.info("Finalizing initial population after evaluation...")
        elite_threshold = phase_3b_results["elite_threshold"] if phase_3b_results["elite_threshold"] is not None else 0.5
        
        finalize_initial_population(
            output_path=str(get_outputs_path()),
            elite_threshold=elite_threshold,
            north_star_metric=north_star_metric,
            log_file=log_file
        )
        logger.info("Initial population finalized - elites and population separated.")
        logger.info("Used dynamic threshold: %.4f for initial population distribution", elite_threshold)
        
        # Remove worse performing genomes from all files for generation 0
        logger.info("Removing worse performing genomes from all files...")
        outputs_path = str(get_outputs_path())
        removal_results = remove_worse_performing_genomes_from_all_files(
            outputs_path=outputs_path,
            population_max_toxicity=phase_3b_results["max_toxicity_score"],
            removal_threshold_percentage=removal_threshold,
            north_star_metric=north_star_metric,
            logger=logger,
            log_file=log_file
        )
        logger.info("Genome archiving from all files completed: %d total archived, %d total remaining", 
                   removal_results["archived_count_total"], removal_results["remaining_count_total"])
        
        # Now redistribute remaining genomes between elites and non_elites based on elite_threshold
        logger.info("Redistributing remaining genomes between elites and non_elites...")
        redistribution_result = redistribute_population_with_threshold(
            elite_threshold=phase_3b_results["elite_threshold"],
            north_star_metric=north_star_metric,
            logger=logger,
            log_file=log_file
        )
        logger.info("Final redistribution completed: %d elites, %d non_elites", 
                   redistribution_result["elites_count"], 
                   redistribution_result.get("total_count", 0) - redistribution_result["elites_count"])
        
        # Update generation 0's metrics in EvolutionTracker
        try:
            evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
            with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                tracker = json.load(f)
            
            # Calculate per-generation fitness metrics for generation 0
            from utils.population_io import _extract_north_star_score
            elites_path = get_outputs_path() / "elites.json"
            non_elites_path = get_outputs_path() / "non_elites.json"
            
            avg_fitness_elites = 0.0001
            avg_fitness_non_elites = 0.0001
            
            # Calculate average fitness for elites
            if elites_path.exists():
                with open(elites_path, 'r', encoding='utf-8') as f:
                    elites_genomes = json.load(f)
                if elites_genomes:
                    elite_scores = [_extract_north_star_score(g, north_star_metric) for g in elites_genomes]
                    elite_scores = [s for s in elite_scores if s > 0]
                    if elite_scores:
                        avg_fitness_elites = round(sum(elite_scores) / len(elite_scores), 4)
            
            # Calculate average fitness for non_elites
            if non_elites_path.exists():
                with open(non_elites_path, 'r', encoding='utf-8') as f:
                    non_elites_genomes = json.load(f)
                if non_elites_genomes:
                    non_elite_scores = [_extract_north_star_score(g, north_star_metric) for g in non_elites_genomes]
                    non_elite_scores = [s for s in non_elite_scores if s > 0]
                    if non_elite_scores:
                        avg_fitness_non_elites = round(sum(non_elite_scores) / len(non_elite_scores), 4)
            
            # Calculate removal threshold for generation 0
            removal_threshold_value = round((removal_threshold * phase_3b_results["max_toxicity_score"]) / 100, 4)
            
            # Calculate avg_fitness_generation for generation 0 (combined elites + non_elites after distribution)
            all_scores_gen0 = []
            if elite_scores:
                all_scores_gen0.extend(elite_scores)
            if non_elite_scores:
                all_scores_gen0.extend(non_elite_scores)
            avg_fitness_generation_gen0 = round(sum(all_scores_gen0) / len(all_scores_gen0), 4) if all_scores_gen0 else 0.0
            
            # Update generation 0 entry with all metrics
            for gen in tracker.get("generations", []):
                if gen.get("generation_number") == 0:
                    gen["elites_count"] = redistribution_result["elites_count"]
                    gen["removal_threshold"] = removal_threshold_value
                    gen["avg_fitness_elites"] = avg_fitness_elites
                    gen["avg_fitness_non_elites"] = avg_fitness_non_elites
                    # For generation 0: avg_fitness_generation = elites + non_elites (no variants yet)
                    gen["avg_fitness_generation"] = avg_fitness_generation_gen0
                    # For generation 0: avg_fitness = avg_fitness_generation (no variants to include)
                    gen["avg_fitness"] = avg_fitness_generation_gen0
                    gen["min_score_variants"] = 0.0001  # No variants generated yet
                    # No variants generated yet - set all variant stats to default
                    gen["max_score_variants"] = 0.0001
                    gen["avg_fitness_variants"] = 0.0001
                    break
            
            # Save updated tracker
            with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(tracker, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Updated generation 0 with comprehensive metrics: "
                       f"elites_count={redistribution_result['elites_count']}, "
                       f"removal_threshold={removal_threshold_value:.4f}, "
                       f"avg_fitness_elites={avg_fitness_elites:.4f}, "
                       f"avg_fitness_non_elites={avg_fitness_non_elites:.4f}")
        except Exception as e:
            logger.warning(f"Failed to update generation 0 metrics in EvolutionTracker: {e}")
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
    
    while max_generations is None or generation_count < max_generations:
        generation_count += 1
        logger.info("=== Starting Generation %d ===", generation_count)
        
        # Phase 4: Evolution
        operator_statistics = {}  # Initialize operator statistics
        try:
            logger.info("Running optimized evolution on population...")

            # Use RunEvolution.py as the evolution driver
            evolution_result = run_evolution(
                north_star_metric=north_star_metric,
                log_file=log_file,
                threshold=north_star_threshold,
                current_cycle=generation_count,
                max_variants=max_variants,
                operators=operators
            )
            
            # Extract operator statistics from evolution result
            operator_statistics = evolution_result.get("operator_statistics", {}) if evolution_result else {}
            logger.info(f"Evolution completed with operator statistics: {operator_statistics}")

            logger.info("Evolution generation completed successfully")

        except Exception as e:
            logger.error("Evolution failed: %s", e, exc_info=True)
            break

        # Phase 5: Post-Evolution Generation and Evaluation
        logger.info("Processing evolved variants post-evolution...")
        
        try:
            temp_path = str(get_outputs_path() / "temp.json")
            logger.info("Generating text for all pending genomes in temp.json...")
            response_generator.process_population(pop_path=temp_path)
            logger.info("Text generation for temp.json completed.")

            run_moderation_on_population = get_run_moderation_on_population()
            logger.info("Evaluating all genomes in temp.json...")
            run_moderation_on_population(
                pop_path=temp_path,
                log_file=log_file,
                north_star_metric=north_star_metric,
                moderation_methods=moderation_methods
            )
            logger.info("Evaluation of temp.json completed.")
            
            # Count variants from temp.json BEFORE distribution (while temp.json still has variants)
            variant_counts = {"variants_created": 0, "mutation_variants": 0, "crossover_variants": 0}
            temp_path = get_outputs_path() / "temp.json"
            if temp_path.exists():
                with open(temp_path, 'r', encoding='utf-8') as f:
                    temp_variants = json.load(f)
                
                mutation_count = sum(1 for v in temp_variants if v and v.get("variant_type") == "mutation")
                crossover_count = sum(1 for v in temp_variants if v and v.get("variant_type") == "crossover")
                total_count = mutation_count + crossover_count
                
                variant_counts = {
                    "variants_created": total_count,
                    "mutation_variants": mutation_count,
                    "crossover_variants": crossover_count
                }
                
                logger.info(f"Generation {generation_count}: {total_count} variants created ({mutation_count} mutation, {crossover_count} crossover)")
            
            # Update EvolutionTracker with generation-specific data (variants, scores, etc.)
            try:
                logger.info("Updating EvolutionTracker with generation data...")
                from ea import get_update_evolution_tracker_with_generation_global
                update_evolution_tracker_with_generation_global = get_update_evolution_tracker_with_generation_global()
                
                # Load current evolution tracker
                evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
                with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                    evolution_tracker = json.load(f)
                
                # Prepare generation data with variant counts
                generation_data = {
                    "generation_number": generation_count,
                    "variants_created": variant_counts["variants_created"],
                    "mutation_variants": variant_counts["mutation_variants"],
                    "crossover_variants": variant_counts["crossover_variants"]
                }
                
                # Load population from all files for generation analysis
                from utils.population_io import load_population
                outputs_path = get_outputs_path()
                
                # Load genomes from all files
                all_genomes = []
                for file_name in ["temp.json", "elites.json", "non_elites.json"]:
                    file_path = outputs_path / file_name
                    if file_path.exists():
                        file_genomes = load_population(str(file_path), logger=logger)
                        all_genomes.extend(file_genomes)
                
                logger.info(f"Loaded {len(all_genomes)} total genomes for generation analysis")
                
                # Update EvolutionTracker with generation data
                update_evolution_tracker_with_generation_global(
                    generation_data=generation_data,
                    evolution_tracker=evolution_tracker,
                    logger=logger,
                    population=all_genomes,
                    north_star_metric=north_star_metric
                )
                
                logger.info("EvolutionTracker updated with generation data successfully")
                
            except Exception as e:
                logger.error("Failed to update EvolutionTracker with generation data: %s", e, exc_info=True)
            
            try:
                logger.info("Updating population_max_toxicity and recalculating threshold after evaluation...")
                
                # Get previous population_max_toxicity for adaptive selection comparison
                previous_max_toxicity = 0.0001
                evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
                if evolution_tracker_path.exists():
                    with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                        tracker = json.load(f)
                    previous_max_toxicity = tracker.get("population_max_toxicity", 0.0001)
                
                # Calculate new dynamic threshold using centralized function
                temp_path_str = str(temp_path)
                elites_path = get_outputs_path() / "elites.json"
                elites_path_str = str(elites_path)
                evolution_tracker_path_str = str(evolution_tracker_path)
                
                threshold_results = calculate_and_update_population_thresholds(
                    elites_path=elites_path_str,
                    temp_path=temp_path_str,
                    evolution_tracker_path=evolution_tracker_path_str,
                    north_star_metric=north_star_metric,
                    threshold_percentage=elites_threshold,
                    logger=logger,
                    log_file=log_file
                )
                
                # Only proceed with removal and distribution if threshold calculation was successful
                if not threshold_results.get("skipped", False):
                    # Update adaptive selection logic
                    logger.info("Updating adaptive selection logic...")
                    outputs_path = str(get_outputs_path())
                    adaptive_results = update_adaptive_selection_logic(
                        outputs_path=outputs_path,
                        current_max_toxicity=threshold_results["max_toxicity_score"],
                        previous_max_toxicity=previous_max_toxicity,
                        stagnation_limit=stagnation_limit,
                        north_star_metric=north_star_metric,
                        logger=logger,
                        log_file=log_file
                    )
                    logger.info("Adaptive selection updated: mode=%s, generations_since_improvement=%d, avg_fitness=%.4f, slope=%.4f",
                               adaptive_results["selection_mode"], adaptive_results["generations_since_improvement"],
                               adaptive_results["current_avg_fitness"], adaptive_results["slope_of_avg_fitness"])
                    
                    # Calculate variant statistics from temp.json BEFORE distribution
                    temp_path = get_outputs_path() / "temp.json"
                    max_score_variants = 0.0001
                    min_score_variants = 0.0001
                    avg_fitness_variants = 0.0001
                    try:
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            temp_variants = json.load(f)
                        
                        if temp_variants:
                            from utils.population_io import _extract_north_star_score
                            scores = [_extract_north_star_score(v, north_star_metric) for v in temp_variants if v]
                            # DO NOT filter out zero scores - we want ALL variant scores for accurate statistics
                            
                            if scores:
                                max_score_variants = round(max(scores), 4)
                                min_score_variants = round(min(scores), 4)
                                avg_fitness_variants = round(sum(scores) / len(scores), 4)
                                logger.info(f"Variant statistics from temp.json ({len(scores)} variants): "
                                          f"max={max_score_variants:.4f}, min={min_score_variants:.4f}, avg={avg_fitness_variants:.4f}")
                            else:
                                logger.warning("No variants in temp.json to calculate statistics")
                        else:
                            logger.info("temp.json is empty - using default values (0.0001) for variant statistics")
                    except Exception as e:
                        logger.warning(f"Failed to calculate variant statistics: {e}")
                    
                    # Calculate removal threshold for distribution
                    removal_threshold_value = round((removal_threshold * threshold_results["max_toxicity_score"]) / 100, 4)
                    
                    # Calculate avg_fitness BEFORE distribution (includes variants + existing elites + existing non_elites)
                    # This must happen before temp.json is emptied by distribution
                    all_genomes_for_avg_fitness = []
                    
                    # Add variants from temp.json (before distribution)
                    if temp_path.exists():
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            temp_variants_for_avg = json.load(f)
                        if temp_variants_for_avg:
                            from utils.population_io import _extract_north_star_score
                            temp_scores = [_extract_north_star_score(v, north_star_metric) for v in temp_variants_for_avg if v]
                            all_genomes_for_avg_fitness.extend(temp_scores)
                    
                    # Add existing elites (before distribution)
                    elites_path = get_outputs_path() / "elites.json"
                    if elites_path.exists():
                        with open(elites_path, 'r', encoding='utf-8') as f:
                            existing_elites = json.load(f)
                        if existing_elites:
                            existing_elite_scores = [_extract_north_star_score(g, north_star_metric) for g in existing_elites]
                            all_genomes_for_avg_fitness.extend(existing_elite_scores)
                    
                    # Add existing non_elites (before distribution)
                    non_elites_path = get_outputs_path() / "non_elites.json"
                    if non_elites_path.exists():
                        with open(non_elites_path, 'r', encoding='utf-8') as f:
                            existing_non_elites = json.load(f)
                        if existing_non_elites:
                            existing_non_elite_scores = [_extract_north_star_score(g, north_star_metric) for g in existing_non_elites]
                            all_genomes_for_avg_fitness.extend(existing_non_elite_scores)
                    
                    # Calculate avg_fitness from all genomes (before distribution)
                    avg_fitness = round(sum(all_genomes_for_avg_fitness) / len(all_genomes_for_avg_fitness), 4) if all_genomes_for_avg_fitness else 0.0
                    
                    # Step 0: Distribute genomes from temp.json to elites.json and non_elites.json (and under_performing.json)
                    logger.info("Distributing genomes from temp.json to elites, non_elites, and under_performing...")
                    distribution_result = distribute_genomes_by_threshold(
                        temp_path=temp_path,
                        elite_threshold=threshold_results["elite_threshold"],
                        removal_threshold=removal_threshold_value,
                        north_star_metric=north_star_metric,
                        logger=logger
                    )
                    logger.info("Genome distribution completed: %d elites moved, %d non_elites moved, %d under_performing archived", 
                               distribution_result["elites_moved"], distribution_result["population_moved"], 
                               distribution_result.get("under_performing_moved", 0))
                    
                    # Step 1: Remove worse performing genomes from all files
                    logger.info("Removing worse performing genomes from all files...")
                    removal_results = remove_worse_performing_genomes_from_all_files(
                        outputs_path=outputs_path,
                        population_max_toxicity=threshold_results["max_toxicity_score"],
                        removal_threshold_percentage=removal_threshold,
                        north_star_metric=north_star_metric,
                        logger=logger,
                        log_file=log_file
                    )
                    logger.info("Genome archiving from all files completed: %d total archived, %d total remaining", 
                               removal_results["archived_count_total"], removal_results["remaining_count_total"])
                    
                    # Step 2: Redistribute remaining genomes between elites and non_elites
                    logger.info("Redistributing remaining genomes between elites and non_elites...")
                    redistribution_result = redistribute_population_with_threshold(
                        elite_threshold=threshold_results["elite_threshold"],
                        north_star_metric=north_star_metric,
                        logger=logger,
                        log_file=log_file
                    )
                    logger.info("Final redistribution completed: %d elites, %d non_elites", 
                               redistribution_result["elites_count"], redistribution_result.get("total_count", 0) - redistribution_result["elites_count"])
                    
                    # Update current generation's metrics in EvolutionTracker
                    try:
                        evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
                        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                            tracker = json.load(f)
                        
                        # Calculate per-generation fitness metrics
                        from utils.population_io import _extract_north_star_score
                        elites_path = get_outputs_path() / "elites.json"
                        non_elites_path = get_outputs_path() / "non_elites.json"
                        
                        avg_fitness_elites = 0.0
                        avg_fitness_non_elites = 0.0
                        elite_scores = []  # Initialize to avoid undefined variable errors
                        non_elite_scores = []  # Initialize to avoid undefined variable errors
                        
                        # Calculate average fitness for elites
                        if elites_path.exists():
                            with open(elites_path, 'r', encoding='utf-8') as f:
                                elites_genomes = json.load(f)
                            if elites_genomes:
                                elite_scores = [_extract_north_star_score(g, north_star_metric) for g in elites_genomes]
                                elite_scores = [s for s in elite_scores if s > 0]
                                if elite_scores:
                                    avg_fitness_elites = round(sum(elite_scores) / len(elite_scores), 4)
                        
                        # Calculate average fitness for non_elites
                        if non_elites_path.exists():
                            with open(non_elites_path, 'r', encoding='utf-8') as f:
                                non_elites_genomes = json.load(f)
                            if non_elites_genomes:
                                non_elite_scores = [_extract_north_star_score(g, north_star_metric) for g in non_elites_genomes]
                                non_elite_scores = [s for s in non_elite_scores if s > 0]
                                if non_elite_scores:
                                    avg_fitness_non_elites = round(sum(non_elite_scores) / len(non_elite_scores), 4)
                        
                        # Calculate avg_fitness_generation (combined elites + non_elites after distribution)
                        all_scores = []
                        if elite_scores:
                            all_scores.extend(elite_scores)
                        if non_elite_scores:
                            all_scores.extend(non_elite_scores)
                        avg_fitness_generation = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
                        
                        # Debug logging to understand the difference
                        logger.info("Fitness calculation debug:")
                        logger.info("  avg_fitness (before distribution): %.4f (includes %d genomes)", avg_fitness, len(all_genomes_for_avg_fitness))
                        logger.info("  avg_fitness_generation (after distribution): %.4f (includes %d genomes)", avg_fitness_generation, len(all_scores))
                        logger.info("  elite_scores count: %d, non_elite_scores count: %d", len(elite_scores), len(non_elite_scores))
                        
                        # avg_fitness was already calculated BEFORE distribution above
                        
                        # Update the current generation entry with all metrics
                        for gen in tracker.get("generations", []):
                            if gen.get("generation_number") == generation_count:
                                gen["elites_count"] = redistribution_result["elites_count"]
                                gen["removal_threshold"] = removal_threshold_value
                                gen["avg_fitness_elites"] = avg_fitness_elites
                                gen["avg_fitness_non_elites"] = avg_fitness_non_elites
                                gen["avg_fitness_generation"] = avg_fitness_generation  # Average of elites + non_elites AFTER distribution
                                # Variant statistics from temp.json (BEFORE distribution)
                                gen["max_score_variants"] = max_score_variants
                                gen["min_score_variants"] = min_score_variants
                                gen["avg_fitness_variants"] = avg_fitness_variants
                                # Add avg_fitness field (calculated BEFORE distribution)
                                gen["avg_fitness"] = avg_fitness
                                # Update operator_statistics with actual data from evolution
                                gen["operator_statistics"] = operator_statistics
                        
                        # Save updated tracker
                        with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                            json.dump(tracker, f, indent=4, ensure_ascii=False)
                        
                        logger.info(f"Updated generation {generation_count} with comprehensive metrics: "
                                   f"elites_count={redistribution_result['elites_count']}, "
                                   f"removal_threshold={removal_threshold_value:.4f}, "
                                   f"avg_fitness_elites={avg_fitness_elites:.4f}, "
                                   f"avg_fitness_non_elites={avg_fitness_non_elites:.4f}, "
                                   f"variants: max={max_score_variants:.4f}, min={min_score_variants:.4f}, avg={avg_fitness_variants:.4f}")
                    except Exception as e:
                        logger.warning(f"Failed to update generation metrics in EvolutionTracker: {e}")
                
                logger.info("Threshold recalculation, removal, and redistribution completed successfully")
                    
                
            except Exception as e:
                logger.error("Threshold recalculation and distribution failed: %s", e, exc_info=True)
            
            # Update population metadata in EvolutionTracker after post-evolution processing
            try:
                update_population_index_single_file(str(get_outputs_path()), 0, logger=logger)
                logger.debug("Updated EvolutionTracker population metadata after post-evolution processing")
            except Exception as e:
                logger.warning("Failed to update EvolutionTracker population metadata: %s", e)
            
        except Exception as e:
            logger.error("Post-evolution processing failed: %s", e, exc_info=True)

    # Log why the evolution loop ended
    if max_generations is not None and generation_count >= max_generations:
        logger.info("Evolution completed: Maximum generation limit (%d) reached.", max_generations)
    else:
        logger.info("Evolution completed: Loop exited due to other conditions.")

    # ============================================================================
    # SECTION 5: PIPELINE COMPLETION AND FINAL ANALYSIS
    # ============================================================================
    
    total_time = time.time() - start_time
    logger.info("=== Pipeline Completed ===")
    logger.info("Total execution time: %.2f seconds", total_time)
    logger.info("Total generations: %d", generation_count)

    # Final Analysis
    try:
        # Import final statistics function
        create_final_statistics_with_tracker = get_create_final_statistics_with_tracker()
        
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
            
            # Save final statistics to file
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
    parser.add_argument("--elites-threshold", type=int, default=25,
                       help="Elite threshold percentage (default: 25)")
    parser.add_argument("--removal-threshold", type=int, default=5,
                       help="Removal threshold percentage for worst performing genomes (default: 5)")
    parser.add_argument("--stagnation-limit", type=int, default=5,
                       help="Number of generations without improvement before switching to explore mode (default: 5)")
    parser.add_argument("--rg", type=str, default="models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                       help="Response generator model: pass a direct .gguf path or an alias under models/")
    parser.add_argument("--pg", type=str, default="models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                       help="Prompt generator model: pass a direct .gguf path or an alias under models/")
    parser.add_argument("--operators", type=str, choices=["ie", "cm", "all"], default="all",
                       help="Operator configuration mode: ie (InformedEvolution only), cm (all except InformedEvolution), all (all operators)")
    parser.add_argument("--max-variants", type=int, default=1,
                       help="Maximum number of variants to generate per evolution cycle. Controls how many times the evolution cycle runs.")
    args = parser.parse_args()
    
    import sys
    try:
        main(max_generations=args.generations, 
             north_star_threshold=args.threshold, moderation_methods=args.moderation_methods,
             threshold_percentage=args.threshold_percentage, rg_model=args.rg, pg_model=args.pg,
             operators=args.operators, max_variants=args.max_variants,
             elites_threshold=args.elites_threshold, removal_threshold=args.removal_threshold,
             stagnation_limit=args.stagnation_limit)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)