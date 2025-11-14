import sys
import time
import json
import os

from typing import Optional
from pathlib import Path
from datetime import datetime

from utils.device_utils import get_optimal_device, get_device_info

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


from utils import get_system_utils
get_project_root, get_config_path, get_data_path, get_outputs_path, _extract_north_star_score, initialize_system = get_system_utils()


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
        logger.info("Updating config files with models: RG=%s, PG=%s", rg_model, pg_model)

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

        if not rg_file and not pg_file:
            logger.error("No models could be resolved for RG=%s, PG=%s", rg_model, pg_model)
            raise ValueError(f"No models could be resolved for RG={rg_model}, PG={pg_model}")

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
                logger.debug("Updated RGConfig.yaml with file: %s", rg_file)
            else:
                logger.warning("Skipped RGConfig.yaml update; no file resolved for alias '%s'", rg_model)

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
                logger.debug("Updated PGConfig.yaml with file: %s", pg_file)
            else:
                logger.warning("Skipped PGConfig.yaml update; no file resolved for alias '%s'", pg_model)

        logger.debug("Model configuration updates completed successfully")

    except Exception as e:
        logger.error("Failed to update model configurations: %s", e)
        raise


def main(max_generations=None, north_star_threshold=0.99, moderation_methods=None, threshold_percentage=25, rg_model="models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf", pg_model="models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf", operators="all", max_variants=1, elites_threshold=25, removal_threshold=5, stagnation_limit=5, seed_file="data/prompt.csv"):
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
    
    get_logger, get_log_filename, log_system_info, _ = get_custom_logging()
    log_file = get_log_filename()
    logger = get_logger("main", log_file)
    
    log_system_info(logger)
    
    if moderation_methods is None:
        moderation_methods = ["google"]
    
    if "all" in moderation_methods:
        moderation_methods = ["google"]
    
    start_time = time.time()
    north_star_metric = "toxicity"
    logger.info("Starting evolutionary run: metric=%s, threshold=%.4f, max_generations=%s", 
                north_star_metric, north_star_threshold, max_generations)

    try:
        update_model_configs(rg_model, pg_model, logger)
    except Exception as e:
        logger.error("Config update failed: %s", e, exc_info=True)
        return

    try:
        response_generator, prompt_generator = initialize_system(logger, log_file, seed_file=seed_file)
    except Exception as e:
        logger.error("System initialization failed: %s", e, exc_info=True)
        return

    try:
        logger.info("Generating responses using response generation model...")
        temp_path = str(get_outputs_path() / "temp.json")
        response_generator.process_population(pop_path=temp_path)
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        return

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
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        return

    try:
        logger.info("Updating evolution tracker for generation 0 and calculating elite threshold...")
        
        temp_path = get_outputs_path() / "temp.json"
        if not temp_path.exists():
            raise FileNotFoundError(f"temp.json not found: {temp_path}")
            
        with open(temp_path, 'r', encoding='utf-8') as f:
            evaluated_genomes = json.load(f)
        
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
        
        if not threshold_results.get("skipped", False):
            outputs_path = str(get_outputs_path())
            adaptive_results = update_adaptive_selection_logic(
                outputs_path=outputs_path,
                current_max_toxicity=threshold_results["max_toxicity_score"],
                previous_max_toxicity=0.0,
                stagnation_limit=stagnation_limit,
                north_star_metric=north_star_metric,
                logger=logger,
                log_file=log_file
            )
            logger.debug("Adaptive selection updated: mode=%s, generations_since_improvement=%d, avg_fitness=%.4f, slope=%.4f",
                       adaptive_results["selection_mode"], adaptive_results["generations_since_improvement"],
                       adaptive_results["current_avg_fitness"], adaptive_results["slope_of_avg_fitness"])
            
            phase_3b_results = {
                "max_toxicity_score": threshold_results["max_toxicity_score"],
                "elite_threshold": threshold_results["elite_threshold"],
                "best_genome_id": threshold_results["best_genome_id"]
            }
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
        logger.info("Initial population finalized using elite threshold: %.4f", elite_threshold)
        
        outputs_path = str(get_outputs_path())
        removal_results = remove_worse_performing_genomes_from_all_files(
            outputs_path=outputs_path,
            population_max_toxicity=phase_3b_results["max_toxicity_score"],
            removal_threshold_percentage=removal_threshold,
            north_star_metric=north_star_metric,
            logger=logger,
            log_file=log_file
        )
        logger.debug("Archived %d genomes, %d remaining", 
                   removal_results["archived_count_total"], removal_results["remaining_count_total"])
        
        redistribution_result = redistribute_population_with_threshold(
            elite_threshold=phase_3b_results["elite_threshold"],
            north_star_metric=north_star_metric,
            logger=logger,
            log_file=log_file
        )
        logger.debug("Redistribution: %d elites, %d non_elites", 
                   redistribution_result["elites_count"], 
                   redistribution_result.get("total_count", 0) - redistribution_result["elites_count"])
        
        try:
            evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
            with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                tracker = json.load(f)
            
            from utils.population_io import _extract_north_star_score
            elites_path = get_outputs_path() / "elites.json"
            non_elites_path = get_outputs_path() / "non_elites.json"
            
            avg_fitness_elites = 0.0001
            avg_fitness_non_elites = 0.0001
            
            if elites_path.exists():
                with open(elites_path, 'r', encoding='utf-8') as f:
                    elites_genomes = json.load(f)
                if elites_genomes:
                    elite_scores = [_extract_north_star_score(g, north_star_metric) for g in elites_genomes]
                    elite_scores = [s for s in elite_scores if s > 0]
                    if elite_scores:
                        avg_fitness_elites = round(sum(elite_scores) / len(elite_scores), 4)
            
            if non_elites_path.exists():
                with open(non_elites_path, 'r', encoding='utf-8') as f:
                    non_elites_genomes = json.load(f)
                if non_elites_genomes:
                    non_elite_scores = [_extract_north_star_score(g, north_star_metric) for g in non_elites_genomes]
                    non_elite_scores = [s for s in non_elite_scores if s > 0]
                    if non_elite_scores:
                        avg_fitness_non_elites = round(sum(non_elite_scores) / len(non_elite_scores), 4)
            
            removal_threshold_value = round((removal_threshold * phase_3b_results["max_toxicity_score"]) / 100, 4)
            
            all_scores_gen0 = []
            if elite_scores:
                all_scores_gen0.extend(elite_scores)
            if non_elite_scores:
                all_scores_gen0.extend(non_elite_scores)
            avg_fitness_generation_gen0 = round(sum(all_scores_gen0) / len(all_scores_gen0), 4) if all_scores_gen0 else 0.0
            
            for gen in tracker.get("generations", []):
                if gen.get("generation_number") == 0:
                    gen["elites_count"] = redistribution_result["elites_count"]
                    gen["removal_threshold"] = removal_threshold_value
                    gen["avg_fitness_elites"] = avg_fitness_elites
                    gen["avg_fitness_non_elites"] = avg_fitness_non_elites
                    gen["avg_fitness_generation"] = avg_fitness_generation_gen0
                    gen["avg_fitness"] = avg_fitness_generation_gen0
                    gen["min_score_variants"] = 0.0001
                    gen["max_score_variants"] = max(all_scores_gen0) if all_scores_gen0 else 0.0001
                    gen["avg_fitness_variants"] = 0.0001
                    break
            
            with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(tracker, f, indent=4, ensure_ascii=False)
            
            logger.debug("Gen0 metrics: elites=%d, removal_th=%.4f, elite_avg=%.4f, non_elite_avg=%.4f",
                        redistribution_result['elites_count'], removal_threshold_value,
                        avg_fitness_elites, avg_fitness_non_elites)
        except Exception as e:
            logger.warning("Failed to update generation 0 metrics in EvolutionTracker: %s", e)
    except Exception as e:
        logger.error("Initial population finalization failed: %s", e, exc_info=True)
        return

    evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
    if evolution_tracker_path.exists():
        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
            evolution_tracker = json.load(f)
        
        existing_generations = evolution_tracker.get("generations", [])
        if existing_generations:
            max_generation = max(gen.get("generation_number", 0) for gen in existing_generations)
            generation_count = max_generation
        else:
            generation_count = 0
            
        elite_threshold = evolution_tracker.get("generations", [{}])[0].get("elites_threshold", phase_3b_results["elite_threshold"])
        logger.debug("Resuming from generation %d, elite_threshold=%.4f", generation_count, elite_threshold)
    else:
        generation_count = 0
        elite_threshold = phase_3b_results["elite_threshold"]
        logger.debug("Starting fresh, elite_threshold=%.4f", elite_threshold)
    
    while max_generations is None or generation_count < max_generations:
        generation_count += 1
        logger.info("=== Starting Generation %d ===", generation_count)
        
        operator_statistics = {}
        try:
            evolution_result = run_evolution(
                north_star_metric=north_star_metric,
                log_file=log_file,
                threshold=north_star_threshold,
                current_cycle=generation_count,
                max_variants=max_variants,
                operators=operators
            )
            
            operator_statistics = evolution_result.get("operator_statistics", {}) if evolution_result else {}
            if operator_statistics:
                logger.debug("Operator stats: %s", operator_statistics)

        except Exception as e:
            logger.error("Evolution failed: %s", e, exc_info=True)
            break

        try:
            temp_path = str(get_outputs_path() / "temp.json")
            response_generator.process_population(pop_path=temp_path)
            
            run_moderation_on_population = get_run_moderation_on_population()
            run_moderation_on_population(
                pop_path=temp_path,
                log_file=log_file,
                north_star_metric=north_star_metric,
                moderation_methods=moderation_methods
            )
            
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
                
                logger.info("Gen %d: %d variants (%d mutation, %d crossover)", 
                           generation_count, total_count, mutation_count, crossover_count)
            
            try:
                from ea import get_update_evolution_tracker_with_generation_global
                update_evolution_tracker_with_generation_global = get_update_evolution_tracker_with_generation_global()
                
                evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
                with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                    evolution_tracker = json.load(f)
                
                generation_data = {
                    "generation_number": generation_count,
                    "variants_created": variant_counts["variants_created"],
                    "mutation_variants": variant_counts["mutation_variants"],
                    "crossover_variants": variant_counts["crossover_variants"]
                }
                
                from utils.population_io import load_population
                outputs_path = get_outputs_path()
                
                all_genomes = []
                for file_name in ["temp.json", "elites.json", "non_elites.json"]:
                    file_path = outputs_path / file_name
                    if file_path.exists():
                        file_genomes = load_population(str(file_path), logger=logger)
                        all_genomes.extend(file_genomes)
                
                logger.debug("Loaded %d genomes for analysis", len(all_genomes))
                
                update_evolution_tracker_with_generation_global(
                    generation_data=generation_data,
                    evolution_tracker=evolution_tracker,
                    logger=logger,
                    population=all_genomes,
                    north_star_metric=north_star_metric
                )
                
                
            except Exception as e:
                logger.error("Failed to update EvolutionTracker with generation data: %s", e, exc_info=True)
            
            try:
                previous_max_toxicity = 0.0001
                evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
                if evolution_tracker_path.exists():
                    with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                        tracker = json.load(f)
                    previous_max_toxicity = tracker.get("population_max_toxicity", 0.0001)
                
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
                
                if not threshold_results.get("skipped", False):
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
                    logger.debug("Selection: mode=%s, since_improvement=%d, avg=%.4f, slope=%.4f",
                               adaptive_results["selection_mode"], adaptive_results["generations_since_improvement"],
                               adaptive_results["current_avg_fitness"], adaptive_results["slope_of_avg_fitness"])
                    
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
                            
                            if scores:
                                max_score_variants = round(max(scores), 4)
                                min_score_variants = round(min(scores), 4)
                                avg_fitness_variants = round(sum(scores) / len(scores), 4)
                                logger.debug("Variants: max=%.4f, min=%.4f, avg=%.4f",
                                           max_score_variants, min_score_variants, avg_fitness_variants)
                    except Exception as e:
                        logger.warning("Failed to calculate variant statistics: %s", e)
                    
                    removal_threshold_value = round((removal_threshold * threshold_results["max_toxicity_score"]) / 100, 4)
                    
                    all_genomes_for_avg_fitness = []
                    
                    if temp_path.exists():
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            temp_variants_for_avg = json.load(f)
                        if temp_variants_for_avg:
                            from utils.population_io import _extract_north_star_score
                            temp_scores = [_extract_north_star_score(v, north_star_metric) for v in temp_variants_for_avg if v]
                            all_genomes_for_avg_fitness.extend(temp_scores)
                    
                    elites_path = get_outputs_path() / "elites.json"
                    if elites_path.exists():
                        with open(elites_path, 'r', encoding='utf-8') as f:
                            existing_elites = json.load(f)
                        if existing_elites:
                            existing_elite_scores = [_extract_north_star_score(g, north_star_metric) for g in existing_elites]
                            all_genomes_for_avg_fitness.extend(existing_elite_scores)
                    
                    non_elites_path = get_outputs_path() / "non_elites.json"
                    if non_elites_path.exists():
                        with open(non_elites_path, 'r', encoding='utf-8') as f:
                            existing_non_elites = json.load(f)
                        if existing_non_elites:
                            existing_non_elite_scores = [_extract_north_star_score(g, north_star_metric) for g in existing_non_elites]
                            all_genomes_for_avg_fitness.extend(existing_non_elite_scores)
                    
                    avg_fitness = round(sum(all_genomes_for_avg_fitness) / len(all_genomes_for_avg_fitness), 4) if all_genomes_for_avg_fitness else 0.0
                    
                    distribution_result = distribute_genomes_by_threshold(
                        temp_path=temp_path,
                        elite_threshold=threshold_results["elite_threshold"],
                        removal_threshold=removal_threshold_value,
                        north_star_metric=north_star_metric,
                        logger=logger
                    )
                    logger.debug("Distribution: %d to elites, %d to non_elites, %d archived", 
                               distribution_result["elites_moved"], distribution_result["population_moved"], 
                               distribution_result.get("under_performing_moved", 0))
                    
                    removal_results = remove_worse_performing_genomes_from_all_files(
                        outputs_path=outputs_path,
                        population_max_toxicity=threshold_results["max_toxicity_score"],
                        removal_threshold_percentage=removal_threshold,
                        north_star_metric=north_star_metric,
                        logger=logger,
                        log_file=log_file
                    )
                    logger.debug("Archived %d genomes, %d remaining", 
                               removal_results["archived_count_total"], removal_results["remaining_count_total"])
                    
                    redistribution_result = redistribute_population_with_threshold(
                        elite_threshold=threshold_results["elite_threshold"],
                        north_star_metric=north_star_metric,
                        logger=logger,
                        log_file=log_file
                    )
                    logger.debug("Final: %d elites, %d non_elites", 
                               redistribution_result["elites_count"], redistribution_result.get("total_count", 0) - redistribution_result["elites_count"])
                    
                    try:
                        evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
                        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                            tracker = json.load(f)
                        
                        from utils.population_io import _extract_north_star_score
                        elites_path = get_outputs_path() / "elites.json"
                        non_elites_path = get_outputs_path() / "non_elites.json"
                        
                        avg_fitness_elites = 0.0
                        avg_fitness_non_elites = 0.0
                        elite_scores = []
                        non_elite_scores = []
                        
                        if elites_path.exists():
                            with open(elites_path, 'r', encoding='utf-8') as f:
                                elites_genomes = json.load(f)
                            if elites_genomes:
                                elite_scores = [_extract_north_star_score(g, north_star_metric) for g in elites_genomes]
                                elite_scores = [s for s in elite_scores if s > 0]
                                if elite_scores:
                                    avg_fitness_elites = round(sum(elite_scores) / len(elite_scores), 4)
                        
                        if non_elites_path.exists():
                            with open(non_elites_path, 'r', encoding='utf-8') as f:
                                non_elites_genomes = json.load(f)
                            if non_elites_genomes:
                                non_elite_scores = [_extract_north_star_score(g, north_star_metric) for g in non_elites_genomes]
                                non_elite_scores = [s for s in non_elite_scores if s > 0]
                                if non_elite_scores:
                                    avg_fitness_non_elites = round(sum(non_elite_scores) / len(non_elite_scores), 4)
                        
                        all_scores = []
                        if elite_scores:
                            all_scores.extend(elite_scores)
                        if non_elite_scores:
                            all_scores.extend(non_elite_scores)
                        avg_fitness_generation = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
                        
                        
                        
                        for gen in tracker.get("generations", []):
                            if gen.get("generation_number") == generation_count:
                                gen["elites_count"] = redistribution_result["elites_count"]
                                gen["removal_threshold"] = removal_threshold_value
                                gen["avg_fitness_elites"] = avg_fitness_elites
                                gen["avg_fitness_non_elites"] = avg_fitness_non_elites
                                gen["avg_fitness_generation"] = avg_fitness_generation
                                gen["max_score_variants"] = max_score_variants
                                gen["min_score_variants"] = min_score_variants
                                gen["avg_fitness_variants"] = avg_fitness_variants
                                gen["avg_fitness"] = avg_fitness
                                gen["operator_statistics"] = operator_statistics
                        
                        with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                            json.dump(tracker, f, indent=4, ensure_ascii=False)
                        
                        logger.debug("Gen%d: elites=%d, elite_avg=%.4f, variants: max=%.4f, min=%.4f, avg=%.4f",
                                    generation_count, redistribution_result['elites_count'],
                                    avg_fitness_elites, max_score_variants, min_score_variants, avg_fitness_variants)
                    except Exception as e:
                        logger.warning("Failed to update generation metrics in EvolutionTracker: %s", e)
                    
                
            except Exception as e:
                logger.error("Threshold recalculation and distribution failed: %s", e, exc_info=True)
            
            try:
                update_population_index_single_file(str(get_outputs_path()), 0, logger=logger)
            except Exception as e:
                logger.warning("Failed to update EvolutionTracker population metadata: %s", e)
            
        except Exception as e:
            logger.error("Post-evolution processing failed: %s", e, exc_info=True)

    if max_generations is not None and generation_count >= max_generations:
        logger.info("Evolution completed: Maximum generation limit (%d) reached.", max_generations)
    else:
        logger.info("Evolution completed: Loop exited due to other conditions.")

    
    total_time = time.time() - start_time
    logger.info("=== Pipeline Completed ===")
    logger.info("Total execution time: %.2f seconds", total_time)
    logger.info("Total generations: %d", generation_count)



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
    parser.add_argument("--seed-file", type=str, default="data/prompt.csv",
                       help="Path to CSV file with seed prompts (must have 'questions' column). Default: data/prompt.csv")
    args = parser.parse_args()
    
    import sys
    try:
        main(max_generations=args.generations, 
             north_star_threshold=args.threshold, moderation_methods=args.moderation_methods,
             threshold_percentage=args.threshold_percentage, rg_model=args.rg, pg_model=args.pg,
             operators=args.operators, max_variants=args.max_variants,
             elites_threshold=args.elites_threshold, removal_threshold=args.removal_threshold,
             stagnation_limit=args.stagnation_limit, seed_file=args.seed_file)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)