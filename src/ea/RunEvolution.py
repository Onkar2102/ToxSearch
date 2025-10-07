## @file RunEvolution.py
# @brief Main script for evolving LLM input prompts using mutation operators.

import json
import time
from typing import Dict, Any, List, Optional
# Lazy import to avoid torch dependency issues
def get_EvolutionEngine():
    """Lazy import of EvolutionEngine to avoid torch dependency issues"""
    from ea.EvolutionEngine import EvolutionEngine
    return EvolutionEngine
from utils import get_population_io, get_custom_logging

from pathlib import Path

# Get the functions at module level to avoid repeated calls
get_logger, _, _, PerformanceLogger = get_custom_logging()

project_root = Path(__file__).resolve().parents[2]

def get_outputs_path():
    """Get the absolute path to the outputs directory"""
    # Create the full path: outputs/
    outputs_dir = project_root / "outputs"
    
    # Ensure the directory exists
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    return outputs_dir

# Dynamic paths that will be set during runtime
population_path = None
evolution_tracker_path = None
parent_selection_tracker_path = None

def initialize_evolution_tracker(logger, log_file=None):
    """Initialize the evolution tracker file if it doesn't exist"""
    with PerformanceLogger(logger, "Initialize Evolution Tracker"):
        try:
            # Get the dynamic path for this run
            outputs_path = get_outputs_path()
            evolution_tracker_path = outputs_path / "EvolutionTracker.json"
            
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

# Function removed - no longer needed for global evolution

def _extract_north_star_score(genome: Dict, north_star_metric: str) -> float:
    """Extract the north star metric score from a genome using the configured metric."""
    if not genome or not genome.get("moderation_result"):
        return 0.0
    
    moderation_result = genome["moderation_result"]
    
    # Try Google API scores first (most comprehensive)
    if "moderation_results" in moderation_result:
        google_scores = moderation_result["moderation_results"].get("google", {})
        if "scores" in google_scores:
            score = google_scores["scores"].get(north_star_metric, 0.0)
            if score > 0:
                return float(score)
    
    # Try OpenAI API scores as fallback
    if "moderation_results" in moderation_result:
        openai_scores = moderation_result["moderation_results"].get("openai", {})
        if "scores" in openai_scores:
            score = openai_scores["scores"].get(north_star_metric, 0.0)
            if score > 0:
                return float(score)
    
    # Fallback to direct scores if available
    if "scores" in moderation_result:
        score = moderation_result["scores"].get(north_star_metric, 0.0)
        if score > 0:
            return float(score)
    
    return 0.0

def check_threshold_and_update_tracker(population, north_star_metric, logger, log_file=None, threshold=0.99):
    """Check threshold achievement and update evolution tracker (global version)"""
    _logger = logger or get_logger("check_threshold", log_file)
    with PerformanceLogger(_logger, "Check Threshold and Update Tracker"):
        try:
            # Get the dynamic path for this run
            outputs_path = get_outputs_path()
            evolution_tracker_path = outputs_path / "EvolutionTracker.json"
            
            # Load existing evolution tracker (global structure)
            if evolution_tracker_path.exists():
                try:
                    with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            evolution_tracker = json.loads(content)
                        else:
                            evolution_tracker = {
                                "scope": "global",
                                "status": "not_complete",
                                "total_generations": 1,
                                "generations": []
                            }
                except (json.JSONDecodeError, FileNotFoundError):
                    evolution_tracker = {
                        "scope": "global",
                        "status": "not_complete",
                        "total_generations": 1,
                        "generations": []
                    }
            else:
                evolution_tracker = {
                    "scope": "global",
                    "status": "not_complete",
                    "total_generations": 1,
                    "generations": []
                }

            # Find current best genome globally
            completed_genomes = [g for g in population if g.get("status") == "complete"]
            if completed_genomes:
                best_genome = max(completed_genomes, key=lambda g: _extract_north_star_score(g, north_star_metric))
                best_score = _extract_north_star_score(best_genome, north_star_metric)
                best_genome_id = best_genome["id"]
                
                if isinstance(best_score, (int, float)) and best_score >= threshold:
                    evolution_tracker["status"] = "complete"
                    _logger.info("Global population achieved threshold with score %.4f (genome %s)", best_score, best_genome_id)
                else:
                    evolution_tracker["status"] = "not_complete"
                    _logger.info("Global population best score: %.4f (genome %s), threshold not reached", best_score, best_genome_id)
            else:
                evolution_tracker["status"] = "not_complete"
                _logger.info("No completed genomes found in global population")

                        # Update evolution tracker for global population
            if not evolution_tracker.get("generations"):
                # Initialize generation 0 if not exists
                gen0_genomes = [g for g in population if g.get("generation") == 0]
                if gen0_genomes:
                    # Find the best genome in generation 0
                    best_gen0_genome = max(gen0_genomes, key=lambda g: _extract_north_star_score(g, north_star_metric))
                    best_gen0_id = best_gen0_genome["id"]
                    best_gen0_score = _extract_north_star_score(best_gen0_genome, north_star_metric)
                    
                    evolution_tracker["generations"] = [{
                        "generation_number": 0,
                        "genome_id": best_gen0_id,  # Best genome ID from generation 0
                        "max_score": best_gen0_score,
                        "mutation": None,
                        "crossover": None
                    }]
                    _logger.info("Created generation 0 entry with best genome %s, score: %.4f", 
                              best_gen0_id, best_gen0_score)

            # Mark all genomes as complete if threshold is achieved globally
            if evolution_tracker["status"] == "complete":
                marked_count = 0
                for genome in population:
                    if genome.get("status") != "complete":
                        genome["status"] = "complete"
                        genome["completion_reason"] = f"Global population achieved {north_star_metric} >= {threshold}"
                        marked_count += 1
                _logger.info("Marked %d genomes as complete (global threshold achieved)", marked_count)
                
                # Save population to elites.json (steady-state mode)
                _, _, _, save_population, _, _, _, _, _, _, _, _, _ = get_population_io()
                elites_path = str(outputs_path / "elites.json")
                save_population(population, elites_path, logger=_logger)
                _logger.info("Updated elites with completed statuses")
            else:
                _logger.info("Global population has not achieved the threshold of %.4f", threshold)
            with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(evolution_tracker, f, indent=4, ensure_ascii=False)
            _logger.info("Updated evolution tracker with threshold check results")
            
            return evolution_tracker
            
        except Exception as e:
            _logger.error("Failed to check threshold and update tracker: %s", e, exc_info=True)
            # Return a default tracker structure on error
            return {
                "scope": "global",
                "status": "error",
                "total_generations": 1,
                "generations": []
            }

def get_pending_status(evolution_tracker, logger):
    """Get status of global evolution tracker"""
    with PerformanceLogger(logger, "Get Pending Status"):
        try:
            status = evolution_tracker.get("status", "not_complete")
            logger.info("Global evolution status: %s", status)
            return status
        except Exception as e:
            logger.error("Failed to get pending status: %s", e, exc_info=True)
            raise

def update_evolution_tracker_with_generation_global(generation_data, evolution_tracker, logger, population=None, north_star_metric=None):
    """Update evolution tracker with generation data for global population"""
    _logger = logger or get_logger("update_evolution_tracker", log_file=None)
    with PerformanceLogger(_logger, "Update Evolution Tracker with Generation Global"):
        try:
            # Use generation number from evolution cycle
            gen_number = generation_data.get("generation_number", evolution_tracker.get("total_generations", 0))
            
            # Calculate max_score from variants created in this generation
            best_genome_id = None
            best_score = 0.0
            
            if population and north_star_metric:
                # Find all genomes created in this generation
                generation_genomes = [g for g in population if g.get("generation") == gen_number]
                
                if generation_genomes:
                    # Calculate scores for all genomes in this generation
                    genome_scores = []
                    for genome in generation_genomes:
                        score = _extract_north_star_score(genome, north_star_metric)
                        if score > 0:  # Only consider genomes with valid scores
                            genome_scores.append((genome["id"], score))
                    
                    if genome_scores:
                        # Find the best genome from this generation
                        best_genome_id, best_score = max(genome_scores, key=lambda x: x[1])
                        _logger.info(f"Generation {gen_number} best score: {best_score} (genome {best_genome_id})")
                    else:
                        _logger.warning(f"No valid scores found for generation {gen_number}")
                else:
                    _logger.warning(f"No genomes found for generation {gen_number}")
            else:
                # Fallback: use parent score if population not available
                if generation_data.get("parents"):
                    best_parent = max(generation_data["parents"], 
                                    key=lambda p: p.get("north_star_score", 0.0))
                    best_genome_id = best_parent["id"]
                    best_score = best_parent["north_star_score"]
                    _logger.warning(f"Using parent score as fallback for generation {gen_number}: {best_score}")
            
            # Enhanced parent tracking with generation information
            parents_info = None
            if generation_data.get("parents"):
                # Find the best mutation parent (highest score)
                mutation_parent_info = None
                crossover_parents_info = []
                
                # Get all parents from the generation data
                all_parents = generation_data["parents"]
                if all_parents:
                    # Sort parents by score to find the best
                    sorted_parents = sorted(all_parents, key=lambda p: p.get("north_star_score", 0.0), reverse=True)
                    
                    # The best parent becomes the mutation parent
                    best_parent = sorted_parents[0]
                    mutation_parent_info = {
                        "genome_id": best_parent["id"],
                        "generation": best_parent.get("generation", 0),
                        "score": best_parent.get("north_star_score", 0.0)
                    }
                    
                    # Other parents become crossover parents
                    for parent in sorted_parents[1:]:  # Take all additional parents (up to 4 for steady-state)
                        crossover_parents_info.append({
                            "genome_id": parent["id"],
                            "generation": parent.get("generation", 0),
                            "score": parent.get("north_star_score", 0.0)
                        })
                
                parents_info = {
                    "mutation_parent": mutation_parent_info,
                    "crossover_parents": crossover_parents_info
                }
            
            new_gen = {
                "generation_number": gen_number,
                "genome_id": best_genome_id,  # Best genome from THIS generation
                "max_score": best_score,     # Best score from THIS generation
                "variants_created": generation_data.get("variants_created", 0),
                "mutation_variants": generation_data.get("mutation_variants", 0),
                "crossover_variants": generation_data.get("crossover_variants", 0),
                "parents": parents_info
            }
            
            # Check if this generation already exists
            existing_gen = None
            for gen in evolution_tracker.get("generations", []):
                if gen["generation_number"] == gen_number:
                    existing_gen = gen
                    break
            
            if existing_gen:
                # Update existing generation
                existing_gen.update(new_gen)
                _logger.info("Updated existing generation %d globally with max_score %.4f", gen_number, best_score)
            else:
                # Add new generation
                evolution_tracker["generations"].append(new_gen)
                evolution_tracker["total_generations"] = max(evolution_tracker["total_generations"], gen_number + 1)
                _logger.info("Added new generation %d globally with max_score %.4f", gen_number, best_score)
            
            # Sort generations by generation number
            evolution_tracker["generations"].sort(key=lambda x: x["generation_number"])
            
            # Calculate and update population max toxicity at the top level
            # Also compute top 25% toxicity threshold and set population_status for each genome in this generation
            if population and north_star_metric:
                all_evaluated_genomes = [g for g in population if g.get("moderation_result")]
                if all_evaluated_genomes:
                    population_scores = []
                    for genome in all_evaluated_genomes:
                        score = _extract_north_star_score(genome, north_star_metric)
                        if score > 0:  # Only consider genomes with valid scores
                            population_scores.append((genome["id"], score))
                    
                    if population_scores:
                        # Max toxicity for the whole population
                        population_best_genome_id, population_max_toxicity = max(population_scores, key=lambda x: x[1])
                        evolution_tracker["population_max_toxicity"] = population_max_toxicity
                        evolution_tracker["population_best_genome_id"] = population_best_genome_id
                        _logger.info(f"Updated population max toxicity: {population_max_toxicity} (genome {population_best_genome_id})")
                    else:
                        _logger.warning(f"No valid scores found in entire population")
                else:
                    _logger.warning(f"No evaluated genomes found in population")

                # --- Top 25% toxicity threshold and elite marking ---
                # Only for genomes in this generation
                generation_genomes = [g for g in population if g.get("generation") == gen_number and g.get("moderation_result")]
                gen_scores = []
                for genome in generation_genomes:
                    score = _extract_north_star_score(genome, north_star_metric)
                    if score > 0:
                        gen_scores.append(score)
                if gen_scores:
                    # Compute 75th percentile (top 25% threshold)
                    import numpy as np
                    top25_threshold = float(np.percentile(gen_scores, 75))
                    # Mark genomes as elite/non-elite
                    for genome in generation_genomes:
                        score = _extract_north_star_score(genome, north_star_metric)
                        if score >= top25_threshold:
                            genome["population_status"] = "elite"
                        else:
                            genome["population_status"] = "non-elite"
                    # Save threshold in tracker for this generation
                    # Attach to the new_gen dict in tracker
                    for gen in evolution_tracker.get("generations", []):
                        if gen["generation_number"] == gen_number:
                            gen["top25_toxicity_threshold"] = top25_threshold
                            break
                    _logger.info(f"Top 25% toxicity threshold for generation {gen_number}: {top25_threshold:.4f}")
                else:
                    _logger.warning(f"No valid scores for top 25% threshold in generation {gen_number}")
            
            # Get the dynamic path for this run
            outputs_path = get_outputs_path()
            evolution_tracker_path = outputs_path / "EvolutionTracker.json"
            
            with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(evolution_tracker, f, indent=4, ensure_ascii=False)
            
            _logger.info("Updated global evolution tracker with generation %d data: %d variants created, max_score %.4f", 
                       gen_number, generation_data.get("variants_created", 0), best_score)
            
        except Exception as e:
            _logger.error("Failed to update global evolution tracker with generation data: %s", e, exc_info=True)
            raise

# Function removed - no longer needed for global evolution

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
            # Calculate basic statistics for global tracker
            total_generations = evolution_tracker.get("total_generations", 0)
            status = evolution_tracker.get("status", "not_complete")
            completed = 1 if status == "complete" else 0
            pending = 1 if status == "not_complete" else 0
            
            # Calculate score statistics for global tracker
            all_scores = []
            best_scores = []
            for gen_entry in evolution_tracker.get("generations", []):
                score = gen_entry.get("max_score", 0.0)
                all_scores.append(score)
                if gen_entry.get("generation_number") == total_generations - 1:  # Latest generation
                    best_scores.append(score)
            
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            best_avg_score = sum(best_scores) / len(best_scores) if best_scores else 0.0
            max_score = max(all_scores) if all_scores else 0.0
            min_score = min(all_scores) if all_scores else 0.0
            
            # Calculate variant statistics for global tracker
            total_variants_created = 0
            total_mutation_variants = 0
            total_crossover_variants = 0
            
            for gen_entry in evolution_tracker.get("generations", []):
                total_variants_created += gen_entry.get("variants_created", 0)
                total_mutation_variants += gen_entry.get("mutation_variants", 0)
                total_crossover_variants += gen_entry.get("crossover_variants", 0)
            
            # Create comprehensive statistics for global tracker
            final_stats = {
                "execution_summary": {
                    "execution_time_seconds": execution_time,
                    "generations_completed": generations_completed,
                    "total_prompts": 1,  # Global tracker represents one population
                    "completed_prompts": completed,
                    "pending_prompts": pending,
                    "completion_rate": (completed * 100)  # Either 0% or 100%
                },
                "generation_statistics": {
                    "total_generations": total_generations,
                    "average_generations_per_prompt": total_generations,  # Same as total for global
                    "max_generations_for_any_prompt": total_generations
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
            
            # Add detailed information for global population
            prompt_detail = {
                "scope": "global",
                "status": status,
                "total_generations": total_generations,
                "best_score": 0.0,
                "initial_score": 0.0,
                "score_improvement": 0.0,
                "variants_created": 0
            }
            
            generations = evolution_tracker.get("generations", [])
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
            
            _logger.info(f"Created comprehensive final statistics for global population")
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
def run_evolution(north_star_metric, log_file=None, threshold=0.99, current_cycle=None):
    """Run one evolution generation with comprehensive logging and steady state support"""
    # Set up dynamic paths for this run
    outputs_path = get_outputs_path()
    population_path = outputs_path / "Population.json"
    evolution_tracker_path = outputs_path / "EvolutionTracker.json"
    
    with PerformanceLogger(get_logger("RunEvolution", log_file), "Run Evolution", 
                          north_star_metric=north_star_metric, population_path=str(population_path)):
        
        logger = get_logger("RunEvolution", log_file)
        logger.info("Starting evolution run using population file: %s", population_path)
        logger.info("Output directory: %s", outputs_path)
        logger.info("North star metric: %s", north_star_metric)
        logger.info("Using steady state population management")
        if current_cycle is not None:
            logger.info("Evolution cycle: %d", current_cycle)

        # Check for population file
        if not population_path.exists():
            logger.error("Population file not found: %s", population_path)
            raise FileNotFoundError(f"Population file not found: {population_path}")

        # Phase 1: Initialize evolution tracker
        initialize_evolution_tracker(logger, log_file)

        # Phase 2: Load population with error handling
        with PerformanceLogger(logger, "Load Population"):
            try:
                _, _, load_population, _, _, _, _, _, _, _, _, _, _ = get_population_io()
                population = load_population(str(outputs_path), logger=logger)
                logger.info("Successfully loaded population with %d genomes", len(population))
            except Exception as e:
                logger.error("Unexpected error loading population: %s", e, exc_info=True)
                raise

        # Phase 3: Check threshold and update evolution tracker
        evolution_tracker = check_threshold_and_update_tracker(population, north_star_metric, logger, log_file, threshold)

        # Phase 4: Check if evolution should continue
        evolution_status = get_pending_status(evolution_tracker, logger)
        
        if evolution_status == "complete":
            logger.info("Evolution completed - threshold achieved globally")
            return

        # Phase 5: Initialize evolution engine
        with PerformanceLogger(logger, "Initialize Evolution Engine"):
            try:
                EvolutionEngine = get_EvolutionEngine()
                engine = EvolutionEngine(north_star_metric, log_file, current_cycle=current_cycle)
                engine.genomes = population
                engine.update_next_id()
                logger.debug("EvolutionEngine next_id set to %d", engine.next_id)
            except Exception as e:
                logger.error("Failed to initialize evolution engine: %s", e, exc_info=True)
                raise

        # Phase 6: Process global evolution
        try:
            logger.info("Processing global evolution")
            logger.debug("Calling generate_variants_global()")
            
            # Get generation data from the engine
            generation_data = engine.generate_variants_global()
            
            logger.info("Generated %d variants (mutation: %d, crossover: %d) globally", 
                       generation_data["variants_created"], 
                       generation_data["mutation_variants"], 
                       generation_data["crossover_variants"])
            
            # Save updated population after evolution to elites.json
            with PerformanceLogger(logger, "Save Population After Evolution"):
                try:
                    _, _, _, save_population, _, _, _, _, _, _, _, _, _ = get_population_io()
                    # Save to elites.json since we're in steady-state mode
                    elites_path = str(outputs_path / "elites.json")
                    save_population(engine.genomes, elites_path, logger=logger)
                    logger.debug("Saved updated elites after global evolution")
                except Exception as e:
                    logger.error("Failed to save population after evolution: %s", e, exc_info=True)
                    raise
                    
        except Exception as e:
            logger.error("Failed to process global evolution: %s", e, exc_info=True)
            raise

        logger.info("Global evolution processing completed successfully")

        # Phase 6.5: Redistribute elites to population (steady state)
        with PerformanceLogger(logger, "Steady State Redistribution"):
                    pass

        # Phase 7: Deduplicate population
        with PerformanceLogger(logger, "Deduplicate Population"):
            from collections import defaultdict
            try:

                # Keep generation 0 genomes as-is
                gen_zero = [g for g in engine.genomes if g["generation"] == 0]
                gen_gt_zero = [g for g in engine.genomes if g["generation"] > 0]

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
                           len(engine.genomes), len(final_population), duplicates_removed)
            except Exception as e:
                logger.error("Failed to deduplicate population: %s", e, exc_info=True)
                raise

        # Phase 8: Save final population to elites.json
        with PerformanceLogger(logger, "Save Final Population"):
            try:
                _, _, _, save_population, _, _, _, _, _, _, _, _, _ = get_population_io()
                # Save to elites.json since we're in steady-state mode
                elites_path = str(outputs_path / "elites.json")
                save_population(final_population, elites_path, logger=logger)
                logger.info("Elites re-saved in sorted and deduplicated order")
            except Exception as e:
                logger.error("Failed to save final population: %s", e, exc_info=True)
                raise

        # Log final summary
        logger.info("Evolution run completed successfully:")
        logger.info("  - Total genomes processed: %d", len(engine.genomes))
        logger.info("  - Final elites size: %d", len(final_population))
        logger.info("  - Evolution tracker updated: %s", evolution_tracker_path)

