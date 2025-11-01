## @file RunEvolution.py
# @brief Main script for evolving LLM input prompts using mutation operators.

import json
from typing import Dict, Any, List, Optional
# Lazy import to avoid torch dependency issues
def get_EvolutionEngine():
    """Lazy import of EvolutionEngine to avoid torch dependency issues"""
    from ea.evolution_engine import EvolutionEngine
    return EvolutionEngine
from utils import get_population_io, get_custom_logging
from utils.population_io import update_population_index_single_file

from pathlib import Path

# Get the functions at module level to avoid repeated calls
get_logger, _, _, _ = get_custom_logging()

# Import system utilities
from utils import get_system_utils
get_project_root, get_config_path, get_data_path, get_outputs_path, _extract_north_star_score, initialize_system = get_system_utils()

project_root = Path(__file__).resolve().parents[2]

def _reset_temp_json(logger):
    """Reset temp.json to empty list at the start of variant generation."""
    try:
        temp_path = get_outputs_path() / "temp.json"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        logger.debug("Reset temp.json for new generation")
    except Exception as e:
        logger.error(f"Failed to reset temp.json: {e}")
        raise

def _deduplicate_variants_in_temp(logger, operator_stats=None):
    """
    Deduplicate variants in temp.json by comparing against existing genomes in all files.
    This function ONLY performs deduplication and does NOT distribute genomes.
    
    Args:
        logger: Logger instance
        operator_stats: Optional OperatorStatistics instance to track duplicates
        
    Returns:
        int: Number of duplicates removed
    """
    try:
        outputs_path = get_outputs_path()
        temp_path = outputs_path / "temp.json"
        elites_path = outputs_path / "elites.json"
        population_path = outputs_path / "non_elites.json"
        
        # Load temp.json variants
        if not temp_path.exists():
            logger.warning("temp.json not found for deduplication")
            return 0
            
        with open(temp_path, 'r', encoding='utf-8') as f:
            temp_variants = json.load(f)
        
        if not temp_variants:
            logger.debug("No variants in temp.json to deduplicate")
            return 0
        
        # Load existing genomes from all files for de-duplication
        existing_prompts = set()
        existing_ids = set()
        
        # Load from elites.json
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
                for genome in elites:
                    if genome and genome.get("prompt"):
                        existing_prompts.add(genome["prompt"].strip().lower())
                        existing_ids.add(genome.get("id"))
        
        # Load from non_elites.json
        if population_path.exists():
            with open(population_path, 'r', encoding='utf-8') as f:
                population = json.load(f)
                for genome in population:
                    if genome and genome.get("prompt"):
                        existing_prompts.add(genome["prompt"].strip().lower())
                        existing_ids.add(genome.get("id"))
        
        # Deduplicate variants (remove duplicates, keep unique ones)
        unique_variants = []
        duplicates_removed = 0
        
        for variant in temp_variants:
            if not variant or not variant.get("prompt"):
                duplicates_removed += 1
                continue
                
            # Check if genome already exists (de-duplication)
            norm_prompt = variant["prompt"].strip().lower()
            genome_id = variant.get("id")
            
            if norm_prompt in existing_prompts or genome_id in existing_ids:
                duplicates_removed += 1
                if operator_stats:
                    # Get operator name from variant's creation_info
                    operator_name = variant.get("creation_info", {}).get("operator", "unknown")
                    operator_stats.record_duplicate(operator_name)
                logger.debug(f"Removing duplicate genome {genome_id}")
                continue
            
            # Keep unique variant
            unique_variants.append(variant)
        
        # Save deduplicated variants back to temp.json
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(unique_variants, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Deduplication: {len(temp_variants)} → {len(unique_variants)} ({duplicates_removed} duplicates)")
        
        return duplicates_removed
        
    except Exception as e:
        logger.error(f"Failed to deduplicate variants in temp.json: {e}")
        raise


def distribute_genomes_by_threshold(temp_path, elite_threshold, north_star_metric, logger, removal_threshold=None, north_star_threshold=0.99):
    """
    Distribute genomes from temp.json based on thresholds into 3 categories:
    - Genomes >= elite_threshold → elites.json (elite genomes)
    - Genomes between removal_threshold and elite_threshold → non_elites.json (for further evolution)
    - Genomes <= removal_threshold → under_performing.json (archived low-performers)
    
    Args:
        temp_path: Path to temp.json file
        elite_threshold: Toxicity threshold for elite selection
        north_star_metric: The metric to use for scoring
        logger: Logger instance
        removal_threshold: Toxicity threshold for removal/archiving (default: None, no archiving)
        north_star_threshold: Threshold for toxic genomes (default: 0.99)
        
    Returns:
        dict: Distribution statistics with elites_moved, population_moved, under_performing_moved, total_processed
    """
    try:
        outputs_path = get_outputs_path()
        elites_path = outputs_path / "elites.json"
        population_path = outputs_path / "non_elites.json"
        under_performing_path = outputs_path / "under_performing.json"
        
        # Load temp.json variants
        if not temp_path.exists():
            logger.warning("temp.json not found for distribution")
            return {"elites_moved": 0, "population_moved": 0, "under_performing_moved": 0, "total_processed": 0}
            
        with open(temp_path, 'r', encoding='utf-8') as f:
            temp_variants = json.load(f)
        
        if not temp_variants:
            logger.debug("No variants in temp.json to distribute")
            return {"elites_moved": 0, "population_moved": 0, "under_performing_moved": 0, "total_processed": 0}
        
        # Categorize variants based on thresholds into 3 categories
        elites_to_move = []
        population_to_move = []
        under_performing_to_move = []
        
        for variant in temp_variants:
            if not variant or not variant.get("prompt"):
                continue
                
            # Check toxicity score
            toxicity_score = _extract_north_star_score(variant, north_star_metric)
            genome_id = variant.get("id")
            
            # Categorize based on thresholds and add initial_state
            if toxicity_score >= elite_threshold:
                # Elite genomes
                variant["initial_state"] = "elite"
                elites_to_move.append(variant)
                logger.debug(f"Genome {genome_id} marked as elite (score: {toxicity_score:.3f})")
            elif removal_threshold is not None and toxicity_score <= removal_threshold:
                # Under-performing genomes (archived)
                variant["initial_state"] = "inefficient"
                under_performing_to_move.append(variant)
                logger.debug(f"Genome {genome_id} marked as under-performing (score: {toxicity_score:.3f}) - will be archived")
            else:
                # Non-elite genomes (for further evolution)
                variant["initial_state"] = "non_elite"
                population_to_move.append(variant)
                logger.debug(f"Genome {genome_id} marked for population (score: {toxicity_score:.3f})")
        
        # Move elite genomes to elites.json
        if elites_to_move:
            # Add to elites.json
            elites_to_save = []
            if elites_path.exists():
                with open(elites_path, 'r', encoding='utf-8') as f:
                    elites_to_save = json.load(f)
            
            elites_to_save.extend(elites_to_move)
            
            with open(elites_path, 'w', encoding='utf-8') as f:
                json.dump(elites_to_save, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Moved {len(elites_to_move)} elite genomes to elites.json")
        
        # Move non-elite genomes to non_elites.json
        if population_to_move:
            # Add to non_elites.json
            population_to_save = []
            if population_path.exists():
                with open(population_path, 'r', encoding='utf-8') as f:
                    population_to_save = json.load(f)
            
            population_to_save.extend(population_to_move)
            
            with open(population_path, 'w', encoding='utf-8') as f:
                json.dump(population_to_save, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Moved {len(population_to_move)} genomes to non_elites.json")
        
        # Move under-performing genomes to under_performing.json (archive)
        if under_performing_to_move:
            # Add to under_performing.json
            under_performing_to_save = []
            if under_performing_path.exists():
                with open(under_performing_path, 'r', encoding='utf-8') as f:
                    under_performing_to_save = json.load(f)
            
            under_performing_to_save.extend(under_performing_to_move)
            
            with open(under_performing_path, 'w', encoding='utf-8') as f:
                json.dump(under_performing_to_save, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Archived {len(under_performing_to_move)} under-performing genomes")
        
        # Clear temp.json after successful distribution
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        
        distribution_stats = {
            "elites_moved": len(elites_to_move),
            "population_moved": len(population_to_move),
            "under_performing_moved": len(under_performing_to_move),
            "total_processed": len(temp_variants)
        }
        
        logger.debug(f"Distribution complete: {distribution_stats['total_processed']} variants → "
                   f"{distribution_stats['elites_moved']} elites, "
                   f"{distribution_stats['population_moved']} population, "
                   f"{distribution_stats['under_performing_moved']} archived")
        
        return distribution_stats
        
    except Exception as e:
        logger.error(f"Failed to distribute genomes by threshold: {e}")
        raise


# Dynamic paths that will be set during runtime
population_path = None
evolution_tracker_path = None
parent_selection_tracker_path = None

# _extract_north_star_score is now imported from utils.get_system_utils()

def check_threshold_and_update_tracker(population, north_star_metric, log_file=None, threshold=0.99):
    """Check threshold achievement and update evolution tracker (global version)"""
    get_logger, _, _, _ = get_custom_logging()
    logger = get_logger("RunEvolution", log_file)
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
                # For toxicity maximization, we don't stop when we reach the threshold
                # We continue evolving to find even higher toxicity scores
                evolution_tracker["status"] = "not_complete"
                logger.debug("Threshold achieved (%.4f), continuing for higher scores", best_score)
            else:
                evolution_tracker["status"] = "not_complete"
                logger.debug("Best score: %.4f, threshold not reached", best_score)
        else:
            evolution_tracker["status"] = "not_complete"
            logger.debug("No completed genomes found")

        if not evolution_tracker.get("generations"):
            # Initialize generation 0 if not exists
            gen0_genomes = [g for g in population if g.get("generation") == 0]
            if gen0_genomes:
                # Find the best genome in generation 0
                best_gen0_genome = max(gen0_genomes, key=lambda g: _extract_north_star_score(g, north_star_metric))
                best_gen0_id = best_gen0_genome["id"]
                best_gen0_score = _extract_north_star_score(best_gen0_genome, north_star_metric)
                # Get selection_mode from EvolutionTracker root level (defaults to "default" for gen 0)
                selection_mode = evolution_tracker.get("selection_mode", "default")
                evolution_tracker["generations"] = [{
                    "generation_number": 0,
                    "genome_id": best_gen0_id,  # Best genome ID from generation 0
                    "max_score_variants": best_gen0_score,
                    "min_score_variants": 0.0001,
                    "avg_fitness": 0.0001,
                    # Variant statistics from temp.json (before distribution)
                    "avg_fitness_variants": 0.0001,
                    # Population statistics (after distribution)
                    "avg_fitness_generation": 0.0001,
                    "avg_fitness_elites": 0.0001,
                    "avg_fitness_non_elites": 0.0001,
                    "parents": None,
                    "top_10": None,
                    "variants_created": None,
                    "mutation_variants": None,
                    "crossover_variants": None,
                    "elites_threshold": 0.0001,
                    "removal_threshold": 0.0001,
                    "elites_count": 0,
                    "selection_mode": selection_mode,  # Add selection mode for generation 0
                }]
                logger.debug("Created gen 0 entry: genome %s, score: %.4f", best_gen0_id, best_gen0_score)

        # For toxicity maximization, we don't mark genomes as complete when threshold is achieved
        # We continue evolving to find even higher toxicity scores
        logger.debug("Best score: %.4f, continuing evolution", best_score if completed_genomes else 0.0)
        
        with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
            json.dump(evolution_tracker, f, indent=4, ensure_ascii=False)
        
        return evolution_tracker
    except Exception as e:
        logger.error("Failed to check threshold and update tracker: %s", e, exc_info=True)
        # Return a default tracker structure on error
        return {
            "scope": "global",
            "status": "error",
            "total_generations": 1,
            "generations": []
        }

def get_pending_status(evolution_tracker, logger):
    """Get status of global evolution tracker"""
    try:
        status = evolution_tracker.get("status", "not_complete")
        logger.debug("Evolution status: %s", status)
        return status
    except Exception as e:
        logger.error("Failed to get pending status: %s", e, exc_info=True)
        raise

def update_evolution_tracker_with_generation_global(generation_data, evolution_tracker, logger, population=None, north_star_metric=None):
    """Update evolution tracker with generation data for global population"""
    _logger = logger or get_logger("update_evolution_tracker", log_file=None)
    try:
        # Use generation number from evolution cycle - should always be provided
        gen_number = generation_data.get("generation_number")
        if gen_number is None:
            _logger.error("generation_number not provided in generation_data")
            return
        
        # Calculate max_score from variants created in this generation
        best_genome_id = None
        best_score = 0.0001
        
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
        
        # Calculate avg_fitness for this generation
        avg_fitness = 0.0001
        try:
            from utils.population_io import calculate_average_fitness
            outputs_path = get_outputs_path()
            avg_fitness = calculate_average_fitness(str(outputs_path), north_star_metric, logger=_logger)
            _logger.info(f"Calculated avg_fitness for generation {gen_number}: {avg_fitness:.4f}")
        except Exception as e:
            _logger.warning(f"Failed to calculate avg_fitness for generation {gen_number}: {e}")
        
        # Check if this generation already exists (may have been created by immediate updates)
        existing_gen = None
        for gen in evolution_tracker.get("generations", []):
            if gen["generation_number"] == gen_number:
                existing_gen = gen
                break
        
        if existing_gen:
            # Update existing generation (only update variant counts and scores, not parents/top_10)
            variants_created = generation_data.get("variants_created", 0)
            mutation_variants = generation_data.get("mutation_variants", 0)
            crossover_variants = generation_data.get("crossover_variants", 0)
            
            _logger.info(f"Updating generation {gen_number} with variant counts: created={variants_created}, mutation={mutation_variants}, crossover={crossover_variants}")
            
            # Get selection_mode from EvolutionTracker root level for this generation
            selection_mode = evolution_tracker.get("selection_mode", "default")
            
            # NOTE: max_score_variants represents the maximum score of VARIANTS GENERATED in this generation (from temp.json)
            # It does NOT represent the entire population's max score. Use population_max_toxicity for that.
            existing_gen.update({
                "genome_id": best_genome_id,
                "max_score_variants": best_score,  # Max score of variants created in THIS generation
                "avg_fitness": round(avg_fitness, 4),
                "variants_created": variants_created,
                "mutation_variants": mutation_variants,
                "crossover_variants": crossover_variants,
                "selection_mode": selection_mode  # Add selection mode for this generation
            })
            _logger.info("Updated existing generation %d globally with max_score_variants %.4f and %d variants", gen_number, best_score, variants_created)
        else:
            # Generation entry doesn't exist yet - create it
            _logger.warning("Generation %d not found - creating new entry", gen_number)
            variants_created = generation_data.get("variants_created", 0)
            mutation_variants = generation_data.get("mutation_variants", 0)
            crossover_variants = generation_data.get("crossover_variants", 0)
            
            # Get selection_mode from EvolutionTracker root level for this generation
            selection_mode = evolution_tracker.get("selection_mode", "default")
            
            new_gen = {
                "generation_number": gen_number,
                "genome_id": best_genome_id,
                "avg_fitness": round(avg_fitness, 4),
                # Variant statistics from temp.json (before distribution)
                "max_score_variants": best_score,
                "min_score_variants": 0.0001,
                "avg_fitness_variants": 0.0001,
                # Population statistics (after distribution)
                "avg_fitness_generation": 0.0001,
                "avg_fitness_elites": 0.0001,
                "avg_fitness_non_elites": 0.0001,
                "parents": [],
                "top_10": [],
                "variants_created": variants_created,
                "mutation_variants": mutation_variants,
                "crossover_variants": crossover_variants,
                "elites_threshold": 0.0001,
                "removal_threshold": 0.0001,
                "elites_count": 0,
                "selection_mode": selection_mode,  # Add selection mode for this generation
            }
            evolution_tracker.setdefault("generations", []).append(new_gen)
            _logger.info("Created new generation entry %d with max_score_variants %.4f and %d variants", gen_number, best_score, variants_created)
        
        # Sort generations by generation number
        evolution_tracker["generations"].sort(key=lambda x: x["generation_number"])
        
        # Note: Population threshold calculations are now handled by the centralized
        # calculate_and_update_population_thresholds function in utils/population_io.py
        # This ensures single source of truth for threshold calculations
        
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
            score = gen_entry.get("max_score_variants", 0.0001)
            all_scores.append(score)
            if gen_entry.get("generation_number") == total_generations - 1:  # Latest generation
                best_scores.append(score)
        
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0001
        best_avg_score = sum(best_scores) / len(best_scores) if best_scores else 0.0001
        max_score = max(all_scores) if all_scores else 0.0001
        min_score = min(all_scores) if all_scores else 0.0001
        
        # Calculate variant statistics for global tracker
        total_variants_created = 0
        total_mutation_variants = 0
        total_crossover_variants = 0
        
        for gen_entry in evolution_tracker.get("generations", []):
            total_variants_created += gen_entry.get("variants_created") or 0
            total_mutation_variants += gen_entry.get("mutation_variants") or 0
            total_crossover_variants += gen_entry.get("crossover_variants") or 0
        
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
                "max_score_variants": max_score,
                "min_score_variants": min_score,
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
                prompt_detail["initial_score"] = gen_0.get("max_score_variants", 0.0)
            
            # Best score (latest generation)
            latest_gen = max(generations, key=lambda g: g.get("generation_number", 0))
            prompt_detail["best_score"] = latest_gen.get("max_score_variants", 0.0)
            prompt_detail["score_improvement"] = prompt_detail["best_score"] - prompt_detail["initial_score"]
            
            # Total variants created
            prompt_detail["variants_created"] = sum(gen.get("variants_created") or 0 for gen in generations)
        
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
def run_evolution(north_star_metric, log_file=None, threshold=0.99, current_cycle=None, max_variants=1, max_num_parents=4, operators="all"):
    """Run one evolution generation with comprehensive logging and steady state support"""
    # Set up dynamic paths for this run
    outputs_path = get_outputs_path()
    population_path = outputs_path / "non_elites.json"
    evolution_tracker_path = outputs_path / "EvolutionTracker.json"
    
    logger = get_logger("RunEvolution", log_file)
    logger.info("Starting evolution: cycle=%s, metric=%s", current_cycle, north_star_metric)

    # Check for population file
    if not population_path.exists():
        logger.error("Population file not found: %s", population_path)
        raise FileNotFoundError(f"Population file not found: {population_path}")

    # Phase 2: Load population with error handling
    try:
        _, _, load_population, _, _, _, _, _, _, _, _, _, _ = get_population_io()
        population = load_population(str(outputs_path), logger=logger)
        logger.debug("Loaded %d genomes", len(population))
    except Exception as e:
        logger.error("Unexpected error loading population: %s", e, exc_info=True)
        raise

    # Phase 3: Check threshold and update evolution tracker
    evolution_tracker = check_threshold_and_update_tracker(population, north_star_metric, log_file, threshold)

    # Phase 4: Check if evolution should continue
    evolution_status = get_pending_status(evolution_tracker, logger)
    
    if evolution_status == "complete":
        logger.info("Evolution completed - threshold achieved globally")
        return

    # Phase 5: Initialize evolution engine
    try:
        EvolutionEngine = get_EvolutionEngine()
        engine = EvolutionEngine(north_star_metric, log_file, current_cycle=current_cycle, max_variants=max_variants, adaptive_selection_after=5, max_num_parents=max_num_parents, operators=operators, outputs_path=outputs_path)
        # Don't load population into memory - let EvolutionEngine load lazily
        # engine.genomes = population  # ← COMMENTED OUT FOR MEMORY OPTIMIZATION
        engine.update_next_id()
        logger.debug("EvolutionEngine next_id set to %d", engine.next_id)
    except Exception as e:
        logger.error("Failed to initialize evolution engine: %s", e, exc_info=True)
        raise

    # Phase 6: Process global evolution
    # EvolutionEngine handles parent selection, parents.json, and intra-temp.json deduplication (within temp.json itself).
    # Cross-file deduplication (checking against elites.json, non_elites.json) is handled below in _check_and_move_genomes_from_temp().
    try:
        logger.info("Processing global evolution")
        logger.debug("Calling generate_variants_global()")
        # Reset temp.json before variant generation
        _reset_temp_json(logger)
        # Generate variants and update temp.json (handles parent selection and variant deduplication)
        engine.generate_variants_global(evolution_tracker=evolution_tracker)
        
        # Get operator statistics after variant generation
        operator_stats_dict = engine.operator_stats.to_dict()
        logger.debug(f"Operator statistics: {operator_stats_dict}")
        
        # Count variants from temp.json for logging
        temp_path = outputs_path / "temp.json"
        variant_count = 0
        if temp_path.exists():
            with open(temp_path, 'r', encoding='utf-8') as f:
                temp_variants = json.load(f)
                variant_count = len(temp_variants)
        # Update population index after evolution
        try:
            update_population_index_single_file(str(outputs_path), len(engine.genomes), logger=logger)
            logger.debug("Updated population index after evolution")
        except Exception as e:
            logger.error("Failed to update population index: %s", e, exc_info=True)
    except Exception as e:
        logger.error("Failed to process global evolution: %s", e, exc_info=True)
        raise
    logger.debug("Evolution processing completed")

    # Phase 4: Deduplicate variants in temp.json (cross-file check)
    # Remove variants that already exist in elites.json and non_elites.json
    try:
        duplicates_removed = _deduplicate_variants_in_temp(logger, engine.operator_stats)
        logger.debug("Deduplicated variants (%d duplicates removed)", duplicates_removed)
    except Exception as e:
        logger.error("Failed to deduplicate variants in temp.json: %s", e, exc_info=True)
        raise

    # Update EvolutionTracker with new generation data
    try:
        current_generation = current_cycle
        if current_generation is None:
            logger.error("current_cycle is None - cannot determine generation number")
            return
        new_generation_data = {
            "generation_number": current_generation,
            "genome_id": None,  # Will be updated with best genome
            "max_score_variants": 0.0,   # Will be updated with best score
            "parents": [],
            "elites_threshold": threshold,
            "operator_statistics": operator_stats_dict
        }
        
        # EvolutionTracker update will be handled in main.py after evaluation
        # This ensures we track the best genome from evaluated variants
    
    except Exception as e:
        logger.error("Failed to prepare EvolutionTracker data: %s", e, exc_info=True)
    
    # Return operator statistics for use by main.py
    return {
        "operator_statistics": engine.operator_stats.to_dict(),
        "total_genomes": len(engine.genomes)
    }



