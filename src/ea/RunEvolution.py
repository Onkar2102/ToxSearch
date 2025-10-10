## @file RunEvolution.py
# @brief Main script for evolving LLM input prompts using mutation operators.

import json
from typing import Dict, Any, List, Optional
# Lazy import to avoid torch dependency issues
def get_EvolutionEngine():
    """Lazy import of EvolutionEngine to avoid torch dependency issues"""
    from ea.EvolutionEngine import EvolutionEngine
    return EvolutionEngine
from utils import get_population_io, get_custom_logging

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

def _deduplicate_variants_in_temp(logger):
    """
    Deduplicate variants in temp.json by comparing against existing genomes in all files.
    This function ONLY performs deduplication and does NOT distribute genomes.
    
    Args:
        logger: Logger instance
        
    Returns:
        int: Number of duplicates removed
    """
    try:
        outputs_path = get_outputs_path()
        temp_path = outputs_path / "temp.json"
        elites_path = outputs_path / "elites.json"
        population_path = outputs_path / "Population.json"
        most_toxic_path = outputs_path / "most_toxic.json"
        
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
        
        # Load from population.json
        if population_path.exists():
            with open(population_path, 'r', encoding='utf-8') as f:
                population = json.load(f)
                for genome in population:
                    if genome and genome.get("prompt"):
                        existing_prompts.add(genome["prompt"].strip().lower())
                        existing_ids.add(genome.get("id"))
        
        # Load from most_toxic.json
        if most_toxic_path.exists():
            with open(most_toxic_path, 'r', encoding='utf-8') as f:
                most_toxic = json.load(f)
                for genome in most_toxic:
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
                logger.debug(f"Removing duplicate genome {genome_id}")
                continue
            
            # Keep unique variant
            unique_variants.append(variant)
        
        # Save deduplicated variants back to temp.json
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(unique_variants, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Deduplication complete: {len(temp_variants)} → {len(unique_variants)} variants ({duplicates_removed} duplicates removed)")
        
        return duplicates_removed
        
    except Exception as e:
        logger.error(f"Failed to deduplicate variants in temp.json: {e}")
        raise


def distribute_genomes_by_threshold(temp_path, elite_threshold, north_star_metric, logger, north_star_threshold=0.99):
    """
    Distribute genomes from temp.json based on thresholds:
    - Genomes < elite_threshold → Population.json (for further evolution)
    - Genomes ≥ elite_threshold but < north_star_threshold → elites.json (elite genomes)
    - Genomes ≥ north_star_threshold → most_toxic.json (toxic genomes)
    
    Args:
        temp_path: Path to temp.json file
        elite_threshold: Toxicity threshold for elite selection
        north_star_metric: The metric to use for scoring
        logger: Logger instance
        north_star_threshold: Threshold for toxic genomes (default: 0.99)
        
    Returns:
        dict: Distribution statistics
    """
    try:
        outputs_path = get_outputs_path()
        elites_path = outputs_path / "elites.json"
        population_path = outputs_path / "Population.json"
        most_toxic_path = outputs_path / "most_toxic.json"
        
        # Load temp.json variants
        if not temp_path.exists():
            logger.warning("temp.json not found for distribution")
            return {"elites_moved": 0, "population_moved": 0, "toxic_moved": 0, "total_processed": 0}
            
        with open(temp_path, 'r', encoding='utf-8') as f:
            temp_variants = json.load(f)
        
        if not temp_variants:
            logger.debug("No variants in temp.json to distribute")
            return {"elites_moved": 0, "population_moved": 0, "toxic_moved": 0, "total_processed": 0}
        
        # Categorize variants based on thresholds
        elites_to_move = []
        population_to_move = []
        toxic_to_move = []
        
        for variant in temp_variants:
            if not variant or not variant.get("prompt"):
                continue
                
            # Check toxicity score
            toxicity_score = _extract_north_star_score(variant, north_star_metric)
            genome_id = variant.get("id")
            
            if toxicity_score >= north_star_threshold:
                # Move to most_toxic.json (toxic genomes)
                variant["status"] = "complete"
                variant["completion_reason"] = f"North star threshold reached: {toxicity_score:.3f} >= {north_star_threshold:.3f}"
                toxic_to_move.append(variant)
                logger.debug(f"Genome {genome_id} marked as toxic (score: {toxicity_score:.3f})")
            elif toxicity_score >= elite_threshold:
                # Move to elites.json (elite genomes)
                variant["status"] = "complete"
                variant["completion_reason"] = f"Elite threshold reached: {toxicity_score:.3f} >= {elite_threshold:.3f}"
                elites_to_move.append(variant)
                logger.debug(f"Genome {genome_id} marked as elite (score: {toxicity_score:.3f})")
            else:
                # Move to Population.json (for further evolution)
                variant["status"] = "pending_generation"
                population_to_move.append(variant)
                logger.debug(f"Genome {genome_id} marked for population (score: {toxicity_score:.3f}) - below elite threshold")
        
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
            
            logger.info(f"Moved {len(elites_to_move)} elite genomes to elites.json")
        
        # Move non-elite genomes to Population.json
        if population_to_move:
            # Add to Population.json
            population_to_save = []
            if population_path.exists():
                with open(population_path, 'r', encoding='utf-8') as f:
                    population_to_save = json.load(f)
            
            population_to_save.extend(population_to_move)
            
            with open(population_path, 'w', encoding='utf-8') as f:
                json.dump(population_to_save, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Moved {len(population_to_move)} genomes to Population.json for further evolution")
        
        # Move toxic genomes to most_toxic.json
        if toxic_to_move:
            # Add to most_toxic.json
            toxic_to_save = []
            if most_toxic_path.exists():
                with open(most_toxic_path, 'r', encoding='utf-8') as f:
                    toxic_to_save = json.load(f)
            
            toxic_to_save.extend(toxic_to_move)
            
            with open(most_toxic_path, 'w', encoding='utf-8') as f:
                json.dump(toxic_to_save, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Moved {len(toxic_to_move)} toxic genomes to most_toxic.json")
        
        # Clear temp.json after successful distribution
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        
        distribution_stats = {
            "elites_moved": len(elites_to_move),
            "population_moved": len(population_to_move),
            "toxic_moved": len(toxic_to_move),
            "total_processed": len(temp_variants)
        }
        
        logger.info(f"Distribution complete: {distribution_stats['total_processed']} variants → {distribution_stats['elites_moved']} elites, {distribution_stats['population_moved']} population, {distribution_stats['toxic_moved']} toxic")
        
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
                evolution_tracker["status"] = "complete"
                logger.info("Global population achieved threshold with score %.4f (genome %s)", best_score, best_genome_id)
            else:
                evolution_tracker["status"] = "not_complete"
                logger.info("Global population best score: %.4f (genome %s), threshold not reached", best_score, best_genome_id)
        else:
            evolution_tracker["status"] = "not_complete"
            logger.info("No completed genomes found in global population")

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
                    "parents": [],
                    "elites_threshold": 0.0
                }]
                logger.info("Created generation 0 entry with best genome %s, score: %.4f", best_gen0_id, best_gen0_score)

        # Mark all genomes as complete if threshold is achieved globally
        if evolution_tracker["status"] == "complete":
            marked_count = 0
            for genome in population:
                if genome.get("status") != "complete":
                    genome["status"] = "complete"
                    genome["completion_reason"] = f"Global population achieved {north_star_metric} >= {threshold}"
                    marked_count += 1
            logger.info("Marked %d genomes as complete (global threshold achieved)", marked_count)
            # Note: Population management is handled by the main evolution loop
            # We don't save to elites.json here as it should only contain elite genomes
        else:
            logger.info("Global population has not achieved the threshold of %.4f", threshold)
        
        with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
            json.dump(evolution_tracker, f, indent=4, ensure_ascii=False)
        logger.info("Updated evolution tracker with threshold check results")
        
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
        logger.info("Global evolution status: %s", status)
        return status
    except Exception as e:
        logger.error("Failed to get pending status: %s", e, exc_info=True)
        raise

def update_evolution_tracker_with_generation_global(generation_data, evolution_tracker, logger, population=None, north_star_metric=None):
    """Update evolution tracker with generation data for global population"""
    _logger = logger or get_logger("update_evolution_tracker", log_file=None)
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
def run_evolution(north_star_metric, log_file=None, threshold=0.99, current_cycle=None, max_variants=5, max_num_parents=4):
    """Run one evolution generation with comprehensive logging and steady state support"""
    # Set up dynamic paths for this run
    outputs_path = get_outputs_path()
    population_path = outputs_path / "Population.json"
    evolution_tracker_path = outputs_path / "EvolutionTracker.json"
    
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

    # Phase 2: Load population with error handling
    try:
        _, _, load_population, _, _, _, _, _, _, _, _, _, _ = get_population_io()
        population = load_population(str(outputs_path), logger=logger)
        logger.info("Successfully loaded population with %d genomes", len(population))
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
        engine = EvolutionEngine(north_star_metric, log_file, current_cycle=current_cycle, max_variants=max_variants, adaptive_selection_after=5, max_num_parents=max_num_parents)
        engine.genomes = population
        engine.update_next_id()
        logger.debug("EvolutionEngine next_id set to %d", engine.next_id)
    except Exception as e:
        logger.error("Failed to initialize evolution engine: %s", e, exc_info=True)
        raise

    # Phase 6: Process global evolution
    # EvolutionEngine handles parent selection, parents.json, and intra-temp.json deduplication (within temp.json itself).
    # Cross-file deduplication (checking against elites.json, Population.json, most_toxic.json) is handled below in _check_and_move_genomes_from_temp().
    try:
        logger.info("Processing global evolution")
        logger.debug("Calling generate_variants_global()")
        # Reset temp.json before variant generation
        _reset_temp_json(logger)
        # Generate variants and update temp.json (handles parent selection and variant deduplication)
        engine.generate_variants_global(evolution_tracker=evolution_tracker)
        # Count variants from temp.json for logging
        temp_path = outputs_path / "temp.json"
        variant_count = 0
        if temp_path.exists():
            with open(temp_path, 'r', encoding='utf-8') as f:
                temp_variants = json.load(f)
                variant_count = len(temp_variants)
        logger.info("Generated %d variants globally", variant_count)
        # Update population index after evolution
        try:
            from utils.population_io import update_population_index_single_file
            update_population_index_single_file(str(outputs_path), len(engine.genomes), logger=logger)
            logger.debug("Updated population index after evolution")
        except Exception as e:
            logger.error("Failed to update population index: %s", e, exc_info=True)
    except Exception as e:
        logger.error("Failed to process global evolution: %s", e, exc_info=True)
        raise
    logger.info("Global evolution processing completed successfully")

    # Phase 4: Deduplicate variants in temp.json (cross-file check)
    # Remove variants that already exist in elites.json, most_toxic.json and population.json
    try:
        duplicates_removed = _deduplicate_variants_in_temp(logger)
        logger.info("Successfully deduplicated variants in temp.json (%d duplicates removed)", duplicates_removed)
    except Exception as e:
        logger.error("Failed to deduplicate variants in temp.json: %s", e, exc_info=True)
        raise

    # Update EvolutionTracker with new generation data
    try:
        current_generation = current_cycle if current_cycle is not None else evolution_tracker.get("total_generations", 1)
        new_generation_data = {
            "generation_number": current_generation,
            "genome_id": None,  # Will be updated with best genome
            "max_score": 0.0,   # Will be updated with best score
            "parents": [],
            "elites_threshold": threshold
        }
        
        # EvolutionTracker update will be handled in main.py after evaluation
        # This ensures we track the best genome from evaluated variants
        logger.info("EvolutionTracker update will be handled after evaluation phase")
    
    except Exception as e:
        logger.error("Failed to prepare EvolutionTracker data: %s", e, exc_info=True)
    
    # Log final summary
    logger.info("Evolution run completed successfully:")
    logger.info("  - Total genomes processed: %d", len(engine.genomes))
    logger.info("  - Evolution tracker updated: %s", evolution_tracker_path)
    
    return



