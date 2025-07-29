## @file RunEvolution.py
# @author Onkar Shelar (os9660@rit.edu)
# @brief Main script for evolving LLM input prompts using mutation operators.

import json
import os
import time
from typing import Dict, Any, List, Optional
from ea.EvolutionEngine import EvolutionEngine
from ea.TextVariationOperators import TextVariationOperators
from utils.initialize_population import load_and_initialize_population
from utils.custom_logging import get_logger, PerformanceLogger
import nltk
import logging

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

def check_threshold_and_update_tracker(population, north_star_metric, logger, log_file=None):
    """Check threshold achievement and update evolution tracker (flat list version)"""
    with PerformanceLogger(logger, "Check Threshold and Update Tracker"):
        try:
            threshold = 0.9999
            # Load existing evolution tracker (flat list)
            if evolution_tracker_path.exists():
                with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                    evolution_tracker = json.load(f)
            else:
                evolution_tracker = []

            # Since population is sorted by prompt_id ASC, north_star_metric DESC, id DESC
            current_prompt_id = None
            completed_prompt_ids = set()
            max_scores = {}
            max_genome_ids = {}
            for genome in population:
                prompt_id = genome["prompt_id"]
                if current_prompt_id != prompt_id:
                    current_prompt_id = prompt_id
                    north_star_score = genome.get("moderation_result", {}).get("scores", {}).get(north_star_metric, 0.0)
                    max_scores[prompt_id] = north_star_score
                    max_genome_ids[prompt_id] = genome["id"]
                    if isinstance(north_star_score, (int, float)) and north_star_score >= threshold:
                        completed_prompt_ids.add(prompt_id)
                        logger.info("Prompt_id %d achieved threshold with score %.4f", prompt_id, north_star_score)

            # Update evolution tracker for each prompt_id
            for prompt_id in max_scores.keys():
                entry = find_tracker_entry(evolution_tracker, prompt_id)
                if entry is None:
                    # Create new entry
                    entry = {
                        "prompt_id": prompt_id,
                        "status": "complete" if prompt_id in completed_prompt_ids else "not_complete",
                        "total_generations": 1,
                        "generations": [
                            {
                                "generation_number": 0,
                                "genome_id": max_genome_ids[prompt_id],
                                "max_score": max_scores[prompt_id],
                                "mutation": None,
                                "crossover": None
                            }
                        ]
                    }
                    evolution_tracker.append(entry)
                else:
                    entry["status"] = "complete" if prompt_id in completed_prompt_ids else "not_complete"
                    # Optionally update max_score/genome_id for gen 0 if needed
                    if entry["generations"] and entry["generations"][0]["generation_number"] == 0:
                        entry["generations"][0]["max_score"] = max_scores[prompt_id]
                        entry["generations"][0]["genome_id"] = max_genome_ids[prompt_id]

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
                        north_star_score = genome.get("moderation_result", {}).get("scores", {}).get(north_star_metric, 0.0)
                        if north_star_score == max_scores[genome["prompt_id"]]:
                            completed_genomes.append(genome)
                logger.info("Marked %d genomes from %d prompt_ids as complete", marked_count, len(completed_prompt_ids))
                logger.info("Completed prompt_ids: %s", sorted(completed_prompt_ids))
                completed_filename = f"outputs/completed_genomes_{north_star_metric}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                with open(completed_filename, 'w', encoding='utf-8') as f:
                    json.dump(completed_genomes, f, indent=2, ensure_ascii=False)
                logger.info("Saved %d completed genomes to %s", len(completed_genomes), completed_filename)
                with open(population_path, 'w', encoding='utf-8') as f:
                    json.dump(population, f, indent=4)
                logger.info("Updated population with completed statuses")
            else:
                logger.info("No prompt_ids achieved the threshold of %.4f", threshold)
            with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(evolution_tracker, f, indent=4, ensure_ascii=False)
            logger.info("Updated evolution tracker with threshold check results")
            return evolution_tracker
        except Exception as e:
            logger.error("Failed to check threshold and update tracker: %s", e, exc_info=True)
            raise

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
            
            new_gen = {
                "generation_number": gen_number,
                "genome_id": best_genome_id,
                "max_score": best_score,
                "mutation": mutation_info,
                "crossover": crossover_info,
                "variants_created": generation_data.get("variants_created", 0),
                "mutation_variants": generation_data.get("mutation_variants", 0),
                "crossover_variants": generation_data.get("crossover_variants", 0)
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

## @brief Main entry point: runs one evolution generation, applying selection and variation to prompts.
# @return None
def run_evolution(north_star_metric, log_file=None):
    """Run one evolution generation with comprehensive logging"""
    with PerformanceLogger(get_logger("RunEvolution", log_file), "Run Evolution", 
                          north_star_metric=north_star_metric, population_path=str(population_path)):
        
        logger = get_logger("RunEvolution", log_file)
        logger.info("Starting evolution run using population file: %s", population_path)
        logger.info("North star metric: %s", north_star_metric)

        if not population_path.exists():
            logger.error("Population file not found: %s", population_path)
            raise FileNotFoundError(f"{population_path} not found.")

        # Phase 1: Initialize evolution tracker
        initialize_evolution_tracker(logger, log_file)

        # Phase 2: Load population with error handling
        with PerformanceLogger(logger, "Load Population"):
            try:
                with open(str(population_path), 'r', encoding='utf-8') as f:
                    population = json.load(f)
                logger.info("Successfully loaded population with %d genomes", len(population))
            except json.JSONDecodeError as e:
                logger.error("Failed to parse population JSON: %s", e, exc_info=True)
                raise
            except Exception as e:
                logger.error("Unexpected error loading population: %s", e, exc_info=True)
                raise

        # Phase 3: Check threshold and update evolution tracker
        evolution_tracker = check_threshold_and_update_tracker(population, north_star_metric, logger, log_file)

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

                    # Update evolution tracker with generation data
                    update_evolution_tracker_with_generation(prompt_id, generation_data, evolution_tracker, logger)

                    # Save updated population after each prompt
                    with PerformanceLogger(logger, "Save Population After Prompt", prompt_id=prompt_id):
                        try:
                            with open(population_path, 'w', encoding='utf-8') as f:
                                json.dump(engine.genomes, f, indent=4)
                            logger.debug("Saved updated population after processing prompt_id=%d", prompt_id)
                        except Exception as e:
                            logger.error("Failed to save population after prompt_id=%d: %s", prompt_id, e, exc_info=True)
                            raise
                            
                except Exception as e:
                    logger.error("Failed to process prompt_id=%d: %s", prompt_id, e, exc_info=True)
                    error_count += 1

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
                with open(population_path, 'w', encoding='utf-8') as f:
                    json.dump(final_population, f, indent=4)
                logger.info("Population re-saved in sorted and deduplicated order")
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

