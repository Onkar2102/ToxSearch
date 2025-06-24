## @file RunEvolution.py
# @author Onkar Shelar (os9660@rit.edu)
# @brief Main script for evolving LLM input prompts using mutation operators.

import json
import os
import time
from typing import Dict, Any, List, Optional
from ea.EvolutionEngine import EvolutionEngine
from ea.TextVariationOperators import TextVariationOperators
from utils.initialize_population import load_and_initialize_population, sort_population_json
from utils.custom_logging import get_logger, PerformanceLogger
import nltk
import logging

from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
population_path = project_root / "outputs" / "Population.json"

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

        # Load population with error handling
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

        # Define status path for evolution status output
        status_path = project_root / "outputs" / "EvolutionStatus.json"

        logger.debug("Sorting population by prompt_id ASC, north_star_metric DESC, id ASC...")

        # Sort population with error handling
        with PerformanceLogger(logger, "Sort Population"):
            try:
                population = sort_population_json(
                    population,
                    sort_keys=["prompt_id", lambda g: g.get(north_star_metric, 0.0), "id"],
                    reverse_flags=[False, True, False],
                    log_file=log_file
                )
                logger.debug("Population sorted successfully")
            except Exception as e:
                logger.error("Failed to sort population: %s", e, exc_info=True)
                raise
        
        # Save sorted population
        with PerformanceLogger(logger, "Save Sorted Population"):
            try:
                with open(population_path, 'w', encoding='utf-8') as f:
                    json.dump(population, f, indent=4)
                logger.info("Population re-saved in sorted order")
            except Exception as e:
                logger.error("Failed to save sorted population: %s", e, exc_info=True)
                raise

        # Initialize evolution engine
        with PerformanceLogger(logger, "Initialize Evolution Engine"):
            try:
                engine = EvolutionEngine(north_star_metric, log_file)
                engine.genomes = population
                engine.update_next_id()
                logger.debug("EvolutionEngine next_id set to %d", engine.next_id)
            except Exception as e:
                logger.error("Failed to initialize evolution engine: %s", e, exc_info=True)
                raise

        # Collect prompt evolution status
        with PerformanceLogger(logger, "Collect Prompt Status"):
            try:
                prompt_status = {}
                for genome in population:
                    pid = genome["prompt_id"]
                    if pid not in prompt_status:
                        prompt_status[pid] = genome.get("status")
                logger.debug("Collected prompt status map for %d prompt_ids", len(prompt_status))

                completed_prompt_ids = {pid for pid, status in prompt_status.items() if status == "complete"}
                pending_prompt_ids = sorted(pid for pid, status in prompt_status.items() if status == "pending_evolution")

                logger.info("Completed prompt_ids (skipped): %s", sorted(completed_prompt_ids))
                logger.info("Pending prompt_ids (to process): %s", pending_prompt_ids)
            except Exception as e:
                logger.error("Failed to collect prompt status: %s", e, exc_info=True)
                raise

        # Track evolution status for all genomes
        with PerformanceLogger(logger, "Track Evolution Status"):
            try:
                evolution_status = []
                for genome in population:
                    evolution_status.append({
                        "generation": genome["generation"],
                        "prompt_id": genome["prompt_id"],
                        "status": genome.get("status", "unknown")
                    })
                logger.debug("Initialized evolution status tracking for %d genomes", len(evolution_status))
            except Exception as e:
                logger.error("Failed to track evolution status: %s", e, exc_info=True)
                raise

        logger.debug("Initialized EvolutionEngine: starting next_id=%d", engine.next_id)

        # Process each pending prompt
        processed_count = 0
        error_count = 0
        
        for prompt_id in pending_prompt_ids:
            with PerformanceLogger(logger, "Process Prompt", prompt_id=prompt_id):
                try:
                    logger.info("Processing prompt_id=%d", prompt_id)
                    logger.debug("Calling generate_variants() for prompt_id=%d", prompt_id)
                    
                    start_time = time.time()
                    engine.generate_variants(prompt_id)
                    processing_time = time.time() - start_time
                    
                    logger.info("Completed variant generation for prompt_id=%d in %.3f seconds", 
                               prompt_id, processing_time)
                    processed_count += 1

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

        # Sort population before deduplication
        with PerformanceLogger(logger, "Sort Population Before Deduplication"):
            try:
                population = engine.genomes
                population = sort_population_json(
                    population,
                    sort_keys=["prompt_id", lambda g: g.get(north_star_metric, 0.0), "id"],
                    reverse_flags=[False, True, False],
                    log_file=log_file
                )
                logger.debug("Sorted population by prompt_id ASC, north_star_metric DESC, id ASC before deduplication")
            except Exception as e:
                logger.error("Failed to sort population before deduplication: %s", e, exc_info=True)
                raise

        # Deduplicate population
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

                # Sort final population again for consistency
                final_population = sort_population_json(
                    final_population,
                    sort_keys=["prompt_id", lambda g: g.get(north_star_metric, 0.0), "id"],
                    reverse_flags=[False, True, False],
                    log_file=log_file
                )

                logger.info("Deduplicated population: %d → %d (removed %d duplicates)", 
                           len(population), len(final_population), duplicates_removed)
            except Exception as e:
                logger.error("Failed to deduplicate population: %s", e, exc_info=True)
                raise

        # Save final population
        with PerformanceLogger(logger, "Save Final Population"):
            try:
                with open(population_path, 'w', encoding='utf-8') as f:
                    json.dump(final_population, f, indent=4)
                logger.info("Population re-saved in sorted and deduplicated order")
            except Exception as e:
                logger.error("Failed to save final population: %s", e, exc_info=True)
                raise

        # Save evolution status to JSON file
        with PerformanceLogger(logger, "Save Evolution Status"):
            try:
                with open(status_path, 'w', encoding='utf-8') as f:
                    json.dump(evolution_status, f, indent=4)
                logger.info("Saved evolution status for %d records to %s", len(evolution_status), status_path)
            except Exception as e:
                logger.error("Failed to save evolution status: %s", e, exc_info=True)
                raise

        # Log final summary
        logger.info("Evolution run completed successfully:")
        logger.info("  - Total genomes processed: %d", len(population))
        logger.info("  - Prompts processed: %d", processed_count)
        logger.info("  - Errors encountered: %d", error_count)
        logger.info("  - Final population size: %d", len(final_population))
        logger.info("  - Evolution status records: %d", len(evolution_status))

