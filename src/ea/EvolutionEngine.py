"""
EvolutionEngine.py

Core evolutionary algorithm engine for text generation optimization.

This module implements the main evolution logic including parent selection,
operator application, and population management. Uses steady-state evolution
with elites preservation and multi-API moderation scoring.

Author: Onkar Shelar (os9660@rit.edu)
"""

import json
import random
from typing import List, Dict, Any, Optional
from utils import get_custom_logging
from .operator_helpers import get_applicable_operators
from .ParentSelector import ParentSelector
from itertools import combinations
from pathlib import Path

class EvolutionEngine:

    def __init__(self, north_star_metric, log_file, current_cycle=None, max_variants=10):
        self.genomes: List[Dict] = []
        self.next_id = 0
        self.north_star_metric = north_star_metric
        self.log_file = log_file
        self.current_cycle = current_cycle  # Current evolution cycle number
        self.use_steady_state = True
        self.max_variants = max_variants  # Maximum number of variants to generate per operator
        get_logger, _, _, _ = get_custom_logging()
        self.logger = get_logger("EvolutionEngine", log_file)
        self.parent_selector = ParentSelector(north_star_metric, log_file)
        self.logger.debug(f"EvolutionEngine initialized with next_id={self.next_id}, north_star_metric={north_star_metric}, current_cycle={current_cycle}, max_variants={max_variants}, use_steady_state=True")

    def _extract_north_star_score(self, genome: Dict) -> float:
        """
        Extract the north star metric score from a genome.
        
        Attempts to retrieve the score for the configured north star metric
        from the genome's moderation results, trying Google API scores first
        and falling back to OpenAI scores if needed.
        
        Args:
            genome (Dict): Genome dictionary containing moderation results
            
        Returns:
            float: Score for the north star metric, or 0.0 if not found
        """
        if not genome or not genome.get("moderation_result"):
            return 0.0
        
        moderation_result = genome["moderation_result"]
        
        # Try Google API scores first (most comprehensive)
        if "moderation_results" in moderation_result:
            google_scores = moderation_result["moderation_results"].get("google", {})
            if "scores" in google_scores:
                score = google_scores["scores"].get(self.north_star_metric, 0.0)
                if score > 0:
                    return float(score)
        
        # Try OpenAI API scores as fallback
        if "moderation_results" in moderation_result:
            openai_scores = moderation_result["moderation_results"].get("openai", {})
            if "scores" in openai_scores:
                score = openai_scores["scores"].get(self.north_star_metric, 0.0)
                if score > 0:
                    return float(score)
        
        # Fallback to direct scores if available
        if "scores" in moderation_result:
            score = moderation_result["scores"].get(self.north_star_metric, 0.0)
            if score > 0:
                return float(score)
        
        return 0.0

    def update_next_id(self):
        if self.genomes:
            # Convert string IDs to integers for arithmetic, then back to string
            max_id = max(int(g["id"]) for g in self.genomes)
            self.next_id = max_id + 1
        else:
            self.next_id = 1  # Start from 1 since we removed "genome_" prefix
        self.logger.debug(f"Updated next_id to {self.next_id}")
    
    def generate_variants_global(self) -> Dict[str, Any]:
        self.logger.debug(f"Generating variants globally for evolution cycle {self.current_cycle}")

        # Ensure ``next_id`` is always in sync with the current population each
        # time we start a new generation cycle.  This prevents reused IDs when
        # the engine object persists across multiple calls.
        self.update_next_id()

        # Get all genomes for global evolution
        all_genomes = [g for g in self.genomes if g is not None]
        if not all_genomes:
            self.logger.error(f"No genomes found. Exiting evolution process.")
            raise SystemExit(1)

        # Log detailed information about available genomes
        completed_genomes = [g for g in all_genomes if g.get("status") == "complete"]
        pending_genomes = [g for g in all_genomes if g.get("status") == "pending_evolution"]
        other_genomes = [g for g in all_genomes if g.get("status") not in ["complete", "pending_evolution"]]
        
        self.logger.info(f"Global genome breakdown: {len(completed_genomes)} completed, {len(pending_genomes)} pending_evolution, {len(other_genomes)} other")
        
        if completed_genomes:
            max_score = max([self._extract_north_star_score(g) for g in completed_genomes])
            self.logger.info(f"Best completed genome score globally: {max_score}")
        else:
            self.logger.warning(f"No completed genomes found")

        mutation_parent, crossover_parents = self.parent_selector.select_parents_adaptive_tournament()
        
        # Log parent selection results
        if mutation_parent is None:
            self.logger.warning(f"No mutation parent selected with adaptive tournament")
        else:
            score = self._extract_north_star_score(mutation_parent)
            self.logger.info(f"Selected mutation parent with adaptive tournament: genome_id={mutation_parent['id']}, score={score}")
        
        if crossover_parents is None:
            self.logger.warning(f"No crossover parents selected with adaptive tournament")
        else:
            self.logger.info(f"Selected {len(crossover_parents)} crossover parents with adaptive tournament: {[p['id'] for p in crossover_parents]}")

        existing_prompts = set(g["prompt"].strip().lower() for g in self.genomes if g is not None)

        offspring = []
        generation_data = {
            "generation_number": self.current_cycle,  # Add generation number
            "parents": [],
            "variants_created": 0,
            "mutation_variants": 0,
            "crossover_variants": 0
        }

        # Track parent information for parent selection tracker
        def get_parent_info(parent):
            return {
                "id": parent["id"],
                "score": self._extract_north_star_score(parent),
                "parents_id": parent.get("parents", None)
            }

        # For generation 0, parents is null
        if all(g["generation"] == 0 for g in all_genomes):
            # Parent tracking has been removed
            pass

        # For mutation, use the topmost genome as parent
        if mutation_parent:
            generation_data["parents"].append({
                "id": mutation_parent["id"],
                "north_star_score": self._extract_north_star_score(mutation_parent),
                "generation": mutation_parent["generation"],
                "type": "mutation_parent"
            })
            # Mutation parent tracking has been removed

        # For crossover, use the top 3 genomes as parents
        if crossover_parents:
            for parent in crossover_parents:
                generation_data["parents"].append({
                    "id": parent["id"],
                    "north_star_score": self._extract_north_star_score(parent),
                    "generation": parent["generation"],
                    "type": "crossover_parent"
                })
            # Crossover parent tracking has been removed

        # --- Mutation phase -------------------------------------------------
        if mutation_parent:
            mutation_operators = get_applicable_operators(1, self.north_star_metric, self.log_file)
            self.logger.debug(f"Running mutation globally with {len(mutation_operators)} operators.")
            for op in mutation_operators:
                if op.operator_type != "mutation":
                    continue

                try:
                    # Pass correct input type based on operator class
                    operator_input = {
                        "parent_data": mutation_parent,
                        "max_variants": self.max_variants
                    }
                    variants = op.apply(operator_input)
                    for vp in variants:
                        norm_vp = vp.strip().lower()
                        if norm_vp in existing_prompts:
                            continue
                        existing_prompts.add(norm_vp)
                        child = {
                            "id": str(self.next_id),
                            "prompt": vp,
                            "model_provider": None,
                            "model_name": None,
                            "moderation_result": None,
                            "operator": op.name,
                            "parents": [mutation_parent["id"]],
                            "generation": self.current_cycle,  # Use current evolution cycle instead of parent + 1
                            "status": "pending_generation",
                            "creation_info": {
                                "type": "mutation",
                                "operator": op.name,
                                "source_generation": mutation_parent["generation"],
                                "evolution_cycle": self.current_cycle  # Track which evolution cycle created this
                            }
                        }
                        self.next_id += 1
                        self.logger.debug(f"Created mutation variant id={child['id']} globally (evolution cycle {self.current_cycle})")
                        self.logger.debug(f"Mutation variant prompt: '{vp[:60]}...'")
                        offspring.append(child)
                except Exception as e:
                    self.logger.error(f"[Mutation Error] {op.name}: {e}")

        # --- Crossover phase -------------------------------------------------
        if crossover_parents:
            crossover_operators = get_applicable_operators(len(crossover_parents), self.north_star_metric, self.log_file)
            self.logger.debug(f"Running crossover globally with {len(crossover_parents)} parents and {len(crossover_operators)} operators.")
            for op in crossover_operators:
                if op.operator_type != "crossover":
                    continue

                for parent_pair in combinations(crossover_parents, 2):  # All pairs of parents
                    try:
                        # Pass enriched parent data and max_variants to crossover operators
                        # Pass correct input type for crossover operators
                        operator_input = {
                            "parent_data": list(parent_pair),
                            "max_variants": self.max_variants
                        }
                        variants = op.apply(operator_input)
                        for vp in variants:
                            norm_vp = vp.strip().lower()
                            if norm_vp in existing_prompts:
                                continue
                            existing_prompts.add(norm_vp)
                            child = {
                                "id": str(self.next_id),
                                "prompt": vp,
                                "model_provider": None,
                                "model_name": None,
                                "moderation_result": None,
                                "operator": op.name,
                                "parents": [p["id"] for p in parent_pair],
                                "generation": self.current_cycle,  # Use current evolution cycle instead of max parent + 1
                                "status": "pending_generation",
                                "creation_info": {
                                    "type": "crossover",
                                    "operator": op.name,
                                    "source_generation": max(p["generation"] for p in parent_pair),
                                    "evolution_cycle": self.current_cycle  # Track which evolution cycle created this
                                }
                            }
                            self.next_id += 1
                            self.logger.debug(f"Created crossover variant id={child['id']} globally (evolution cycle {self.current_cycle})")
                            self.logger.debug(f"Crossover variant prompt: '{vp[:60]}...'")
                            offspring.append(child)
                    except Exception as e:
                        self.logger.error(f"[Crossover Error] {op.name} with parents {[p['id'] for p in parent_pair]}: {e}")

        # --------------------------------------------------------------------
        # Final deduplication – run **once** over the combined offspring list
        # to avoid double-inserting the mutation children that are already
        # present when we process the crossover phase.
        # --------------------------------------------------------------------

        unique_offspring = {}
        for child in offspring:
            key = child["prompt"].strip().lower()
            if key not in unique_offspring:
                unique_offspring[key] = child

        # Classify counts
        generation_data["mutation_variants"] = sum(
            1 for c in unique_offspring.values() if c["creation_info"]["type"] == "mutation"
        )
        generation_data["crossover_variants"] = sum(
            1 for c in unique_offspring.values() if c["creation_info"]["type"] == "crossover"
        )
        generation_data["variants_created"] = len(unique_offspring)

        # Add new variants to the in-memory population (no separate generation files)
        self.genomes.extend(unique_offspring.values())
        
        self.logger.info(
            "Added %d unique variants to the population (mutation: %d, crossover: %d) for evolution cycle %d.",
            generation_data["variants_created"],
            generation_data["mutation_variants"],
            generation_data["crossover_variants"],
            self.current_cycle
        )
        
        return generation_data



    # Method removed - no longer needed for global evolution

