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
import os
from typing import List, Dict, Any, Optional
from utils import get_custom_logging
from .ParentSelector import ParentSelector
from itertools import combinations
from pathlib import Path

class EvolutionEngine:

    def __init__(self, north_star_metric, log_file, current_cycle=None, max_variants=3, adaptive_selection_after=5):
        self.genomes: List[Dict] = []
        self.next_id = 0
        self.north_star_metric = north_star_metric
        self.log_file = log_file
        self.current_cycle = current_cycle  # Current evolution cycle number
        self.use_steady_state = True
        self.max_variants = max_variants  # Maximum number of variants to generate per operator
        get_logger, _, _, _ = get_custom_logging()
        self.logger = get_logger("EvolutionEngine", log_file)
        self.parent_selector = ParentSelector(north_star_metric, log_file, adaptive_selection_after=adaptive_selection_after, max_variants=max_variants)
        # Initialize the shared generator instance
        self.generator = self._get_generator()
        
        self.logger.debug(f"EvolutionEngine initialized with next_id={self.next_id}, north_star_metric={north_star_metric}, current_cycle={current_cycle}, max_variants={max_variants}, adaptive_selection_after={adaptive_selection_after}, use_steady_state=True")


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
        
        # Try Google API scores first
        if "google" in moderation_result:
            google_scores = moderation_result["google"]
            if "scores" in google_scores:
                score = google_scores["scores"].get(self.north_star_metric, 0.0)
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
            # Use numeric IDs directly for arithmetic
            max_id = max(g["id"] if isinstance(g["id"], int) else int(g["id"]) for g in self.genomes)
            self.next_id = max_id + 1
        else:
            self.next_id = 1  # Start from 1
        self.logger.debug(f"Updated next_id to {self.next_id}")
    
    def _get_generator(self):
        """Get the shared LLaMA text generator instance."""
        from .operator_helpers import get_generator
        return get_generator()

    def _get_single_parent_operators(self):
        """Return list of mutation operators that require only a single parent."""
        from .llm_pos_aware_synonym_replacement import LLM_POSAwareSynonymReplacement
        from .mlm_operator import MLMOperator
        from .paraphrasing_operator import LLMBasedParaphrasingOperator
        from .llm_pos_aware_antonym_replacement import LLM_POSAwareAntonymReplacement
        from .stylistic_mutator import StylisticMutator
        from .llm_back_translation_operators import (
            LLMBackTranslationHIOperator, LLMBackTranslationFROperator, LLMBackTranslationDEOperator, LLMBackTranslationJAOperator, LLMBackTranslationZHOperator)
        return [
            LLM_POSAwareSynonymReplacement(log_file=self.log_file, max_variants=self.max_variants, num_POS_tags=1, generator=self.generator),
            MLMOperator(log_file=self.log_file, generator=self.generator),
            LLMBasedParaphrasingOperator(self.north_star_metric, log_file=self.log_file, generator=self.generator),
            LLM_POSAwareAntonymReplacement(log_file=self.log_file, max_variants=self.max_variants, num_POS_tags=1, generator=self.generator),
            StylisticMutator(log_file=self.log_file, generator=self.generator),
            LLMBackTranslationHIOperator(log_file=self.log_file, generator=self.generator),
            LLMBackTranslationFROperator(log_file=self.log_file, generator=self.generator),
            LLMBackTranslationDEOperator(log_file=self.log_file, generator=self.generator),
            LLMBackTranslationJAOperator(log_file=self.log_file, generator=self.generator),
            LLMBackTranslationZHOperator(log_file=self.log_file, generator=self.generator),
        ]

    def _get_multi_parent_operators(self):
        """Return list of crossover operators that require multiple parents."""
        from .semantic_similarity_crossover import SemanticSimilarityCrossover
        from .instruction_preserving_crossover import InstructionPreservingCrossover
        return [
            SemanticSimilarityCrossover(log_file=self.log_file),
            InstructionPreservingCrossover(north_star_metric=self.north_star_metric, log_file=self.log_file, generator=self.generator)
        ]



    def _reset_temp_json(self):
        """Reset temp.json to empty list at the start of variant generation."""
        try:
            temp_path = Path("outputs/temp.json")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2, ensure_ascii=False)
            self.logger.debug("Reset temp.json for new generation")
        except Exception as e:
            self.logger.error(f"Failed to reset temp.json: {e}")
            raise

    def generate_variants_global(self, x: int = 1, y: int = 1) -> Dict[str, Any]:
        """
        Generate variants globally for evolution cycle.
        
        Args:
            x (int): Number of random parents to select from elites.json
            y (int): Number of random parents to select from population.json
            
        Returns:
            Dict[str, Any]: Generation data including variants created
        """
        self.logger.debug(f"Generating variants globally for evolution cycle {self.current_cycle} with x={x}, y={y}")
        
        # Reset temp.json at the start of each generation
        self._reset_temp_json()

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

        # Call ParentSelector to select parents and save to parents.json
        selected_parents = self.parent_selector.adaptive_tournament_selection(x, y)
        
        # Log parent selection results
        if not selected_parents:
            self.logger.warning(f"No parents selected")
            return {
                "generation_number": self.current_cycle,
                "parents": [],
                "variants_created": 0,
                "mutation_variants": 0,
                "crossover_variants": 0
            }
        
        self.logger.info(f"Selected {len(selected_parents)} parents: {[p['id'] for p in selected_parents]}")

        # Load parents from parents.json for operators to use
        parents = self._load_parents_from_file()
        if not parents:
            self.logger.error("Failed to load parents from file")
            return {
                "generation_number": self.current_cycle,
                "parents": [],
                "variants_created": 0,
                "mutation_variants": 0,
                "crossover_variants": 0
            }

        generation_data = {
            "generation_number": self.current_cycle,
            "parents": [],
            "variants_created": 0,
            "mutation_variants": 0,
            "crossover_variants": 0
        }

        # Track parent information
        for parent in parents:
            generation_data["parents"].append({
                "id": parent["id"],
                "north_star_score": self._extract_north_star_score(parent),
                "generation": parent["generation"],
                "type": "parent"
            })

        # --- Crossover phase (done first) -----------------------------------
        crossover_variants_count = 0
        if len(parents) >= 2:
            crossover_operators = self._get_multi_parent_operators()
            self.logger.debug(f"Running crossover globally with {len(parents)} parents and {len(crossover_operators)} operators.")
            
            for op in crossover_operators:
                if op.operator_type != "crossover":
                    continue

                for parent_pair in combinations(parents, 2):  # All pairs of parents
                    try:
                        # Pass correct input type for crossover operators
                        operator_input = {
                            "parent_data": list(parent_pair),
                            "max_variants": self.max_variants
                        }
                        variants = op.apply(operator_input)
                        
                        # Save variants immediately to temp.json
                        if variants:
                            variants_to_save = []
                            for vp in variants:
                                child = {
                                    "id": self.next_id,  # Use numeric ID instead of string
                                    "prompt": vp,
                                    "model_name": None,
                                    "moderation_result": None,
                                    "operator": op.name,
                                    "parents": [p["id"] for p in parent_pair],
                                    "generation": self.current_cycle,
                                    "status": "pending_generation",
                                    "creation_info": {
                                        "type": "crossover",
                                        "operator": op.name,
                                        "source_generation": max(p["generation"] for p in parent_pair),
                                        "evolution_cycle": self.current_cycle
                                    }
                                }
                                self.next_id += 1
                                variants_to_save.append(child)
                                crossover_variants_count += 1
                            
                            # Save variants immediately to temp.json
                            self._append_variants_to_temp(variants_to_save)
                            self.logger.debug(f"Saved {len(variants_to_save)} crossover variants from {op.name}")
                            
                    except Exception as e:
                        self.logger.error(f"[Crossover Error] {op.name} with parents {[p['id'] for p in parent_pair]}: {e}")

        # --- Mutation phase (done after crossover) --------------------------
        mutation_variants_count = 0
        if len(parents) >= 1:
            mutation_operators = self._get_single_parent_operators()
            self.logger.debug(f"Running mutation globally with {len(parents)} parents and {len(mutation_operators)} operators.")
            
            for op in mutation_operators:
                if op.operator_type != "mutation":
                    continue

                # Apply mutation to all parents
                for parent in parents:
                    try:
                        # Pass correct input type based on operator class
                        operator_input = {
                            "parent_data": parent,
                            "max_variants": self.max_variants
                        }
                        variants = op.apply(operator_input)
                        
                        # Save variants immediately to temp.json
                        if variants:
                            variants_to_save = []
                            for vp in variants:
                                child = {
                                    "id": self.next_id,  # Use numeric ID instead of string
                                    "prompt": vp,
                                    "model_name": None,
                                    "moderation_result": None,
                                    "operator": op.name,
                                    "parents": [parent["id"]],
                                    "generation": self.current_cycle,
                                    "status": "pending_generation",
                                    "creation_info": {
                                        "type": "mutation",
                                        "operator": op.name,
                                        "source_generation": parent["generation"],
                                        "evolution_cycle": self.current_cycle
                                    }
                                }
                                self.next_id += 1
                                variants_to_save.append(child)
                                mutation_variants_count += 1
                            
                            # Save variants immediately to temp.json
                            self._append_variants_to_temp(variants_to_save)
                            self.logger.debug(f"Saved {len(variants_to_save)} mutation variants from {op.name}")
                            
                    except Exception as e:
                        self.logger.error(f"[Mutation Error] {op.name} with parent {parent['id']}: {e}")

        # Update generation data
        generation_data["mutation_variants"] = mutation_variants_count
        generation_data["crossover_variants"] = crossover_variants_count
        generation_data["variants_created"] = mutation_variants_count + crossover_variants_count

        # Perform deduplication on temp.json
        self._deduplicate_temp_json()
        
        self.logger.info(
            "Generated %d unique variants (mutation: %d, crossover: %d) for evolution cycle %d.",
            generation_data["variants_created"],
            generation_data["mutation_variants"],
            generation_data["crossover_variants"],
            self.current_cycle
        )
        
        # Clean up parent files after variant generation is complete
        self.clean_parent_files()
        
        return generation_data
    
    def _load_parents_from_file(self) -> List[Dict]:
        """
        Load parents from parents.json file.
        
        Returns:
            List[Dict]: List of parent genomes
        """
        try:
            parents_path = Path("outputs/parents.json")
            if not parents_path.exists():
                self.logger.warning("Parents file not found: %s", parents_path)
                return []
            
            with open(parents_path, 'r', encoding='utf-8') as f:
                parents_data = json.load(f)
            
            parents = parents_data.get("parents", [])
            
            self.logger.debug(f"Loaded {len(parents)} parents from file: {[p['id'] for p in parents]}")
            
            return parents
            
        except Exception as e:
            self.logger.error(f"Failed to load parents from file: {e}")
            return []
    

    def _append_variants_to_temp(self, variants: List[Dict]) -> None:
        """
        Append variants to temp.json file.
        
        Args:
            variants: List of variant genomes to append
        """
        try:
            temp_path = Path("outputs/temp.json")
            
            # Load existing variants
            if temp_path.exists():
                with open(temp_path, 'r', encoding='utf-8') as f:
                    existing_variants = json.load(f)
            else:
                existing_variants = []
            
            # Append new variants
            existing_variants.extend(variants)
            
            # Save back to file
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(existing_variants, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Appended {len(variants)} variants to temp.json (total: {len(existing_variants)})")
            
        except Exception as e:
            self.logger.error(f"Failed to append variants to temp.json: {e}")
            raise
    
    def _deduplicate_temp_json(self) -> None:
        """
        Deduplicate variants in temp.json and remove duplicates from elites.json and population.json.
        """
        try:
            temp_path = Path("outputs/temp.json")
            elites_path = Path("outputs/elites.json")
            population_path = Path("outputs/Population.json")
            
            # Load temp.json variants
            if not temp_path.exists():
                self.logger.warning("temp.json not found for deduplication")
                return
                
            with open(temp_path, 'r', encoding='utf-8') as f:
                temp_variants = json.load(f)
            
            if not temp_variants:
                self.logger.debug("No variants in temp.json to deduplicate")
                return
            
            # Load existing genomes from elites.json and population.json
            existing_prompts = set()
            
            # Load from elites.json
            if elites_path.exists():
                with open(elites_path, 'r', encoding='utf-8') as f:
                    elites = json.load(f)
                    for genome in elites:
                        if genome and genome.get("prompt"):
                            existing_prompts.add(genome["prompt"].strip().lower())
            
            # Load from population.json
            if population_path.exists():
                with open(population_path, 'r', encoding='utf-8') as f:
                    population = json.load(f)
                    for genome in population:
                        if genome and genome.get("prompt"):
                            existing_prompts.add(genome["prompt"].strip().lower())
            
            # Deduplicate within temp.json first
            seen_prompts = set()
            unique_variants = []
            duplicates_removed_internal = 0
            
            for variant in temp_variants:
                if variant and variant.get("prompt"):
                    norm_prompt = variant["prompt"].strip().lower()
                    if norm_prompt not in seen_prompts:
                        seen_prompts.add(norm_prompt)
                        unique_variants.append(variant)
                    else:
                        duplicates_removed_internal += 1
            
            # Remove duplicates that exist in elites.json or population.json
            final_variants = []
            duplicates_removed_external = 0
            
            for variant in unique_variants:
                if variant and variant.get("prompt"):
                    norm_prompt = variant["prompt"].strip().lower()
                    if norm_prompt not in existing_prompts:
                        final_variants.append(variant)
                    else:
                        duplicates_removed_external += 1
            
            # Save deduplicated variants back to temp.json
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(final_variants, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Deduplicated temp.json: {len(temp_variants)} → {len(final_variants)} variants (removed {duplicates_removed_internal} internal + {duplicates_removed_external} external duplicates)")
            
        except Exception as e:
            self.logger.error(f"Failed to deduplicate temp.json: {e}")
            raise

    def move_toxic_genomes_to_most_toxic(self, toxicity_threshold: float = 0.5):
        """
        Move genomes with toxicity scores above threshold or status 'complete' to most_toxic.json.
        Also mark genomes as complete if their score >= threshold.
        Update EvolutionTracker.json with most_toxic_counts.
        Args:
            toxicity_threshold (float): Threshold above which genomes are considered toxic
        """
        try:
            from pathlib import Path
            import json
            
            # Check all relevant JSON files for toxic genomes
            files_to_check = ["elites.json", "Population.json", "temp.json"]
            most_toxic_path = Path("outputs/most_toxic.json")
            
            # Load existing most_toxic.json
            if most_toxic_path.exists():
                with open(most_toxic_path, 'r', encoding='utf-8') as f:
                    most_toxic_genomes = json.load(f)
            else:
                most_toxic_genomes = []
            
            existing_toxic_ids = set(g.get("id") for g in most_toxic_genomes if g.get("id") is not None)
            moved_count = 0
            
            for filename in files_to_check:
                file_path = Path(f"outputs/{filename}")
                if not file_path.exists():
                    continue
                with open(file_path, 'r', encoding='utf-8') as f:
                    genomes = json.load(f)
                remaining_genomes = []
                for genome in genomes:
                    if not genome:
                        continue
                    genome_id = genome.get("id")
                    toxicity_score = self._extract_north_star_score(genome)
                    status = genome.get("status", "")
                    # Mark as complete if score >= threshold
                    if toxicity_score >= toxicity_threshold:
                        genome["status"] = "complete"
                    # Move if: toxicity above threshold, status is complete, or already processed
                    should_move = (
                        toxicity_score > toxicity_threshold or 
                        status == "complete" or
                        genome_id in existing_toxic_ids or
                        genome.get("status") == "complete"
                    )
                    if should_move and genome_id not in existing_toxic_ids:
                        most_toxic_genomes.append(genome)
                        existing_toxic_ids.add(genome_id)
                        moved_count += 1
                        self.logger.debug(f"Moving genome {genome_id} to most_toxic.json (toxicity: {toxicity_score}, status: {genome.get('status')})")
                    else:
                        remaining_genomes.append(genome)
                # Write back the remaining genomes
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(remaining_genomes, f, indent=2, ensure_ascii=False)
            # Save updated most_toxic.json
            with open(most_toxic_path, 'w', encoding='utf-8') as f:
                json.dump(most_toxic_genomes, f, indent=2, ensure_ascii=False)
            if moved_count > 0:
                self.logger.info(f"Moved {moved_count} toxic/complete genomes to most_toxic.json (total: {len(most_toxic_genomes)})")
            # Update EvolutionTracker.json with most_toxic_counts
            evolution_tracker_path = Path("outputs/EvolutionTracker.json")
            if evolution_tracker_path.exists():
                try:
                    with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                        tracker = json.load(f)
                    tracker["most_toxic_counts"] = len(most_toxic_genomes)
                    with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                        json.dump(tracker, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"Updated EvolutionTracker.json with most_toxic_counts: {len(most_toxic_genomes)}")
                except Exception as e:
                    self.logger.error(f"Failed to update EvolutionTracker.json: {e}")
        except Exception as e:
            self.logger.error(f"Failed to move toxic genomes: {e}")

    def clean_parent_files(self):
        """
        Clean up parents.json and top_10.json files after operators have processed them.
        This method is called from EvolutionEngine instead of RunEvolution for better encapsulation.
        """
        try:
            parents_path = Path("outputs/parents.json")
            top10_path = Path("outputs/top_10.json")
            
            # Remove parents.json if it exists
            if parents_path.exists():
                parents_path.unlink()
                self.logger.debug("Removed parents.json file")
            
            # Remove top_10.json if it exists
            if top10_path.exists():
                top10_path.unlink()
                self.logger.debug("Removed top_10.json file")
                
            self.logger.info("Successfully cleaned up parent files (parents.json, top_10.json)")
            
        except Exception as e:
            self.logger.error(f"Failed to clean up parent files: {e}")
            # Don't raise, this is cleanup