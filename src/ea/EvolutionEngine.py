"""
EvolutionEngine.py

Core evolutionary algorithm engine for text generation optimization.

This module implements the main evolution logic including parent selection,
operator application, and population management. Uses steady-state evolution
with elites preservation and multi-API moderation scoring.
"""

import json
import random
from typing import List, Dict, Any, Optional
from utils import get_custom_logging
from .ParentSelector import ParentSelector
from itertools import combinations
from pathlib import Path
from .synonym_replacement import LLM_POSAwareSynonymReplacement
from .mlm_operator import MLMOperator
from .paraphrasing import LLMBasedParaphrasingOperator
from .antonym_replacement import LLM_POSAwareAntonymReplacement
from .stylistic_mutator import StylisticMutator
from .back_translation import (
    LLMBackTranslationHIOperator, LLMBackTranslationFROperator, 
    LLMBackTranslationDEOperator, LLMBackTranslationJAOperator, 
    LLMBackTranslationZHOperator
)
from .semantic_similarity_crossover import SemanticSimilarityCrossover
from .fusion_crossover import SemanticFusionCrossover

# Global generator instances - will be set by main.py
_global_response_generator = None
_global_prompt_generator = None

def set_global_generators(response_generator, prompt_generator):
    """Set the global generator instances to be used by the system."""
    global _global_response_generator, _global_prompt_generator
    _global_response_generator = response_generator
    _global_prompt_generator = prompt_generator

def get_response_generator():
    """Get the shared response generator instance."""
    global _global_response_generator
    if _global_response_generator is None:
        raise RuntimeError("No global response generator set. Call set_global_generators() first from main.py")
    return _global_response_generator

def get_prompt_generator():
    """Get the shared prompt generator instance."""
    global _global_prompt_generator
    if _global_prompt_generator is None:
        raise RuntimeError("No global prompt generator set. Call set_global_generators() first from main.py")
    return _global_prompt_generator

# Legacy function for backward compatibility
def get_generator():
    """Get the shared prompt generator instance (legacy function)."""
    return get_prompt_generator()

class EvolutionEngine:

    def __init__(self, north_star_metric, log_file, current_cycle=None, max_variants=1, adaptive_selection_after=5, max_num_parents=4):
        self.genomes: List[Dict] = []
        self.next_id = 0
        self.north_star_metric = north_star_metric
        self.log_file = log_file
        self.current_cycle = current_cycle  # Current evolution cycle number
        self.use_steady_state = True
        self.max_variants = max_variants  # Maximum number of variants to generate per operator
        get_logger, _, _, _ = get_custom_logging()
        self.logger = get_logger("EvolutionEngine", log_file)
        self.parent_selector = ParentSelector(north_star_metric, log_file, adaptive_selection_after=adaptive_selection_after, max_num_parents=max_num_parents)
        # Initialize the shared generator instances
        self.prompt_generator = get_prompt_generator()
        self.response_generator = get_response_generator()
        
        self.logger.debug(f"EvolutionEngine initialized with next_id={self.next_id}, north_star_metric={north_star_metric}, current_cycle={current_cycle}, max_variants={max_variants}, adaptive_selection_after={adaptive_selection_after}, max_num_parents={max_num_parents}, use_steady_state=True")


    def _extract_north_star_score(self, genome: Dict) -> float:
        """Extract the north star metric score from a genome."""
        if not genome or not genome.get("moderation_result"):
            return 0.0
        
        moderation_result = genome["moderation_result"]
        if "google" in moderation_result:
            google_scores = moderation_result["google"]
            if "scores" in google_scores:
                score = google_scores["scores"].get(self.north_star_metric, 0.0)
                if score > 0:
                    return float(score)
        
        return 0.0

    def update_next_id(self):
        """Update next_id based on current genomes."""
        if self.genomes:
            max_id = max(g["id"] for g in self.genomes if g.get("id") is not None)
            self.next_id = max_id + 1
        else:
            self.next_id = 1
        self.logger.debug(f"Updated next_id to {self.next_id}")
    

    def _count_variants_from_temp(self) -> Dict[str, int]:
        """Count variants in temp.json by type (mutation/crossover)."""
        try:
            temp_path = Path("data/outputs/temp.json")
            if not temp_path.exists():
                return {"mutation_variants": 0, "crossover_variants": 0, "variants_created": 0}
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                temp_variants = json.load(f)
            
            mutation_count = sum(1 for v in temp_variants if v and v.get("creation_info", {}).get("type") == "mutation")
            crossover_count = sum(1 for v in temp_variants if v and v.get("creation_info", {}).get("type") == "crossover")
            total_count = mutation_count + crossover_count
            
            self.logger.debug(f"Counted variants from temp.json: {mutation_count} mutation, {crossover_count} crossover, {total_count} total")
            
            return {
                "mutation_variants": mutation_count,
                "crossover_variants": crossover_count,
                "variants_created": total_count
            }
            
        except Exception as e:
            self.logger.error(f"Failed to count variants from temp.json: {e}")
            return {"mutation_variants": 0, "crossover_variants": 0, "variants_created": 0}

    def _analyze_generation_data(self, parents: List[Dict], variant_counts: Dict[str, int]) -> Dict[str, Any]:
        """Analyze and create generation data for tracking and analytics."""
        generation_data = {
            "generation_number": self.current_cycle,
            "parents": [],
            "variants_created": variant_counts["variants_created"],
            "mutation_variants": variant_counts["mutation_variants"],
            "crossover_variants": variant_counts["crossover_variants"]
        }
        
        # Record selected parent metadata
        for parent in parents:
            generation_data["parents"].append({
                "id": parent["id"],
                "north_star_score": self._extract_north_star_score(parent),
                "generation": parent["generation"],
                "type": "parent"
            })
        
        return generation_data

    def _create_child_genome(self, prompt: str, operator: Any, parents: List[Dict], variant_type: str) -> Dict:
        """Create a child genome from a prompt and operator."""
        child = {
            "id": self.next_id,
            "prompt": prompt,
            "model_name": None,
            "moderation_result": None,
            "operator": operator.name,
            "parents": [p["id"] for p in parents],
            "generation": self.current_cycle,
            "status": "pending_generation",
            "creation_info": {
                "type": variant_type,
                "operator": operator.name,
                "source_generation": max(p["generation"] for p in parents) if len(parents) > 1 else parents[0]["generation"],
                "evolution_cycle": self.current_cycle
            }
        }
        
        # Add operator timing if available
        if hasattr(operator, '_last_operation_time'):
            child['operator_timing'] = operator._last_operation_time.copy()
        
        self.next_id += 1
        return child

    def generate_variants_global(self, evolution_tracker: Dict[str, Any] = None) -> None:
        """
        Generate variants globally for evolution cycle.
        Updates temp.json with unique variants created.
        
        Args:
            evolution_tracker (Dict[str, Any]): Evolution tracker data for determining parent counts
        """
        self.logger.debug(f"Generating variants globally for evolution cycle {self.current_cycle}")

        # Step 1: Synchronize next_id with current genomes
        # Prevents ID reuse if engine persists across cycles.
        self.update_next_id()

        # Step 2: Run adaptive parent selection (writes parents.json & top_10.json)
        self.parent_selector.adaptive_tournament_selection(evolution_tracker)
        
        # Step 3: Validate that parents were selected by checking parents.json
        parents = self._load_parents_from_file()
        if not parents:
            self.logger.error("No parents selected or failed to load parents from file")
            return
        
        # Step 3.5: Update EvolutionTracker with parent information
        self._update_evolution_tracker_with_parents(evolution_tracker, parents)

        # Step 4: Crossover phase (multi-parent recombination first for diversity)
        if len(parents) >= 2:
            crossover_operators = self._get_multi_parent_operators()
            self.logger.debug(f"Running crossover globally with {len(parents)} parents and {len(crossover_operators)} operators.")
            
            for op in crossover_operators:
                if op.operator_type != "crossover":
                    continue

                for parent_pair in combinations(parents, 2):  # All pairs of parents
                    try:
                        # Call crossover operator max_variant times since it outputs one variant
                        variants_to_save = []
                        for _ in range(self.max_variants):
                            operator_input = {
                                "parent_data": list(parent_pair),
                                "max_variants": 1  # Each call produces one variant
                            }
                            variants = op.apply(operator_input)
                            
                            if variants:
                                variants_to_save.extend([self._create_child_genome(vp, op, list(parent_pair), "crossover") for vp in variants])
                        
                        # Save variants immediately to temp.json
                        if variants_to_save:
                            self._append_variants_to_temp(variants_to_save)
                            self.logger.debug(f"Saved {len(variants_to_save)} crossover variants from {op.name}")
                            
                    except Exception as e:
                        self.logger.error(f"[Crossover Error] {op.name} with parents {[p['id'] for p in parent_pair]}: {e}")

        # Step 5: Mutation phase (single-parent exploration)
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
                            variants_to_save = [self._create_child_genome(vp, op, [parent], "mutation") for vp in variants]
                            self._append_variants_to_temp(variants_to_save)
                            self.logger.debug(f"Saved {len(variants_to_save)} mutation variants from {op.name}")
                            
                    except Exception as e:
                        self.logger.error(f"[Mutation Error] {op.name} with parent {parent['id']}: {e}")

        # Step 6: Deduplicate variants within temp.json (intra-file de-duplication only)
        # This removes duplicates within temp.json itself, not cross-file duplicates
        duplicates_removed = self._deduplicate_temp_json()
        if duplicates_removed:
            self.logger.debug(f"Deduplicated temp.json: removed {duplicates_removed} intra-file duplicates")

        # Step 7: Read variant counts from temp.json (after all operators have run and deduped)
        variant_counts = self._count_variants_from_temp()
        
        # Step 8: Generate analytics data (for tracking and logging)
        generation_data = self._analyze_generation_data(parents, variant_counts)
        self._last_generation_data = generation_data  # Store for later retrieval

        # Step 9: Update EvolutionTracker with final variant counts
        self.logger.debug(f"Updating EvolutionTracker with variant counts: {generation_data}")
        self._update_evolution_tracker_with_variants(evolution_tracker, generation_data)
        
        # Step 10: Clean up transient selection artifacts
        self.clean_parents_file()
        
        # Step 11: Final logging summary
        self.logger.info(
            "Generated %d unique variants (mutation: %d, crossover: %d) for evolution cycle %d.",
            generation_data["variants_created"],
            generation_data["mutation_variants"],
            generation_data["crossover_variants"],
            self.current_cycle
        )
    
    def _load_parents_from_file(self) -> List[Dict]:
        """
        Load parents from parents.json file.
        
        Returns:
            List[Dict]: List of parent genomes
        """
        try:
            parents_path = Path("data/outputs/parents.json")
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
            temp_path = Path("data/outputs/temp.json")
            
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
    
    def clean_parents_file(self) -> None:
        """Empty the parents.json and top_10.json files after all operators have processed the parents."""
        try:
            parents_path = Path("data/outputs/parents.json")
            top10_path = Path("data/outputs/top_10.json")
            emptied = []
            for path in [parents_path, top10_path]:
                # Ensure parent directory exists
                path.parent.mkdir(parents=True, exist_ok=True)
                # Write empty list to file (creates or overwrites)
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=2, ensure_ascii=False)
                emptied.append(str(path))
            if emptied:
                self.logger.info(f"Emptied files: {', '.join(emptied)}")
        except Exception as e:
            self.logger.error(f"Failed to empty parents/top_10 file: {e}")

    def _get_single_parent_operators(self):
        """Return list of mutation operators that require only a single parent."""
        return [
            LLM_POSAwareSynonymReplacement(self.north_star_metric, log_file=self.log_file, max_variants=self.max_variants, num_POS_tags=1, generator=self.prompt_generator),
            MLMOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
            LLMBasedParaphrasingOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
            LLM_POSAwareAntonymReplacement(self.north_star_metric, log_file=self.log_file, max_variants=self.max_variants, num_POS_tags=1, generator=self.prompt_generator),
            StylisticMutator(log_file=self.log_file, generator=self.prompt_generator),
            LLMBackTranslationHIOperator(log_file=self.log_file, generator=self.prompt_generator),
            LLMBackTranslationFROperator(log_file=self.log_file, generator=self.prompt_generator),
            LLMBackTranslationDEOperator(log_file=self.log_file, generator=self.prompt_generator),
            LLMBackTranslationJAOperator(log_file=self.log_file, generator=self.prompt_generator),
            LLMBackTranslationZHOperator(log_file=self.log_file, generator=self.prompt_generator),
        ]

    def _get_multi_parent_operators(self):
        """Return list of crossover operators that require multiple parents."""
        return [
            SemanticSimilarityCrossover(log_file=self.log_file),
            SemanticFusionCrossover(north_star_metric=self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator)
        ]

    def _update_evolution_tracker_with_parents(self, evolution_tracker: Dict[str, Any], parents: List[Dict]) -> None:
        """
        Update EvolutionTracker with parent information immediately after parent selection.
        
        Args:
            evolution_tracker: Evolution tracker dictionary to update
            parents: List of selected parent genomes
        """
        try:
            if not evolution_tracker or not parents:
                return
            
            # Extract parent information with scores
            parent_info = []
            for parent in parents:
                # Use scores from slimmed parent data, fallback to extraction if available
                scores = parent.get("scores", {})
                north_star_score = scores.get(self.north_star_metric, 0.0)
                
                parent_data = {
                    "id": parent.get("id"),
                    "generation": parent.get("generation", 0),
                    "north_star_score": north_star_score
                }
                parent_info.append(parent_data)
            
            # Sort parents by score for consistent ordering
            sorted_parents = sorted(parent_info, key=lambda p: p.get("north_star_score", 0.0), reverse=True)
            
            # Create simplified parent information
            parents_info = []
            for parent in sorted_parents:
                parents_info.append({
                    "genome_id": parent["id"],
                    "generation": parent.get("generation", 0),
                    "score": parent.get("north_star_score", 0.0)
                })
            
            # Update the current generation in evolution tracker
            current_generation = self.current_cycle if self.current_cycle is not None else len(evolution_tracker.get("generations", []))
            
            # Find the current generation entry
            current_gen = None
            for gen in evolution_tracker.get("generations", []):
                if gen.get("generation_number") == current_generation:
                    current_gen = gen
                    break
            
            # If generation doesn't exist, create it
            if current_gen is None:
                current_gen = {
                    "generation_number": current_generation,
                    "genome_id": None,
                    "max_score": 0.0,
                    "parents": [],
                    "elites_threshold": 0.0
                }
                evolution_tracker.setdefault("generations", []).append(current_gen)
            
            # Update parent information
            current_gen["parents"] = parents_info
            
            # Save updated evolution tracker
            from pathlib import Path
            outputs_path = Path("data/outputs")
            evolution_tracker_path = outputs_path / "EvolutionTracker.json"
            
            with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(evolution_tracker, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Updated EvolutionTracker with {len(parents_info)} parents: {[p['genome_id'] for p in parents_info]}")
            
        except Exception as e:
            self.logger.error(f"Failed to update EvolutionTracker with parent information: {e}")

    def _update_evolution_tracker_with_variants(self, evolution_tracker: Dict[str, Any], generation_data: Dict[str, Any]) -> None:
        """
        Update EvolutionTracker with variant counts after variant generation is complete.
        
        Args:
            evolution_tracker: Evolution tracker dictionary to update
            generation_data: Generation data containing variant counts
        """
        try:
            if not evolution_tracker or not generation_data:
                self.logger.warning("Cannot update EvolutionTracker: evolution_tracker or generation_data is None")
                return
            
            self.logger.debug(f"Updating EvolutionTracker with variant counts for generation {self.current_cycle}")
            
            # Find the current generation entry
            current_generation = self.current_cycle if self.current_cycle is not None else len(evolution_tracker.get("generations", []))
            
            current_gen = None
            for gen in evolution_tracker.get("generations", []):
                if gen.get("generation_number") == current_generation:
                    current_gen = gen
                    break
            
            if current_gen is None:
                self.logger.warning(f"Generation {current_generation} not found in EvolutionTracker for variant update, creating it")
                # Create the generation entry if it doesn't exist
                current_gen = {
                    "generation_number": current_generation,
                    "genome_id": None,
                    "max_score": 0.0,
                    "parents": [],
                    "elites_threshold": 0.0
                }
                evolution_tracker.setdefault("generations", []).append(current_gen)
            
            # Update variant counts
            current_gen["variants_created"] = generation_data.get("variants_created", 0)
            current_gen["mutation_variants"] = generation_data.get("mutation_variants", 0)
            current_gen["crossover_variants"] = generation_data.get("crossover_variants", 0)
            
            # Save updated evolution tracker
            from pathlib import Path
            outputs_path = Path("data/outputs")
            evolution_tracker_path = outputs_path / "EvolutionTracker.json"
            
            with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(evolution_tracker, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Updated EvolutionTracker with variant counts: {generation_data['variants_created']} total ({generation_data['mutation_variants']} mutation, {generation_data['crossover_variants']} crossover)")
            
        except Exception as e:
            self.logger.error(f"Failed to update EvolutionTracker with variant information: {e}")

    def get_last_generation_data(self) -> Dict[str, Any]:
        """Get the generation data from the last run for tracking purposes."""
        return getattr(self, '_last_generation_data', {
            "generation_number": self.current_cycle,
            "parents": [],
            "variants_created": 0,
            "mutation_variants": 0,
            "crossover_variants": 0
        })

    def _deduplicate_temp_json(self) -> int:
        """
        Remove duplicate variants within data/outputs/temp.json based on normalized prompt.
        Keeps the first occurrence and discards subsequent duplicates.
        
        Returns:
            int: Number of duplicates removed
        """
        try:
            temp_path = Path("data/outputs/temp.json")
            if not temp_path.exists():
                return 0

            with open(temp_path, 'r', encoding='utf-8') as f:
                variants = json.load(f)

            if not isinstance(variants, list) or not variants:
                return 0

            seen_prompts = set()
            seen_ids = set()
            unique_variants = []
            duplicates_removed = 0

            for v in variants:
                if not isinstance(v, dict):
                    # Skip invalid entries but count as removed when rewriting list
                    duplicates_removed += 1
                    continue

                prompt = v.get("prompt")
                vid = v.get("id")

                # Normalize prompt for dedup
                norm = prompt.strip().lower() if isinstance(prompt, str) else None

                # Criteria: duplicate if same normalized prompt OR duplicate id already seen
                if (norm is not None and norm in seen_prompts) or (vid is not None and vid in seen_ids):
                    duplicates_removed += 1
                    continue

                if norm is not None:
                    seen_prompts.add(norm)
                if vid is not None:
                    seen_ids.add(vid)
                unique_variants.append(v)

            # Only rewrite file if duplicates were removed
            if duplicates_removed > 0:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(unique_variants, f, indent=2, ensure_ascii=False)

            return duplicates_removed
        except Exception as e:
            self.logger.error(f"Failed to deduplicate temp.json: {e}")
            return 0

