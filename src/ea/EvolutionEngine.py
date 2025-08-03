import json
import random
from typing import List, Dict, Any, Optional
from utils.custom_logging import get_logger
from ea.TextVariationOperators import get_applicable_operators
from ea.ParentSelector import ParentSelector
from itertools import combinations

class EvolutionEngine:

    def __init__(self, north_star_metric, log_file, current_cycle=None):
        self.genomes: List[Dict] = []
        self.next_id = 0
        self.north_star_metric = north_star_metric
        self.log_file = log_file
        self.current_cycle = current_cycle  # Current evolution cycle number
        self.logger = get_logger("EvolutionEngine", log_file)
        self.parent_selector = ParentSelector(north_star_metric, log_file)
        self.logger.debug(f"EvolutionEngine initialized with next_id={self.next_id}, north_star_metric={north_star_metric}, current_cycle={current_cycle}")

    def update_next_id(self):
        if self.genomes:
            # Convert string IDs to integers for arithmetic, then back to string
            max_id = max(int(g["id"]) for g in self.genomes)
            self.next_id = max_id + 1
        else:
            self.next_id = 1  # Start from 1 since we removed "genome_" prefix
        self.logger.debug(f"Updated next_id to {self.next_id}")
    
    def generate_variants(self, prompt_id: int) -> Dict[str, Any]:
        self.logger.debug(f"Generating variants for prompt_id={prompt_id} (evolution cycle {self.current_cycle})")

        # Ensure ``next_id`` is always in sync with the current population each
        # time we start a new generation cycle.  This prevents reused IDs when
        # the engine object persists across multiple calls.
        self.update_next_id()

        prompt_genomes = [g for g in self.genomes if g is not None and g["prompt_id"] == prompt_id]
        if not prompt_genomes:
            self.logger.error(f"No genomes found for prompt_id={prompt_id}. Exiting evolution process.")
            raise SystemExit(1)

        # Log detailed information about available genomes
        completed_genomes = [g for g in prompt_genomes if g.get("status") == "complete"]
        pending_genomes = [g for g in prompt_genomes if g.get("status") == "pending_evolution"]
        other_genomes = [g for g in prompt_genomes if g.get("status") not in ["complete", "pending_evolution"]]
        
        self.logger.info(f"Prompt {prompt_id} genome breakdown: {len(completed_genomes)} completed, {len(pending_genomes)} pending_evolution, {len(other_genomes)} other")
        
        if completed_genomes:
            max_score = max([(g.get("moderation_result") or {}).get("scores", {}).get(self.north_star_metric, 0.0) for g in completed_genomes])
            self.logger.info(f"Best completed genome score for prompt {prompt_id}: {max_score}")
        else:
            self.logger.warning(f"No completed genomes found for prompt {prompt_id}")

        mutation_parent, crossover_parents = self.parent_selector.select_parents(prompt_genomes, prompt_id)
        
        # Log parent selection results
        if mutation_parent is None:
            self.logger.warning(f"No mutation parent selected for prompt_id={prompt_id}")
        else:
            self.logger.info(f"Selected mutation parent for prompt {prompt_id}: genome_id={mutation_parent['id']}, score={(mutation_parent.get('moderation_result') or {}).get('scores', {}).get(self.north_star_metric, 0.0)}")
        
        if crossover_parents is None:
            self.logger.warning(f"No crossover parents selected for prompt_id={prompt_id}")
        else:
            self.logger.info(f"Selected {len(crossover_parents)} crossover parents for prompt {prompt_id}: {[p['id'] for p in crossover_parents]}")

        existing_prompts = set(g["prompt"].strip().lower() for g in self.genomes if g is not None and g["prompt_id"] == prompt_id)

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
                "score": (parent.get("moderation_result") or {}).get("scores", {}).get(self.north_star_metric, 0.0),
                "parents_id": parent.get("parents", None)
            }

        # For generation 0, parents is null
        if all(g["generation"] == 0 for g in prompt_genomes):
            # Parent tracking has been removed
            pass

        # For mutation, use the topmost genome as parent
        if mutation_parent:
            generation_data["parents"].append({
                "id": mutation_parent["id"],
                "north_star_score": (mutation_parent.get("moderation_result") or {}).get("scores", {}).get(self.north_star_metric, 0.0),
                "generation": mutation_parent["generation"],
                "type": "mutation_parent"
            })
            # Mutation parent tracking has been removed

        # For crossover, use the top 3 genomes as parents
        if crossover_parents:
            for parent in crossover_parents:
                generation_data["parents"].append({
                    "id": parent["id"],
                    "north_star_score": (parent.get("moderation_result") or {}).get("scores", {}).get(self.north_star_metric, 0.0),
                    "generation": parent["generation"],
                    "type": "crossover_parent"
                })
            # Crossover parent tracking has been removed

        # --- Mutation phase -------------------------------------------------
        if mutation_parent:
            mutation_operators = get_applicable_operators(1, self.north_star_metric, self.log_file)
            self.logger.debug(f"Running mutation on prompt_id={prompt_id} with {len(mutation_operators)} operators.")
            for op in mutation_operators:
                if op.operator_type != "mutation":
                    continue

                try:
                    variants = op.apply(mutation_parent["prompt"])
                    for vp in variants:
                        norm_vp = vp.strip().lower()
                        if norm_vp in existing_prompts:
                            continue
                        existing_prompts.add(norm_vp)
                        child = {
                            "id": str(self.next_id),
                            "prompt_id": prompt_id,
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
                        self.logger.debug(f"Created mutation variant id={child['id']} for prompt_id={prompt_id} (evolution cycle {self.current_cycle})")
                        self.logger.debug(f"Mutation variant prompt: '{vp[:60]}...'")
                        offspring.append(child)
                except Exception as e:
                    self.logger.error(f"[Mutation Error] {op.name}: {e}")

        # --- Crossover phase -------------------------------------------------
        if crossover_parents:
            crossover_operators = get_applicable_operators(len(crossover_parents), self.north_star_metric, self.log_file)
            self.logger.debug(f"Running crossover on prompt_id={prompt_id} with {len(crossover_parents)} parents and {len(crossover_operators)} operators.")
            for op in crossover_operators:
                if op.operator_type != "crossover":
                    continue

                for parent_pair in combinations(crossover_parents, 2):  # All pairs of parents
                    try:
                        prompts = [p["prompt"] for p in parent_pair]
                        variants = op.apply(prompts)  # Send both prompts
                        for vp in variants:
                            norm_vp = vp.strip().lower()
                            if norm_vp in existing_prompts:
                                continue
                            existing_prompts.add(norm_vp)
                            child = {
                                "id": str(self.next_id),
                                "prompt_id": prompt_id,
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
                            self.logger.debug(f"Created crossover variant id={child['id']} for prompt_id={prompt_id} (evolution cycle {self.current_cycle})")
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

        # Create new generation file for the new variants
        if unique_offspring:
            self._create_new_generation_file(unique_offspring.values(), self.current_cycle)

        self.genomes.extend(unique_offspring.values())
        self.logger.debug(
            "Saved %d unique variants to the population (mutation: %d, crossover: %d) for evolution cycle %d.",
            generation_data["variants_created"],
            generation_data["mutation_variants"],
            generation_data["crossover_variants"],
            self.current_cycle
        )
        
        return generation_data

    def _create_new_generation_file(self, new_variants, generation_number):
        """Create a new generation file for the new variants"""
        from pathlib import Path
        import json
        
        generation_file_path = Path("outputs") / f"gen{generation_number}.json"
        
        # Save new variants to generation file
        variants_list = list(new_variants)
        
        try:
            with open(generation_file_path, 'w', encoding='utf-8') as f:
                json.dump(variants_list, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Created new generation file: {generation_file_path} with {len(variants_list)} variants")
        except Exception as e:
            self.logger.error(f"Failed to create generation file {generation_file_path}: {e}")
            raise

    def load_parents_from_tracker(self, prompt_id: int, generation_number: int, evolution_tracker: List[dict]) -> List[Dict[str, Any]]:
        """
        Load parent genomes for a specific prompt_id and generation using tracker information
        
        This method provides an efficient way to load parents from previous generations
        without loading the entire population.
        
        Parameters
        ----------
        prompt_id : int
            The prompt ID to load parents for
        generation_number : int
            The generation number to load parents for
        evolution_tracker : List[dict]
            The evolution tracker containing parent information
            
        Returns
        -------
        List[Dict[str, Any]]
            List of parent genomes found
        """
        from ea.RunEvolution import load_parents_from_tracker
        return load_parents_from_tracker(prompt_id, generation_number, evolution_tracker, 
                                       logger=self.logger, log_file=self.log_file)

