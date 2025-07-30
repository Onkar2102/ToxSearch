import json
import random
from typing import List, Dict, Any, Optional
from utils.custom_logging import get_logger
from ea.TextVariationOperators import get_applicable_operators
from ea.ParentSelector import ParentSelector
from itertools import combinations
from utils.evolution_utils import append_parents_by_generation_entry

class EvolutionEngine:

    def __init__(self, north_star_metric, log_file):
        self.genomes: List[Dict] = []
        self.next_id = 0
        self.north_star_metric = north_star_metric
        self.log_file = log_file
        self.logger = get_logger("EvolutionEngine", log_file)
        self.parent_selector = ParentSelector(north_star_metric, log_file)
        self.logger.debug(f"EvolutionEngine initialized with next_id={self.next_id}, north_star_metric={north_star_metric}")

    def update_next_id(self):
        if self.genomes:
            # Convert string IDs to integers for arithmetic, then back to string
            max_id = max(int(g["id"]) for g in self.genomes)
            self.next_id = max_id + 1
        else:
            self.next_id = 1  # Start from 1 since we removed "genome_" prefix
        self.logger.debug(f"Updated next_id to {self.next_id}")
    
    def generate_variants(self, prompt_id: int) -> Dict[str, Any]:
        self.logger.debug(f"Generating variants for prompt_id={prompt_id}")

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
            max_score = max([g.get("moderation_result", {}).get("scores", {}).get(self.north_star_metric, 0.0) for g in completed_genomes])
            self.logger.info(f"Best completed genome score for prompt {prompt_id}: {max_score}")
        else:
            self.logger.warning(f"No completed genomes found for prompt {prompt_id}")

        mutation_parent, crossover_parents = self.parent_selector.select_parents(prompt_genomes, prompt_id)
        
        # Log parent selection results
        if mutation_parent is None:
            self.logger.warning(f"No mutation parent selected for prompt_id={prompt_id}")
        else:
            self.logger.info(f"Selected mutation parent for prompt {prompt_id}: genome_id={mutation_parent['id']}, score={mutation_parent.get('moderation_result', {}).get('scores', {}).get(self.north_star_metric, 0.0)}")
        
        if crossover_parents is None:
            self.logger.warning(f"No crossover parents selected for prompt_id={prompt_id}")
        else:
            self.logger.info(f"Selected {len(crossover_parents)} crossover parents for prompt {prompt_id}: {[p['id'] for p in crossover_parents]}")

        existing_prompts = set(g["prompt"].strip().lower() for g in self.genomes if g is not None and g["prompt_id"] == prompt_id)

        offspring = []
        generation_data = {
            "parents": [],
            "variants_created": 0,
            "mutation_variants": 0,
            "crossover_variants": 0
        }

        # Track parent information for parent selection tracker
        def get_parent_info(parent):
            return {
                "id": parent["id"],
                "score": parent.get("moderation_result", {}).get("scores", {}).get(self.north_star_metric, 0.0),
                "parents_id": parent.get("parents", None)
            }

        # For generation 0, parents is null
        if all(g["generation"] == 0 for g in prompt_genomes):
            append_parents_by_generation_entry(prompt_id, 0, None, "initial", self.logger)

        # For mutation, use the topmost genome as parent
        if mutation_parent:
            generation_data["parents"].append({
                "id": mutation_parent["id"],
                "north_star_score": mutation_parent.get("moderation_result", {}).get("scores", {}).get(self.north_star_metric, 0.0),
                "generation": mutation_parent["generation"],
                "type": "mutation_parent"
            })
            # Track mutation parents for this generation
            append_parents_by_generation_entry(
                prompt_id, 
                mutation_parent["generation"] + 1, 
                [mutation_parent["id"]], 
                "mutation", 
                self.logger
            )

        if crossover_parents:
            for parent in crossover_parents:
                generation_data["parents"].append({
                    "id": parent["id"],
                    "north_star_score": parent.get("moderation_result", {}).get("scores", {}).get(self.north_star_metric, 0.0),
                    "generation": parent["generation"],
                    "type": "crossover_parent"
                })
            # Track crossover parents for this generation
            crossover_parent_ids = [p["id"] for p in crossover_parents]
            max_generation = max(p["generation"] for p in crossover_parents)
            append_parents_by_generation_entry(
                prompt_id, 
                max_generation + 1, 
                crossover_parent_ids, 
                "crossover", 
                self.logger
            )

        mutation_operators = get_applicable_operators(1, self.north_star_metric, self.log_file)
        self.logger.debug(f"Running mutation on prompt_id={prompt_id} using parent id={mutation_parent['id'] if mutation_parent else 'None'} with {len(mutation_operators)} operators.")
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
                        "generation": mutation_parent["generation"] + 1,
                        "status": "pending_generation",
                        "creation_info": {
                            "type": "mutation",
                            "operator": op.name,
                            "source_generation": mutation_parent["generation"]
                        }
                    }
                    self.next_id += 1
                    self.logger.debug(f"Created mutation variant id={child['id']} for prompt_id={prompt_id}")
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
                                "generation": max(p["generation"] for p in parent_pair) + 1,
                                "status": "pending_generation",
                                "creation_info": {
                                    "type": "crossover",
                                    "operator": op.name,
                                    "source_generation": max(p["generation"] for p in parent_pair)
                                }
                            }
                            self.next_id += 1
                            self.logger.debug(f"Created crossover variant id={child['id']} for prompt_id={prompt_id}")
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

        self.genomes.extend(unique_offspring.values())
        self.logger.debug(
            "Saved %d unique variants to the population (mutation: %d, crossover: %d).",
            generation_data["variants_created"],
            generation_data["mutation_variants"],
            generation_data["crossover_variants"],
        )
        
        return generation_data

