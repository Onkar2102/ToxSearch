import json
import random
from typing import List, Dict, Any, Optional
from utils.custom_logging import get_logger
from ea.TextVariationOperators import get_applicable_operators
from itertools import combinations

class EvolutionEngine:

    def __init__(self, north_star_metric, log_file):
        self.genomes: List[Dict] = []
        self.next_id = 0
        self.north_star_metric = north_star_metric
        self.logger = get_logger("EvolutionEngine", log_file)
        self.logger.debug(f"EvolutionEngine initialized with next_id={self.next_id}, north_star_metric={self.north_star_metric}")

    def update_next_id(self):
        if self.genomes:
            # Convert string IDs to integers for arithmetic, then back to string
            max_id = max(int(g["id"]) for g in self.genomes)
            self.next_id = max_id + 1
        else:
            self.next_id = 1  # Start from 1 since we removed "genome_" prefix
        self.logger.debug(f"Updated next_id to {self.next_id}")
    
    def select_parents(self, prompt_id: int):
        prompt_genomes = [g for g in self.genomes if g["prompt_id"] == prompt_id]

        if len(prompt_genomes) == 1:
            mutation_parent = prompt_genomes[0]
            crossover_parents = None
        elif 2 <= len(prompt_genomes) < 5:
            sorted_genomes = sorted(
                prompt_genomes,
                key=lambda g: -(g.get(self.north_star_metric) if isinstance(g.get(self.north_star_metric), (int, float)) else 0.0)
            )
            mutation_parent = sorted_genomes[0]
            crossover_parents = sorted_genomes
        elif len(prompt_genomes) >= 5:
            sorted_genomes = sorted(
                prompt_genomes,
                key=lambda g: -(g.get(self.north_star_metric) if isinstance(g.get(self.north_star_metric), (int, float)) else 0.0)
            )
            top_5_score = sorted(set(
                g.get(self.north_star_metric, 0.0)
                for g in sorted_genomes
                if isinstance(g.get(self.north_star_metric, (int, float)))
            ), reverse=True)[:5]
            top_5_genomes = [g for g in sorted_genomes if g.get(self.north_star_metric, 0.0) in top_5_score]

            if not top_5_score:
                self.logger.warning(f"No valid north star scores for prompt_id={prompt_id}. Skipping parent selection.")
                mutation_parent = None
                crossover_parents = None
            else:
                max_score = max(top_5_score)
                mutation_candidates = [g for g in top_5_genomes if g.get(self.north_star_metric, 0.0) == max_score]
                mutation_parent = random.choice(mutation_candidates)
                crossover_parents = random.sample(top_5_genomes, min(5, len(top_5_genomes)))
        else:
            mutation_parent = None
            crossover_parents = None

        return mutation_parent, crossover_parents
    
    def generate_variants(self, prompt_id: int) -> List[Dict]:
        self.logger.debug(f"Generating variants for prompt_id={prompt_id}")
        prompt_genomes = [g for g in self.genomes if g["prompt_id"] == prompt_id]
        if not prompt_genomes:
            self.logger.error(f"No genomes found for prompt_id={prompt_id}. Exiting evolution process.")
            raise SystemExit(1)

        mutation_parent, crossover_parents = self.select_parents(prompt_id)
        if mutation_parent is None:
            self.logger.warning(f"No suitable genomes found for prompt_id={prompt_id}")

        existing_prompts = set(g["prompt"].strip().lower() for g in self.genomes if g["prompt_id"] == prompt_id)

        offspring = []

        mutation_operators = get_applicable_operators(1, self.north_star_metric)
        self.logger.debug(f"Running mutation on prompt_id={prompt_id} using parent id={mutation_parent['id']} with {len(mutation_operators)} operators.")
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

        # Deduplicate and save unique mutation offspring to population
        unique_mutation_offspring = {}
        for child in offspring:
            key = child["prompt"].strip().lower()
            if key not in unique_mutation_offspring:
                unique_mutation_offspring[key] = child

        self.genomes.extend(unique_mutation_offspring.values())
        self.logger.debug(f"Saved {len(unique_mutation_offspring)} unique mutation variants to the population.")

        if crossover_parents:
            crossover_operators = get_applicable_operators(len(crossover_parents), self.north_star_metric)
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

        # Deduplicate and save unique crossover offspring to population
        unique_crossover_offspring = {}
        for child in offspring:
            key = child["prompt"].strip().lower()
            if key not in unique_crossover_offspring:
                unique_crossover_offspring[key] = child

        self.genomes.extend(unique_crossover_offspring.values())
        self.logger.debug(f"Saved {len(unique_crossover_offspring)} unique crossover variants to the population.")

