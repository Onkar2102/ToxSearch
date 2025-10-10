"""
ParentSelector.py

Redesigned parent selection system with adaptive tournament selection.
Selects parents from elites.json and population.json based on toxicity scores.
"""

import random
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from utils import get_custom_logging
from utils.population_io import load_elites, load_population

class ParentSelector:
    """
    A class responsible for selecting parents for genetic operations using adaptive tournament selection.
    
    Selection Strategy:
    - Mutation parent: Genome with highest toxicity score from elites.json
    - Crossover parents: x random parents from elites.json + y random parents from population.json
    - Adaptive logic: Increases x and y based on population_max_toxicity stagnation
    """
    
    def __init__(self, north_star_metric: str, log_file: Optional[str] = None, adaptive_selection_after: int = 5, max_num_parents: int = 4):
        """
        Initialize the ParentSelector.
        
        Args:
            north_star_metric (str): The primary fitness metric to use for selection
            log_file (Optional[str]): Log file path for logging
            adaptive_selection_after (int): Size of moving window for adaptive selection
            max_num_parents (int): Maximum number of parents to select (default: 4)
        """
        self.north_star_metric = north_star_metric
        get_logger, _, _, _ = get_custom_logging()
        self.logger = get_logger("ParentSelector", log_file)
        self.logger.debug(f"ParentSelector initialized with north_star_metric={north_star_metric}")
        self.adaptive_selection_after = adaptive_selection_after
        self.max_num_parents = max_num_parents
    
    def _extract_toxicity_score(self, genome: Dict) -> float:
        """Extract toxicity score from a genome using the north star metric."""
        from utils import get_system_utils
        _, _, _, _, _extract_north_star_score, _ = get_system_utils()
        return _extract_north_star_score(genome, self.north_star_metric)
    
    def _determine_parent_counts(self, evolution_tracker: Dict[str, Any]) -> Tuple[int, int]:
        """
        Determine the number of parents to select from elites (x) and population (y) using adaptive logic.
        
        Logic:
        - Initially: x=2, y=0 (top performing genome + 1 random from elites)
        - Until generation adaptive_selection_after: no checking, keep (2, 0)
        - After adaptive_selection_after: check if population_max_toxicity increased
        - If increased: keep current (x, y)
        - If same: alternate adding 1 to x or y, max total = max_num_parents
        
        Args:
            evolution_tracker: Evolution tracker data

        Returns:
            Tuple[int, int]: (x, y) - number of parents from elites and population
        """
        try:
            generations = evolution_tracker.get("generations", [])
            total_generations = len(generations)
            
            # Initial selection: x=2, y=0 (top performing + 1 random from elites)
            if total_generations <= self.adaptive_selection_after:
                self.logger.debug(f"Within initial window ({total_generations} <= {self.adaptive_selection_after}), using x=2, y=0")
                return 2, 0
            
            # After initial window, check for stagnation by looking at score progression
            # Check if max_score has increased in the last adaptive_selection_after generations
            found_improvement = False
            current_max_score = 0.0
            
            # Find the current maximum score from recent generations
            for i in range(min(self.adaptive_selection_after, total_generations)):
                gen_idx = total_generations - 1 - i
                if gen_idx < 0:
                    break
                gen = generations[gen_idx]
                gen_score = gen.get("max_score", 0.0)
                current_max_score = max(current_max_score, gen_score)
            
            # Check if there was any improvement in the last adaptive_selection_after generations
            for i in range(1, self.adaptive_selection_after + 1):
                gen_idx = total_generations - i
                if gen_idx < 0:
                    break
                gen = generations[gen_idx]
                gen_score = gen.get("max_score", 0.0)
                
                # Check if this generation had a higher score than previous generations
                if gen_score > current_max_score - 1e-8:  # Allow for small floating point differences
                    # Look at previous generations to see if there was improvement
                    for j in range(i + 1, min(i + self.adaptive_selection_after + 1, total_generations + 1)):
                        prev_gen_idx = total_generations - j
                        if prev_gen_idx < 0:
                            break
                        prev_gen = generations[prev_gen_idx]
                        prev_score = prev_gen.get("max_score", 0.0)
                        if gen_score > prev_score + 1e-8:  # Significant improvement
                            found_improvement = True
                            self.logger.info(f"Score improvement detected: {prev_score:.4f} -> {gen_score:.4f} in generation {gen_idx}")
                            break
                    if found_improvement:
                        break
            
            if found_improvement:
                # Recent improvement detected, fallback to original logic (2, 0)
                self.logger.info(f"Recent score improvement detected, falling back to x=2, y=0")
                return 2, 0
            else:
                # Stagnation detected, increase parent counts
                stagnation_windows = (total_generations - self.adaptive_selection_after) // self.adaptive_selection_after + 1
                x = 2 + (stagnation_windows + 1) // 2
                y = stagnation_windows // 2
                
                # Cap at max limits: x=1+max_num_parents, y=max_num_parents
                # (where 1 is the topmost performing genome)
                x = min(x, 1 + self.max_num_parents)
                y = min(y, self.max_num_parents)
                
                self.logger.info(f"Stagnation detected for {stagnation_windows} windows, using x={x}, y={y}")
                return x, y

        except Exception as e:
            self.logger.warning(f"Error determining parent counts: {e}")
            return 2, 0
    
    def adaptive_tournament_selection(self, evolution_tracker: Dict[str, Any] = None) -> None:
        """
        Perform adaptive tournament selection.
        Updates parents.json and top_10.json with selected parents.
        
        Args:
            evolution_tracker (Dict[str, Any]): Evolution tracker data for determining parent counts
        """
        try:
            # Determine parent counts using evolution tracker
            if evolution_tracker:
                x, y = self._determine_parent_counts(evolution_tracker)
                self.logger.info(f"Determined parent counts: x={x} (elites), y={y} (population)")
            else:
                x, y = 2, 0
                self.logger.warning("No evolution tracker provided, using default parent counts: x=2, y=0")
            
            # Load elites and population
            elites = load_elites(log_file=None)
            population = load_population(log_file=None)
            
            if not elites:
                self.logger.warning("No genomes in elites, cannot select parents")
                return
            
            # Select parents: genome with highest toxicity score from elites + x random from elites + y random from population
            selected_parents = []
            
            # Add genome with highest toxicity score from elites
            topmost_elite = max(elites, key=lambda g: self._extract_toxicity_score(g))
            selected_parents.append(topmost_elite)
            
            # Add x-1 random parents from elites (excluding topmost, since topmost is already added)
            elites_candidates = [g for g in elites if g != topmost_elite]
            elites_to_select = min(x - 1, len(elites_candidates))
            
            if elites_to_select > 0:
                selected_elites = random.sample(elites_candidates, elites_to_select)
                selected_parents.extend(selected_elites)
                self.logger.info(f"Selected {elites_to_select} random elite parents (excluding topmost)")
            elif x > 1:
                self.logger.warning(f"Requested {x-1} additional elite parents but only {len(elites_candidates)} available")
            
            # Select y random parents from population if requested
            if y > 0 and len(population) >= y:
                selected_population = random.sample(population, y)
                selected_parents.extend(selected_population)
                self.logger.info(f"Selected {y} random parents from population")
            elif y > 0:
                self.logger.warning(f"Requested {y} parents from population but only {len(population)} available")
            
            self.logger.info(f"Selected {len(selected_parents)} parents: {[p['id'] for p in selected_parents]}")
            
            # Save parents to parents.json
            self._save_parents_to_file(selected_parents)
            
            # Save top 10 genomes by toxicity to top_10.json
            self._save_top_10_by_toxicity()
            
        except Exception as e:
            self.logger.error(f"Error in adaptive tournament selection: {e}")
    
    def _save_parents_to_file(self, parents: List[Dict]) -> None:
        """
        Save selected parents to parents.json file for operators to fetch.
        Only saves essential fields: prompt, generated_text, and scores.
        
        Args:
            parents: List of selected parent genomes
        """
        try:
            # Create slimmed-down parents data structure with only essential fields
            slim_parents = []
            for parent in parents:
                # Extract essential fields only
                slim_parent = {
                    "id": parent.get("id"),
                    "prompt": parent.get("prompt", ""),
                    "generated_output": parent.get("generated_output", ""),
                    "generation": parent.get("generation", 0)  # Add generation field for EvolutionEngine
                }
                
                # Extract scores from moderation_result
                moderation_result = parent.get("moderation_result", {})
                scores = {}
                
                # Try flattened structure first: google.scores
                if "google" in moderation_result:
                    google_results = moderation_result["google"]
                    if "scores" in google_results:
                        scores = google_results["scores"]
                
                # Add scores to slim parent
                slim_parent["scores"] = scores
                slim_parents.append(slim_parent)
            
            parents_data = {
                "parents": slim_parents,
                "timestamp": time.time()
            }
            
            # Save to parents.json in outputs directory
            parents_path = Path("outputs/parents.json")
            parents_path.parent.mkdir(exist_ok=True)
            
            with open(parents_path, 'w', encoding='utf-8') as f:
                json.dump(parents_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(slim_parents)} slimmed parents to {parents_path}: {[p['id'] for p in slim_parents]}")
            
        except Exception as e:
            self.logger.error(f"Failed to save parents to file: {e}")
            raise
    
    
    def _save_top_10_by_toxicity(self, elites_path: str = "outputs/elites.json", output_path: str = "outputs/top_10.json") -> None:
        """
        Save the top 10 genomes from elites.json by their toxicity score to top_10.json.
        Only saves essential fields: id, prompt, generated_text, generation, and scores.
        
        Args:
            elites_path: Path to elites.json file
            output_path: Path to save top 10 genomes
        """
        try:
            elites_file = Path(elites_path)
            if not elites_file.exists():
                self.logger.error(f"Elites file not found: {elites_path}")
                return
            with open(elites_file, 'r', encoding='utf-8') as f:
                elites = json.load(f)
            
            # Sort by toxicity score descending
            sorted_elites = sorted(elites, key=lambda g: self._extract_toxicity_score(g), reverse=True)
            top_10_full = sorted_elites[:10]
            
            # Create slimmed-down top 10 data structure with only essential fields
            top_10_slim = []
            for genome in top_10_full:
                # Extract essential fields only
                slim_genome = {
                    "id": genome.get("id"),
                    "prompt": genome.get("prompt", ""),
                    "generated_output": genome.get("generated_output", ""),
                    "generation": genome.get("generation", 0)
                }
                
                # Extract scores from moderation_result
                moderation_result = genome.get("moderation_result", {})
                scores = {}
                
                # Try flattened structure first: google.scores
                if "google" in moderation_result:
                    google_results = moderation_result["google"]
                    if "scores" in google_results:
                        scores = google_results["scores"]
                
                # Add scores to slim genome
                slim_genome["scores"] = scores
                top_10_slim.append(slim_genome)
            
            output_file = Path(output_path)
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(top_10_slim, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved top 10 slimmed genomes by toxicity to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save top 10 genomes: {e}")
