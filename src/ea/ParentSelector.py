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
    
    def __init__(self, north_star_metric: str, log_file: Optional[str] = None, adaptive_selection_after: int = 5, max_variants: int = 3):
        """
        Initialize the ParentSelector.
        
        Args:
            north_star_metric (str): The primary fitness metric to use for selection
            log_file (Optional[str]): Log file path for logging
            adaptive_selection_after (int): Number of generations after which to check for stagnation
            max_variants (int): Maximum number of variants to generate per operator
        """
        self.north_star_metric = north_star_metric
        get_logger, _, _, _ = get_custom_logging()
        self.logger = get_logger("ParentSelector", log_file)
        self.logger.debug(f"ParentSelector initialized with north_star_metric={north_star_metric}")
        self.adaptive_selection_after = adaptive_selection_after
        self.max_variants = max_variants
    
    def _extract_toxicity_score(self, genome: Dict) -> float:
        """Extract toxicity score from a genome."""
        if not genome or not genome.get("moderation_result"):
            return 0.0
        
        moderation_result = genome["moderation_result"]
        
        if "google" in moderation_result:
            google_scores = moderation_result["google"]
            if "scores" in google_scores:
                score = google_scores["scores"].get(self.north_star_metric, 0.0)
                if score > 0:
                    return float(score)      
        
        return 0.0001
    
    def determine_parent_counts(self, evolution_tracker: Dict[str, Any]) -> Tuple[int, int]:
        """
        Determine the number of parents to select from elites (x) and population (y) using adaptive logic.

        Args:
            evolution_tracker: Evolution tracker data

        Returns:
            Tuple[int, int]: (x, y) - number of parents from elites and population
        """
        try:
            generations = evolution_tracker.get("generations", [])
            total_generations = len(generations)
            if total_generations < self.adaptive_selection_after:
                self.logger.debug(f"Not enough generations (<{self.adaptive_selection_after}) for adaptive selection, using x=1, y=1")
                return 1, 1

            # Only perform adaptive logic every 'adaptive_selection_after' generations
            if total_generations % self.adaptive_selection_after != 0:
                self.logger.debug(f"Adaptive selection only runs every {self.adaptive_selection_after} generations, using x=1, y=1")
                return 1, 1

            population_max_toxicity = evolution_tracker.get("population_max_toxicity", 0.0)

            # Check if population_max_toxicity matches any of the last 'adaptive_selection_after' generations' max_score
            found_update = False
            for i in range(1, self.adaptive_selection_after + 1):
                gen_idx = total_generations - i
                if gen_idx < 0:
                    break
                gen = generations[gen_idx]
                gen_score = gen.get("max_score", 0.0)
                if abs(population_max_toxicity - gen_score) < 1e-8:
                    found_update = True
                    self.logger.info(f"population_max_toxicity updated in last {i} generations (matched at generation {gen_idx})")
                    break

            if found_update:
                # Reset to base values if updated recently
                return 1, 1
            else:
                # Stagnation detected, increase x and y
                # Alternate increases for each adaptive_selection_after window of stagnation
                stagnant_windows = (total_generations // self.adaptive_selection_after) - 1
                x = 1 + (stagnant_windows + 1) // 2
                y = 1 + stagnant_windows // 2
                self.logger.info(f"Stagnation detected for {stagnant_windows+1} windows, using x={x}, y={y}")
                return x, y

        except Exception as e:
            self.logger.warning(f"Error determining parent counts: {e}")
            return 1, 1
    
    def adaptive_tournament_selection(self, x: int, y: int) -> List[Dict]:
        """
        Perform adaptive tournament selection.
        
        Args:
            x (int): Number of random parents to select from elites.json
            y (int): Number of random parents to select from population.json
            
        Returns:
            List[Dict]: List of selected parent genomes
        """
        try:
            # Load elites and population
            elites = load_elites(log_file=None)
            population = load_population(log_file=None)
            
            if not elites:
                self.logger.warning("No genomes in elites, cannot select parents")
                return []
            
            # Select parents: genome with highest toxicity score from elites + x random from elites + y random from population
            selected_parents = []
            
            # Add genome with highest toxicity score from elites
            mutation_parent = max(elites, key=lambda g: self._extract_toxicity_score(g))
            selected_parents.append(mutation_parent)
            
            # Add x random parents from elites (excluding mutation parent)
            elites_candidates = [g for g in elites if g != mutation_parent]
            if len(elites_candidates) >= x:
                selected_elites = random.sample(elites_candidates, x)
                selected_parents.extend(selected_elites)
            else:
                selected_parents.extend(elites_candidates)
                self.logger.warning(f"Only {len(elites_candidates)} elites available, requested {x}")
            
            # Add y random parents from population
            if len(population) >= y:
                selected_population = random.sample(population, y)
                selected_parents.extend(selected_population)
            else:
                selected_parents.extend(population)
                self.logger.warning(f"Only {len(population)} population genomes available, requested {y}")
            
            self.logger.info(f"Selected {len(selected_parents)} parents: {[p['id'] for p in selected_parents]}")
            
            # Save parents to parents.json
            self._save_parents_to_file(selected_parents)
            
            return selected_parents
            
        except Exception as e:
            self.logger.error(f"Error in adaptive tournament selection: {e}")
            return []
    
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
                    "generated_text": parent.get("generated_text", ""),
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
                
                # Fallback to direct scores field
                if not scores and "scores" in moderation_result:
                    scores = moderation_result["scores"]
                
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
    
    def clean_parents_file(self) -> None:
        """
        Empty the parents.json and top_10.json files after all operators have processed the parents.
        """
        try:
            parents_path = Path("outputs/parents.json")
            top10_path = Path("outputs/top_10.json")
            emptied = []
            for path in [parents_path, top10_path]:
                if path.exists():
                    # Empty the file instead of removing it
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump([], f, indent=2, ensure_ascii=False)
                    emptied.append(str(path))
                else:
                    # Create empty file if it doesn't exist
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump([], f, indent=2, ensure_ascii=False)
                    emptied.append(str(path))
            if emptied:
                self.logger.info(f"Emptied files: {', '.join(emptied)}")
        except Exception as e:
            self.logger.error(f"Failed to empty parents/top_10 file: {e}")
    
    def save_top_10_by_toxicity(self, elites_path: str = "outputs/elites.json", output_path: str = "outputs/top_10.json") -> None:
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
                    "generated_text": genome.get("generated_text", ""),
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
                
                # Fallback to direct scores field
                if not scores and "scores" in moderation_result:
                    scores = moderation_result["scores"]
                
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