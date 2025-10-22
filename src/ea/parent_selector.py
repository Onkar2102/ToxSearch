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
from utils.population_io import load_elites, load_population, _extract_north_star_score

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
    
    def _determine_parent_counts(self, evolution_tracker: Dict[str, Any]) -> Tuple[int, int]:
        """
        Determine the number of parents to select from elites (x) and population (y) using adaptive selection logic.
        
        Selection modes:
        - DEFAULT: 1 random from elites + 1 random from non_elites
        - EXPLORE: 1 random from elites + 2 random from non_elites (when stagnation > limit)
        - EXPLOIT: 2 random from elites + 1 random from non_elites (when fitness slope < 0)
        
        Args:
            evolution_tracker: Evolution tracker data containing selection_mode

        Returns:
            Tuple[int, int]: (x, y) - number of parents from elites and non_elites
        """
        selection_mode = evolution_tracker.get("selection_mode", "default")
        
        if selection_mode == "exploit":
            # EXPLOIT mode: 2 from elites, 1 from non_elites
            x, y = 2, 1
            self.logger.info(f"Using EXPLOIT mode: x={x} (elites), y={y} (non_elites)")
        elif selection_mode == "explore":
            # EXPLORE mode: 1 from elites, 2 from non_elites
            x, y = 1, 2
            self.logger.info(f"Using EXPLORE mode: x={x} (elites), y={y} (non_elites)")
        else:
            # DEFAULT mode: 1 from elites, 1 from non_elites
            x, y = 1, 1
            self.logger.info(f"Using DEFAULT mode: x={x} (elites), y={y} (non_elites)")
        
        return x, y
    
    def adaptive_tournament_selection(self, evolution_tracker: Dict[str, Any] = None, outputs_path: str = None, current_generation: int = None) -> None:
        """
        Perform adaptive tournament selection.
        Updates parents.json and top_10.json with selected parents.
        
        Args:
            evolution_tracker (Dict[str, Any]): Evolution tracker data for determining parent counts
            outputs_path (str): Path to outputs directory
            current_generation (int): Current generation number (if None, will calculate from total_generations)
        """
        try:
            # Determine parent counts using evolution tracker

            x, y = self._determine_parent_counts(evolution_tracker)
            self.logger.info(f"Determined parent counts: x={x} (elites), y={y} (population)")

            
            # Load elites and population
            elites_path = str(Path(outputs_path) / "elites.json")
            population_path = str(Path(outputs_path) / "non_elites.json")
            
            elites = load_elites(elites_path, log_file=None)
            population = load_population(population_path, log_file=None)
            
            if not elites:
                self.logger.error("CRITICAL ERROR: No genomes in elites.json - this indicates a fundamental problem with the evolution process")
                self.logger.error("Possible causes:")
                self.logger.error("1. Elite threshold calculation is incorrect")
                self.logger.error("2. Genome redistribution logic is faulty")
                self.logger.error("3. All genomes are being removed during cleanup")
                self.logger.error("4. Initial population setup failed")
                self.logger.error("Project must stop immediately to prevent further issues")
                raise RuntimeError("No elites available - evolution cannot continue. This indicates a critical system failure.")
            
            # Select parents: genome with highest toxicity score from elites + x random from elites + y random from population
            selected_parents = []
            
            # Add genome with highest toxicity score from elites
            topmost_elite = max(elites, key=lambda g: _extract_north_star_score(g, "toxicity"))
            selected_parents.append(topmost_elite)
            
            # Add x-1 random parents from elites (excluding topmost, since topmost is already added)
            elites_candidates = [g for g in elites if g != topmost_elite]
            elites_to_select = min(x - 1, len(elites_candidates))
            
            if elites_to_select > 0:
                selected_elites = random.sample(elites_candidates, elites_to_select)
                selected_parents.extend(selected_elites)
                self.logger.info(f"Selected {elites_to_select} random elite parents (excluding topmost)")
            elif x > 1:
                # Not enough unique elites - select the topmost elite multiple times
                additional_needed = x - 1
                for _ in range(additional_needed):
                    selected_parents.append(topmost_elite)
                self.logger.info(f"Not enough unique elites ({len(elites_candidates)} available). Selected topmost elite {additional_needed} additional times")
            
            # Select y random parents from population if requested
            if y > 0 and len(population) >= y:
                selected_population = random.sample(population, y)
                selected_parents.extend(selected_population)
                self.logger.info(f"Selected {y} random parents from population")
            elif y > 0 and len(population) > 0:
                # Not enough unique population parents - select with replacement
                selected_population = random.choices(population, k=y)
                selected_parents.extend(selected_population)
                self.logger.info(f"Not enough unique population parents ({len(population)} available). Selected {y} parents with replacement")
            elif y > 0:
                self.logger.warning(f"Requested {y} parents from population but population is empty")
            
            self.logger.info(f"Selected {len(selected_parents)} parents: {[p['id'] for p in selected_parents]}")
            
            # Save parents to parents.json
            self._save_parents_to_file(selected_parents, outputs_path)
            
            # Save top 10 genomes by toxicity to top_10.json
            elites_path = str(Path(outputs_path) / "elites.json")
            top_10_path = str(Path(outputs_path) / "top_10.json")
            self._save_top_10_by_toxicity(elites_path, top_10_path)
            
        except Exception as e:
            self.logger.error(f"Error in adaptive tournament selection: {e}")
            raise
    
    
    def _save_parents_to_file(self, parents: List[Dict], outputs_path: str = None) -> None:
        """
        Save selected parents to parents.json file for operators to fetch.
        Only saves essential fields: id, prompt, and toxicity score (like top_10.json).
        
        Args:
            parents: List of selected parent genomes
        """
        try:
            # Create slimmed-down parents data structure with only essential fields
            slim_parents = []
            for parent in parents:
                # Extract only id, prompt, and toxicity score (like top_10.json)
                toxicity_score = round(_extract_north_star_score(parent, "toxicity"), 4)
                slim_parent = {
                    "id": parent.get("id"),
                    "prompt": parent.get("prompt", ""),
                    "toxicity": toxicity_score
                }
                slim_parents.append(slim_parent)
            
            # Save to parents.json in outputs directory (direct array like top_10.json)
            parents_path = Path(outputs_path) / "parents.json"
            parents_path.parent.mkdir(exist_ok=True)
            
            with open(parents_path, 'w', encoding='utf-8') as f:
                json.dump(slim_parents, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(slim_parents)} slimmed parents to {parents_path}: {[p['id'] for p in slim_parents]}")
            
        except Exception as e:
            self.logger.error(f"Failed to save parents to file: {e}")
            raise
    
    
    def _save_top_10_by_toxicity(self, elites_path: str = None, output_path: str = None) -> None:
        """
        Save the top 10 genomes from elites.json and population.json combined by their toxicity score to top_10.json.
        Only saves essential fields: id, prompt, generated_text, generation, and scores.
        
        Args:
            elites_path: Path to elites.json file. If None, uses centralized get_outputs_path()
            output_path: Path to top_10.json file. If None, uses centralized get_outputs_path()
        """
        try:
            from utils.population_io import get_outputs_path
            
            # Use centralized paths if not provided
            if elites_path is None:
                outputs_path = get_outputs_path()
                elites_path = str(outputs_path / "elites.json")
            if output_path is None:
                outputs_path = get_outputs_path()
                output_path = str(outputs_path / "top_10.json")
            
            # Load elites
            elites_file = Path(elites_path)
            elites = []
            if elites_file.exists():
                with open(elites_file, 'r', encoding='utf-8') as f:
                    elites = json.load(f)
            else:
                self.logger.warning(f"Elites file not found: {elites_path}")
            
            # Load population
            population_path = elites_file.parent / "non_elites.json"
            population = []
            if population_path.exists():
                with open(population_path, 'r', encoding='utf-8') as f:
                    population = json.load(f)
            else:
                self.logger.warning(f"Population file not found: {population_path}")
            
            # Combine elites and population
            all_genomes = elites + population
            
            if not all_genomes:
                self.logger.error("No genomes found in elites or population")
                return
            
            # Sort by toxicity score descending
            sorted_genomes = sorted(all_genomes, key=lambda g: _extract_north_star_score(g, "toxicity"), reverse=True)
            top_10_full = sorted_genomes[:10]
        
            # Create slimmed-down top 10 data structure with only essential fields
            top_10_slim = []
            for genome in top_10_full:
                # Extract only id, prompt, and north star metric score (no scaling)
                original_score = round(_extract_north_star_score(genome, "toxicity"), 4)
                slim_genome = {
                    "id": genome.get("id"),
                    "prompt": genome.get("prompt", ""),
                    "toxicity": original_score
                }
                top_10_slim.append(slim_genome)
            
            output_file = Path(output_path)
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(top_10_slim, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved top 10 slimmed genomes by toxicity to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save top 10 genomes: {e}")
