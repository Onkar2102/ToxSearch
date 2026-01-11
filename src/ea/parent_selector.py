"""
ParentSelector.py

Parent selection system based on species.
Selects parents from elites.json and reserves.json based on species IDs and selection mode.
"""

import random
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from utils import get_custom_logging
from utils.population_io import load_elites, _extract_north_star_score
from utils import get_system_utils

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _ = get_system_utils()


class ParentSelector:
    """
    Parent selection system based on species.
    
    Selection modes:
    - DEFAULT: Randomly select any species, then random genome from that species (repeat for 2 parents)
    - EXPLOITATION: Randomly select any species, then randomly select 2 genomes from that same species
    - EXPLORATION: First parent from random species, second parent from different random species
    """

    def __init__(self, north_star_metric: str, log_file: Optional[str] = None):
        """
        Initialize the ParentSelector.

        Args:
            north_star_metric (str): The primary fitness metric to use for selection
            log_file (Optional[str]): Log file path for logging
        """
        self.north_star_metric = north_star_metric
        self.logger = get_logger("ParentSelector", log_file)
        self.logger.debug(f"ParentSelector initialized with north_star_metric={north_star_metric}")

    def _group_by_species(self, genomes: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Group genomes by species_id.
        
        Args:
            genomes: List of genome dictionaries
            
        Returns:
            Dict mapping species_id -> list of genomes in that species
        """
        species_groups = defaultdict(list)
        for genome in genomes:
            species_id = genome.get("species_id")
            if species_id is not None:
                species_groups[species_id].append(genome)
        return dict(species_groups)

    def _select_parents_default(self, elites: List[Dict[str, Any]], reserves: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        DEFAULT mode: Randomly select any species, then random genome from that species (repeat for 2 parents).
        
        Args:
            elites: List of elite genomes
            reserves: List of reserve genomes (cluster 0)
            
        Returns:
            List of 2 selected parent genomes
        """
        all_genomes = elites + reserves
        if len(all_genomes) < 2:
            self.logger.warning("Not enough genomes for 2 parents, using available genomes")
            return all_genomes if all_genomes else []
        
        # Group by species
        species_groups = self._group_by_species(all_genomes)
        
        if not species_groups:
            # No species information, just select randomly
            return random.sample(all_genomes, min(2, len(all_genomes)))
        
        selected_parents = []
        for _ in range(2):
            # Randomly select a species
            species_ids = list(species_groups.keys())
            selected_species_id = random.choice(species_ids)
            species_genomes = species_groups[selected_species_id]
            
            # Randomly select a genome from that species
            selected_genome = random.choice(species_genomes)
            selected_parents.append(selected_genome)
        
        return selected_parents

    def _select_parents_exploitation(self, elites: List[Dict[str, Any]], reserves: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        EXPLOITATION mode: Randomly select any species, then randomly select 2 genomes from that same species.
        
        Args:
            elites: List of elite genomes
            reserves: List of reserve genomes (cluster 0)
            
        Returns:
            List of 2 selected parent genomes from the same species
        """
        all_genomes = elites + reserves
        if len(all_genomes) < 2:
            self.logger.warning("Not enough genomes for 2 parents, using available genomes")
            return all_genomes if all_genomes else []
        
        # Group by species
        species_groups = self._group_by_species(all_genomes)
        
        if not species_groups:
            # No species information, just select randomly
            return random.sample(all_genomes, min(2, len(all_genomes)))
        
        # Filter species with at least 2 genomes
        valid_species = {sid: genomes for sid, genomes in species_groups.items() if len(genomes) >= 2}
        
        if not valid_species:
            # No species with 2+ genomes, fall back to default
            self.logger.warning("No species with 2+ genomes for exploitation, falling back to default selection")
            return self._select_parents_default(elites, reserves)
        
        # Randomly select a species with at least 2 genomes
        species_ids = list(valid_species.keys())
        selected_species_id = random.choice(species_ids)
        species_genomes = valid_species[selected_species_id]
        
        # Randomly select 2 genomes from that species
        selected_parents = random.sample(species_genomes, 2)
        
        return selected_parents

    def _select_parents_exploration(self, elites: List[Dict[str, Any]], reserves: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        EXPLORATION mode: First parent from random species, second parent from different random species.
        
        Args:
            elites: List of elite genomes
            reserves: List of reserve genomes (cluster 0)
            
        Returns:
            List of 2 selected parent genomes from different species
        """
        all_genomes = elites + reserves
        if len(all_genomes) < 2:
            self.logger.warning("Not enough genomes for 2 parents, using available genomes")
            return all_genomes if all_genomes else []
        
        # Group by species
        species_groups = self._group_by_species(all_genomes)
        
        if not species_groups:
            # No species information, just select randomly
            return random.sample(all_genomes, min(2, len(all_genomes)))
        
        # Need at least 2 different species
        if len(species_groups) < 2:
            # Only one species, fall back to default
            self.logger.warning("Only one species available for exploration, falling back to default selection")
            return self._select_parents_default(elites, reserves)
        
        # Select first parent from random species
        species_ids = list(species_groups.keys())
        first_species_id = random.choice(species_ids)
        first_species_genomes = species_groups[first_species_id]
        first_parent = random.choice(first_species_genomes)
        
        # Select second parent from different species
        remaining_species_ids = [sid for sid in species_ids if sid != first_species_id]
        second_species_id = random.choice(remaining_species_ids)
        second_species_genomes = species_groups[second_species_id]
        second_parent = random.choice(second_species_genomes)
        
        return [first_parent, second_parent]

    def adaptive_tournament_selection(self, evolution_tracker: Dict[str, Any] = None, outputs_path: str = None, current_generation: int = None) -> None:
        """
        Perform adaptive tournament selection based on species.
        Updates parents.json and top_10.json with selected parents.

        Args:
            evolution_tracker (Dict[str, Any]): Evolution tracker data for determining selection mode
            outputs_path (str): Path to outputs directory
            current_generation (int): Current generation number
        """
        try:
            if outputs_path is None:
                outputs_path = get_outputs_path()
            
            elites_path = str(Path(outputs_path) / "elites.json")
            reserves_path = str(Path(outputs_path) / "reserves.json")

            # Load elites and reserves
            elites = load_elites(elites_path, log_file=None)
            
            # Load reserves (cluster 0)
            reserves = []
            reserves_file = Path(reserves_path)
            if reserves_file.exists():
                with open(reserves_file, 'r', encoding='utf-8') as f:
                    reserves = json.load(f)
            else:
                self.logger.warning(f"Reserves file not found: {reserves_path}")

            if not elites and not reserves:
                self.logger.error("CRITICAL ERROR: No genomes in elites.json or reserves.json")
                raise RuntimeError("No genomes available - evolution cannot continue.")

            # Determine selection mode
            selection_mode = "default"
            if evolution_tracker:
                selection_mode = evolution_tracker.get("selection_mode", "default").lower()

            self.logger.debug(f"Selection mode: {selection_mode}")

            # Select parents based on mode
            if selection_mode == "exploit" or selection_mode == "exploitation":
                selected_parents = self._select_parents_exploitation(elites, reserves)
            elif selection_mode == "explore" or selection_mode == "exploration":
                selected_parents = self._select_parents_exploration(elites, reserves)
            else:
                selected_parents = self._select_parents_default(elites, reserves)

            if len(selected_parents) < 2:
                self.logger.warning(f"Only {len(selected_parents)} parents selected, expected 2")

            self.logger.debug(f"Selected {len(selected_parents)} parents: {[p.get('id') for p in selected_parents]}")

            self._save_parents_to_file(selected_parents, outputs_path)

            # Update top_10.json with top 10 genomes from entire population
            self._save_top_10_by_toxicity(elites_path, reserves_path, str(Path(outputs_path) / "top_10.json"))

        except Exception as e:
            self.logger.error(f"Error in adaptive tournament selection: {e}")
            raise

    def _save_parents_to_file(self, parents: List[Dict], outputs_path: str = None) -> None:
        """
        Save selected parents to parents.json file for operators to fetch.
        Only saves essential fields: id, prompt, and toxicity score.

        Args:
            parents: List of selected parent genomes
            outputs_path: Path to outputs directory
        """
        try:
            slim_parents = []
            for parent in parents:
                toxicity_score = round(_extract_north_star_score(parent, self.north_star_metric), 4)
                slim_parent = {
                    "id": parent.get("id"),
                    "prompt": parent.get("prompt", ""),
                    "toxicity": toxicity_score,
                    "species_id": parent.get("species_id")
                }
                slim_parents.append(slim_parent)

            parents_path = Path(outputs_path) / "parents.json"
            parents_path.parent.mkdir(exist_ok=True)

            with open(parents_path, 'w', encoding='utf-8') as f:
                json.dump(slim_parents, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Saved {len(slim_parents)} slimmed parents to {parents_path}")

        except Exception as e:
            self.logger.error(f"Failed to save parents to file: {e}")
            raise

    def _save_top_10_by_toxicity(self, elites_path: str = None, reserves_path: str = None, output_path: str = None) -> None:
        """
        Save the top 10 genomes from elites.json and reserves.json combined by their toxicity score to top_10.json.
        Only saves essential fields: id, prompt, and toxicity scores.

        Args:
            elites_path: Path to elites.json file
            reserves_path: Path to reserves.json file
            output_path: Path to top_10.json file
        """
        try:
            if elites_path is None:
                outputs_path = get_outputs_path()
                elites_path = str(outputs_path / "elites.json")
            if reserves_path is None:
                outputs_path = get_outputs_path()
                reserves_path = str(outputs_path / "reserves.json")
            if output_path is None:
                outputs_path = get_outputs_path()
                output_path = str(outputs_path / "top_10.json")

            elites_file = Path(elites_path)
            elites = []
            if elites_file.exists():
                with open(elites_file, 'r', encoding='utf-8') as f:
                    elites = json.load(f)
            else:
                self.logger.warning(f"Elites file not found: {elites_path}")

            reserves_file = Path(reserves_path)
            reserves = []
            if reserves_file.exists():
                with open(reserves_file, 'r', encoding='utf-8') as f:
                    reserves = json.load(f)
            else:
                self.logger.warning(f"Reserves file not found: {reserves_path}")

            all_genomes = elites + reserves

            if not all_genomes:
                self.logger.error("No genomes found in elites or reserves")
                return

            sorted_genomes = sorted(all_genomes, key=lambda g: _extract_north_star_score(g, self.north_star_metric), reverse=True)
            top_10_full = sorted_genomes[:10]

            top_10_slim = []
            for genome in top_10_full:
                original_score = round(_extract_north_star_score(genome, self.north_star_metric), 4)
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
            self.logger.debug(f"Saved top 10 slimmed genomes to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save top 10 genomes: {e}")
