"""
ParentSelector.py

Parent selection system based on species.
Selects parents from elites.json and reserves.json based on species IDs and selection mode.
Species are sorted by fitness (descending), and frozen species are excluded.
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

# Species ID for reserves (cluster 0)
CLUSTER_0_ID = 0


class ParentSelector:
    """
    Parent selection system based on species.

    Species are sorted by best_fitness (descending). Frozen species are excluded.
    Species 0 (reserves/cluster0) is included in selection.

    Selection modes:
    - DEFAULT: Randomly select species 1 and species 2, then randomly select genomes from each
    - EXPLOITATION: Select species with highest fitness, select genome with highest fitness as parent 1,
                    then randomly select parent 2 from same species
    - EXPLORATION: Select top species (highest fitness), select genome with highest fitness as parent 1,
                   then randomly select different species, select genome with highest fitness as parent 2
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

    def _load_speciation_state(self, outputs_path: str) -> Dict[str, Any]:
        """
        Load speciation_state.json to get species information.
        
        Args:
            outputs_path: Path to outputs directory
            
        Returns:
            Dict containing speciation state, or empty dict if not found
        """
        speciation_state_path = Path(outputs_path) / "speciation_state.json"
        if not speciation_state_path.exists():
            self.logger.warning(f"Speciation state file not found: {speciation_state_path}")
            return {}
        
        try:
            with open(speciation_state_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load speciation state: {e}")
            return {}

    def _get_active_species_ids(self, speciation_state: Dict[str, Any], all_species_in_genomes: set = None) -> set:
        """
        Get IDs of active species (not frozen).
        
        Species are considered active if:
        1. They exist in speciation_state with species_state != "frozen"
        2. They exist in genomes but not in speciation_state (assumed active)
        3. Species 0 (cluster0/reserves) is always included
        
        Args:
            speciation_state: Loaded speciation state dictionary
            all_species_in_genomes: Optional set of all species IDs found in genomes
            
        Returns:
            Set of active species IDs (including cluster 0)
        """
        frozen_ids = set()
        tracked_ids = set()
        
        # Check species states in speciation state
        species_dict = speciation_state.get("species", {})
        for sid_str, sp_data in species_dict.items():
            sid = int(sid_str)
            tracked_ids.add(sid)
            species_state = sp_data.get("species_state", "active")
            if species_state == "frozen":
                frozen_ids.add(sid)
        
        # Start with all species in genomes (if provided), otherwise use tracked species
        if all_species_in_genomes is not None:
            active_ids = all_species_in_genomes.copy()
        else:
            active_ids = tracked_ids.copy()
        
        # Always include cluster 0 (reserves)
        active_ids.add(CLUSTER_0_ID)
        
        # Remove frozen species
        active_ids -= frozen_ids
        
        self.logger.debug(f"Active species: {sorted(active_ids)}, Frozen: {sorted(frozen_ids)}")
        
        return active_ids

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

    def _calculate_species_best_fitness(self, species_groups: Dict[int, List[Dict[str, Any]]]) -> Dict[int, float]:
        """
        Calculate best fitness for each species.
        
        Args:
            species_groups: Dict mapping species_id -> list of genomes
            
        Returns:
            Dict mapping species_id -> best fitness score
        """
        best_fitness = {}
        for species_id, genomes in species_groups.items():
            if genomes:
                max_fitness = max(_extract_north_star_score(g, self.north_star_metric) for g in genomes)
                best_fitness[species_id] = max_fitness
            else:
                best_fitness[species_id] = 0.0
        return best_fitness

    def _get_sorted_active_species(
        self, 
        species_groups: Dict[int, List[Dict[str, Any]]], 
        active_species_ids: set
    ) -> List[Tuple[int, List[Dict[str, Any]], float]]:
        """
        Get list of active species sorted by best fitness (descending).
        
        Args:
            species_groups: Dict mapping species_id -> list of genomes
            active_species_ids: Set of active (non-frozen) species IDs
            
        Returns:
            List of tuples (species_id, genomes, best_fitness) sorted by best_fitness descending
        """
        # Calculate best fitness per species
        species_fitness = self._calculate_species_best_fitness(species_groups)
        
        # Filter to active species only
        active_species = []
        for species_id, genomes in species_groups.items():
            if species_id in active_species_ids and genomes:
                best_fit = species_fitness.get(species_id, 0.0)
                active_species.append((species_id, genomes, best_fit))
        
        # Sort by best fitness descending
        active_species.sort(key=lambda x: x[2], reverse=True)
        
        return active_species

    def _get_genome_with_highest_fitness(self, genomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get the genome with the highest fitness score from a list.
        
        Args:
            genomes: List of genome dictionaries
            
        Returns:
            Genome with highest fitness score
        """
        if not genomes:
            return {}
        return max(genomes, key=lambda g: _extract_north_star_score(g, self.north_star_metric))

    def _select_parents_default(
        self, 
        elites: List[Dict[str, Any]], 
        reserves: List[Dict[str, Any]],
        active_species_ids: set
    ) -> List[Dict[str, Any]]:
        """
        DEFAULT mode: Randomly select species 1 and species 2 from sorted list,
        then randomly select one genome from each as parents.
        
        Args:
            elites: List of elite genomes
            reserves: List of reserve genomes (cluster 0)
            active_species_ids: Set of active (non-frozen) species IDs
            
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
            self.logger.warning("No species information, falling back to random selection")
            return random.sample(all_genomes, min(2, len(all_genomes)))
        
        # Get sorted active species
        sorted_species = self._get_sorted_active_species(species_groups, active_species_ids)
        
        if not sorted_species:
            # No active species, fall back to random
            self.logger.warning("No active species found, falling back to random selection")
            return random.sample(all_genomes, min(2, len(all_genomes)))
        
        selected_parents = []
        
        # Randomly select 2 species (can be same or different)
        for _ in range(2):
            selected_species = random.choice(sorted_species)
            species_id, species_genomes, _ = selected_species
            
            # Randomly select a genome from the species
            selected_genome = random.choice(species_genomes)
            selected_parents.append(selected_genome)
        
        self.logger.debug(f"DEFAULT mode: Selected parents from species {[p.get('species_id') for p in selected_parents]}")
        return selected_parents

    def _select_parents_exploitation(
        self, 
        elites: List[Dict[str, Any]], 
        reserves: List[Dict[str, Any]],
        active_species_ids: set
    ) -> List[Dict[str, Any]]:
        """
        EXPLOITATION mode: Select species with highest fitness, select genome with 
        highest fitness as parent 1, then randomly select parent 2 from same species.
        
        Args:
            elites: List of elite genomes
            reserves: List of reserve genomes (cluster 0)
            active_species_ids: Set of active (non-frozen) species IDs
            
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
            self.logger.warning("No species information, falling back to random selection")
            return random.sample(all_genomes, min(2, len(all_genomes)))
        
        # Get sorted active species
        sorted_species = self._get_sorted_active_species(species_groups, active_species_ids)
        
        if not sorted_species:
            # No active species, fall back to random
            self.logger.warning("No active species found, falling back to random selection")
            return random.sample(all_genomes, min(2, len(all_genomes)))
        
        # Filter species with at least 2 genomes
        valid_species = [(sid, genomes, fit) for sid, genomes, fit in sorted_species if len(genomes) >= 2]
        
        if not valid_species:
            # No species with 2+ genomes, fall back to default
            self.logger.warning("No species with 2+ genomes for exploitation, falling back to default selection")
            return self._select_parents_default(elites, reserves, active_species_ids)
        
        # Select species with highest fitness (first in sorted list)
        top_species_id, top_species_genomes, top_fitness = valid_species[0]
        
        # Parent 1: genome with highest fitness
        parent1 = self._get_genome_with_highest_fitness(top_species_genomes)
        
        # Parent 2: random genome from same species (excluding parent 1 if possible)
        remaining_genomes = [g for g in top_species_genomes if g.get("id") != parent1.get("id")]
        if remaining_genomes:
            parent2 = random.choice(remaining_genomes)
        else:
            # Only one unique genome, use it again
            parent2 = parent1
        
        self.logger.debug(f"EXPLOITATION mode: Selected parents from species {top_species_id} (fitness={top_fitness:.4f})")
        return [parent1, parent2]

    def _select_parents_exploration(
        self, 
        elites: List[Dict[str, Any]], 
        reserves: List[Dict[str, Any]],
        active_species_ids: set
    ) -> List[Dict[str, Any]]:
        """
        EXPLORATION mode: Select top species (highest fitness), select genome with highest fitness as parent 1.
        Then randomly select a different species, select genome with highest fitness as parent 2.
        
        Args:
            elites: List of elite genomes
            reserves: List of reserve genomes (cluster 0)
            active_species_ids: Set of active (non-frozen) species IDs

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
            self.logger.warning("No species information, falling back to random selection")
            return random.sample(all_genomes, min(2, len(all_genomes)))
        
        # Get sorted active species
        sorted_species = self._get_sorted_active_species(species_groups, active_species_ids)
        
        if not sorted_species:
            # No active species, fall back to random
            self.logger.warning("No active species found, falling back to random selection")
            return random.sample(all_genomes, min(2, len(all_genomes)))
        
        # Need at least 2 different species
        if len(sorted_species) < 2:
            # Only one species, fall back to default
            self.logger.warning("Only one species available for exploration, falling back to default selection")
            return self._select_parents_default(elites, reserves, active_species_ids)
        
        # Select top species (first in sorted list - highest fitness)
        first_species_id, first_species_genomes, first_fitness = sorted_species[0]
        
        # Parent 1: genome with highest fitness from top species
        parent1 = self._get_genome_with_highest_fitness(first_species_genomes)
        
        # Select second species (different from first, randomly chosen)
        remaining_species = sorted_species[1:]  # All species except the top one
        second_species_id, second_species_genomes, second_fitness = random.choice(remaining_species)
        
        # Parent 2: genome with highest fitness from second species
        parent2 = self._get_genome_with_highest_fitness(second_species_genomes)
        
        self.logger.debug(f"EXPLORATION mode: Parent 1 from top species {first_species_id} (fitness={first_fitness:.4f}), "
                         f"Parent 2 from randomly selected species {second_species_id} (fitness={second_fitness:.4f})")
        return [parent1, parent2]

    def adaptive_tournament_selection(self, evolution_tracker: Dict[str, Any] = None, outputs_path: str = None, current_generation: int = None) -> None:
        """
        Perform adaptive tournament selection based on species.
        Updates parents.json and top_10.json with selected parents.
        
        Species are sorted by best_fitness (descending).
        Frozen species are excluded from selection.
        Species 0 (reserves) is included.

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

            # Get all species IDs present in genomes
            all_genomes = elites + reserves
            species_groups = self._group_by_species(all_genomes)
            all_species_in_genomes = set(species_groups.keys())
            
            # Load speciation state to get frozen species
            speciation_state = self._load_speciation_state(outputs_path)
            active_species_ids = self._get_active_species_ids(speciation_state, all_species_in_genomes)
            
            self.logger.debug(f"Active species IDs: {sorted(active_species_ids)}")

            # Determine selection mode
            selection_mode = "default"
            if evolution_tracker:
                selection_mode = evolution_tracker.get("selection_mode", "default").lower()

            self.logger.debug(f"Selection mode: {selection_mode}")

            # Select parents based on mode
            if selection_mode == "exploit" or selection_mode == "exploitation":
                selected_parents = self._select_parents_exploitation(elites, reserves, active_species_ids)
            elif selection_mode == "explore" or selection_mode == "exploration":
                selected_parents = self._select_parents_exploration(elites, reserves, active_species_ids)
            else:
                selected_parents = self._select_parents_default(elites, reserves, active_species_ids)

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
