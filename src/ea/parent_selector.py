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
        self._last_outputs_path = None  # Store for fallback logic
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

    def _fallback_to_reserves_or_frozen(
        self,
        elites: List[Dict[str, Any]],
        reserves: List[Dict[str, Any]],
        species_groups: Dict[int, List[Dict[str, Any]]],
        active_species_ids: set,
        outputs_path: str = None,
        num_parents: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Fallback logic when no active species found.
        
        Priority:
        1. Select from reserves if available and sufficient
        2. If reserves empty/insufficient, include frozen species and use normal selection
        
        Args:
            elites: List of elite genomes
            reserves: List of reserve genomes
            species_groups: Dict mapping species_id -> genomes
            active_species_ids: Set of active species IDs
            outputs_path: Path to outputs directory (for loading speciation state)
            num_parents: Number of parents needed
            
        Returns:
            List of selected parents, or empty list if fallback fails
        """
        all_genomes = elites + reserves
        
        # First try: select from reserves only
        if reserves and len(reserves) >= num_parents:
            self.logger.info(f"Selecting {num_parents} parents from reserves (all species frozen)")
            return random.sample(reserves, num_parents)
        
        # Second try: if reserves empty or insufficient, include frozen species
        if outputs_path is None:
            outputs_path = get_outputs_path()
        
        speciation_state = self._load_speciation_state(outputs_path)
        frozen_species_ids = set()
        species_dict = speciation_state.get("species", {})
        historical_species = speciation_state.get("historical_species", {})
        
        # Check both active species dict and historical_species for frozen
        for sid_str, sp_data in list(species_dict.items()) + list(historical_species.items()):
            sid = int(sid_str) if isinstance(sid_str, str) else sid_str
            species_state = sp_data.get("species_state", "active")
            if species_state == "frozen":
                frozen_species_ids.add(sid)
        
        if not frozen_species_ids:
            # No frozen species either, return empty to trigger final fallback
            return []
        
        # Include frozen species in active set for selection
        active_species_ids_with_frozen = active_species_ids.copy()
        active_species_ids_with_frozen.update(frozen_species_ids)
        active_species_ids_with_frozen.add(CLUSTER_0_ID)  # Always include reserves
        
        # Try again with frozen species included
        sorted_species_with_frozen = self._get_sorted_active_species(species_groups, active_species_ids_with_frozen)
        if sorted_species_with_frozen:
            self.logger.info(f"Using frozen species for parent selection (reserves empty/insufficient, {len(frozen_species_ids)} frozen species available)")
            # Use normal selection logic with frozen species
            # For simplicity, select randomly from available species
            selected_species = random.choice(sorted_species_with_frozen)
            species_id, species_genomes, _ = selected_species
            if len(species_genomes) >= num_parents:
                return random.sample(species_genomes, num_parents)
            else:
                # Species has < num_parents members, select what we can and supplement from all
                selected = list(species_genomes)
                remaining = [g for g in all_genomes if g not in selected]
                needed = num_parents - len(selected)
                if remaining and needed > 0:
                    selected.extend(random.sample(remaining, min(needed, len(remaining))))
                return selected[:num_parents] if len(selected) >= num_parents else selected
        
        return []

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
        DEFAULT mode: Randomly select one species, then randomly select 2 genomes from that species.
        
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
            # No active species - prefer reserves first, then frozen species
            self.logger.warning("No active species found, checking reserves and frozen species")
            
            # First try: select from reserves only
            if reserves and len(reserves) >= 2:
                self.logger.info("Selecting parents from reserves (all species frozen)")
                return random.sample(reserves, min(2, len(reserves)))
            
            # Second try: if reserves empty or insufficient, include frozen species
            # Get frozen species IDs from speciation state
            speciation_state = self._load_speciation_state(outputs_path if hasattr(self, '_last_outputs_path') else get_outputs_path())
            frozen_species_ids = set()
            species_dict = speciation_state.get("species", {})
            historical_species = speciation_state.get("historical_species", {})
            
            # Check both active species dict and historical_species for frozen
            for sid_str, sp_data in list(species_dict.items()) + list(historical_species.items()):
                sid = int(sid_str) if isinstance(sid_str, str) else sid_str
                species_state = sp_data.get("species_state", "active")
                if species_state == "frozen":
                    frozen_species_ids.add(sid)
            
            # Include frozen species in active set for selection
            active_species_ids_with_frozen = active_species_ids.copy()
            active_species_ids_with_frozen.update(frozen_species_ids)
            active_species_ids_with_frozen.add(CLUSTER_0_ID)  # Always include reserves
            
            # Try again with frozen species included
            sorted_species_with_frozen = self._get_sorted_active_species(species_groups, active_species_ids_with_frozen)
            if sorted_species_with_frozen:
                self.logger.info(f"Using frozen species for parent selection (reserves empty, {len(frozen_species_ids)} frozen species available)")
                selected_species = random.choice(sorted_species_with_frozen)
                species_id, species_genomes, _ = selected_species
                if len(species_genomes) >= 2:
                    return random.sample(species_genomes, 2)
                else:
                    # Fallback to all genomes if species has < 2 members
                    return random.sample(all_genomes, min(2, len(all_genomes)))
            
            # Final fallback: random from all genomes
            self.logger.warning("No active or frozen species found, falling back to random selection from all genomes")
            return random.sample(all_genomes, min(2, len(all_genomes)))
        
        # Randomly select one species
        selected_species = random.choice(sorted_species)
        species_id, species_genomes, _ = selected_species
        
        # Check if species has at least 2 genomes
        if len(species_genomes) < 2:
            # If only one genome, select it twice (or select from all genomes as fallback)
            self.logger.warning(f"Species {species_id} has only {len(species_genomes)} genome(s), selecting from all genomes")
            return random.sample(all_genomes, min(2, len(all_genomes)))
        
        # Randomly select 2 genomes from the selected species
        selected_parents = random.sample(species_genomes, 2)
        
        self.logger.debug(f"DEFAULT mode: Selected 2 parents from species {species_id}")
        return selected_parents

    def _select_parents_exploitation(
        self, 
        elites: List[Dict[str, Any]], 
        reserves: List[Dict[str, Any]],
        active_species_ids: set
    ) -> List[Dict[str, Any]]:
        """
        EXPLOITATION mode: Select species with highest fitness, select 3 parents from same species.
        - Parent 1: Highest fitness genome from top species
        - Parent 2: Random genome from same top species (excluding parent 1)
        - Parent 3: Random genome from same top species (excluding parent 1 and 2)
        
        This ensures intensive local search around the best region.
        
        Args:
            elites: List of elite genomes
            reserves: List of reserve genomes (cluster 0)
            active_species_ids: Set of active (non-frozen) species IDs
            
        Returns:
            List of 3 selected parent genomes from the same species
        """
        all_genomes = elites + reserves
        if len(all_genomes) < 3:
            self.logger.warning(f"Not enough genomes for 3 parents (have {len(all_genomes)}), using available genomes")
            return all_genomes if all_genomes else []
        
        # Group by species
        species_groups = self._group_by_species(all_genomes)
        
        if not species_groups:
            # No species information, just select randomly
            self.logger.warning("No species information, falling back to random selection")
            return random.sample(all_genomes, min(3, len(all_genomes)))
        
        # Get sorted active species
        sorted_species = self._get_sorted_active_species(species_groups, active_species_ids)
        
        if not sorted_species:
            # No active species - prefer reserves first, then frozen species
            self.logger.warning("No active species found for exploitation, checking reserves and frozen species")
            selected_parents = self._fallback_to_reserves_or_frozen(elites, reserves, species_groups, active_species_ids, outputs_path=getattr(self, '_last_outputs_path', None), num_parents=3)
            if selected_parents:
                return selected_parents
            # Final fallback
            return random.sample(all_genomes, min(3, len(all_genomes)))
        
        # Filter species with at least 3 genomes
        valid_species = [(sid, genomes, fit) for sid, genomes, fit in sorted_species if len(genomes) >= 3]
        
        if not valid_species:
            # No species with 3+ genomes, try with 2+ genomes and reuse if needed
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
            # Parent 3: random genome from same species (excluding parent 1 and 2)
            remaining_genomes_2 = [g for g in remaining_genomes if g.get("id") != parent2.get("id")]
            if remaining_genomes_2:
                parent3 = random.choice(remaining_genomes_2)
            else:
                # Only 2 unique genomes, reuse parent1 for parent3
                parent3 = parent1
        else:
            # Only one unique genome, reuse for parent2 and parent3
            parent2 = parent1
            parent3 = parent1
        
        self.logger.debug(f"EXPLOITATION mode: Selected 3 parents from species {top_species_id} (fitness={top_fitness:.4f})")
        return [parent1, parent2, parent3]

    def _select_parents_exploration(
        self, 
        elites: List[Dict[str, Any]], 
        reserves: List[Dict[str, Any]],
        active_species_ids: set
    ) -> List[Dict[str, Any]]:
        """
        EXPLORATION mode: Select 3 parents from 3 different species.
        - Parent 1: Highest fitness genome from top species
        - Parent 2: Highest fitness genome from random species 2 (different from top)
        - Parent 3: Highest fitness genome from random species 3 (different from top and species 2)
        
        This ensures maximum diversity and better coverage of the fitness landscape.

        Args:
            elites: List of elite genomes
            reserves: List of reserve genomes (cluster 0)
            active_species_ids: Set of active (non-frozen) species IDs

        Returns:
            List of 3 selected parent genomes from 3 different species
        """
        all_genomes = elites + reserves
        if len(all_genomes) < 3:
            self.logger.warning(f"Not enough genomes for 3 parents (have {len(all_genomes)}), using available genomes")
            return all_genomes if all_genomes else []
        
        # Group by species
        species_groups = self._group_by_species(all_genomes)
        
        if not species_groups:
            # No species information, just select randomly
            self.logger.warning("No species information, falling back to random selection")
            return random.sample(all_genomes, min(3, len(all_genomes)))
        
        # Get sorted active species
        sorted_species = self._get_sorted_active_species(species_groups, active_species_ids)
        
        if not sorted_species:
            # No active species - prefer reserves first, then frozen species
            self.logger.warning("No active species found for exploration, checking reserves and frozen species")
            selected_parents = self._fallback_to_reserves_or_frozen(elites, reserves, species_groups, active_species_ids, outputs_path=getattr(self, '_last_outputs_path', None), num_parents=3)
            if selected_parents:
                return selected_parents
            # Final fallback
            return random.sample(all_genomes, min(3, len(all_genomes)))
        
        # Need at least 3 different species
        if len(sorted_species) < 3:
            if len(sorted_species) < 2:
                # Only one species, fall back to default
                self.logger.warning("Only one species available for exploration, falling back to default selection")
                return self._select_parents_default(elites, reserves, active_species_ids)
            else:
                # Only 2 species, use both and reuse one
                self.logger.warning("Only 2 species available for exploration, using both and reusing one")
                first_species_id, first_species_genomes, first_fitness = sorted_species[0]
                second_species_id, second_species_genomes, second_fitness = sorted_species[1]
                
                parent1 = self._get_genome_with_highest_fitness(first_species_genomes)
                parent2 = self._get_genome_with_highest_fitness(second_species_genomes)
                # Reuse parent1 for parent3
                parent3 = parent1
                
                self.logger.debug(f"EXPLORATION mode: Parent 1 from species {first_species_id} (fitness={first_fitness:.4f}), "
                                 f"Parent 2 from species {second_species_id} (fitness={second_fitness:.4f}), "
                                 f"Parent 3 reused from species {first_species_id}")
                return [parent1, parent2, parent3]
        
        # Select top species (first in sorted list - highest fitness)
        first_species_id, first_species_genomes, first_fitness = sorted_species[0]
        
        # Parent 1: genome with highest fitness from top species
        parent1 = self._get_genome_with_highest_fitness(first_species_genomes)
        
        # Select second species (different from first, randomly chosen)
        remaining_species = sorted_species[1:]  # All species except the top one
        second_species_id, second_species_genomes, second_fitness = random.choice(remaining_species)
        
        # Parent 2: genome with highest fitness from second species
        parent2 = self._get_genome_with_highest_fitness(second_species_genomes)
        
        # Select third species (different from first and second, randomly chosen)
        remaining_species_2 = [sp for sp in remaining_species if sp[0] != second_species_id]
        if remaining_species_2:
            third_species_id, third_species_genomes, third_fitness = random.choice(remaining_species_2)
            # Parent 3: genome with highest fitness from third species
            parent3 = self._get_genome_with_highest_fitness(third_species_genomes)
        else:
            # Should not happen if we have 3+ species, but fallback
            parent3 = parent1
        
        self.logger.debug(f"EXPLORATION mode: Parent 1 from top species {first_species_id} (fitness={first_fitness:.4f}), "
                         f"Parent 2 from species {second_species_id} (fitness={second_fitness:.4f}), "
                         f"Parent 3 from species {third_species_id} (fitness={third_fitness:.4f})")
        return [parent1, parent2, parent3]

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

            # Expected parent counts: 2 for DEFAULT, 3 for EXPLORATION/EXPLOITATION
            expected_count = 3 if selection_mode in ["exploit", "exploitation", "explore", "exploration"] else 2
            if len(selected_parents) < expected_count:
                self.logger.warning(f"Only {len(selected_parents)} parents selected, expected {expected_count}")

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
