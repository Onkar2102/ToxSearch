import random
from typing import List, Dict, Any, Optional, Tuple
from utils import get_custom_logging
from utils.population_io import load_elites, load_population

class ParentSelector:
    """
    A class responsible for selecting parents for genetic operations in evolutionary algorithms.
    Supports different selection strategies and is configurable for various fitness metrics.
    """
    
    def __init__(self, north_star_metric: str, log_file: Optional[str] = None):
        """
        Initialize the ParentSelector.
        
        Args:
            north_star_metric (str): The primary fitness metric to use for selection
            log_file (Optional[str]): Log file path for logging
            use_steady_state (bool): Whether to use steady state population management
        """
        self.north_star_metric = north_star_metric
        self.use_steady_state = True
        get_logger, _, _, _ = get_custom_logging()
        self.logger = get_logger("ParentSelector", log_file)
        self.logger.debug(f"ParentSelector initialized with north_star_metric={north_star_metric}, use_steady_state=True")
    
    def _extract_tournament_score(self, genome: Dict) -> float:
        """Extract score for tournament selection using the configured north star metric."""
        if not genome or not genome.get("moderation_result"):
            return 0.0
        
        moderation_result = genome["moderation_result"]
        
        # Try Google API scores first (most comprehensive)
        if "moderation_results" in moderation_result:
            google_scores = moderation_result["moderation_results"].get("google", {})
            if "scores" in google_scores:
                score = google_scores["scores"].get(self.north_star_metric, 0.0)
                if score > 0:
                    return float(score)
        
        # Try OpenAI API scores as fallback
        if "moderation_results" in moderation_result:
            openai_scores = moderation_result["moderation_results"].get("openai", {})
            if "scores" in openai_scores:
                score = openai_scores["scores"].get(self.north_star_metric, 0.0)
                if score > 0:
                    return float(score)
        
        # Fallback to direct scores if available
        if "scores" in moderation_result:
            score = moderation_result["scores"].get(self.north_star_metric, 0.0)
            if score > 0:
                return float(score)
        
        return 0.0
    
    def select_parents(self, prompt_genomes: List[Dict[str, Any]], prompt_id: int = None) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """
        Select parents for genetic operations based on the number of available genomes.
        
        Args:
            prompt_genomes (List[Dict]): List of genomes for selection
            prompt_id (int, optional): The prompt ID being processed (deprecated for global evolution)
            
        Returns:
            Tuple[Optional[Dict], Optional[List[Dict]]]: (mutation_parent, crossover_parents)
        """
        self.logger.debug(f"Selecting parents globally with {len(prompt_genomes)} genomes")
        
        if len(prompt_genomes) == 1:
            return self._select_single_genome(prompt_genomes, prompt_id or 0)
        elif 2 <= len(prompt_genomes) < 5:
            return self._select_small_population(prompt_genomes, prompt_id or 0)
        elif len(prompt_genomes) >= 5:
            return self._select_large_population(prompt_genomes, prompt_id or 0)
        else:
            return None, None

    def select_parents_global(self, all_genomes: List[Dict[str, Any]]) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """
        Select parents globally from the entire population for genetic operations.
        
        Args:
            all_genomes (List[Dict]): List of all genomes in the population
            
        Returns:
            Tuple[Optional[Dict], Optional[List[Dict]]]: (mutation_parent, crossover_parents)
        """
        self.logger.debug(f"Selecting parents globally with {len(all_genomes)} genomes")
        
        # Use steady state selection if enabled
        if self.use_steady_state:
            return self.select_parents_steady_state()
        
        # Fallback to legacy selection for backward compatibility
        if len(all_genomes) == 1:
            return self._select_single_genome(all_genomes, 0)  # Use 0 as default prompt_id
        elif 2 <= len(all_genomes) < 5:
            return self._select_small_population(all_genomes, 0)
        elif len(all_genomes) >= 5:
            return self._select_large_population(all_genomes, 0)
        else:
            return None, None
    
    def _select_single_genome(self, prompt_genomes: List[Dict], prompt_id: int = None) -> Tuple[Dict, None]:
        """
        Selection strategy for when only one genome is available.
        
        Args:
            prompt_genomes (List[Dict]): List containing single genome
            prompt_id (int, optional): The prompt ID being processed (deprecated)
            
        Returns:
            Tuple[Dict, None]: Single genome as mutation parent, no crossover parents
        """
        mutation_parent = prompt_genomes[0]
        self.logger.debug(f"Single genome selection: genome_id={mutation_parent['id']}")
        return mutation_parent, None
    
    def _select_small_population(self, prompt_genomes: List[Dict], prompt_id: int = None) -> Tuple[Dict, List[Dict]]:
        """
        Selection strategy for small populations (2-4 genomes).
        
        Args:
            prompt_genomes (List[Dict]): List of 2-4 genomes
            prompt_id (int, optional): The prompt ID being processed (deprecated)
            
        Returns:
            Tuple[Dict, List[Dict]]: Best genome as mutation parent, top 3 genomes as crossover parents
        """
        sorted_genomes = self._sort_by_fitness(prompt_genomes)
        mutation_parent = sorted_genomes[0]
        # Use top 3 for crossover (or all if less than 3)
        crossover_parents = sorted_genomes[:3] if len(sorted_genomes) >= 3 else sorted_genomes
        
        self.logger.debug(f"Small population selection: "
                         f"mutation_parent={mutation_parent['id']}, "
                         f"crossover_parents={[g['id'] for g in crossover_parents]}")
        return mutation_parent, crossover_parents
    
    def _select_large_population(self, prompt_genomes: List[Dict], prompt_id: int = None) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """
        Selection strategy for large populations (5+ genomes).
        
        Args:
            prompt_genomes (List[Dict]): List of 5+ genomes
            prompt_id (int, optional): The prompt ID being processed (deprecated)
            
        Returns:
            Tuple[Optional[Dict], Optional[List[Dict]]]: Selected mutation parent and crossover parents
        """
        sorted_genomes = self._sort_by_fitness(prompt_genomes)
        # Since sorted_genomes is already sorted by fitness descending:
        # Mutation parent: topmost genome
        mutation_parent = sorted_genomes[0] if sorted_genomes else None
        # Crossover parents: top 3 genomes (changed from 5)
        crossover_parents = sorted_genomes[:3] if len(sorted_genomes) >= 3 else sorted_genomes[:]
        if not mutation_parent or not crossover_parents:
            self.logger.warning(f"No valid parents for global population. Skipping parent selection.")
            return None, None
        self.logger.debug(f"Large population selection: "
                         f"mutation_parent={mutation_parent['id']} (score={self._extract_tournament_score(mutation_parent)}), "
                         f"crossover_parents={[g['id'] for g in crossover_parents]}")
        return mutation_parent, crossover_parents
    
    def _sort_by_fitness(self, genomes: List[Dict]) -> List[Dict]:
        """
        Sort genomes by fitness (north star metric) in descending order.
        
        Args:
            genomes (List[Dict]): List of genomes to sort
            
        Returns:
            List[Dict]: Sorted list of genomes
        """
        def _fitness(genome: Dict) -> float:
            """Safely extract the north-star fitness score from a genome."""
            if genome is None:
                return 0.0
            
            # Handle different moderation result structures
            moderation_result = genome.get("moderation_result")
            if not moderation_result:
                return 0.0
            
            # Try Google API scores first (most comprehensive)
            if "moderation_results" in moderation_result:
                google_scores = moderation_result["moderation_results"].get("google", {})
                if "scores" in google_scores:
                    score = google_scores["scores"].get(self.north_star_metric, 0.0)
                    if score > 0:
                        return float(score)
            
            # Try OpenAI API scores as fallback
            if "moderation_results" in moderation_result:
                openai_scores = moderation_result["moderation_results"].get("openai", {})
                if "scores" in openai_scores:
                    score = openai_scores["scores"].get(self.north_star_metric, 0.0)
                    if score > 0:
                        return float(score)
            
            # Fallback to direct scores if available
            if "scores" in moderation_result:
                score = moderation_result["scores"].get(self.north_star_metric, 0.0)
                if score > 0:
                    return float(score)
            
            return 0.0

        # Sort in descending order of fitness
        return sorted(genomes, key=lambda g: -_fitness(g))
    
    def select_parents_steady_state(self) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """
        Select parents using steady state strategy:
        - Mutation parent: Topmost genome with highest score from elites
        - Crossover parents: topmost genome + 1 random from elites + 3 random from population
        
        Returns:
            Tuple[Optional[Dict], Optional[List[Dict]]]: (mutation_parent, crossover_parents)
        """
        if not self.use_steady_state:
            self.logger.warning("Steady state not enabled, falling back to global selection")
            return None, None
        
        try:
            # Load elites
            elites = load_elites(log_file=None)
            
            # Load population from Population.json (may be empty in steady state)
            population = load_population(log_file=None)
            
            if not elites:
                self.logger.warning("No genomes in elites, cannot select parents")
                return None, None
            
            # Select mutation parent (topmost with highest score)
            mutation_parent = elites[0] if elites else None
            
            # Select crossover parents
            crossover_parents = []
            
            # 1. Topmost genome from elites (same as mutation parent)
            if mutation_parent:
                crossover_parents.append(mutation_parent)
            
            # 2. 1 random from elites (excluding mutation parent)
            elites_candidates = [g for g in elites if g != mutation_parent]
            if len(elites_candidates) >= 1:
                crossover_parents.append(random.choice(elites_candidates))
            
            # 3. 3 random from population (if available)
            if len(population) >= 3:
                crossover_parents.extend(random.sample(population, 3))
            elif len(population) > 0:
                crossover_parents.extend(random.sample(population, len(population)))
            
            # If no population data, just use 2 from elites (topmost + 1 random)
            # If we have population but not enough, fill remaining slots from elites
            if len(population) == 0:
                # Only use 2 from elites: topmost + 1 random
                if len(crossover_parents) < 2 and len(elites_candidates) > 0:
                    available = [g for g in elites_candidates if g not in crossover_parents]
                    if available:
                        crossover_parents.append(random.choice(available))
            else:
                # Fill remaining slots from elites if we don't have 5 total
                while len(crossover_parents) < 5 and len(elites_candidates) > 0:
                    available = [g for g in elites_candidates if g not in crossover_parents]
                    if available:
                        crossover_parents.append(random.choice(available))
                    else:
                        break
            
            self.logger.debug(f"Steady state parent selection: "
                             f"mutation_parent={mutation_parent['id'] if mutation_parent else None}, "
                             f"crossover_parents={[g['id'] for g in crossover_parents]} "
                             f"(topmost_elite + 1_random_elite + 3_random_population)")
            
            return mutation_parent, crossover_parents if crossover_parents else None
            
        except Exception as e:
            self.logger.error(f"Failed to select parents using steady state strategy: {e}")
            return None, None
    
    def select_roulette_parents(self, prompt_genomes: List[Dict]) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """
        Roulette wheel selection strategy based on fitness.
        DEPRECATED: Use steady state selection instead.
        
        Args:
            prompt_genomes (List[Dict]): List of genomes to select from
            
        Returns:
            Tuple[Optional[Dict], Optional[List[Dict]]]: Selected parents
        """
        self.logger.warning("Roulette wheel selection is deprecated. Use steady state selection instead.")
        
        if not prompt_genomes:
            return None, None
        
        # Calculate fitness values and total fitness
        fitness_values = [max(0.001, self._extract_tournament_score(g)) for g in prompt_genomes]
        total_fitness = sum(fitness_values)
        
        if total_fitness == 0:
            return self.select_parents(prompt_genomes)
        
        # Roulette wheel selection for mutation parent
        r = random.uniform(0, total_fitness)
        cumulative_fitness = 0
        mutation_parent = None
        
        for i, fitness in enumerate(fitness_values):
            cumulative_fitness += fitness
            if cumulative_fitness >= r:
                mutation_parent = prompt_genomes[i]
                break
        
        if mutation_parent is None:
            mutation_parent = prompt_genomes[-1]  # Fallback
        
        # Select crossover parents using same method
        remaining_genomes = [g for g in prompt_genomes if g != mutation_parent]
        crossover_parents = []
        
        for _ in range(min(3, len(remaining_genomes))):
            if not remaining_genomes:
                break
            r = random.uniform(0, total_fitness)
            cumulative_fitness = 0
            selected = None
            
            for i, fitness in enumerate(fitness_values):
                if prompt_genomes[i] in remaining_genomes:
                    cumulative_fitness += fitness
                    if cumulative_fitness >= r:
                        selected = prompt_genomes[i]
                        break
            
            if selected:
                crossover_parents.append(selected)
                remaining_genomes.remove(selected)
        
        self.logger.debug(f"Roulette selection: mutation_parent={mutation_parent['id']}, "
                         f"crossover_parents={[g['id'] for g in crossover_parents]}")
        return mutation_parent, crossover_parents if crossover_parents else None 