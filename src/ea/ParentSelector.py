import random
from typing import List, Dict, Any, Optional, Tuple
from utils.custom_logging import get_logger

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
        """
        self.north_star_metric = north_star_metric
        self.logger = get_logger("ParentSelector", log_file)
        self.logger.debug(f"ParentSelector initialized with north_star_metric={north_star_metric}")
    
    def select_parents(self, prompt_genomes: List[Dict[str, Any]], prompt_id: int) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """
        Select parents for genetic operations based on the number of available genomes.
        
        Args:
            prompt_genomes (List[Dict]): List of genomes for the specific prompt_id
            prompt_id (int): The prompt ID being processed
            
        Returns:
            Tuple[Optional[Dict], Optional[List[Dict]]]: (mutation_parent, crossover_parents)
        """
        self.logger.debug(f"Selecting parents for prompt_id={prompt_id} with {len(prompt_genomes)} genomes")
        
        if len(prompt_genomes) == 1:
            return self._select_single_genome(prompt_genomes, prompt_id)
        elif 2 <= len(prompt_genomes) < 5:
            return self._select_small_population(prompt_genomes, prompt_id)
        elif len(prompt_genomes) >= 5:
            return self._select_large_population(prompt_genomes, prompt_id)
        else:
            return None, None
    
    def _select_single_genome(self, prompt_genomes: List[Dict], prompt_id: int) -> Tuple[Dict, None]:
        """
        Selection strategy for when only one genome is available.
        
        Args:
            prompt_genomes (List[Dict]): List containing single genome
            prompt_id (int): The prompt ID being processed
            
        Returns:
            Tuple[Dict, None]: Single genome as mutation parent, no crossover parents
        """
        mutation_parent = prompt_genomes[0]
        self.logger.debug(f"Single genome selection for prompt_id={prompt_id}: genome_id={mutation_parent['id']}")
        return mutation_parent, None
    
    def _select_small_population(self, prompt_genomes: List[Dict], prompt_id: int) -> Tuple[Dict, List[Dict]]:
        """
        Selection strategy for small populations (2-4 genomes).
        
        Args:
            prompt_genomes (List[Dict]): List of 2-4 genomes
            prompt_id (int): The prompt ID being processed
            
        Returns:
            Tuple[Dict, List[Dict]]: Best genome as mutation parent, all genomes as crossover parents
        """
        sorted_genomes = self._sort_by_fitness(prompt_genomes)
        mutation_parent = sorted_genomes[0]
        crossover_parents = sorted_genomes
        
        self.logger.debug(f"Small population selection for prompt_id={prompt_id}: "
                         f"mutation_parent={mutation_parent['id']}, "
                         f"crossover_parents={[g['id'] for g in crossover_parents]}")
        return mutation_parent, crossover_parents
    
    def _select_large_population(self, prompt_genomes: List[Dict], prompt_id: int) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """
        Selection strategy for large populations (5+ genomes).
        
        Args:
            prompt_genomes (List[Dict]): List of 5+ genomes
            prompt_id (int): The prompt ID being processed
            
        Returns:
            Tuple[Optional[Dict], Optional[List[Dict]]]: Selected mutation parent and crossover parents
        """
        sorted_genomes = self._sort_by_fitness(prompt_genomes)
        # Since sorted_genomes is already sorted by fitness descending:
        # Mutation parent: topmost genome
        mutation_parent = sorted_genomes[0] if sorted_genomes else None
        # Crossover parents: top 5 genomes
        crossover_parents = sorted_genomes[:5] if len(sorted_genomes) >= 5 else sorted_genomes[:]
        if not mutation_parent or not crossover_parents:
            self.logger.warning(f"No valid parents for prompt_id={prompt_id}. Skipping parent selection.")
            return None, None
        self.logger.debug(f"Large population selection for prompt_id={prompt_id}: "
                         f"mutation_parent={mutation_parent['id']} (score={mutation_parent.get(self.north_star_metric, 0.0)}), "
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
            return float(
                genome.get("moderation_result", {})
                      .get("scores", {})
                      .get(self.north_star_metric, 0.0)
            )

        # Sort in descending order of fitness
        return sorted(genomes, key=lambda g: -_fitness(g))
    
    def select_tournament_parents(self, prompt_genomes: List[Dict], tournament_size: int = 3) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """
        Tournament selection strategy.
        
        Args:
            prompt_genomes (List[Dict]): List of genomes to select from
            tournament_size (int): Size of tournament for selection
            
        Returns:
            Tuple[Optional[Dict], Optional[List[Dict]]]: Selected parents
        """
        if len(prompt_genomes) < tournament_size:
            return self.select_parents(prompt_genomes, prompt_genomes[0]["prompt_id"])
        
        # Tournament selection for mutation parent
        tournament = random.sample(prompt_genomes, tournament_size)
        mutation_parent = max(tournament, key=lambda g: g.get(self.north_star_metric, 0.0))
        
        # Select crossover parents from remaining genomes
        remaining_genomes = [g for g in prompt_genomes if g != mutation_parent]
        crossover_parents = random.sample(remaining_genomes, min(3, len(remaining_genomes))) if remaining_genomes else None
        
        self.logger.debug(f"Tournament selection: mutation_parent={mutation_parent['id']}, "
                         f"crossover_parents={[g['id'] for g in crossover_parents] if crossover_parents else None}")
        return mutation_parent, crossover_parents
    
    def select_roulette_parents(self, prompt_genomes: List[Dict]) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """
        Roulette wheel selection strategy based on fitness.
        
        Args:
            prompt_genomes (List[Dict]): List of genomes to select from
            
        Returns:
            Tuple[Optional[Dict], Optional[List[Dict]]]: Selected parents
        """
        if not prompt_genomes:
            return None, None
        
        # Calculate fitness values and total fitness
        fitness_values = [max(0.001, g.get(self.north_star_metric, 0.0)) for g in prompt_genomes]
        total_fitness = sum(fitness_values)
        
        if total_fitness == 0:
            return self.select_parents(prompt_genomes, prompt_genomes[0]["prompt_id"])
        
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