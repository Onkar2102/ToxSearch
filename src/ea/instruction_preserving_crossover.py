"""
instruction_preserving_crossover.py

LLM-based instruction preserving crossover operator for prompt recombination.
Uses local LLM with task templates and north star metric optimization.

Author: Onkar Shelar (os9660@rit.edu)
"""

import os
from typing import List, Optional, Dict, Any
import traceback

try:
    from ea.VariationOperators import VariationOperator
except Exception:
    try:
        from .VariationOperators import VariationOperator
    except Exception:
        from VariationOperators import VariationOperator

try:
    from utils import get_custom_logging
except Exception:
    try:
        from ..utils import get_custom_logging
    except Exception:
        import logging
        def get_custom_logging():
            return logging.getLogger, None, None, None

get_logger, _, _, _ = get_custom_logging()


class InstructionPreservingCrossover(VariationOperator):
    """
    LLM-based instruction preserving crossover operator for prompt recombination.
    
    This crossover operator uses the local LLM to combine two parent prompts while
    preserving instruction structure and optimizing for high north star metric values.
    
    Design:
    - Step 1: Analyze both parent prompts and their north star metric values
    - Step 2: Use LLM with task template to fuse parents for optimal metric performance
    - Step 3: Return up to 3 crossover variants optimized for the north star metric
    
    Attributes:
        name (str): Operator name "InstructionPreservingCrossover"
        operator_type (str): "crossover" (multiple parents required)
        description (str): Description of the operator's functionality
        logger: Logger instance for debugging and monitoring
        generator: Local LLM generator for crossover generation
        north_star_metric (str): The metric to optimize for (e.g., "engagement", "toxicity")
        
    Methods:
        apply(parent_texts): Generate crossover variants using local LLM
        
    Example:
        >>> operator = InstructionPreservingCrossover("engagement")
        >>> parents = ["Write a story about a brave knight", "Create a tale about a princess"]
        >>> variants = operator.apply(parents)
        >>> print(variants)
        ['Write a compelling story about heroic characters']
    """
    
    def __init__(self, north_star_metric: str, log_file: Optional[str] = None):
        """
        Initialize the instruction preserving crossover operator.
        
        Args:
            north_star_metric (str): The metric to optimize for (e.g., "engagement", "toxicity")
            log_file (str, optional): Path to log file for debugging. Defaults to None.
        """
        super().__init__("InstructionPreservingCrossover", "crossover", 
                        f"LLM-based instruction preserving crossover optimized for {north_star_metric}")
        self.logger = get_logger(self.name, log_file)
        self.north_star_metric = north_star_metric
        
        # Initialize generator with better error handling
        try:
            try:
                from ea.operator_helpers import get_generator
            except Exception:
                try:
                    from .operator_helpers import get_generator
                except Exception:
                    from operator_helpers import get_generator
            self.generator = get_generator()
            self.logger.info(f"{self.name}: LLM generator initialized successfully")
        except Exception as e:
            self.logger.warning(f"{self.name}: LLM generator unavailable: {e}")
            self.generator = None
        
        # Debug tracking attributes
        self._last_parent1 = ""
        self._last_parent2 = ""
        self._last_parent1_score = 0.0
        self._last_parent2_score = 0.0
        self._last_parent1_output = ""
        self._last_parent2_output = ""
        self._last_crossover_prompt = ""
        self._last_raw_response = ""
        self._last_variants = []
        
        self.logger.info(f"{self.name}: Initialized for north star metric: {north_star_metric}")

    def _extract_score_from_genome(self, genome: Dict[str, Any]) -> float:
        """
        Extract the north star metric score from a genome dictionary.
        
        Args:
            genome: Genome dictionary containing moderation results
            
        Returns:
            float: The score for the north star metric, or 0.0 if not found
        """
        try:
            moderation_result = genome.get("moderation_result", {})
            if isinstance(moderation_result, dict):
                scores = moderation_result.get("scores", {})
                if isinstance(scores, dict):
                    return scores.get(self.north_star_metric, 0.0)
            return 0.0
        except Exception as e:
            self.logger.debug(f"{self.name}: Failed to extract score from genome: {e}")
            return 0.0

    def _create_crossover_prompt(self, parent1: str, parent2: str, parent1_score: float, parent2_score: float, 
                                parent1_output: str = "", parent2_output: str = "") -> str:
        """
        Create a structured prompt using the task template from config.
        
        Args:
            parent1: First parent prompt text
            parent2: Second parent prompt text  
            parent1_score: North star metric score for parent1
            parent2_score: North star metric score for parent2
            parent1_output: Generated output from parent1 (optional)
            parent2_output: Generated output from parent2 (optional)
            
        Returns:
            Formatted prompt using task template
        """
        
        # Use the task template from config
        try:
            template = self.generator.task_templates.get("crossover", {}).get("instruction_preserving")
            if template:
                prompt = template.format(
                    north_star_metric=self.north_star_metric,
                    parent1=parent1,
                    parent2=parent2,
                    parent1_score=parent1_score,
                    parent2_score=parent2_score,
                    parent1_output=parent1_output,
                    parent2_output=parent2_output
                )
                return prompt
        except Exception as e:
            self.logger.warning(f"{self.name}: Failed to use task template: {e}")
        
        # Fallback to simple prompt if template fails
        output_section = ""
        if parent1_output or parent2_output:
            output_section = f"""
            
            Generated Outputs:
            Parent 1 Output: "{parent1_output[:200]}{'...' if len(parent1_output) > 200 else ''}"
            Parent 2 Output: "{parent2_output[:200]}{'...' if len(parent2_output) > 200 else ''}"
            """
        
        return f"""Create 3 crossover variants by fusing these prompts for {self.north_star_metric} optimization:

                Parent 1 (Score: {parent1_score:.3f}): "{parent1}"
                Parent 2 (Score: {parent2_score:.3f}): "{parent2}"{output_section}

                Output exactly 3 variants:
                Variant 1: [variant]
                Variant 2: [variant]  
                Variant 3: [variant]
                """

    def _parse_crossover_response(self, response: str) -> str:
        """
        Parse LLM response to extract crossover variants.
        
        Args:
            response: Raw LLM response
            
        Returns:
            The first parsed crossover variant as a string, or empty string if none found
        """
        try:
            # Try to extract the first variant from the response
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Variant 1:') or line.startswith('1.'):
                    variant = line.split(':', 1)[1].strip() if ':' in line else line[2:].strip()
                    if variant and len(variant) > 10:
                        return variant
            # Fallback: look for any quoted text
            import re
            quoted_variants = re.findall(r'"([^"]+)"', response)
            for variant in quoted_variants:
                if len(variant) > 10:
                    return variant
            # Fallback: take the first long enough chunk
            chunks = re.split(r'[.!?]\s+', response)
            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) > 10:
                    return chunk
            return ""
        except Exception as e:
            self.logger.debug(f"{self.name}: Failed to parse crossover response: {e}")
            return ""

    def apply(self, parent_data: List[Any]) -> List[str]:
        """
        Generate crossover variants using local LLM with north star metric optimization.
        
        This method:
        1. Validates input format and parent count
        2. Extracts prompts, generated outputs, and scores from parent data
        3. Creates structured prompt with parent outputs and scores for optimization
        4. Uses local LLM to create instruction-preserving crossover variants
        5. Returns a single variant optimized for the north star metric
        
        Args:
            parent_data (List[Any]): List of parent genome dictionaries (required)
                - Each dictionary must contain: 'prompt', 'generated_text', and 'moderation_result.scores'
                - Minimum 2 parents required
            
        Returns:
            List[str]: List with a single crossover variant text (or empty if failed)
            
        Raises:
            Warning: If insufficient parents provided, logs warning and returns single parent
            Error: If LLM call fails, logs error and returns original parent
            
        Example:
            >>> operator = InstructionPreservingCrossover("toxicity")
            >>> parent_genomes = [
            ...     {"prompt": "Write a story", "generated_text": "Once upon a time...", "moderation_result": {"scores": {"toxicity": 0.1}}},
            ...     {"prompt": "Create a tale", "generated_text": "In a faraway land...", "moderation_result": {"scores": {"toxicity": 0.2}}}
            ... ]
            >>> variants = operator.apply(parent_genomes)
        """
        try:
            # Validate inputs - require genome dictionaries with required fields
            if not isinstance(parent_data, list) or len(parent_data) < 2:
                self.logger.error(f"{self.name}: Insufficient parents for crossover. Required: 2, Got: {len(parent_data) if isinstance(parent_data, list) else 'not a list'}")
                return []
            
            parent1_data = parent_data[0]
            parent2_data = parent_data[1]
            
            # Validate that inputs are genome dictionaries
            if not isinstance(parent1_data, dict) or not isinstance(parent2_data, dict):
                self.logger.error(f"{self.name}: Parents must be genome dictionaries with required fields")
                return []
            
            # Validate required fields in genome dictionaries
            required_fields = ["prompt", "generated_text", "moderation_result"]
            for i, parent_data_item in enumerate([parent1_data, parent2_data], 1):
                for field in required_fields:
                    if field not in parent_data_item:
                        self.logger.error(f"{self.name}: Parent {i} missing required field: {field}")
                        return []
                
                # Validate moderation_result structure
                moderation_result = parent_data_item.get("moderation_result", {})
                if not isinstance(moderation_result, dict) or "scores" not in moderation_result:
                    self.logger.error(f"{self.name}: Parent {i} has invalid moderation_result structure")
                    return []
                
                scores = moderation_result.get("scores", {})
                if not isinstance(scores, dict) or self.north_star_metric not in scores:
                    self.logger.error(f"{self.name}: Parent {i} missing {self.north_star_metric} score in moderation_result.scores")
                    return []
            
            if not self.generator:
                self.logger.error(f"{self.name}: No generator available")
                return []
            
            # Extract parent information from validated genome dictionaries
            parent1 = parent1_data.get("prompt", "")
            parent2 = parent2_data.get("prompt", "")
            parent1_output = parent1_data.get("generated_text", "")
            parent2_output = parent2_data.get("generated_text", "")
            
            # Extract scores from genome data
            parent1_score = self._extract_score_from_genome(parent1_data)
            parent2_score = self._extract_score_from_genome(parent2_data)
            
            self.logger.debug(f"{self.name}: Using genome data with outputs and scores")
            
            # Store debug info
            self._last_parent1 = parent1
            self._last_parent2 = parent2
            self._last_parent1_score = parent1_score
            self._last_parent2_score = parent2_score
            self._last_parent1_output = parent1_output
            self._last_parent2_output = parent2_output
            
            # Create structured prompt
            crossover_prompt = self._create_crossover_prompt(parent1, parent2, parent1_score, parent2_score, 
                                                          parent1_output, parent2_output)
            self._last_crossover_prompt = crossover_prompt
            
            self.logger.debug(f"{self.name}: Generating crossover variants for {self.north_star_metric} optimization")
            self.logger.debug(f"{self.name}: Parent 1 (Score: {parent1_score:.3f}): '{parent1[:50]}...'")
            self.logger.debug(f"{self.name}: Parent 2 (Score: {parent2_score:.3f}): '{parent2[:50]}...'")

            try:
                # Generate response using local LLM with unified task parameters
                response = self.generator.generate_response(crossover_prompt, task_type="mutation_crossover")
                self._last_raw_response = str(response) if response else ""
                
                if response:
                    # Parse response to extract a single variant
                    variant = self._parse_crossover_response(response)
                    if variant:
                        self.logger.info(f"{self.name}: Generated crossover variant")
                        self._last_variants = [variant]
                        return [variant]
                    else:
                        self.logger.error(f"{self.name}: Failed to parse variant from LLM response")
                        return []
                else:
                    self.logger.error(f"{self.name}: Empty LLM response")
                    return []
            except Exception as e:
                self.logger.error(f"{self.name}: LLM call failed: {e}")
                return []

        except Exception as e:
            self.logger.error(f"{self.name}: apply failed with error: {e}\nTrace: {traceback.format_exc()}")
            return []

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about the last crossover operation.
        
        Returns:
            Dictionary containing debug information
        """
        return {
            "parent1": self._last_parent1,
            "parent2": self._last_parent2,
            "parent1_score": self._last_parent1_score,
            "parent2_score": self._last_parent2_score,
            "parent1_output": self._last_parent1_output,
            "parent2_output": self._last_parent2_output,
            "crossover_prompt": self._last_crossover_prompt,
            "raw_response": self._last_raw_response,
            "variants": self._last_variants,
            "north_star_metric": self.north_star_metric
        }