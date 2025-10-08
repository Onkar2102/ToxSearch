"""
paraphrasing_operator.py

This module contains the LLM-based paraphrasing operator for the evolutionary algorithm.
This mutation operator generates paraphrased variants using the local LLaMA model's paraphrase method.

Version: 3.0
"""

import os
import traceback
from typing import List, Optional, Dict, Any
from .VariationOperators import VariationOperator
from .EvolutionEngine import EvolutionEngine
from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()


class LLMBasedParaphrasingOperator(VariationOperator):
    """
    Paraphrasing operator using local LLaMA model for text mutation.

    This operator generates paraphrased versions of input text by leveraging
    the local LLaMA model's paraphrasing capabilities. The paraphrasing process 
    is guided by a specified optimization metric (north_star_metric) to ensure 
    the generated variants align with desired objectives.

    Process:
    1. Receive input text string for paraphrasing
    2. Use LLaMA model's paraphrase method with metric-specific prompts
    3. Generate semantically equivalent but stylistically different variants
        def __init__(self, north_star_metric: str, log_file: Optional[str] = None, generator=None):

    Attributes:
        north_star_metric (str): The optimization metric guiding paraphrasing
        logger: Logger instance for debugging and monitoring
        generator: Local LLaMA generator for text generation

    Methods:
        apply(text): Generates paraphrased variants of the input text string

    Example:
        >>> operator = LLMBasedParaphrasingOperator(north_star_metric="engagement")
        >>> text = "Write a story about a brave knight"
        >>> variants = operator.apply(text)
        >>> print(variants)
            # Use provided generator
            if generator is not None:
                self.generator = generator
            else:
                from .operator_helpers import get_generator
                self.generator = get_generator()
    """

    def __init__(self, north_star_metric: str, log_file: Optional[str] = None, generator=None):
        """
        Initialize the LLM-based paraphrasing operator.
        
        Args:
            north_star_metric (str): The optimization metric for paraphrasing direction
            log_file (str, optional): Path to log file for debugging. Defaults to None.
            generator: LLaMA generator instance to use. If None, will create own instance.
        """
        super().__init__("LLMBasedParaphrasing", "mutation", 
                        f"Uses local LLaMA model paraphrase method with {north_star_metric} optimization.")
        self.north_star_metric = north_star_metric
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name} with north_star_metric: {self.north_star_metric}")
        
        # Initialize generator - use provided or get global one
        if generator is not None:
            self.generator = generator
            self.logger.info(f"{self.name}: Using provided LLM generator")
        else:
            from .operator_helpers import get_generator
            self.generator = get_generator()
            self.logger.info(f"{self.name}: Using global LLM generator")
        
        # Debug tracking attributes
        self._last_genome = {}
        self._last_original_prompt = ""
        self._last_paraphrased_prompt = ""

    def apply(self, operator_input: Dict[str, Any]) -> List[str]:
        """
        Generate paraphrased variant using local LLaMA model's paraphrase method.
        
        This method:
        1. Validates input format and extracts parent data
        2. Uses generator.paraphrase() method with north_star_metric
        3. Returns paraphrased prompt if different from original
        4. Falls back to original prompt if paraphrasing fails
        
        Args:
            operator_input (Dict[str, Any]): Operator input containing:
                - 'parent_data': Enriched parent genome dictionary containing:
                    - 'prompt': Original prompt text to paraphrase
                    - 'generated_text': Generated output from the prompt (optional)
                    - 'scores': Moderation scores dictionary
                    - 'north_star_score': Primary optimization metric score
                - 'max_variants': Maximum number of variants to generate
                
        Returns:
            List[str]: List containing paraphrased prompt variants (or original if failed)
            
        Raises:
            Warning: If LLM generation fails, logs warning and returns original prompt
            
        Example:
            >>> operator = LLMBasedParaphrasingOperator("engagement")
            >>> input_data = {
            ...     "parent_data": {"prompt": "Write a story", "generated_text": "...", "scores": {"engagement": 0.8}},
            ...     "max_variants": 5
            ... }
            >>> variants = operator.apply(input_data)
            >>> print(variants)
            ['Craft an engaging narrative tale']
        """
        try:
            # Validate input format
            if not isinstance(operator_input, dict):
                self.logger.error(f"{self.name}: Input must be a dictionary")
                return []
            
            # Extract parent data and max_variants
            parent_data = operator_input.get("parent_data", {})
            max_variants = operator_input.get("max_variants", 1)
            
            if not isinstance(parent_data, dict):
                self.logger.error(f"{self.name}: parent_data must be a dictionary")
                return []
            
            # Extract prompt from parent data
            original_prompt = parent_data.get("prompt", "")
            
            if not original_prompt:
                self.logger.error(f"{self.name}: Parent data missing required 'prompt' field")
                return []
            
            # Store debug information
            self._last_genome = parent_data
            self._last_original_prompt = original_prompt
            
            # Extract optional fields for enhanced paraphrasing
            generated_output = parent_data.get("generated_text", "")
            current_score = parent_data.get("north_star_score", 0.0)
            
            # Use generator's paraphrase method
            paraphrased_prompt = self.generator.paraphrase(
                original_prompt, 
                self.north_star_metric, 
                generated_output, 
                current_score
            )
            
            self._last_paraphrased_prompt = paraphrased_prompt
            
            if paraphrased_prompt and paraphrased_prompt.lower() != original_prompt.lower():
                self.logger.info(f"{self.name}: Generated paraphrased prompt")
                return [paraphrased_prompt]
            else:
                self.logger.warning(f"{self.name}: Paraphrasing returned same or empty text, returning original")
                return [original_prompt]
                
        except Exception as e:
            self.logger.error(f"{self.name}: apply failed with error: {e}\nTrace: {traceback.format_exc()}")
            return []

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about the last operation.
        
        Returns:
            Dict containing debug information about the last paraphrasing operation
        """
        return {
            "genome": self._last_genome,
            "original_prompt": self._last_original_prompt,
            "paraphrased_prompt": self._last_paraphrased_prompt,
            "north_star_metric": self.north_star_metric
        }