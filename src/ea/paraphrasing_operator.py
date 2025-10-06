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
from .operator_helpers import get_generator
from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()


class LLMBasedParaphrasingOperator(VariationOperator):
    """
    Paraphrasing operator using local LLaMA model's paraphrase method.

    This operator generates paraphrased versions of input prompts by leveraging
    the local LLaMA model's built-in paraphrase method. The paraphrasing process 
    is guided by a specified optimization metric (north_star_metric) to ensure 
    the generated variants align with desired objectives.

    Design:
    - Step 1: Extract prompt from genome dictionary
    - Step 2: Use generator.paraphrase() method with north_star_metric
    - Step 3: Return paraphrased prompt if different from original
    - Step 4: Fallback to original prompt if paraphrasing fails

    Attributes:
        north_star_metric (str): The optimization metric guiding the paraphrasing process.
        logger: Logger instance for debugging and monitoring.
        generator: Local LLaMA generator for paraphrasing generation.

    Methods:
        apply(genome): Generates paraphrased variant of the input genome's prompt.

    Example:
        >>> operator = LLMBasedParaphrasingOperator(north_star_metric="engagement")
        >>> genome = {"prompt": "Write a story", "generated_text": "...", "moderation_result": {"scores": {"engagement": 0.8}}}
        >>> variants = operator.apply(genome)
        >>> print(variants)
        ['Craft an engaging narrative tale']
    """

    def __init__(self, north_star_metric: str, log_file: Optional[str] = None):
        """
        Initialize the LLM-based paraphrasing operator.
        
        Args:
            north_star_metric (str): The optimization metric for paraphrasing direction
            log_file (str, optional): Path to log file for debugging. Defaults to None.
        """
        super().__init__("LLMBasedParaphrasing", "mutation", 
                        f"Uses local LLaMA model paraphrase method with {north_star_metric} optimization.")
        self.north_star_metric = north_star_metric
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name} with north_star_metric: {self.north_star_metric}")
        
        # Initialize generator
        self.generator = get_generator()
        self.logger.info(f"{self.name}: Local LLaMA generator initialized successfully")
        
        # Debug tracking attributes
        self._last_genome = {}
        self._last_original_prompt = ""
        self._last_paraphrased_prompt = ""

    def apply(self, genome: Dict[str, Any]) -> List[str]:
        """
        Generate paraphrased variant using local LLaMA model's paraphrase method.
        
        This method:
        1. Validates input genome format and extracts prompt
        2. Uses generator.paraphrase() method with north_star_metric
        3. Returns paraphrased prompt if different from original
        4. Falls back to original prompt if paraphrasing fails
        
        Args:
            genome (Dict[str, Any]): Genome dictionary containing:
                - 'prompt': Original prompt text to paraphrase
                - 'generated_text': Generated output from the prompt (optional)
                - 'moderation_result': Dictionary with 'scores' containing north_star_metric (optional)
                
        Returns:
            List[str]: List containing single paraphrased prompt (or original if failed)
            
        Raises:
            Warning: If LLM generation fails, logs warning and returns original prompt
            
        Example:
            >>> operator = LLMBasedParaphrasingOperator("engagement")
            >>> genome = {"prompt": "Write a story", "generated_text": "...", "moderation_result": {"scores": {"engagement": 0.8}}}
            >>> variants = operator.apply(genome)
            >>> print(variants)
            ['Craft an engaging narrative tale']
        """
        try:
            # Validate input format
            if not isinstance(genome, dict):
                self.logger.error(f"{self.name}: Input must be a genome dictionary")
                return []
            
            # Extract prompt
            original_prompt = genome.get("prompt", "")
            
            if not original_prompt:
                self.logger.error(f"{self.name}: Genome missing required 'prompt' field")
                return []
            
            # Store debug information
            self._last_genome = genome
            self._last_original_prompt = original_prompt
            
            # Extract optional fields for enhanced paraphrasing
            generated_output = genome.get("generated_text", "")
            current_score = 0.0
            
            # Extract north star metric score if available
            moderation_result = genome.get("moderation_result", {})
            if isinstance(moderation_result, dict):
                scores = moderation_result.get("scores", {})
                if isinstance(scores, dict):
                    current_score = scores.get(self.north_star_metric, 0.0)

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