"""
stylistic_mutator.py

This module contains the stylistic mutation operator for the evolutionary algorithm.
This mutation operator alters the style of text while preserving core semantic content,
modifying attributes such as formality, politeness, sentiment, and arbitrary styles.

Version: 1.0
"""

import os
import traceback
import random
from typing import List, Optional, Dict, Any
from .VariationOperators import VariationOperator
from .EvolutionEngine import EvolutionEngine
from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()


class StylisticMutator(VariationOperator):
    """
    Stylistic mutation operator that alters text style while preserving semantic content.

    This operator systematically modifies stylistic attributes of text including:
    - Formality level (formal/informal)
    - Politeness (polite/impolite/neutral)
    - Sentiment (positive/negative/neutral)
    - Tone (authoritative/casual/academic)
    - Voice (active/passive)
    - Arbitrary styles (poetic, technical, conversational, etc.)

    The operator creates stylistic variations by modifying presentation style
    while preserving the core semantic meaning of the text.

    Process:
    1. Receive input text string for stylistic modification
    2. Randomly select a stylistic attribute to modify
    3. Use LLaMA model to generate stylistically modified version
    4. Return list of stylistically varied text versions
    5. Fallback to original text if mutation fails

    Attributes:
        logger: Logger instance for debugging and monitoring
        generator: Local LLaMA generator for stylistic mutation
        style_attributes: List of available stylistic attributes to modify

    Methods:
        apply(text): Generates stylistically modified variants of input text string

    Example:
        >>> operator = StylisticMutator()
        >>> genome = {"prompt": "Write a story about a brave knight"}
        >>> variants = operator.apply(genome)
        >>> print(variants)
        ['Compose an elegant narrative concerning a valiant warrior']
    """

    # Available stylistic attributes for mutation
    STYLE_ATTRIBUTES = [
        "formality",      # formal/informal
        "politeness",     # polite/impolite/neutral
        "sentiment",      # positive/negative/neutral
        "tone",          # authoritative/casual/academic
        "voice",         # active/passive
        "complexity",    # simple/complex
        "poetic",        # poetic/plain
        "technical",     # technical/layman
        "conversational", # conversational/formal
        "emphatic",      # emphatic/subtle
        "concise",       # concise/verbose
        "persuasive"     # persuasive/neutral
    ]

    def __init__(self, log_file: Optional[str] = None, seed: Optional[int] = 42, generator=None):
        """
        Initialize the stylistic mutation operator.
        
        Args:
            log_file (str, optional): Path to log file for debugging. Defaults to None.
            seed (int, optional): Random seed for reproducible style selection. Defaults to 42.
            generator: LLaMA generator instance to use
        """
        super().__init__("StylisticMutator", "mutation", 
                        "Alters text style while preserving semantic content.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")
        
        # Use provided generator
        self.generator = generator
        self.logger.info(f"{self.name}: Local LLaMA generator initialized successfully")
        # Initialize random number generator for style selection
        self.rng = random.Random(seed)
        # Debug tracking attributes
        self._last_genome = {}
        self._last_original_prompt = ""
        self._last_selected_style = ""
        self._last_stylistic_prompt = ""

    def _select_random_style(self) -> str:
        """
        Select a random stylistic attribute to modify.
        
        Returns:
            str: Selected stylistic attribute
        """
        selected_style = self.rng.choice(self.STYLE_ATTRIBUTES)
        self.logger.debug(f"{self.name}: Selected style attribute: {selected_style}")
        return selected_style

    def apply(self, operator_input: Dict[str, Any]) -> List[str]:
        """
        Generate stylistically modified variant using local LLaMA model.
        
        This method:
        1. Validates input format and extracts parent data
        2. Selects a random stylistic attribute to modify
        3. Uses generator.stylistic_mutate() method with selected style
        4. Returns stylistically modified prompt if different from original
        5. Falls back to original prompt if mutation fails
        
        Args:
            operator_input (Dict[str, Any]): Operator input containing:
                - 'parent_data': Enriched parent genome dictionary containing:
                    - 'prompt': Original prompt text to modify stylistically
                    - 'generated_text': Generated output from the prompt (optional)
                    - 'scores': Moderation scores dictionary
                    - 'north_star_score': Primary optimization metric score
                - 'max_variants': Maximum number of variants to generate
                
        Returns:
            List[str]: List containing stylistically modified prompt variants (or original if failed)
            
        Raises:
            Warning: If LLM generation fails, logs warning and returns original prompt
            
        Example:
            >>> operator = StylisticMutator()
            >>> input_data = {
            ...     "parent_data": {"prompt": "Write a story about a brave knight"},
            ...     "max_variants": 3
            ... }
            >>> variants = operator.apply(input_data)
            >>> print(variants)
            ['Compose an elegant narrative concerning a valiant warrior']
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
            
            # Select random stylistic attribute
            selected_style = self._select_random_style()
            self._last_selected_style = selected_style
            
            # Use generator's stylistic_mutate method
            stylistic_prompt = self.generator.stylistic_mutate(
                original_prompt, 
                selected_style
            )
            
            self._last_stylistic_prompt = stylistic_prompt
            
            if stylistic_prompt and stylistic_prompt.lower() != original_prompt.lower():
                self.logger.info(f"{self.name}: Generated stylistic variant with {selected_style} style")
                return [stylistic_prompt]
            else:
                self.logger.warning(f"{self.name}: Stylistic mutation returned same or empty text, returning original")
                return [original_prompt]
                
        except Exception as e:
            self.logger.error(f"{self.name}: apply failed with error: {e}\nTrace: {traceback.format_exc()}")
            return []

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about the last operation.
        
        Returns:
            Dict containing debug information about the last stylistic mutation operation
        """
        return {
            "genome": self._last_genome,
            "original_prompt": self._last_original_prompt,
            "selected_style": self._last_selected_style,
            "stylistic_prompt": self._last_stylistic_prompt,
            "available_styles": self.STYLE_ATTRIBUTES
        }

    def get_available_styles(self) -> List[str]:
        """
        Get list of available stylistic attributes.
        
        Returns:
            List[str]: Available stylistic attributes for mutation
        """
        return self.STYLE_ATTRIBUTES.copy()
