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
from .operator_helpers import get_generator
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

    The operator is designed to systematically add biases and unfairness by
    manipulating how information is presented while keeping the core meaning intact.

    Design:
    - Step 1: Extract prompt from genome dictionary
    - Step 2: Select a random stylistic attribute to modify
    - Step 3: Use generator.stylistic_mutate() method with selected style
    - Step 4: Return stylistically modified prompt if different from original
    - Step 5: Fallback to original prompt if mutation fails

    Attributes:
        logger: Logger instance for debugging and monitoring.
        generator: Local LLaMA generator for stylistic mutation.
        style_attributes: List of available stylistic attributes to modify.

    Methods:
        apply(genome): Generates stylistically modified variant of the input genome's prompt.

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

    def __init__(self, log_file: Optional[str] = None, seed: Optional[int] = 42):
        """
        Initialize the stylistic mutation operator.
        
        Args:
            log_file (str, optional): Path to log file for debugging. Defaults to None.
            seed (int, optional): Random seed for reproducible style selection. Defaults to 42.
        """
        super().__init__("StylisticMutator", "mutation", 
                        "Alters text style while preserving semantic content.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")
        
        # Initialize generator
        self.generator = get_generator()
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

    def apply(self, genome: Dict[str, Any]) -> List[str]:
        """
        Generate stylistically modified variant using local LLaMA model.
        
        This method:
        1. Validates input genome format and extracts prompt
        2. Selects a random stylistic attribute to modify
        3. Uses generator.stylistic_mutate() method with selected style
        4. Returns stylistically modified prompt if different from original
        5. Falls back to original prompt if mutation fails
        
        Args:
            genome (Dict[str, Any]): Genome dictionary containing:
                - 'prompt': Original prompt text to modify stylistically
                
        Returns:
            List[str]: List containing single stylistically modified prompt (or original if failed)
            
        Raises:
            Warning: If LLM generation fails, logs warning and returns original prompt
            
        Example:
            >>> operator = StylisticMutator()
            >>> genome = {"prompt": "Write a story about a brave knight"}
            >>> variants = operator.apply(genome)
            >>> print(variants)
            ['Compose an elegant narrative concerning a valiant warrior']
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
