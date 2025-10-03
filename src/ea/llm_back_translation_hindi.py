"""
llm_back_translation_hindi.py

Author: Onkar Shelar os9660@rit.edu

This module contains the LLaMA-based Hindi back-translation operator for the evolutionary algorithm.
This mutation operator performs back-translation using LLaMA and Hindi as the intermediate language.

Author: EOST CAM LLM Team
Version: 1.0
"""

from .base_operators import _GenericLLMBackTranslationOperator


class LLMBackTranslationHIOperator(_GenericLLMBackTranslationOperator):
    """
    LLaMA-based Hindi back-translation operator.

    This operator performs back-translation using LLaMA and Hindi as the intermediate language.
    It translates the input text from English to Hindi and then back to English using LLaMA.

    Attributes:
        Inherits attributes from _GenericLLMBackTranslationOperator.

    Example:
        >>> operator = LLMBackTranslationHIOperator()
        >>> variants = operator.apply("Write a story about a brave knight")
        >>> print(variants)
        ['Write a story about a brave knight', 'Write a story about a courageous warrior']
    """

    def __init__(self, log_file=None):
        """
        Initialize the LLaMA-based Hindi back-translation operator.
        
        Args:
            log_file (str, optional): Path to log file for debugging. Defaults to None.
            
        Note:
            Uses cached LLaMA generator for translation.
        """
        super().__init__(
            name="LLMBackTranslation_HI",
            target_lang="Hindi",
            target_lang_code="hi",
            log_file=log_file
        )

