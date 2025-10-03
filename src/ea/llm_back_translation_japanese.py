"""
llm_back_translation_japanese.py

This module contains the LLaMA-based Japanese back-translation operator for the evolutionary algorithm.
This mutation operator performs back-translation using LLaMA and Japanese as the intermediate language.
"""

from .base_operators import _GenericLLMBackTranslationOperator


class LLMBackTranslationJAOperator(_GenericLLMBackTranslationOperator):
    """
    LLaMA-based Japanese back-translation operator.

    This operator performs back-translation using LLaMA and Japanese as the intermediate language.
    It translates the input text from English to Japanese and then back to English using LLaMA.

    Attributes:
        Inherits attributes from _GenericLLMBackTranslationOperator.

    Example:
        >>> operator = LLMBackTranslationJAOperator()
        >>> variants = operator.apply("Write a story about a brave knight")
        >>> print(variants)
        ['Write a story about a brave knight', 'Write a story about a courageous warrior']
    """

    def __init__(self, log_file=None):
        """
        Initialize the LLaMA-based Japanese back-translation operator.
        
        Args:
            log_file (str, optional): Path to log file for debugging. Defaults to None.
            
        Note:
            Uses cached LLaMA generator for translation.
        """
        super().__init__(
            name="LLMBackTranslation_JA",
            target_lang="Japanese",
            target_lang_code="ja",
            log_file=log_file
        )

