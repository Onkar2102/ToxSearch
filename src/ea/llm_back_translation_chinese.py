"""
llm_back_translation_chinese.py

This module contains the LLaMA-based Chinese back-translation operator for the evolutionary algorithm.
This mutation operator performs back-translation using LLaMA and Chinese as the intermediate language.
"""

from .base_operators import _GenericLLMBackTranslationOperator


class LLMBackTranslationZHOperator(_GenericLLMBackTranslationOperator):
    """
    LLaMA-based Chinese back-translation operator.

    This operator performs back-translation using LLaMA and Chinese as the intermediate language.
    It translates the input text from English to Chinese and then back to English using LLaMA.

    Attributes:
        Inherits attributes from _GenericLLMBackTranslationOperator.

    Example:
        >>> operator = LLMBackTranslationZHOperator()
        >>> variants = operator.apply("Write a story about a brave knight")
        >>> print(variants)
        ['Write a story about a brave knight', 'Write a story about a courageous warrior']
    """

    def __init__(self, log_file=None):
        """
        Initialize the LLaMA-based Chinese back-translation operator.
        
        Args:
            log_file (str, optional): Path to log file for debugging. Defaults to None.
            
        Note:
            Uses cached LLaMA generator for translation.
        """
        super().__init__(
            name="LLMBackTranslation_ZH",
            target_lang="Chinese",
            target_lang_code="zh",
            log_file=log_file
        )

