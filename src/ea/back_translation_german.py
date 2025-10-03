"""
back_translation_german.py

This module contains the German back-translation operator for the evolutionary algorithm.
This mutation operator performs back-translation using German as the intermediate language.
"""

from .base_operators import _GenericBackTranslationOperator


class BackTranslationDEOperator(_GenericBackTranslationOperator):
    """
    German back-translation operator.

    This operator performs back-translation using German as the intermediate language.
    It translates the input text from English to German and then back to English to
    generate diverse variants.

    Attributes:
        Inherits attributes from _GenericBackTranslationOperator.

    Example:
        >>> operator = BackTranslationDEOperator()
        >>> variants = operator.apply("Write a story about a brave knight")
        >>> print(variants)
        ['Write a story about a brave knight', 'Write a story about a courageous warrior']
    """

    def __init__(self, log_file=None):
        """
        Initialize the German back-translation operator.
        
        Args:
            log_file (str, optional): Path to log file for debugging. Defaults to None.
            
        Note:
            Uses Helsinki-NLP translation models for EN↔German translation.
        """
        super().__init__(
            name="BackTranslation_DE",
            lang_code="de",
            en_to_lang_repo="Helsinki-NLP/opus-mt-en-de",
            lang_to_en_repo="Helsinki-NLP/opus-mt-de-en",
            pipeline_task_en_to_lang="translation_en_to_de",
            pipeline_task_lang_to_en="translation_de_to_en",
            description_suffix="German back-translation.",
            log_file=log_file,
        )

