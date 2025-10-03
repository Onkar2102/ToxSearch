"""
back_translation_japanese.py

Author: Onkar Shelar os9660@rit.edu

This module contains the Japanese back-translation operator for the evolutionary algorithm.
This mutation operator performs back-translation using Japanese as the intermediate language.

Author: EOST CAM LLM Team
Version: 1.0
"""

from .base_operators import _GenericBackTranslationOperator


class BackTranslationJAOperator(_GenericBackTranslationOperator):
    """
    Japanese back-translation operator.

    This operator performs back-translation using Japanese as the intermediate language.
    It translates the input text from English to Japanese and then back to English to
    generate diverse variants.

    Attributes:
        Inherits attributes from _GenericBackTranslationOperator.

    Example:
        >>> operator = BackTranslationJAOperator()
        >>> variants = operator.apply("Write a story about a brave knight")
        >>> print(variants)
        ['Write a story about a brave knight', 'Write a story about a courageous warrior']
    """

    def __init__(self, log_file=None):
        """
        Initialize the Japanese back-translation operator.
        
        Args:
            log_file (str, optional): Path to log file for debugging. Defaults to None.
            
        Note:
            Uses Helsinki-NLP translation models for EN↔Japanese translation.
        """
        super().__init__(
            name="BackTranslation_JA",
            lang_code="ja",
            en_to_lang_repo="Helsinki-NLP/opus-mt-en-jap",
            lang_to_en_repo="Helsinki-NLP/opus-mt-jap-en",
            pipeline_task_en_to_lang="translation_en_to_ja",
            pipeline_task_lang_to_en="translation_ja_to_en",
            description_suffix="Japanese back-translation.",
            log_file=log_file,
        )

