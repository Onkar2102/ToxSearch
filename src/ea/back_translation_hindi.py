"""
back_translation_hindi.py

Author: Onkar Shelar os9660@rit.edu

This module contains the Hindi back-translation operator for the evolutionary algorithm.
This mutation operator performs back-translation using Hindi as the intermediate language.

Author: EOST CAM LLM Team
Version: 1.0
"""

from .base_operators import _GenericBackTranslationOperator


class BackTranslationHIOperator(_GenericBackTranslationOperator):
    """
    Hindi back-translation operator.

    This operator performs back-translation using Hindi as the intermediate language.
    It translates the input text from English to Hindi and then back to English to
    generate diverse variants.

    Attributes:
        Inherits attributes from _GenericBackTranslationOperator.

    Example:
        >>> operator = BackTranslationHIOperator()
        >>> variants = operator.apply("Write a story about a brave knight")
        >>> print(variants)
        ['Write a story about a brave knight', 'Write a story about a courageous warrior']
    """

    def __init__(self, log_file=None):
        """
        Initialize the Hindi back-translation operator.
        
        Args:
            log_file (str, optional): Path to log file for debugging. Defaults to None.
            
        Note:
            Uses Helsinki-NLP translation models for EN↔Hindi translation.
        """
        super().__init__(
            name="BackTranslation_HI",
            lang_code="hi",
            en_to_lang_repo="Helsinki-NLP/opus-mt-en-hi",
            lang_to_en_repo="Helsinki-NLP/opus-mt-hi-en",
            pipeline_task_en_to_lang="translation_en_to_hi",
            pipeline_task_lang_to_en="translation_hi_to_en",
            description_suffix="Hindi back-translation.",
            log_file=log_file,
        )

