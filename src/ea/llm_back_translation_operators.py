"""
Combined llm_back_translation_operators.py

This module contains all LLM-based back-translation operators for the evolutionary algorithm.
Supported languages: Hindi, French, German, Japanese, Chinese
"""


# Standalone base and helpers
from .VariationOperators import VariationOperator
from .operator_helpers import get_generator
import logging
get_logger = logging.getLogger

class _GenericLLMBackTranslationOperator(VariationOperator):
    """
    Generic LLaMA-based back-translation operator (EN → target_lang → EN).
    This operator uses the cached LLaMA generator from `get_generator()` and its
    task-specific translate() method to:
    1) Translate English to a target language using config templates
    2) Translate back to English with natural phrasing
    It logs the intermediate translated text and returns up to 3 unique variants.
    Subclasses should pass appropriate language names and codes.
    """
    def __init__(self, name: str, target_lang: str, target_lang_code: str, log_file=None):
        super().__init__(name, "mutation", f"LLaMA-based EN→{target_lang_code.upper()}→EN back-translation.")
        self.logger = get_logger(self.name)
        self.target_lang = target_lang
        self.target_lang_code = target_lang_code
        self.generator = get_generator()

    def apply(self, text: str) -> list:
        self._last_input = text
        self._last_intermediate = None
        self._last_final = None
        try:
            inter = self.generator.translate(text, self.target_lang, "English")
            self._last_intermediate = inter
            if inter and inter != text:
                back_en = self.generator.translate(inter, "English", self.target_lang)
                cleaned = back_en.strip()
                self._last_final = cleaned
                if cleaned and cleaned.lower() != text.strip().lower():
                    return [cleaned]
        except Exception:
            pass
        self._last_final = text
        return [text]

class LLMBackTranslationHIOperator(_GenericLLMBackTranslationOperator):
    """LLaMA-based Hindi back-translation operator."""
    def __init__(self, log_file=None):
        super().__init__(
            name="LLMBackTranslation_HI",
            target_lang="Hindi",
            target_lang_code="hi",
            log_file=log_file,
        )

class LLMBackTranslationFROperator(_GenericLLMBackTranslationOperator):
    """LLaMA-based French back-translation operator."""
    def __init__(self, log_file=None):
        super().__init__(
            name="LLMBackTranslation_FR",
            target_lang="French",
            target_lang_code="fr",
            log_file=log_file,
        )

class LLMBackTranslationDEOperator(_GenericLLMBackTranslationOperator):
    """LLaMA-based German back-translation operator."""
    def __init__(self, log_file=None):
        super().__init__(
            name="LLMBackTranslation_DE",
            target_lang="German",
            target_lang_code="de",
            log_file=log_file,
        )

class LLMBackTranslationJAOperator(_GenericLLMBackTranslationOperator):
    """LLaMA-based Japanese back-translation operator."""
    def __init__(self, log_file=None):
        super().__init__(
            name="LLMBackTranslation_JA",
            target_lang="Japanese",
            target_lang_code="ja",
            log_file=log_file,
        )

class LLMBackTranslationZHOperator(_GenericLLMBackTranslationOperator):
    """LLaMA-based Chinese back-translation operator."""
    def __init__(self, log_file=None):
        super().__init__(
            name="LLMBackTranslation_ZH",
            target_lang="Chinese",
            target_lang_code="zh",
            log_file=log_file,
        )
